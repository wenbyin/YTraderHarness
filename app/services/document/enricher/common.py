"""
通用文本 Chunk 增强：把文档级上下文继承到每个 chunk。

解决的问题
----------
按公司维度组织的一份长文档，中间的纯数据 chunk（一长串股东户数、表格
行、时间线）**正文里不会重复公司名**。于是查询"东方财富 2026年4月
股东户数"时，"东方财富"这个条件在那些 chunk 上**根本不可见** ——
它们只能靠"股东户数"匹配一半，在融合排序里输给恰好提到公司名的
摘要/页眉 chunk。

增强阶段把这些信息**从文档级继承到每条 chunk 的 metadata**，让查询侧
能把它当**过滤条件**用（而不是靠相似度碰运气）。

    增强前：{content: "截止2026-03-31，公司A股股东户数为1024127户…"}
    增强后：metadata 里多了 title / company_code / source

格式识别：后缀名优先，内容兜底
------------------------------
"按格式走不同规则"需要一个格式判断，而它有**两个来源**，可靠性不同：

    1. 文件名后缀（document_metadata.file_name）—— 可信，但可能缺
    2. 正文形状嗅探                        —— 兜底，可能误判

所以顺序是**先后缀、后嗅探**，且嗅探只在后缀缺失或认不出时才跑。
两者都不敢把话说死：认不出就用 `plain` 规则，不报错 —— 增强是
锦上添花，不该因为"这是个没见过的格式"就把整条流水线搞挂。

支持的格式与各自的规则
----------------------
    plain   .txt            只加标题/来源/业务标识
    markdown .md            + 从正文里认出的最近标题（作为 section）
    json    .json           + 顶层键名（作为 section）
    yaml    .yaml/.yml      + 顶层键名
    xml     .xml            + 根元素名
    html    .html/.htm      + <title>，去掉标签后取最近标题

后继标题（section）为什么有用：一份 250+ chunk 的文档，"第 3 节的第
12 条"这种定位光靠 chunk_index 是读不出来的，而 section 能让人和
LLM 都立刻知道"这块讲的是什么"。

产出什么
--------
    metadata     继承的文档级键 + section
    locator      归属定位（chunk_index / section / document_title）
    embed_text   前缀 + 空格 + 正文，专供阶段 9 的向量化

前两者是对既有 chunk 的补全，第三个是**派生字段**。

不改 content
------------
本实现**绝不碰 content 与 chunk_index** —— 理由见 base.py 的说明
（唯一约束与 content_hash）。

`embed_text` 正是"为了向量召回补上下文"的落点，但它是独立字段而不是
把前缀并进正文。这个区别不是洁癖：`content == 正文[start_index:end_index]`
是硬不变式，qa/overlap.py 的区间求交与 eval 的 offset 切片都依赖它，
并进去会**静默错位**。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .base import DocumentChunkEnricher

__all__ = ["TxtCommonChunkEnricher"]


#: 后缀名 -> 格式。
_EXTENSION_FORMATS: dict[str, str] = {
    ".txt": "plain",
    ".text": "plain",
    ".md": "markdown",
    ".markdown": "markdown",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".xml": "xml",
    ".html": "html",
    ".htm": "html",
}

#: metadata 里要继承下去的**文档级**键。
#:
#: 白名单而不是"整个 document_metadata 拷一份"：那份里混着产物级的
#: 东西（checksum / byte_size / file_id 之类），逐条 chunk 复制只会
#: 把每行的 metadata 撑大，还让查询侧的过滤条件无从下手。
_INHERITED_KEYS: tuple[str, ...] = (
    "title",
    "company_code",
    "company_name",
    "source",
    "file_name",
    "document_key",
)

#: embed_text 前缀的元素，**顺序即输出顺序**。
#:
#: 五者都是"这条 chunk 属于谁"，而正文里往往不重复它们 —— 一份按公司
#: 组织的长文档，中间那些纯数据 chunk（一长串股东户数）正文里没有公司名，
#: 查询带上公司名时，向量那两路在 chunk 层面根本看不见这个条件。
#:
#: ⚠️ 与 indexing.py 的 _CONTEXT_KEYS（喂 BM25 的 context_text）、上面的
#: _INHERITED_KEYS（继承白名单）是**三份独立的键表**，刻意不合并：
#:
#:     _INHERITED_KEYS   写：哪些文档级键要拷进 chunk.metadata   （无序，存在即可）
#:     _CONTEXT_KEYS     读：拼给 ES 做 BM25 的上下文           （顺序 = 权重语义）
#:     _EMBED_PREFIX_KEYS 读：拼给向量模型的输入                （顺序 = 输出顺序）
#:
#: 合并它们会让"往继承白名单加一个键"顺带改掉 BM25 的权重或前缀的顺序，
#: 那种耦合不报错、只在检索质量上悄悄体现。
_EMBED_PREFIX_KEYS: tuple[str, ...] = (
    # 公司名/主体：**这条最要紧**，而且是后来补的。
    #
    # 此前前缀里只有 `file_name`（`07_风险、法律与上市状态.md`），而这家
    # 语料的公司名住在 **title**（`荣安地产·风险、法律与上市状态`）——
    # 于是向量里根本没有公司名，而问句里几乎必带公司名。实测后果：问
    # 「荣安地产2007年4月12日…」时，那条正文里不写公司名的金标**连
    # dense 的 top-200 都进不去**（只被 keyword 召回，于是在融合里吃单路
    # 命中的亏）。
    #
    # BM25 那一侧一直看得见它（`_CONTEXT_KEYS` 里有 title）—— 也就是说
    # 两条路此前**看到的上下文不是同一份**。
    "company_name",  # 公司名（元数据里有则带上）
    "title",         # 文档标题，通常含公司名
    "file_name",     # 文件名
    "section",       # 本阶段从正文标题推出来
    "dtd_name",      # DTD 的 "名称(编码)"，见 metadata_extraction
    "document_key",  # business_key
)

#: Markdown / HTML 里的标题行。
_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_TITLE = re.compile(
    r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL
)


class TxtCommonChunkEnricher(DocumentChunkEnricher):
    """
    通用文本增强：继承文档级元数据 + 补定位。

    名字里的 TXT 表示它服务**文本类**内容（含 md/json/xml/…），
    与 TEXT_TXT_COMMON_CLEANER / TEXT_TXT_RECURSIVE_CHUNKER 同一个
    命名逻辑。键一旦发布**永不复用**（会冻结进 PipelineConfigSnapshot）。
    """

    name = "TEXT_COMMON_CHUNK_ENRICHMENT"


    async def enrich(
            self,
            chunks: list,
            *,
            config: dict[str, Any],
            document_metadata: dict[str, Any],
    ) -> list:
        """
        一批 chunk 就地增强，返回同一个列表。

        顺序不变（按下标一一对应），content / chunk_index 不动。

        产出三样东西，都挂在 metadata / locator / embed_text 上：
            继承的文档级元数据、section（从正文标题推）、embed_text
            （喂阶段 9 的向量输入，见 _EMBED_PREFIX_KEYS）。
        """

        if not chunks:
            return chunks

        inherited = _inherited(document_metadata)

        # ⚠️ 这里**不能**因为 inherited 为空就早退。
        #
        # section 与 embed_text 都是**内容派生**的 —— 前者只看正文里的
        # 标题，后者只需要 section + 正文，两者都不依赖任何文档级元数据。
        # 早退会让它们一起消失，而且不报错（实测 document_metadata={} 时
        # markdown 的 section 也拿不到）。inherited 为空只意味着"没有东西
        # 可继承"，不是"没事可做"。

        fmt = _detect_format(document_metadata, chunks)

        # 每份切分产出内部的 section 是**顺序**关系，所以要按 chunk_index
        # 走一遍并在途中累积，不能各算各的。
        ordered = sorted(
            range(len(chunks)),
            key=lambda i: getattr(chunks[i], "chunk_index", i),
        )

        #: 游标：走到当前位置时"最近见到的那个标题"。
        cursor: str | None = None

        for index in ordered:

            chunk = chunks[index]

            # ⚠️ 两个值：本块的 section，与**游标**（走到块尾时处在哪个
            # 小节）。此前是一个值兼两用，于是"这一块属于哪一节"与"下一块
            # 从哪一节开始"只能取同一个答案 —— 而它们在这批文档上**就是
            # 不同的**：事件按时间倒序排，一个块往往以某件事的正文开头，
            # 那件事的标题被切在**上一块**。
            section, cursor = _section_of(chunk, fmt, cursor)

            # 没有可继承的东西就别白重建一次 dict。
            chunk.metadata = (
                {**inherited, **dict(chunk.metadata or {})}
                if inherited
                else dict(chunk.metadata or {})
            )

            if section:
                chunk.metadata["section"] = section

            # 定位信息：只补**缺的**，不覆盖已有的。
            #
            # chunker 已经写进去的（start_index/end_index 之类）是它
            # 真正知道的事实，这里只是把"属于哪一块"补上，不该反过来
            # 盖掉更精确的值。
            locator = dict(chunk.locator or {})

            locator.setdefault("chunk_index", chunk.chunk_index)

            if section:
                locator.setdefault("section", section)

            title = inherited.get("title")

            if title:
                locator.setdefault("document_title", title)

            chunk.locator = locator

            # 向量输入。放在最后算：它要用到这一步才齐的 metadata 与
            # section，还要用到只在 document_metadata 里的 dtd_name / update。
            chunk.embed_text = _embed_text_of(
                chunk,
                section=section,
                document_metadata=document_metadata,
            )

        return chunks


def _embed_text_of(
        chunk,
        *,
        section: str | None,
        document_metadata: dict[str, Any],
) -> str:
    """
    组装喂给阶段 9 的向量输入：`前缀 + 空格 + 正文`。

    前缀是 _EMBED_PREFIX_KEYS 那几个元素的**裸值**，空格相连 —— 与
    indexing.py 拼 context_text 是同一套约定，不引入第二种拼法。

    缺值的元素**跳过**：plain 文本没有 section，business_key 也可能为空，
    给它们留空占位只会往向量里塞无意义的空白。

    一个元素都凑不出来时退回**正文本身**而不是空串 —— 下游只看这个字段，
    让"没有前缀"与"字段缺失"保持两件事，阶段 9 的观测才分得清。

    前缀放在正文**之前**：查询往往先提"哪家公司/哪份文档"，再提"看什么
    数据"，而这个顺序也是人读的时候的顺序。

    注意这里**没有**把结果拼回 chunk.content —— content 是切分的产物，
    `content == 正文[start_index:end_index]` 是硬不变式，qa/overlap.py 的
    区间求交与 eval 的 offset 切片都靠它，改了会静默错位。
    """

    view = {
        **dict(getattr(chunk, "metadata", None) or {}),
        "section": section,
        # 这两个键**刻意不进 _INHERITED_KEYS** —— 它们只为前缀存在，
        # 没有理由跟着进 chunk.metadata（那会顺带改 ES 的 metadata 字段
        # 与对外 API 的形状）。
        "dtd_name": document_metadata.get("dtd_name"),
        "update": document_metadata.get("update"),
    }

    parts = [
        str(view[key]).strip()
        for key in _EMBED_PREFIX_KEYS
        if view.get(key) not in (None, "")
    ]

    content = str(getattr(chunk, "content", "") or "")

    if not parts:
        return content

    return " ".join(parts) + " " + content


# ============================================================
# 文档级元数据
# ============================================================


def _inherited(
        document_metadata: dict[str, Any],
) -> dict[str, Any]:
    """
    从文档级元数据里挑出要继承的键。

    白名单，且**丢掉空值** —— 一个 `"company_code": None` 继承下去
    只会让查询侧的过滤条件多一个永远匹配不上的字段。
    """

    return {
        key: document_metadata[key]
        for key in _INHERITED_KEYS
        if document_metadata.get(key) not in (None, "")
    }


# ============================================================
# 格式识别
# ============================================================


def _detect_format(
        document_metadata: dict[str, Any],
        chunks: list,
) -> str:
    """
    后缀名优先，认不出再嗅探正文，都不行就当 plain。
    """

    file_name = str(document_metadata.get("file_name") or "")

    suffix = _suffix_of(file_name)

    if suffix in _EXTENSION_FORMATS:
        return _EXTENSION_FORMATS[suffix]

    # 后缀缺失或认不出。**注意这里不用 content 的返回值做判断依据** ——
    # 只有嗅探能确定时才算数，否则落回 plain。
    return _sniff(chunks)


def _suffix_of(file_name: str) -> str:
    """取小写后缀（含点）。没有后缀时返回空串。"""

    if "." not in file_name:
        return ""

    return "." + file_name.rsplit(".", 1)[-1].lower()


def _sniff(chunks: list) -> str:
    """
    按正文形状猜格式。

    只看**开头一小段**：整篇扫一遍既慢又没必要 —— 形状在前几百字符
    里就已经能看出来了，而误判的代价只是选错一组增强规则，不影响正确性。
    """

    head = "\n".join(
        str(getattr(chunk, "content", "") or "")[:400]
        for chunk in chunks[:3]
    ).lstrip()

    if not head:
        return "plain"

    if head.startswith("<"):
        lowered = head[:200].lower()
        return "html" if "<html" in lowered or "<body" in lowered else "xml"

    if head.startswith("{") or head.startswith("["):
        try:
            json.loads(head)
            return "json"
        except (ValueError, TypeError):
            # 截断的 JSON 解析不了是常态 —— 形状对了就认它。
            return "json"

    # `key: value` 开头且没有 markdown 标题，当 yaml。
    first_line = head.splitlines()[0] if head.splitlines() else ""

    if (
        ":" in first_line
        and not first_line.lstrip().startswith("#")
        and not first_line.lstrip().startswith("-")
    ):
        return "yaml"

    if _MD_HEADING.search(head):
        return "markdown"

    return "plain"


# ============================================================
# 逐 chunk 的 section
# ============================================================


def _section_of(
        chunk,
        fmt: str,
        cursor: str | None,
) -> tuple[str | None, str | None]:
    """
    算出这条 chunk 所属的 section，并推进入参游标。

    返回 `(本块的 section, 新的游标)` —— **两个值**，因为它们不是一回事：

        本块的 section   **块首**那个位置处在哪一小节（检索要的就是它：
                          问题是从块首那段正文生成的）
        新的游标         **块尾**那个位置处在哪一小节（下一块要看它）

    此前是一个值兼两用（取块内**最后**一个标题），于是"这一块属于哪一节"
    只能等于"下一块从哪一节开始" —— 而在这批文档上它们是**不同的**：

        #### 2007-04-04 ｜ 对外担保        <- 切在这条的**末尾**
        2007年4月12日，公司披露「对外担保」…   <- 而这条以**正文**开头
        #### 2007-04-06 ｜ 对外担保
        …

    事件按时间**倒序**排，所以块首那件事的标题常常落在上一块里。取块内
    最后一个标题会让金标块的 section 写着**另一个日期**（实测四条深 case
    全部取反：2006-08-29 对 2007-04-12、2014-10-16 对 2015-05-20、
    2015-04-28 对 2015-05-22、2023-04-10 对 2022-02-17），而 section 会进
    `embed_text` 前缀与 ES 的 `context_text` —— 等于往向量里塞一个矛盾日期。
    """

    content = str(getattr(chunk, "content", "") or "")

    if fmt == "markdown":

        found = _MD_HEADING.findall(content)

        if not found:
            # 块内没有标题：首尾都在上一条那个小节里。
            return cursor, cursor

        last = found[-1][1].strip()

        first_match = _MD_HEADING.search(content)

        starts_with_heading = (
            first_match is not None
            and not content[:first_match.start()].strip()
        )

        if starts_with_heading:
            # 块首就是标题：它自己就说明了这一块属于哪一节。
            return found[0][1].strip(), last

        # 块首是正文 —— 覆盖它的小节标题在**上一块**里，就是游标指向的
        # 那个。游标还是空的（文档第一块就以正文开头、且上一块不存在）
        # 时退回块内第一个标题：没有更好的答案了。
        return (cursor or found[0][1].strip()), last

    # 其余格式没有"块内多个小节"这回事：整块一个 section，首尾相同 ——
    # 所以两个返回值取同一个（游标也因此原样前进）。
    if fmt == "html":
        section = _html_section(content) or cursor

    elif fmt in ("json", "yaml"):
        section = _structured_section(content) or cursor

    elif fmt == "xml":
        section = _xml_section(content) or cursor

    else:
        section = cursor

    return section, section


def _html_section(content: str) -> str | None:
    """HTML：优先 <title>，否则第一个 h1~h6 的文本。"""

    title = _HTML_TITLE.search(content)

    if title:
        return _HTML_TAG.sub("", title.group(1)).strip()

    heading = re.search(
        r"<h[1-6][^>]*>(.*?)</h[1-6]>",
        content,
        re.IGNORECASE | re.DOTALL,
    )

    if heading:
        return _HTML_TAG.sub("", heading.group(1)).strip()

    return None


def _xml_section(content: str) -> str | None:
    """XML：根元素名 —— 它通常就是这份数据的类型（events / rows…）。"""

    root = re.search(r"<\s*([A-Za-z_][\w.:-]*)", content)

    return root.group(1) if root else None


def _structured_section(content: str) -> str | None:
    """
    JSON / YAML：顶层键名。

    只取**第一层**的键 —— 嵌套键对"这块讲什么"没有帮助，反而会让
    section 变成一串路径。
    """

    text = content.strip()

    try:
        if text.startswith(("{", "[")):
            data = json.loads(text)
        else:
            import yaml

            data = yaml.safe_load(text)

    except Exception:
        # 截断／不完整是常态（chunk 就是被切开的），猜不出就返回 None，
        # 让调用方沿用上一条的 section。这里**不该报错** —— 增强失败
        # 不该把流水线搞挂。
        return None

    if isinstance(data, dict) and data:
        return str(next(iter(data)))

    if isinstance(data, list) and data and isinstance(data[0], dict):
        first = data[0]
        return str(next(iter(first))) if first else None

    return None
