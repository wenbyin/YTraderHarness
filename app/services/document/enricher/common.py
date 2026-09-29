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

不改 content
------------
本实现**只写 metadata 与 locator**，绝不碰 content / chunk_index ——
理由见 base.py 的说明（唯一约束与 content_hash）。

想把上下文拼进正文以改善**向量**召回的话，那是向量化那一步的事
（拼给 embedder 的文本），与这里正交。
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
        """

        if not chunks:
            return chunks

        inherited = _inherited(document_metadata)

        if not inherited:
            # 没有任何可继承的东西 —— 原样返回而不是硬塞一个空壳。
            # 表现与占位实现一致，但这是**真的没料可加**，不是没实现。
            return chunks

        fmt = _detect_format(document_metadata, chunks)

        # 每份切分产出内部的 section 是**顺序**关系，所以要按 chunk_index
        # 走一遍并在途中累积，不能各算各的。
        ordered = sorted(
            range(len(chunks)),
            key=lambda i: getattr(chunks[i], "chunk_index", i),
        )

        section: str | None = None

        for index in ordered:

            chunk = chunks[index]

            section = _section_of(chunk, fmt, section)

            chunk.metadata = {
                **inherited,
                **dict(chunk.metadata or {}),
            }

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

        return chunks


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
        previous: str | None,
) -> str | None:
    """
    算出这条 chunk 所属的 section。

    `previous` 是上一条的 section —— 大多数 chunk 里没有标题，它们
    继承上一条的，所以必须顺序累积（调用方按 chunk_index 遍历）。
    """

    content = str(getattr(chunk, "content", "") or "")

    if fmt == "markdown":
        found = _MD_HEADING.findall(content)
        # 取**最后一个**标题：一条 chunk 可能跨几个小节，末尾那个才是
        # 这条 chunk 收尾时所在的位置，对"这块讲什么"更有代表性。
        return found[-1][1].strip() if found else previous

    if fmt == "html":
        return _html_section(content) or previous

    if fmt in ("json", "yaml"):
        return _structured_section(content) or previous

    if fmt == "xml":
        return _xml_section(content) or previous

    return previous


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
