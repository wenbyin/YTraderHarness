"""
用大模型给候选打分（`RerankerType.LLM`）。

它和 cross-encoder 的分工
------------------------
两者都是"把 query 与正文放一起看"的第二路信号（与融合那套"被几路命中"
无关），区别在成本形状：

    cross-encoder   本地 0.3s/条，只懂相关性，分数稳定但读不懂领域约定
    llm（这一份）   一次调用读一批，能带上"这批语料里什么算相关"的提示

所以这条提示词里除了"打分"，还要写清**这类问题的判别词在哪**：这批语料是
上市公司公告，同一个模板（「自YYYY-MM-DD起，X对Y进行担保，担保金额为N
万元」）会在很多块里重复出现 —— 真正把两条候选区分开的往往只有一个日期
或一个人名。模型如果只看"长得像不像"就会给一堆同分，精排等于没做。

⚠️ 分数必须**有区分度**（见下面"三"）：全打 7 分等于原序返回。

输出契约写在这里（不设 response_format）
------------------------------------
与出题/改写那两份同理：JSON 形状写在提示词里，回包由调用方用 pydantic
校验；对不上就**整批降级**（原序返回），而不是按位置硬配 —— 见
`retrieve/rerank_llm.py` 的说明。
"""

from __future__ import annotations

from typing import Any

from app.agent.model import ModelRole
from app.agent.prompt.contracts import PromptDefinition

__all__ = [
    "PROMPT",
    "render_user_message",
]

#: 每条候选在提示词里最多给多少字。
#:
#: 与 `rerank.py` 的 `_MAX_CHARS` 同量同理由：打分看的是"这条讲的是不是
#: 这件事"，前几百字足够；给全篇只会把 token 花在无关的后续条目上。
MAX_CHARS_PER_CANDIDATE = 400

TEXT = """\
你在为一个中文检索系统做**候选相关性打分**。

给一个问题和若干条候选正文，你要给**每一条**打一个 0~10 的相关性分数：
这条正文里有没有能回答该问题的信息。

一、只按"能不能回答问题"打分，不按别的
    - 正文长、格式漂亮、有很多数字 —— 这些都**不加分**
    - 出现相同的词但答不上问题的，**不给高分**
    - 10 = 直接、完整地回答了问题；0 = 完全无关
    - 中间的分要给得细：6 分与 7 分应当真的存在差别

二、这批语料的特点（判断时请利用它）
    它们是上市公司的公告，同一个句式会在很多块里重复出现，例如
    「自2015-06-06起，X公司对Y公司进行担保，担保金额为3.896亿元…」。
    两条候选**看起来几乎一样**时，把它们区分开的通常只有一个：
    **日期**（自哪一天起 / 哪一天披露）或**主体名**（哪家公司、哪个被
    担保方）。所以：
        · 日期或主体与问题对不上的，分数要明显低
        · 日期与主体都对上的那一条，给最高分

三、分数要有区分度（**这一条最容易做错**）
    所有候选打成同一个分数 = 这次打分没有任何作用（下游按分数排序，
    同分保持原序）。除非两条候选确实同等相关，否则**不要给相同的分**。

四、逐条对应，不得串号
    - 输出 `items` 的条数必须与输入候选条数**完全一致**
    - 每条回它自己的编号，且保持输入顺序
    - 拿不准的也要**照样回一条**（给一个你认为最合适的分数），不许省略

输出：**只输出一个 JSON 对象**，不要代码围栏、不要任何其它文字。
    {"items": [{"index": 1, "relevance": 7}, {"index": 2, "relevance": 2}, ...]}
"""


def render_user_message(
        *,
        query: str,
        candidates: list[str],
) -> str:
    """
    用户消息：问题 + 带编号的候选正文。

    候选正文在这里截断（见 `MAX_CHARS_PER_CANDIDATE`）—— 截断是**提示词
    的一部分**，不是调用方的事：给多少正文直接决定这次调用读得懂什么。
    """

    lines = [
        f"问题：{query}",
        "",
        f"候选正文（共 {len(candidates)} 条）：",
        "",
    ]

    for index, content in enumerate(candidates, start=1):
        lines.append(f"--- {index} ---")
        lines.append(content[:MAX_CHARS_PER_CANDIDATE])
        lines.append("")

    lines.append(
        f"请给上面 {len(candidates)} 条各打一个 0~10 的相关性分数，"
        "按系统提示里的规则输出 JSON（条数与编号必须一致）。"
    )

    return "\n".join(lines)


def _render(**_: Any) -> str:
    """系统提示词无变量，可变部分全在用户消息里。"""

    return TEXT


PROMPT = PromptDefinition(
    key=ModelRole.RERANK,
    description=(
        "检索期精排（LLM 版）：给每条候选打 0~10 的相关性分，"
        "要求有区分度、逐条对应不串号。"
    ),
    render=_render,
)
