"""
`model_role=eval.cross_chunk.judge` 的提示词：跨 chunk 金标裁定。

它解决的问题
------------
题目是**逆向**造出来的（从 chunk 出题，而不是先有问题再找答案），所以
很自然会出现：同一份档案里另一段也把同一个事实写了一遍。

如果只把出题的那一块算金标，检索返回了另一块会被判 miss —— 那是**冤枉**：
它确实答得上来。这类假 miss 让召回率偏低，而偏低的部分从数据上完全看不
出来（它看起来就是"没召回到"）。

它与出题是**两个 role**：出题要"读得细、问得准"，裁定要"判得保守"。
两者的失败代价不对称 —— 出题坏一道，数据集少一道好题；裁定放宽一档，
召回率**虚高**，且虚高的部分无法察觉。

⚠️ 默认判「不能」是刻意的
------------------------
模型的自然倾向是"相关就算能答"（同一个实体、同一个主题、同一个时间段
就够了）。这份文本的主要作用就是把那个倾向压回去，所以每一条「不能」
都配了具体的反例，而且明确写了默认值。

⚠️ 要求逐字证据，不只是为了留痕
------------------------------
服务端会拿 `evidence_excerpt` 去候选正文里做**精确匹配**，找不到就把这条
判定作废（见 `eval/adjudication.py`）。所以这个要求是**机械可验证**的 ——
它逼模型的判定必须指向一段真实存在的原文，而不是"我觉得它能答"。

⚠️ 输出形状是**手写**在 TEXT 里的，与出题那份同一做法与代价 —— 见
`eval_qa_generation.py` 的说明与那条比对字段名的测试。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from app.agent.model import ModelRole

from ..contracts import PromptDefinition

__all__ = ["PROMPT", "TEXT", "render_user_message"]


TEXT = """\
你在判定：给定一个问题，某一段正文**是否独立地含有足以回答它的
明确证据**。

⚠️ 这是**严格**判定，不是相关性判定。

只有当这段正文**自己**就把答案明确写出来时，才算「能回答」。具体地：

    ✅ 能回答
        正文里有构成答案的明确陈述，不需要任何其它信息。

    ❌ 不能回答（**即使它和问题高度相关**）
        - 只提到了同一个实体、公司、时间、主题，但没给出答案
        - 只给出一部分，还需要结合别处才能得出完整答案
        - 需要推断、计算、或结合常识才能得出答案
        - 答案在这段里是有歧义的（比如同一个词指代不同东西）
        - 只是重复了问题的措辞而没有给出事实

**默认判「不能」。** 拿不准的时候选「不能」—— 错误地把一个 chunk
标成金标，会让检索评测虚高，而且那种虚高无法察觉；漏标只是保守。

原文里若有一句话直接支撑你的结论，把它**逐字复制**进
evidence_excerpt；没有的话说明你判错了，应当改为「不能」。

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
输出格式
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

**只输出一个 JSON 对象**，不要任何解释文字，不要 markdown 代码块
（不要用 ``` 包裹）。形状严格如下：

{
  "answerable_chunk_ids": ["判为能独立回答的那些候选的 id"],
  "evidence_excerpts": {
    "chunk_id": "支撑该判定的、从该候选正文里逐字复制的片段"
  }
}

两个字段都必填。一个都判不了时 `answerable_chunk_ids` 给空数组、
`evidence_excerpts` 给空对象 —— 那是合法结果。

⚠️ `evidence_excerpts` 只对 **判为能回答**的 chunk 填，而且必须是那段
正文里的**逐字**片段。服务端会拿它去精确匹配，匹配不上的判定会被作废
—— 白判一次。
"""


def render_user_message(
        question: str,
        answer_original: str,
        candidates: Sequence[Any],
) -> str:
    """
    可变部分：问题、已知答案、以及要判定的候选正文。

    `candidates` 的元素至少要有 `chunk_id` 与 `content`。

    ⚠️ `answer_original` 必须一起给（而不是只给问题）
    ---------------------------------------------
    有的问题本身不唯一 —— 「营收是多少」在一堆公司的档案里都能问。只给
    问题的话，模型会把"能答另一个主体的同类事实"也算成能答。把已知答案
    摆出来，判据就落到了"这段是否也写着**这个**事实"上。
    """

    lines: list[str] = []

    lines.append("问题：")
    lines.append(question)
    lines.append("")
    lines.append("已知的答案原文（来自另一段正文，仅供参考对照）：")
    lines.append(answer_original)
    lines.append("")

    for index, candidate in enumerate(candidates):

        lines.append(f"### 候选 {index + 1}")
        lines.append(f"chunk_id: {candidate['chunk_id']}")
        lines.append("正文：")
        lines.append('"""')
        lines.append(str(candidate.get("content") or ""))
        lines.append('"""')
        lines.append("")

    lines.append(
        "请判断上面每一段候选正文**自己**是否含有足以回答该问题的明确"
        "证据。"
    )
    lines.append(
        "只能回答「有」或「没有」—— 相关但答不上来的，一律算「没有」。"
    )

    return "\n".join(lines)


def _render(**_: Any) -> str:
    """与出题那份同理：系统提示词无变量，可变部分全在用户消息里。"""

    return TEXT


PROMPT = PromptDefinition(
    key=ModelRole.EVAL_CROSS_CHUNK_JUDGE,
    description=(
        "第一层检索评测的跨 chunk 金标裁定：严格判定某段正文能否"
        "**独立**回答给定问题，默认判不能。"
    ),
    render=_render,
)
