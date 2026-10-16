"""
把抽中的 chunk 切成若干次 LLM 调用。

为什么**两个**约束都要有
------------------------
只按 chunk 个数分组（L 个一组）是不够的：不同 chunk 的字符数能差一个
数量级 —— 表格段几千字，事件段几十字。按个数分组会让某一批直接把
上下文撑爆，而症状是「那个窗口每次都失败」，看起来像模型的问题。

只按 token 分组也是不够的：token 预算只约束**容量**，不约束**并行度**。
一个全是短 chunk 的语料会把几百个 chunk 塞进一次调用，模型对「请给
每一段各出 3 道题」这种长清单的遵守度会明显下降（漏段、错配 id）。

所以两个都留着，取更紧的那个。

确定性
------
切分是**纯函数**：给定同样的输入与参数，永远得到同样的窗口划分。
这不是可有可无的性质 —— 重试一个失败窗口时靠的是 `window_index`
定位，如果重新切分得到的划分与首轮不同，那个序号指向的就是别的
chunk 集合，重试等于往数据集里塞另一批数据。

超大单 chunk 的处理
-------------------
单个 chunk 就超过预算时，**跳过它并记下原因**，绝不截断。

截断是这里最诱人的错误做法：它能让流程继续跑、看起来一切正常，但
截断后的正文里可能就不含答案了 —— 而数据集里那道题仍然记着"答案在
这一块"。于是检索把这块排第一也判不中，召回率莫名偏低，且没有任何
报错指向截断。宁可少一个 chunk（它会在统计里显示成 skipped），
也不要一个错的目标标注。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ..tokenization import token_length

__all__ = [
    "DEFAULT_ENCODING_NAME",
    "INSTRUCTION_RESERVE_TOKENS",
    "OUTPUT_TOKENS_PER_QUESTION",
    "PER_CHUNK_OVERHEAD_TOKENS",
    "SkippedChunk",
    "WindowPlan",
    "default_token_counter",
    "plan_windows",
]


#: token 计数的编码名。
#:
#: ⚠️ 取 `o200k_base` 是**跟着正文的实际切分口径走**的：文本 chunker
#: （chunker/txt.py）就是用它度量 chunk 长短的。换一个编码去算预算，
#: 得到的是另一个口径下的数，而两边不一致时不会有任何报错 ——
#: 只是某个窗口突然装不下、或者装下了却超了。
DEFAULT_ENCODING_NAME = "o200k_base"

#: 给系统提示 + JSON Schema + 输出格式说明留的余量。
#:
#: 这个数不是估的：出题的系统提示本身就有两千多字（禁止清单很长），
#: 加上 `WindowQuestions` 的 JSON Schema。**低估它的后果是每个窗口都
#: 刚好超一点点** —— 而 API 报的是"超出上下文长度"，看起来像语料
#: 太大，不会有人怀疑到提示词头上。
INSTRUCTION_RESERVE_TOKENS = 2_400

#: 每道题期望的输出 token（问题 + 答案 + JSON 结构开销）。
OUTPUT_TOKENS_PER_QUESTION = 120

#: 每个 chunk 在输出里的固定结构开销（chunk_id 字段等）。
PER_CHUNK_OVERHEAD_TOKENS = 40

#: 计数函数：文本 -> token 数。可注入是为了测试不必依赖 tiktoken
#: （它首次调用会联网下载词表，见 tokenization.encoding_for 的说明）。
TokenCounter = Callable[[str], int]


def default_token_counter(text: str) -> int:
    """按仓库既有口径数 token。"""

    return token_length(text, DEFAULT_ENCODING_NAME)


@dataclass(frozen=True)
class SkippedChunk:
    """
    没能进任何窗口的 chunk。

    带上原因而不是静默丢掉：`skipped` 是「为什么实际参与的 chunk 比
    抽中的少」的唯一答案，而那个差额直接影响数据集的覆盖面。
    """

    chunk_id: Any
    reason: str


@dataclass
class WindowPlan:
    """
    一个窗口的划分方案。

    `chunks` 是**快照**（含正文），不是 id 列表 —— 理由见
    models/sqlalchemy/retrieval_eval.py 里 generation_window 的说明。
    """

    window_index: int
    chunks: list[dict[str, Any]] = field(default_factory=list)
    token_estimate: int = 0

    @property
    def chunk_ids(self) -> list[Any]:
        return [chunk["chunk_id"] for chunk in self.chunks]


def plan_windows(
        chunks: list[dict[str, Any]],
        *,
        chunks_per_window: int,
        max_input_tokens: int,
        questions_per_chunk: int,
        count_tokens: TokenCounter = default_token_counter,
) -> tuple[list[WindowPlan], list[SkippedChunk]]:
    """
    切窗口。返回 `(窗口列表, 被跳过的 chunk)`。

    贪心按序装填：顺序由调用方决定（抽样那一层已经排好），因此结果
    是确定的。

    **为什么保留原顺序而不是排序后装箱**：排序（比如按长度降序）能把
    箱子装得更满，但会让窗口与"文档里的相邻关系"脱钩 —— 而相邻的
    chunk 往往属于同一份文档，判金标那一阶段要拿同窗口的邻居做候选，
    排序装箱会让候选变得毫无关联。省下的那点调用次数不值得。

    预算算法（每个 chunk 的代价）：

        cost = 正文 token + 每 chunk 结构开销
                          + 每 chunk 的期望输出 token

    输出也要算进预算，因为它是**同一个**上下文窗口的一部分 —— 只算
    输入的话，一个装得很满的窗口会因为输出放不下而被截断，而截断的
    表现是 JSON 不完整、整个窗口解析失败。
    """

    if chunks_per_window < 1:
        raise ValueError("chunks_per_window 至少为 1。")

    if max_input_tokens < 1:
        raise ValueError("max_input_tokens 至少为 1。")

    #: 输入侧真正能用的预算。
    available = max_input_tokens - INSTRUCTION_RESERVE_TOKENS

    #: 单个 chunk 的固定输出开销（与它内容长短无关）。
    per_chunk_output = (
        PER_CHUNK_OVERHEAD_TOKENS
        + questions_per_chunk * OUTPUT_TOKENS_PER_QUESTION
    )

    windows: list[WindowPlan] = []
    skipped: list[SkippedChunk] = []

    current: list[dict[str, Any]] = []
    current_tokens = 0

    def flush() -> None:

        nonlocal current, current_tokens

        if not current:
            return

        windows.append(
            WindowPlan(
                window_index=len(windows),
                chunks=current,
                token_estimate=current_tokens,
            )
        )

        current = []
        current_tokens = 0

    for chunk in chunks:

        content = str(chunk.get("content") or "")

        content_tokens = count_tokens(content)
        cost = content_tokens + per_chunk_output

        # ---- 单个就超预算：跳过，不截断 ----
        #
        # 判据是"这一个 chunk 独占一个窗口也放不下"，而不是"加上它
        # 这个窗口就满了"—— 后者只是该换窗口，前者才是真的装不下。
        if cost > available:

            flush()

            skipped.append(
                SkippedChunk(
                    chunk_id=chunk.get("chunk_id"),
                    reason=(
                        f"单个 chunk 超出输入预算：正文约 {content_tokens} "
                        f"token + 预留输出 {per_chunk_output}，"
                        f"而可用预算 {available}"
                        f"（max_input_tokens={max_input_tokens} 减去指令预留 "
                        f"{INSTRUCTION_RESERVE_TOKENS}）。"
                        "不截断正文是刻意的 —— 截断后正文可能已不含答案，"
                        "而数据集仍会记着答案在这一块。"
                    ),
                )
            )

            continue

        # ---- 装不下就换窗口 ----
        if current and (
                len(current) >= chunks_per_window
                or current_tokens + cost > available
        ):
            flush()

        current.append(chunk)
        current_tokens += cost

    flush()

    return windows, skipped
