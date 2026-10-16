"""
第一层检索评测的离线内核：抽样、切窗口、出题、校验、跨 chunk 判金标。

这一包**只做数据集侧**，不含评测执行（那在 application 层，且不需要
LLM）。分模块的理由是每块各有自己的失败模式与可测边界：

    sampling.py      从候选池里抽哪 N 个 chunk（确定性，纯函数）
    windowing.py     把这 N 个切成若干次 LLM 调用（L + token 双约束，纯函数）
    output.py        模型输出的形状与解析
    generator.py     单窗口：调 ModelRuntime -> 解析 -> 服务端校验
    adjudication.py  跨 chunk 判金标

提示词**不在这里**
------------------
两份系统提示词住在 `app/agent/prompt/prompts/eval_qa_generation.py` 与
`eval_cross_chunk_judge.py`，按 `model_role` 注册、由 prompt provider 取。
理由与本仓库其它模型调用一致：提示词是**配置驱动**的（改它不该动服务层
代码），而"取哪个提示词"由这次调用用的是哪个 role 决定。

所以本包只负责"取"，不负责"写" —— 这是 prompts/__init__.py 立下的分工。

为什么校验（output.py + generator.py 的 validate_items）要单独可测
----------------------------------------------------------------
它是这一层唯一能**机械判定**的部分（id 认不认识、答案是不是逐字子串、
问题重不重复），而模型输出的合法性不能只靠"JSON 解析成功"来推断。
把它做成纯函数，就能脱离 LLM 反复验证 —— 这类规则一旦错了，数据集会
悄悄带上错误的 ground truth，不报错。
"""

from .adjudication import (
    ADJUDICATE_AGENT_TYPE,
    AdjudicationOutcome,
    GoldCandidate,
    adjudicate_question,
)
from .generator import (
    EVAL_AGENT_TYPE,
    GeneratedCase,
    RejectedItem,
    WindowOutcome,
    generate_for_window,
    normalize_question,
    validate_items,
)
from .output import (
    AdjudicationVerdict,
    ModelOutputError,
    QAGenerationItem,
    WindowQuestions,
    parse_adjudication_verdict,
    parse_window_questions,
)
from .sampling import sample_chunks
from .windowing import SkippedChunk, WindowPlan, plan_windows

__all__ = [
    "ADJUDICATE_AGENT_TYPE",
    "EVAL_AGENT_TYPE",
    "AdjudicationOutcome",
    "AdjudicationVerdict",
    "GeneratedCase",
    "GoldCandidate",
    "ModelOutputError",
    "QAGenerationItem",
    "RejectedItem",
    "SkippedChunk",
    "WindowOutcome",
    "WindowPlan",
    "WindowQuestions",
    "adjudicate_question",
    "generate_for_window",
    "normalize_question",
    "parse_adjudication_verdict",
    "parse_window_questions",
    "plan_windows",
    "sample_chunks",
    "validate_items",
]
