"""
QA 测试数据集生成的请求与响应。

这是**手动调用**的接口（人工触发一次、看结果），不是管线里的自动步骤 ——
所以它同步返回，没有 job id、没有轮询接口。代价是它不能跑大批量：一次
HTTP 请求挂几小时是不现实的。批量跑批留给将来的异步 job 版。
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from app.common.schemas.base import APIModel

__all__ = [
    "QADatasetGenerateRequest",
    "QADatasetGenerateResponse",
    "QADatasetFailure",
]


#: 一次同步调用允许生成的最大窗口数。
#:
#: 存在的理由不是"技术上跑不动"，而是**同步接口不该挂太久**：窗口数 ×
#: 单次 LLM 耗时就约等于整个请求的时长，而云端 LLM 单次通常在几秒到
#: 几十秒。100 个窗口已经是分钟级，再多就可能撞上反向代理/网关的超时，
#: 表现是"请求失败了但任务其实跑完了"—— 那种失败最难查。
#:
#: 要跑全量（实测某份档案有 1257 个窗口）请加分批调用，或等异步 job 版。
MAX_WINDOWS_PER_CALL = 100


class QADatasetGenerateRequest(APIModel):
    """生成一批 QA 数据集。"""

    document_version_id: UUID = Field(
        description="对哪个文档版本生成。取它的 clean 正文与 chunk 区间。",
    )

    max_windows: int = Field(
        default=10,
        ge=1,
        le=MAX_WINDOWS_PER_CALL,
        description=(
            "本次最多处理多少个**事件窗口**。"
            "\n\n"
            "窗口按 2~4 级标题切出来，一个窗口 = 一条事件"
            "（`#### 2026-03-20 ｜ 股东户数` 那种）。分得细、单次调用快，"
            "但问题只覆盖单条事件。"
            "\n\n"
            "正文最开头那段（文档头，含公司名）不算窗口 —— 它作为**背景**"
            "喂给模型，不参与出题。"
            "\n\n"
            "**默认 10**：实测某份档案切出 1257 个窗口，全量要跑很久，"
            "而同步接口要挂到跑完。先小批量试出效果，再分批放大。"
        ),
    )

    sample_seed: int = Field(
        default=0,
        description=(
            "抽样种子。窗口是从全部窗口里**随机抽** max_windows 个的"
            "（而不是取前 N 个）—— 取前 N 个会永远只看得到文档开头，"
            "那部分的事件类型和中间、结尾完全不同。\n\n"
            "固定种子 = 同样参数抽出同一批窗口，便于比较两次改动的差异。"
        ),
    )

    max_questions_per_window: int = Field(
        default=3,
        ge=1,
        le=100,
        description=(
            "单个窗口最多出几道题。"
            "\n\n"
            "上限放到 100 是给「一条事件里塞了大量可比数字」的场景用的"
            "（比如对外担保那种几千字的表格段，一段里几十个金额、期限、"
            "比例都能单独成题）。\n\n"
            "⚠️ 这是**每窗口**的上限，总题数约等于 窗口数 × 这个值 —— "
            "而耗时与 token 都跟着涨。配合 `max_windows` 一起用。"
        ),
    )

    llm_timeout_seconds: float = Field(
        default=600.0,
        ge=10.0,
        le=3600.0,
        description=(
            "单次 LLM 调用的超时（秒）。**默认 600。**"
            "\n\n"
            "⚠️ 原默认是 120 秒，那是按小窗口定的 —— 实测一个 83 字符的"
            "小节就要 42 秒，思考型模型的耗时随输入长度增长得很快。窗口一大"
            "就会撞上它。表现为「请求超时」，而看不出是哪一层超的。"
            ""
            "\n\n"
            "块调大时记得同步调大它。"
        ),
    )

    concurrency: int = Field(
        default=1,
        ge=1,
        le=16,
        description=(
            "并发调用云端 LLM 的条数。**默认 1。**"
            "\n\n"
            "⚠️ 这个值必须 ≤ 你账号的**并发上限**，而不只是 RPM。两者是"
            "不同的限制，而且并发限制往往小得多：实测某个账号 RPM=3、"
            "**并发上限只有 1** —— 默认并发 4 时每个窗口都直接 429："
            "\n\n"
            "    request reached max organization concurrency: 1\n\n"
            "代价是慢：并发 1 时窗口只能串行。但撞限流后重试"
            "只会更慢，还可能把 key 短时间封掉。**先确认账号上限再调大。**"
        ),
    )

    provider: str | None = Field(
        default=None,
        description=(
            "用哪个 LLM 提供方（见 app/common/llm.yaml）。"
            "None = 用那里的 default_provider。"
        ),
    )

    temperature: float | None = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description=(
            "采样温度。**None（默认）= 根本不传这个参数**，由服务端用它"
            "自己的默认值。"
            "\n\n"
            "为什么默认是不传：temperature 的**合法范围由模型决定**。"
            "实测某个思考型模型只接受 1 —— 传 0 会得到"
            " `400 invalid temperature: only 1 is allowed for this model`，"
            "而且**整个窗口**的生成直接失败，不是一条样本失败。"
            "\n\n"
            "只有确实需要「同一份输入每次出一模一样结果」时才显式指定，"
            "而且要填该模型接受的值。"
        ),
    )


class QADatasetFailure(APIModel):
    """一个窗口没能贡献样本的原因。"""

    section: str
    span: tuple[int, int]
    reason: str


class QADatasetGenerateResponse(APIModel):
    """生成结果。**产物本身在磁盘上**，这里只回路径与统计。"""

    dataset_path: str = Field(
        description="落盘路径。",
    )

    case_count: int = Field(
        description="最终保留下来的样本数。",
    )

    windows_total: int
    windows_ok: int = Field(
        description="LLM 调用成功的窗口数。",
    )
    windows_failed: int = Field(
        description="LLM 调用失败（超时、限流、格式不对）的窗口数。",
    )

    questions_generated: int = Field(
        description="LLM 一共出了多少道题。",
    )
    questions_kept: int = Field(
        description="保留下来的。",
    )
    questions_dropped: int = Field(
        description=(
            "丢弃的。**这个数偏高不一定是坏事** —— 最常见的原因是模型"
            "没有逐字引用原文（它改写了），那种题的目标位置无法确定，"
            "留着只会污染数据集。"
        ),
    )

    drop_reasons: dict[str, int] = Field(
        default_factory=dict,
        description="丢弃原因分类计数，用来判断问题出在哪一环。",
    )

    failures: list[QADatasetFailure] = Field(
        default_factory=list,
        description="失败的窗口明细（最多保留前若干个）。",
    )

    elapsed_seconds: float
