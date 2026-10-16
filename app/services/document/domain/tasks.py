"""
检索评测的后台任务参数。

`Task.params` 必须是 pydantic 模型（裸 dict 会被 submit 拒绝，见
app/common/task/models.py 的 `_reject_bare_dict`），而且**只放业务 ID 与
执行所需的最小参数** —— 大状态按 ID 在执行阶段重新加载。

这条纪律在这里的落点很具体：**不要把 chunk 正文塞进来**。

    ✗ 带 chunks      抽中的 chunk 正文已经在
                     `retrieval_eval_generation_window.input_chunks` 里
                     （那是权威的输入快照），任务里再存一份就是从第二个
                     地方回答"当时喂给模型的是什么"。而且 Task 会被反复
                     读出（Dispatcher 每轮轮询读、Worker claim 读、查询
                     接口读、重试读）—— 塞正文等于把最热的那些查询全变重。
    ✓ 带 run_id      执行阶段按它把窗口与快照取回来。

⚠️ handler 名字是**持久化数据**
---------------------------
`handler="retrieval_eval.generate"` 这个字符串会按字面存进 tasks 表。
改名 = 让库里已有的行解析失败（判 FAILED）。要改就得新旧名字都注册一段
时间 —— 见 app/infra/task/registry.py。

⚠️ 命名空间是 `retrieval_eval.*`（表名），不是 `eval.*`
----------------------------------------------------
`eval.*` 在 `app/agent/model/types.py` 里是 **model_role** 的命名空间
（`eval.qa_generation`）。两者刻意不共用前缀：handler 是"跑哪条任务"，
model_role 是"这次调模型干什么"，一条任务里可能有多次不同 role 的调用。
混用会让"改 handler 名"看起来像"改模型角色"，而后者不该影响持久化数据。
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "RETRIEVAL_EVAL_EVALUATE_HANDLER",
    "RETRIEVAL_EVAL_GENERATE_HANDLER",
    "RetrievalEvalEvaluateParams",
    "RetrievalEvalGenerateParams",
]


#: 造题任务的逻辑名。稳定契约，不要跟着模块改名走。
RETRIEVAL_EVAL_GENERATE_HANDLER = "retrieval_eval.generate"

#: 评测执行任务的逻辑名。
#:
#: 与造题是**两条独立的任务**：造题要 LLM（分钟级、按窗口重试），评测
#: 不用 LLM（只调检索、按 case 判命中）。混成一条会让"只想重跑评测"
#: 变成"重新出一遍题" —— 那是完全不同的成本。
RETRIEVAL_EVAL_EVALUATE_HANDLER = "retrieval_eval.evaluate"


class RetrievalEvalGenerateParams(BaseModel):
    """
    一次造题执行的参数。

    只有 `generation_run_id` —— 因为**其余一切都能从它推导**：抽样配置、
    种子、窗口划分与它们的输入快照全在库里（`generation_run`
    .config_snapshot 与 `generation_window.input_chunks`）。

    这也正是"重试同一个窗口"能复现当时输入的原因：任务参数里没有可漂移
    的东西，执行时读到的就是首轮写下的那份快照。
    """

    model_config = ConfigDict(extra="forbid")

    generation_run_id: UUID = Field(
        description="要执行的造题运行。",
    )


class RetrievalEvalEvaluateParams(BaseModel):
    """
    一次评测执行的参数。

    与造题同理只有 ID：要跑哪些 case、金标是谁、用多大 max_k，全在
    `retrieval_eval_run` 与它的 bundle 里 —— 执行时按 ID 重新加载。

    ⚠️ **不要**把 case 列表塞进来。一轮评测可能有上千条 case，而 Task
    会被反复读出（Dispatcher 轮询读、Worker claim 读、查询接口读）。
    塞进来等于把最热的那些查询全变重。
    """

    model_config = ConfigDict(extra="forbid")

    eval_run_id: UUID = Field(
        description="要执行的评测轮次。",
    )

    #: ⚠️ **刻意不放 `attempts`。**
    #:
    #: 逐轮的阶梯存在 `retrieval_eval_run.effective_config_snapshot` 上，
    #: 执行时从那里读。理由有二：
    #:
    #:   重跑/重试读的是同一份阶梯，不依赖"提交那一刻的参数"；
    #:   它同时是可查的快照 —— 这几个数是按哪套阶梯跑出来的必须答得上来。
    #:
    #: 曾经在这里加过一个 `attempts` 字段但**没人往里写**，而执行侧用
    #: `getattr(params, "attempts", None)` 兜底 —— 于是"字段不存在"被
    #: 伪装成"这次没传"，前端传的阶梯被静默丢弃、永远跑内置档。
