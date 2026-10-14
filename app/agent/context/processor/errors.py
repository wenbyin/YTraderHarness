"""
Context Builder 的异常层级（规格 §二十八）。

不发明新的错误体系
----------------
顶层继承 `app.common.exceptions.ApplicationError` —— 与 tool / document
两边一致（那是本项目"应用层异常"的共享根）。规格 §二十八 明确要求优先
复用现有 Exception，因此这里只**细分**，不另起一套。

Fatal 与 Optional 的边界
----------------------
    ContextBuildError              所有上下文构建失败的根
      ├── ContextStateError        State 绑定的问题（**Fatal**）
      │     when 路径不存在
      │     when 不是 bool（含 None）
      │     必需输入缺失
      ├── ContextProcessorError    Processor 执行失败
      │     optional=False -> Fatal
      │     optional=True  -> 记 FAILED + warning，流水线继续
      ├── ContextBudgetError       预算不够且裁不动（**Fatal**）
      └── ContextPolicyError       拿不到有效的 Context Policy（**Fatal**）

为什么这些必须是 Fatal
--------------------
"静默降级"在这里的后果是**模型看到一份不完整的上下文，然后自信地
答错**。规格 §二十八 把 `when` 缺失、必需输入缺失、预算超限列为
Fatal，理由正是这个：宁可这一轮 Run 明确失败，也不要让一次
"没有历史、没有画像、没有工具"的调用看起来成功。

反过来，Memory / 外部检索这类旁路信息的失败**不该**让整轮失败 ——
那由 Processor 自己声明 `spec.optional=True` 表达（见 spec.py），
而不是由这里猜。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.common.exceptions import ApplicationError
from app.agent.model import redact_mapping

__all__ = [
    "ContextBudgetError",
    "ContextBuildError",
    "ContextPolicyError",
    "ContextProcessorError",
    "ContextStateError",
]


class ContextBuildError(ApplicationError):
    """
    上下文构建失败的根。

    带 `stage` / `processor` 两个可选字段：上下文构建的每一步都在
    流水线里，报错不指明"哪一步"的话，看日志的人得从头读一遍。
    """

    def __init__(
            self,
            message: str,
            *,
            processor: str | None = None,
            stage: str | None = None,
            details: Mapping[str, Any] | None = None,
    ) -> None:

        self.message = str(message)
        self.processor = processor
        self.stage = stage
        self.details: dict[str, Any] = dict(redact_mapping(dict(details or {})))

        super().__init__(self.message)

    def as_metadata(self) -> dict[str, Any]:

        return {
            "error_type": type(self).__name__,
            "message": self.message,
            "processor": self.processor,
            "stage": self.stage,
            "details": dict(self.details),
        }

    def __str__(self) -> str:

        where = " / ".join(
            item for item in (self.stage, self.processor) if item
        )

        return f"{self.message}（{where}）" if where else self.message


class ContextStateError(ContextBuildError):
    """
    State 绑定的问题：MISSING / INVALID / 必需输入缺失。

    这三件事有一个共同点：**上游（Agent Builder / Policy 层 / 图）
    写错了或漏写了**，而不是运行环境出了故障。因此它不带任何
    "重试"含义 —— 重试一次还是同样的 State。
    """


class ContextProcessorError(ContextBuildError):
    """
    一个 Processor 执行时抛了。

    只有 `spec.optional=False` 的 Processor 才会把它抛到调用方；
    optional 的那些由流水线吞掉并记 FAILED（见 pipeline.py）。
    """


class ContextBudgetError(ContextBuildError):
    """
    预算不够，且裁无可裁。

    两种触发点：
        分配阶段预算就是负的（窗口太小 / output + reserve 已超过窗口）
        裁剪之后仍然超（当前消息本身就超出窗口）
    """


class ContextPolicyError(ContextBuildError):
    """
    拿不到有效的 Context Policy。

    规格 §二十八 把"无法获取必要的 Agent Policy"列为 Fatal：没有
    policy 就没有 gate、没有预算、没有 system prompt —— 那种情况下
    继续构建出来的"上下文"没有任何依据。
    """
