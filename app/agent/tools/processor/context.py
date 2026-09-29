"""
处理器**执行期**看到的东西。

这是一份**契约形状**，不是已经接通的数据结构
------------------------------------------------
仓库里目前**没有工具调用路径**（`app/infra/agent/cutom_tool.py`
是空文件，grep `invoke` 找不到入口）。所以本模块定义的是"将来那条
路径要把什么交给处理器"，本次不接任何执行链。

先把它定下来的理由与 `capability.py` 先定 `config_schema` 形状同理：
形状固定了，处理器实现就能先写完，调用方接进来时不必回头改实现。

为什么 arguments 是**可变 dict**
--------------------------------
BEFORE / AROUND 拦截器要能改写参数（脱敏、补默认值、改单位）。
如果交给它们一个不可变结构，每个实现都得自己复制一份再往下传，
而"谁复制了谁没复制"会变成一类难查的 bug。因此这里**明确约定**：
`ToolInvocation.arguments` 是**这次调用私有**的副本，处理器就地改
不会影响调用方手里的原始参数。

为什么 next_call 是**无参**可调用对象
-------------------------------------
`AsyncToolCall` 收 0 个参数、返回 `ToolOutcome`。处理器想改参数，
改的是 `invocation.arguments`（它已经持有同一个对象），而不是把
参数传给 next_call —— 后者会逼着每个 AROUND 实现把整份 invocation
一路手动传下去，且"改了参数但传的是旧的 invocation"这种错会静默
通过。

无参形态还让"调多次"（重试）与"不调"（短路）都不需要额外机制：
调几次就是 `await next_call()` 几次，不调就是不写这一行。

为什么错误放在 outcome 里而不是抛异常
--------------------------------------
重试要靠"这次失败了吗"来决定下一步，超时要靠"这是超时"与其它
失败区分开。用异常表达这两件事会逼着重试实现 `except Exception`
一把抓，然后连"参数校验失败不该重试"这种基本判断都做不了。
因此失败是一个**可判定的事实**（`outcome.error`），而不是控制流。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

__all__ = [
    "AsyncToolCall",
    "ToolCallDenied",
    "ToolInvocation",
    "ToolOutcome",
]


class ToolCallDenied(Exception):
    """
    拦截器**主动**拒绝了这次调用（审批不通过、参数不合规……）。

    与"工具自己失败"区分开：那是 `ToolOutcome.error` 里的东西，
    工具确实跑了。拒绝是**根本没跑** —— 重试实现据此判断"不该重试"
    （重试一个被拒绝的调用只会再被拒一次），审计也据此区分
    "失败"与"未执行"。

    继承 Exception 而不是 BaseException：它会被写进
    `ToolOutcome.error`，而不是当作控制流往外抛。
    """


@dataclass(frozen=True)
class ToolInvocation:
    """
    一次工具调用的**输入**。

    frozen 冻的是字段本身（不能换一个 arguments 对象），
    不是字典内容 —— 处理器就地改键值正是它该做的事，
    见模块 docstring 里关于"私有副本"的说明。
    """

    #: 被调用的工具，形如 document.retrieve_documents。
    tool_key: str

    #: 这次调用的参数。**本次调用私有的副本**，可写。
    arguments: dict[str, Any] = field(default_factory=dict)

    #: tool_definition 的主键。挂载链是按 tool_id 存的，
    #: 因此执行前必然已经解析出来了；留 None 只是为了
    #: "还没落库的工具"这种边缘情形不炸。
    tool_id: UUID | None = None

    #: 调用方带下来的上下文（租户、会话、追踪 id……），
    #: 同时也是本次调用的**临时共享区**。
    #:
    #: 处理器可以往这里写**带命名空间前缀**的临时键 —— 挂在同一次
    #: 调用上的多个处理器拿到的是同一个对象，这是 BEFORE 那一半
    #: 把信息传给 AFTER 那一半的唯一通道（例如指标记下的开始时刻，
    #: 键名 metrics.started_at）。
    #:
    #: 但**不要改调用方放进去的键**：那些是环境，不是你的便签。
    context: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolOutcome:
    """
    一次工具调用的**结果**，成功与失败共用一种形状。

    `value` 与 `error` 恰好有一个非空；两个都空表示工具正常返回了
    None（是合法结果，不是失败）。

    `duration_ms` 由**执行侧**填（真正调用工具的那一层），处理器不要
    自己算 —— 一个 AFTER 观察者算出来的耗时不包含它自己之前那些
    AROUND 的开销，报出去会偏小且没人看得出来。
    """

    value: Any = None

    #: 失败原因。BaseException 而不是 Exception：取消与超时
    #: （asyncio.CancelledError 在 3.8+ 继承 BaseException）
    #: 也要能被处理器看见并回填。
    error: BaseException | None = None

    duration_ms: float | None = None

    #: 这个结果是**缓存命中**来的，工具并没有真的被调用。
    #:
    #: 为什么要单列一个位，而不是靠 duration≈0 去猜：观测侧要据此
    #: 区分"一次真实执行"与"一次命中" —— 指标里的调用次数、耗时
    #: 统计、成功率的分母全部依赖于它。少了这一位，缓存一开，
    #: "工具被调了多少次"就永久失真，而且失真得没有任何迹象。
    cached: bool = False

    @property
    def ok(self) -> bool:
        """这次调用成功了吗 —— 重试 / 审计的判断依据。"""

        return self.error is None


#: "调用下一步"。
#:
#: AROUND 处理器拿到它，决定调不调、调几次。BEFORE / AFTER 不涉及。
AsyncToolCall = Callable[[], Awaitable[ToolOutcome]]
