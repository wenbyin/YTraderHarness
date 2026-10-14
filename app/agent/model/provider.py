"""
Provider Adapter：**唯一**与具体 provider SDK 打交道的地方。

它在链上的位置
-------------
    ModelRegistry -> ModelFactory -> ProviderAdapter -> Provider SDK
                                          ^^^^^^^^^^^^^ 这里

边界（规格 §7、§20）
-------------------
Adapter 负责：

    provider 客户端构造（SDK 初始化、参数翻译）
    把统一的 ModelRequest 翻成 provider 调用（绑工具、结构化输出）
    provider 异常的**归一化**（→ Model 层异常）
    把 provider 的响应用量读成统一的 ModelUsage

Adapter **不**负责：fallback、retry、超时策略、能力判断、Graph 控制流、
HITL、工具执行。那些全在 Runtime / Registry —— 适配器换一家 SDK 时
不该动到任何可靠性语义。

为什么"provider"与"adapter"是两个概念
------------------------------------
    provider   实现维度：deepseek / kimi / openai …（规格 §4.1）
    adapter    代码维度：怎么跟这一类 provider 说话

一个 adapter 可以服务多个 provider：本项目所有 provider 都是
OpenAI 兼容的，因此它们共用 `adapters.LangChainProviderAdapter`，
只靠 provider 名去 `app/common/llm.py` 里取连接信息。反过来，将来某家
provider 需要专用 SDK（比如某个自研网关），就只加一个 adapter 类，
注册表那边只多一行 `register_provider(...)`。

⚠️ 这一层**刻意不 import** langchain / openai / anthropic
-------------------------------------------------------
`ProviderAdapter` 是 Protocol，`ModelClient` 是数据类，都不需要真的
import 那些包。默认实现在 adapters.py，它对客户端只做鸭子调用
（`bind_tools` / `ainvoke` / `usage_metadata`），因此"import 本模块"
不会拉起任何 provider SDK —— 与 `app/common/llm.py` 的惰性 import
是同一条纪律。
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from .types import ModelDefinition, ModelRequest, ProviderResult

__all__ = [
    "DeltaCallback",
    "ModelClient",
    "ProviderAdapter",
    "ProviderSpec",
]


#: 正文增量的回调：`await on_delta(text) -> None`。
#:
#: 流式调用时**逐块**回调**用户可见的正文**增量 —— 不含思维链内容，
#: 也不含 tool-call 参数的碎片（判据见 adapters.py 的 `_chunk_text`）。
#:
#: 实现方必须自己吞掉异常：这里抛出来会顺着调用链把这次模型调用一起
#: 带走，而增量的投递是**旁路**（见 app/common/stream）。
DeltaCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ModelClient:
    """
    Factory 的产出：**一个已经建好、可复用的 provider 客户端**。

    它把"客户端对象"与"它是按哪份定义、哪个超时、哪份凭据建的"绑在
    一起 —— 少了这些，适配器在报错时就说不清是哪个模型失败了，缓存也
    没法按这些维度分键。

    ⚠️ 调用方（LLM Node）**看不到**这个类型：它只在 Factory 与
    Adapter 之间流转（见 runtime.py）。节点看到的是 ModelResponse。
    """

    definition: ModelDefinition

    #: provider SDK / langchain 的客户端对象。类型是 Any：不同 provider
    #: 的类不同，而且本模块对它们惰性依赖。
    client: Any

    adapter: "ProviderAdapter"

    credential_ref: str | None = None

    #: 构造这个客户端时用的超时（进缓存键：同一个模型配不同超时是
    #: 两个客户端）。
    timeout: float = 60.0

    #: 构造时用的客户端内部重试次数。**默认 0** —— retry 归 Runtime
    #: 管，见 factory.py 的说明。
    max_retries: int = 0

    @property
    def model_id(self) -> str:
        return self.definition.id


@dataclass(frozen=True, slots=True)
class ProviderSpec:
    """
    注册表里一条 provider 登记。

    `name` 必须与 `app/common/llm.yaml` 里的 provider 键一致（那是连接
    信息与密钥的所在），除非这个 adapter 自己解决连接信息。

    ⚠️ 刻意**没有** `options` 之类的"适配器专属配置"
    --------------------------------------------
    适配器是**由调用方构造**的（`build_model_registry(adapters={...})`），
    因此"某个 adapter 需要额外参数"的答案就是它的构造参数 —— 放在这里
    等于给同一件事开第二个入口，而那个入口没有任何消费方（Factory 只
    用它找 adapter，Adapter 看不到自己的 spec）。
    一个没人读的配置字段比没有字段更糟：它让人以为自己配上了。
    """

    name: str
    adapter: "ProviderAdapter"
    label: str = ""
    description: str = ""

    def __post_init__(self) -> None:

        if not self.name.strip():
            raise ValueError("ProviderSpec.name 不能为空。")


@runtime_checkable
class ProviderAdapter(Protocol):
    """
    一家（或一类）provider 的适配器。

    ⚠️ 实现必须做两件事
    ------------------
    1. `create_client` **不阻塞**：只做客户端对象构造，不建连接、不探活。
       它跑在模型调用的路径上（Factory 缓存未命中时），网络往返会让
       第一次调用的延迟凭空多一跳。
    2. `ainvoke` **自己归一化异常**：provider SDK 的异常不许冒出去。
       Runtime 那边还有一层兜底网，但兜底只能把未知异常变成
       `ModelProviderError`（可重试）—— 分类信息在这一层才完整。
    """

    #: 适配器标识（配置里 `adapter:` 用它选实现，如 "langchain"）。
    name: str

    def create_client(
            self,
            definition: ModelDefinition,
            *,
            credential: str | None,
            timeout: float,
            max_retries: int,
    ) -> Any:
        """按定义建一个 provider 客户端。**同步、不阻塞**。"""
        ...

    async def ainvoke(
            self,
            client: ModelClient,
            request: ModelRequest,
            *,
            timeout: float,
            on_delta: DeltaCallback | None = None,
            run_metadata: Mapping[str, Any] | None = None,
    ) -> ProviderResult:
        """
        发一次请求。失败时抛 Model 层异常。

        `on_delta` 非空时**尽量**走流式：逐块回调正文增量，最后返回的
        仍是一个完整的 `ProviderResult`（消息是拼好的，不是"块"）。

        ⚠️ 但它只是"增量"，**不保证逐 token**：provider 不支持流式、或
        本次是结构化输出时，实现应当退化成"一次性回调整段正文"。调用方
        因此不必判断走没走流式，只需要接受"可能一次来一大段"。

        `run_metadata`
            **这次调用是谁**（`agent_type` / `model_role`），由 Runtime
            给。实现应当把它挂到**底层调用框架的运行元信息**上（langchain
            就是 `config={"metadata": ...}`），**不要**发给 provider ——
            provider 不认识它，混进请求参数里在某些端点上会 400。

            它存在的唯一理由：下游要能**区分同一次运行里并发的多次模型
            调用**。langgraph 的 `messages` 事件只带 `langgraph_node`，而
            压缩、分类这些调用与主循环**共用节点名** —— 只靠节点名分不出
            谁是谁（见 run_event_projector 的 message_roles）。
        """
        ...

    async def aclose(self, client: ModelClient) -> None:
        """释放客户端持有的连接（进程关闭时调）。"""
        ...
