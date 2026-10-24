"""
Model 模块：为 Agent / LangGraph 的 LLM Node 提供统一的模型调用能力。

它在系统里的位置（规格 §2）
--------------------------
    Agent / LangGraph
            │
            ▼
       LLM Node                      只知道 ModelRegistry / ModelRuntime
            │
            ▼
     Model Registry                  知道 Resolver 与 Factory
            ├── Resolver              (agent_type, model_role) -> ModelProfile
            ├── Factory               ModelDefinition -> 具体客户端（带缓存）
            ▼
     Model Runtime                   invoke / timeout / retry / fallback /
            │                        异常归一化 / usage / latency
            ▼
     Provider Adapter / SDK          真正发请求

**禁止反向依赖**（规格 §2）：

    Resolver  -> LLM Node            ❌
    Factory   -> LLM Node            ❌
    Runtime   -> LangGraph State     ❌
    Runtime   -> Agent business      ❌
    Runtime   -> HITL                ❌
    Runtime   -> Tool execution      ❌

模块一览
--------
    types.py        领域类型：capabilities / definition / profile / policy /
                    request / response / usage / attempt / resolve context
    errors.py       Model 层异常层级 + provider 异常归一化
    redaction.py    脱敏（日志与观测里不允许出现密钥）
    credentials.py  credential_ref -> 秘密（默认走 llm.yaml，不重复造配置）
    provider.py     ProviderAdapter 契约 + ModelClient
    adapters.py     默认适配器：转交 app/common/llm.py 建客户端
    factory.py      ModelClientFactory：建 / 缓存 / 复用客户端
    resolver.py     Resolver：确定性映射，不调 LLM
    runtime.py      ModelRuntime：一次调用的全部可靠性
    registry.py     ModelRegistry：对 Agent 的唯一入口
    observer.py     调用级观测端口（默认标准库 logging）
    catalog.py      model.yaml 的加载与装配
    model.yaml      模型 / 能力 / profile / 别名 / 绑定的声明

LLM Node 的用法（规格 §15）
-------------------------
    runtime = model_registry.get_runtime(
        agent_type="react",
        model_role="reasoning",
        context=ModelResolveContext(...),
    )
    response = await runtime.ainvoke(
        ModelRequest(messages=..., tools=...)
    )

节点里**不允许**出现：`ChatOpenAI(...)` / `factory.get(...)` /
`resolver.resolve(...)` / `model.bind_tools(...)`。

一次 invoke ≠ 一次 provider 调用（规格 §9）
-----------------------------------------
    一次 LLM Node execution
        ↓
    一次 ModelRuntime.ainvoke      <- 调用方只见这一层
        ↓
    N 次 Provider Attempt          <- retry / fallback 在这里

因此 fallback **不会**重跑节点、不会重建上下文、不会重执行工具、
不会重放 Graph State。

HITL 不在这里（规格 §10）
-----------------------
Model Runtime 是无状态的模型调用层：

    Request -> Model Runtime -> Response

它不知道当前 Graph State、用户是谁、是否需要人工审批、工具是否高风险、
Agent 是否该暂停。HITL 属于 **Graph 控制流**：

    LLM Node 之后        需要用户补充信息 -> HITL（After LLM Node）
    进入 Tool Node 之前   需要人工审批     -> HITL（Before Tool Node）
                         （工具侧的审批拦截器在
                           app/agent/tools/processor/interceptor/approval.py）

本模块**没有**任何等待人工的代码，将来也不该有。

反模式清单（规格 §19）
--------------------
    ❌ LLM Node 自己创建模型
    ❌ LLM Node 自己做 fallback
    ❌ Runtime 调 graph / agent / tool
    ❌ Runtime 实现 HITL
    ❌ Resolver 调 LLM
    ❌ Factory 决定业务模型策略
    ❌ provider 异常泄漏到 LLM Node
    ❌ fallback 重执行整个 Graph Node

⚠️ 与 `app/common/model_factory.py::ModelFactory` 的区别
-----------------------------------------------------
那是**本地模型加载器**（embedder / reranker 的 torch 权重，启动预热）；
这里的是 **provider 客户端的构造与缓存**。因此本模块的工厂叫
`ModelClientFactory` —— 同名会让人 import 错一个而不报错。
"""

from .adapters import LangChainProviderAdapter, default_chat_model_factory
from .catalog import (
    DEFAULT_CATALOG_PATH,
    ModelCatalog,
    ProviderDeclaration,
    build_model_registry,
    get_model_registry,
    load_model_catalog,
    reset_model_registry,
    set_model_registry,
)
from .credentials import (
    CredentialResolver,
    EnvVarCredentialResolver,
    LiteralCredentialResolver,
    NullCredentialResolver,
)
from .errors import (
    DEFAULT_RETRYABLE_ERRORS,
    ErrorDisposition,
    ModelAuthenticationError,
    ModelCandidateFailure,
    ModelCandidatesExhaustedError,
    ModelCapabilityError,
    ModelConfigurationError,
    ModelError,
    ModelInvalidRequestError,
    ModelInvocationError,
    ModelProviderError,
    ModelRateLimitError,
    ModelResolutionError,
    ModelTimeoutError,
    ModelUnavailableError,
    normalize_error,
)
from .factory import FactoryStats, ModelClientFactory
from .observer import (
    CompositeModelObserver,
    LoggingModelObserver,
    ModelCallContext,
    ModelCallObserver,
    NullModelObserver,
)
from .provider import (
    DeltaCallback,
    ModelClient,
    ProviderAdapter,
    ProviderSpec,
)
from .redaction import REDACTED, redact_mapping, redact_text
from .registry import DefaultModelRegistry, ModelRegistry
from .resolver import CatalogModelResolver, ModelResolver, ModelRoleBinding
from .runtime import ModelRuntime
from .trace import ModelTrace
from .types import (
    ModelAttempt,
    ModelCallSummary,
    ModelCapabilities,
    ModelDefinition,
    ModelPricing,
    ModelProfile,
    ModelRequest,
    ModelResolveContext,
    ModelResponse,
    ModelRole,
    ModelRuntimePolicy,
    ModelUsage,
    ProviderResult,
    new_invocation_id,
    required_capabilities_for,
)

__all__ = [
    # ---- 入口（LLM Node 只看这几个） ----
    "ModelRegistry",
    "ModelRuntime",
    "ModelRequest",
    "ModelResponse",
    "ModelResolveContext",
    "ModelTrace",
    "ModelRole",
    "ModelCapabilities",
    #: 流式调用的增量回调类型（LLM Node 用它标注 on_delta）。
    "DeltaCallback",

    # ---- 注册表的实现与装配 ----
    "DefaultModelRegistry",
    "ModelCatalog",
    "ProviderDeclaration",
    "build_model_registry",
    "load_model_catalog",
    "get_model_registry",
    "set_model_registry",
    "reset_model_registry",
    "DEFAULT_CATALOG_PATH",

    # ---- 定义 / 策略 ----
    "ModelDefinition",
    "ModelProfile",
    "ModelRuntimePolicy",
    "ModelPricing",
    "ModelUsage",
    "ModelAttempt",
    "ModelCallSummary",
    "ProviderResult",
    "new_invocation_id",
    "required_capabilities_for",

    # ---- 内部实现（组合根与测试用；LLM Node 不该碰） ----
    "ModelResolver",
    "CatalogModelResolver",
    "ModelRoleBinding",
    "ModelClientFactory",
    "FactoryStats",
    "ProviderAdapter",
    "ProviderSpec",
    "ModelClient",
    "LangChainProviderAdapter",
    "default_chat_model_factory",
    "CredentialResolver",
    "EnvVarCredentialResolver",
    "LiteralCredentialResolver",
    "NullCredentialResolver",
    "ModelCallObserver",
    "ModelCallContext",
    "LoggingModelObserver",
    "NullModelObserver",
    "CompositeModelObserver",
    "REDACTED",
    "redact_mapping",
    "redact_text",

    # ---- 异常 ----
    "ModelError",
    "ModelConfigurationError",
    "ModelResolutionError",
    "ModelCapabilityError",
    "ModelInvocationError",
    "ModelTimeoutError",
    "ModelRateLimitError",
    "ModelUnavailableError",
    "ModelAuthenticationError",
    "ModelInvalidRequestError",
    "ModelProviderError",
    "ModelCandidateFailure",
    "ModelCandidatesExhaustedError",
    "ErrorDisposition",
    "DEFAULT_RETRYABLE_ERRORS",
    "normalize_error",
]
