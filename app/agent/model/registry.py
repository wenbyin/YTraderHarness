"""
Model Registry：Model 模块对 Agent 的**主要入口**（规格 §3、§4、§14）。

依赖方向（规格 §2）
------------------
    Agent / LangGraph
            ↓
       LLM Node
            ↓
     Model Registry   <- 这里
            ├── Resolver
            ├── Factory
            ↓
     Model Runtime
            ↓
     Provider / SDK

Registry **知道** Resolver 与 Factory；LLM Node **只知道** Registry 与
Runtime 的接口。因此节点里永远不该出现：

    ChatOpenAI(...)          ❌ 自己建模型
    factory.get(...)         ❌ 越过 Registry 拿客户端
    resolver.resolve(...)    ❌ 自己解析模型策略
    model.bind_tools(...)    ❌ 直接操作模型实例

节点只做一件事（规格 §15）：

    runtime = model_registry.get_runtime(
        agent_type="react", model_role="reasoning",
    )
    response = await runtime.ainvoke(request)

Registry 管什么
--------------
    Provider    实现维度（deepseek / kimi / openai…）
    Model       具体模型（provider + model name + 能力 + 凭据 + 参数）
    Alias       逻辑名（reasoning_default / fast_default…）
    Profile     模型策略（primary + fallbacks + policy）
    Binding     (agent_type, model_role) -> Profile
    Runtime     按 profile 缓存的执行实例

命名空间是**统一**的
-------------------
模型 id、profile id、别名共享一个命名空间，冲突在注册时就报错。理由
是 `get_runtime` / `definition(ref)` 这些入口拿到的只是一个字符串：
允许"同一个名字既是模型也叫 profile"会让解析结果取决于查表顺序，
而那种 bug 表现为"有时候选到另一个模型"。

⚠️ Registry **无状态**吗？不完全是 —— 注册是启动期行为
---------------------------------------------------
    Application Startup
        ↓ Registry 初始化 + 注册（providers -> models -> profiles -> bindings）
    Agent execution
        ↓ get_runtime(...)   <- 只读 + LRU 缓存
        ↓ runtime.ainvoke(...)

`get_runtime` 会缓存 Runtime 实例（同一个 profile + 同一份解析上下文
共用同一个），因此它**必须**是无状态可共享的（见 runtime.py 的说明）。
注册方法在启动后继续调用是允许的（幂等校验仍在），但不该发生在
请求路径上。
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

from .credentials import CredentialResolver
from .errors import ModelConfigurationError, ModelResolutionError
from .factory import ModelClientFactory
from .observer import ModelCallObserver, NullModelObserver
from .provider import ProviderAdapter, ProviderSpec
from .resolver import CatalogModelResolver, ModelResolver, ModelRoleBinding
from .runtime import ModelRuntime
from .types import (
    ModelDefinition,
    ModelProfile,
    ModelResolveContext,
)

__all__ = [
    "DEFAULT_RUNTIME_CACHE_SIZE",
    "DefaultModelRegistry",
    "ModelRegistry",
]

logger = logging.getLogger(__name__)


#: Runtime 缓存的默认容量。
#:
#: runtime 的键是 (profile, agent_type, model_role, 解析上下文)，而解析
#: 上下文里可以有 attributes —— 理论上租户多了会撑大缓存。Runtime 很轻
#: （几个引用），但有界总比无界好；LRU 保证热的那几组一直在。
DEFAULT_RUNTIME_CACHE_SIZE = 256


@runtime_checkable
class ModelRegistry(Protocol):
    """
    LLM Node 看到的**全部**模型能力。

    一个取 runtime、四个注册、两个只读查询。节点只用第一个；注册是
    组合根（启动期）用的；两个只读查询是 Context Builder 那种"需要知道
    模型性质（比如窗口）"的旁路用的（见 definition 的说明）。

    实现**不要求**是 async：`get_runtime` 是纯查表 + 缓存命中的同步操作
    （它跑在每次模型调用之前，任何 await 都是白加一次的调度）。
    """

    def get_runtime(
            self,
            *,
            agent_type: str,
            model_role: str,
            context: ModelResolveContext | None = None,
    ) -> ModelRuntime:
        ...

    def register_provider(
            self,
            name: str,
            adapter: ProviderAdapter,
            *,
            label: str = "",
            description: str = "",
    ) -> ProviderSpec:
        ...

    def register_model(
            self,
            definition: ModelDefinition,
    ) -> ModelDefinition:
        ...

    def register_profile(
            self,
            profile: ModelProfile,
    ) -> ModelProfile:
        ...

    def register_alias(self, alias: str, target: str) -> str:
        ...

    def definition(self, reference: str) -> ModelDefinition:
        """
        按模型 id / 别名取**模型定义**（只读）。

        为什么它在 Protocol 上（而不只是实现里有）：Context Builder 要用它
        读 `context_window` 算 token 预算（规格 §二十一：必须优先用
        Registry，而不是硬编码模型名与窗口）。让那个读法进 Protocol，
        "窗口从哪来"就只有一个答案，也不会有人为了拿到它去旁边再抄一张
        模型表。

        它**不**改变 Registry 的核心职责：解析、客户端构造、可靠性仍然只
        在 Resolver / Factory / Runtime 三处，这里只是把已经登记的定义读
        出来（纯查表，不构造客户端）。
        """
        ...

    def profile(self, reference: str) -> ModelProfile:
        """按 profile id / 别名取 profile（只读）。"""
        ...


class DefaultModelRegistry:
    """
    默认实现：Resolver + Factory + LRU 的 Runtime 缓存。

    它把三件事拼在一起，每一件都能单独替换：

        resolver    可注入（默认 CatalogModelResolver）
        factory     可注入（默认 ModelClientFactory，可带 CredentialResolver）
        observer    可注入（默认什么都不做）
    """

    def __init__(
            self,
            *,
            resolver: ModelResolver | None = None,
            bindings: Sequence[ModelRoleBinding] | None = None,
            observer: ModelCallObserver | None = None,
            credentials: CredentialResolver | None = None,
            default_max_retries: int = 0,
            runtime_cache_size: int = DEFAULT_RUNTIME_CACHE_SIZE,
    ) -> None:

        if runtime_cache_size <= 0:
            raise ModelConfigurationError(
                "runtime_cache_size 必须为正 —— 设成 0 等于每次模型调用"
                "都重建一个 Runtime，那是没有意义的开销。"
            )

        self._factory = ModelClientFactory(
            credentials=credentials,
            default_max_retries=default_max_retries,
        )

        self._observer: ModelCallObserver = observer or NullModelObserver()

        self._models: dict[str, ModelDefinition] = {}
        self._profiles: dict[str, ModelProfile] = {}
        self._aliases: dict[str, str] = {}
        self._bindings: list[ModelRoleBinding] = list(bindings or ())

        self._runtimes: OrderedDict[tuple[Any, ...], ModelRuntime] = (
            OrderedDict()
        )
        self._runtime_cache_size = runtime_cache_size

        # 外部注入的 resolver 由注入方负责它的绑定来源；默认那个跟着
        # register_binding 一起重建。
        self._resolver: ModelResolver | None = resolver

        if resolver is None:
            self._rebuild_resolver()

    # ========================================================
    # 取用（LLM Node 唯一的入口）
    # ========================================================

    def get_runtime(
            self,
            *,
            agent_type: str,
            model_role: str,
            context: ModelResolveContext | None = None,
    ) -> ModelRuntime:
        """
        `(agent_type, model_role)` -> ModelRuntime（带缓存）。

        这是 LLM Node 与 Model 模块之间**唯一**的调用点。

        Parameters
        ----------
        agent_type
            Agent Builder / JEV 已经决定好的 Agent 类型。本模块不认识
            它的含义，只拿它查表（规格 §6）。
        model_role
            这次模型调用在**模型维度**上的角色（reasoning / summarization…）。
        context
            解析上下文（能力要求 / 凭据覆盖 / 路由属性）。None 相当于
            一个空上下文。

        缓存键含 context 的 `cache_key()`：同一个 profile、不同的能力
        要求会拿到不同的 Runtime（因为要求会被带进 Runtime 供候选链
        过滤）。Runtime 本身无状态，共享是安全的。
        """

        resolved = context or ModelResolveContext()

        profile = self._resolver.resolve(  # type: ignore[union-attr]
            agent_type=agent_type,
            model_role=model_role,
            context=resolved,
        )

        key = (
            profile.id,
            agent_type,
            model_role,
            resolved.cache_key(),
        )

        cached = self._runtimes.get(key)

        if cached is not None:
            self._runtimes.move_to_end(key)
            return cached

        runtime = ModelRuntime(
            profile=profile,
            factory=self._factory,
            definition_lookup=self.definition_optional,
            agent_type=agent_type,
            model_role=model_role,
            context=resolved,
            observer=self._observer,
        )

        self._runtimes[key] = runtime
        self._runtimes.move_to_end(key)

        while len(self._runtimes) > self._runtime_cache_size:
            self._runtimes.popitem(last=False)

        return runtime

    # ========================================================
    # 注册：provider / model / profile / alias / binding
    # ========================================================

    def register_provider(
            self,
            name: str,
            adapter: ProviderAdapter,
            *,
            label: str = "",
            description: str = "",
    ) -> ProviderSpec:
        """登记一个 provider 的适配器（转发给 Factory，理由见 factory.py）。"""

        return self._factory.register_provider(
            name,
            adapter,
            label=label,
            description=description,
        )

    def register_model(
            self,
            definition: ModelDefinition,
    ) -> ModelDefinition:
        """
        登记一个具体模型（连同它声明的别名）。

        重复登记**报错**而不是覆盖：覆盖的后果是"某处以为自己在用的
        模型被另一个定义顶掉了"，而症状是"参数不生效 / 能力声明不对"，
        几乎指不到这里。
        """

        existing = self._models.get(definition.id)

        if existing is not None:
            raise ModelConfigurationError(
                f"模型 {definition.id!r} 已经登记过（"
                f"{existing.target}）。一个模型 id 只能有一个定义 —— "
                "要改参数请改那一处，不要在两处各写一份。"
            )

        self._assert_name_free(definition.id, kind="模型")

        # 别名也先校验再插入：不然一个冲突的别名会让"插入模型 -> 注册
        # 别名"这条两步操作停在中途，留下一个只有模型、没有别名的
        # 半成品注册表 —— 而启动期报错之后它还会被继续用。
        for alias in definition.aliases:

            if alias != definition.id:
                self._assert_name_free(alias, kind="别名")

        self._models[definition.id] = definition

        for alias in definition.aliases:
            self.register_alias(alias, definition.id)

        return definition

    def register_profile(
            self,
            profile: ModelProfile,
    ) -> ModelProfile:
        """登记一个 model profile。重复登记同样报错。"""

        existing = self._profiles.get(profile.id)

        if existing is not None:
            raise ModelConfigurationError(
                f"profile {profile.id!r} 已经登记过。"
                "一个 profile id 只能有一个定义。"
            )

        self._assert_name_free(profile.id, kind="profile")

        self._profiles[profile.id] = profile

        return profile

    def register_alias(self, alias: str, target: str) -> str:
        """
        登记一个逻辑别名。

        解析只有**一跳**（别名 -> 模型 id 或 profile id），不支持
        别名指向别名：多跳除了制造环，没有任何实际用途，而环的检测
        与报错是纯粹的额外复杂度。

        目标必须**先**注册：这条约束让"别名指向一个拼错的名字"在启动期
        就炸，而不是等某次模型调用。
        """

        name = alias.strip()

        if not name:
            raise ModelConfigurationError("别名不能为空。")

        if name != target:
            self._assert_name_free(name, kind="别名")

        existing_alias = self._aliases.get(name)

        if existing_alias is not None and existing_alias != target:
            raise ModelConfigurationError(
                f"别名 {name!r} 已经指向 {existing_alias!r}，"
                f"不能再指向 {target!r}。"
            )

        if target not in self._models and target not in self._profiles:
            raise ModelConfigurationError(
                f"别名 {name!r} 指向的 {target!r} 既不是已登记的模型，"
                f"也不是已登记的 profile。\n"
                f"已登记的模型：{sorted(self._models)}\n"
                f"已登记的 profile：{sorted(self._profiles)}\n"
                "别名只能指向**已经注册**的目标 —— 这条约束让拼错的名字"
                "在启动期就暴露。"
            )

        self._aliases[name] = target

        return name

    def register_binding(
            self,
            binding: ModelRoleBinding,
    ) -> ModelRoleBinding:
        """
        登记一条 `(agent_type, model_role) -> profile` 绑定。

        只对**默认 resolver** 生效。注入了自定义 resolver 时，绑定要
        通过它自己的构造参数给（本方法会明确报错，而不是静默无效 ——
        静默无效的后果是"我加了绑定但模型没变"）。
        """

        if not isinstance(self._resolver, CatalogModelResolver):
            raise ModelConfigurationError(
                "本 Registry 用的是自定义 resolver"
                f"（{type(self._resolver).__name__}），"
                "register_binding 对它不生效。\n"
                "两条路：把绑定交给那个 resolver 的构造函数，或改用默认 "
                "resolver（catalog.build_model_registry）。"
            )

        self._bindings.append(binding)

        self._rebuild_resolver()

        return binding

    # ========================================================
    # 查询
    # ========================================================

    def definition(self, reference: str) -> ModelDefinition:
        """
        按模型 id 或别名取定义；取不到抛 `ModelResolutionError`。

        （Runtime 用的是 `definition_optional`：它自己要把"引用不到"
        包装成一条能说清是哪个 profile 出错的报错。）
        """

        definition = self.definition_optional(reference)

        if definition is not None:
            return definition

        raise ModelResolutionError(
            f"模型 {reference!r} 没有登记。"
            f"已登记：{sorted(self._models) or '（无）'}；"
            f"别名：{sorted(self._aliases) or '（无）'}。"
        )

    def definition_optional(
            self,
            reference: str,
    ) -> ModelDefinition | None:
        """按模型 id / 别名取定义；取不到返回 None。"""

        if reference in self._models:
            return self._models[reference]

        target = self._aliases.get(reference)

        if target is not None and target in self._models:
            return self._models[target]

        # 别名也可能指向 profile（那是 profile 的别名，不是模型的）。
        return None

    def profile(self, reference: str) -> ModelProfile:
        """
        按 profile id 或别名取 profile；取不到抛 `ModelResolutionError`。

        它同时是 `CatalogModelResolver` 的查表函数（构造时注入），因此
        别名解析只有这一处实现。
        """

        if reference in self._profiles:
            return self._profiles[reference]

        target = self._aliases.get(reference)

        if target is not None and target in self._profiles:
            return self._profiles[target]

        raise ModelResolutionError(
            f"model profile {reference!r} 没有登记。"
            f"已登记：{sorted(self._profiles) or '（无）'}；"
            f"别名：{sorted(self._aliases) or '（无）'}。\n"
            "profile 是“用哪个模型、失败退到哪”的那份声明 —— 缺了它，"
            "对应的 (agent_type, model_role) 绑定就解析不出来。"
        )

    def models(self) -> tuple[ModelDefinition, ...]:
        return tuple(self._models[name] for name in sorted(self._models))

    def profiles(self) -> tuple[ModelProfile, ...]:
        return tuple(self._profiles[name] for name in sorted(self._profiles))

    def aliases(self) -> Mapping[str, str]:
        return dict(self._aliases)

    def providers(self) -> tuple[ProviderSpec, ...]:
        return self._factory.providers()

    def bindings(self) -> tuple[ModelRoleBinding, ...]:
        return tuple(self._bindings)

    @property
    def factory(self) -> ModelClientFactory:
        """底层工厂（组合根在关闭阶段要用它释放客户端）。"""

        return self._factory

    @property
    def resolver(self) -> ModelResolver:
        return self._resolver  # type: ignore[return-value]

    # ========================================================
    # 生命周期
    # ========================================================

    async def aclose(self) -> None:
        """关闭全部缓存的 provider 客户端（进程关闭时调一次）。"""

        await self._factory.aclose()

    def describe(self) -> dict[str, Any]:
        """
        整个注册表的一份摘要（日志 / 将来的 `/models` 目录接口）。

        能力声明与 profile 策略都在里面，但**没有任何秘密**：凭据只以
        `credential_ref` 的名字形式出现（见 redaction.py）。
        """

        return {
            "providers": [
                {
                    "name": spec.name,
                    "label": spec.label,
                    "adapter": getattr(spec.adapter, "name", type(spec.adapter).__name__),
                }
                for spec in self.providers()
            ],
            "models": [item.describe() for item in self.models()],
            "profiles": [item.describe() for item in self.profiles()],
            "aliases": self.aliases(),
            "bindings": [
                binding.describe() for binding in self._bindings
            ],
            "factory": self._factory.stats.as_metadata(),
        }

    # ========================================================
    # internals
    # ========================================================

    def _rebuild_resolver(self) -> None:
        """
        用当前的绑定表重建默认 resolver。

        每次 `register_binding` 都重建：`CatalogModelResolver` 持有的是
        **一份绑定的快照**（它要在解析时做特异性比较，热路径上不该反复
        遍历一个可变列表）。绑定是启动期数据，重建成本可以忽略。
        """

        self._resolver = CatalogModelResolver(
            bindings=self._bindings,
            profile_lookup=self.profile,
        )

    def _assert_name_free(self, name: str, *, kind: str) -> None:
        """
        名字必须未被占用（模型 id / profile id / 别名同域）。

        这是"统一命名空间"的执法点。若允许 `fast_default` 既是 profile
        别名又是模型 id，`definition("fast_default")` 与
        `profile("fast_default")` 的结果就取决于查表顺序 —— 而调用方
        以为自己在说同一个东西。
        """

        if name in self._models:
            raise ModelConfigurationError(
                f"{kind}名 {name!r} 与已登记的**模型**同名。"
                "模型 id、profile id、别名共享一个命名空间 —— 重名会让"
                "解析结果取决于查表顺序。"
            )

        if name in self._profiles:
            raise ModelConfigurationError(
                f"{kind}名 {name!r} 与已登记的 **profile** 同名。"
                "模型 id、profile id、别名共享一个命名空间。"
            )

        if name in self._aliases:
            raise ModelConfigurationError(
                f"{kind}名 {name!r} 与已登记的**别名**同名"
                f"（已指向 {self._aliases[name]!r}）。"
            )
