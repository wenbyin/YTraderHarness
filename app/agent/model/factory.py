"""
Model Factory（本模块的实现名：`ModelClientFactory`）：建、缓存、复用
provider 客户端。

⚠️ 与 `app/common/model_factory.py::ModelFactory` 的区别
-----------------------------------------------------
那个是**进程级本地模型加载器**（embedder / reranker，torch 权重几 GB，
启动时后台预热）。这里的是**provider 客户端的构造与缓存**（HTTP 客户端，
建一次几毫秒）。

两者同名会出事：`from ... import ModelFactory` 拿错一个不会报错，
只会在运行到 `.get(key)` 时莫名其妙。因此本模块刻意叫
`ModelClientFactory`，并且**不复用**那个工厂 —— 它们的生命周期、
失效语义、成本都不同（一个要预热、一个不要；一个占显存、一个占连接）。

Factory 负责什么（规格 §7）
--------------------------
    provider adapter 创建   -> 按 definition.provider 找适配器
    SDK client 初始化       -> 交给 adapter.create_client
    client cache / 连接复用 -> 按 (定义, 凭据, 超时, 参数) 分键缓存
    provider 特有初始化     -> 在 adapter 里

Factory **不**负责：fallback、retry、Graph 控制流、Agent reasoning、
HITL、工具选择、Context Builder。它只有"给定模型定义，给我一个能用的
客户端，同一个定义不要建两次"这一件事。

⚠️ 客户端内部重试默认**关掉**
---------------------------
retry 的所有者是 Model Runtime（规格 §8.2）。客户端再自带一份重试会
叠乘：Runtime 3 次 × 客户端 3 次 = 9 次**可计费**调用，而日志里只看到
"重试了 3 次"（旧的 agent_node 注释里记着这个坑）。
`default_max_retries=0` 就是这个决定；要恢复客户端重试必须**显式**
在模型定义的 config 里写 `max_retries`，那是一个看得见的动作。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

from .credentials import CredentialResolver, NullCredentialResolver
from .errors import ModelConfigurationError, ModelError, normalize_error
from .provider import ModelClient, ProviderAdapter, ProviderSpec
from .types import ModelDefinition

__all__ = [
    "FactoryStats",
    "ModelClientFactory",
]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class FactoryStats:
    """
    工厂的计数（测试与运维都要看）。

    `created` 是最重要的那个：规格 §17 要求"不要在每一次 invoke() 里
    重新初始化 SDK client"，而这条约束在代码上唯一可验证的形态就是
    "同一个定义建了几次"。测试直接断言它。
    """

    hits: int = 0
    misses: int = 0
    created: int = 0
    closed: int = 0

    @property
    def cached(self) -> int:
        return self.misses

    def as_metadata(self) -> dict[str, int]:

        return {
            "hits": self.hits,
            "misses": self.misses,
            "created": self.created,
            "closed": self.closed,
        }


class ModelClientFactory:
    """
    按模型定义建/取 provider 客户端。

    线程与并发
    ----------
    `get()` 是**同步**的（客户端构造不联网，见 ProviderAdapter 的约定），
    因此不需要锁：`dict` 的读写在本进程里是原子的，最坏情况是两个并发
    调用同时建了一个客户端、其中一个立刻被覆盖 —— 代价是一次多余的
    构造，而不是错误。为它加一把 asyncio 锁反而会把"不阻塞"这条约定
    变成"可能阻塞"，那更糟。

    上面这段成立的前提是 `get()` **同步走到头、中间没有让出点**。
    `aget()`（异步调用方该用的那个）打破了这个前提——它把建客户端丢进
    线程池，于是有了让出点，也就**需要**一把锁：见 `aget()` 的说明。

    （真要严格"同一定义只建一次"，应当在启动时预热 —— 那属于组合根，
    见 catalog.build_model_registry 的说明。）
    """

    def __init__(
            self,
            *,
            credentials: CredentialResolver | None = None,
            default_max_retries: int = 0,
    ) -> None:

        if default_max_retries < 0:
            raise ValueError(
                "default_max_retries 不能为负 —— 它是客户端内部重试次数，"
                "而 Runtime 才是 retry 的所有者（见模块 docstring）。"
            )

        self._providers: dict[str, ProviderSpec] = {}
        self._clients: dict[tuple[Any, ...], ModelClient] = {}
        self._credentials = credentials or NullCredentialResolver()
        self._default_max_retries = default_max_retries

        #: `aget()` 未命中路径上的锁。**延迟创建**，因为它绑定事件循环
        #: —— 详见 `_lock_for_running_loop()`。
        self._create_lock: asyncio.Lock | None = None
        self._create_lock_loop: asyncio.AbstractEventLoop | None = None

        self.stats = FactoryStats()

    # ========================================================
    # provider 登记
    # ========================================================

    def register_provider(
            self,
            name: str,
            adapter: ProviderAdapter,
            *,
            label: str = "",
            description: str = "",
    ) -> ProviderSpec:
        """
        登记一个 provider 的适配器。

        provider 表放在 Factory 而不是 Registry：Runtime 与 Registry 都
        要读它，但**只有 Factory 消费它**（Registry 只是转发注册）。放在
        消费方那一侧，就不存在"两处各存一份 provider 表"的可能。

        重复登记同名 provider 且适配器不是同一个对象 -> 报错：那意味着
        两处想接不同的实现，静默取其一比报错危险得多（与 tool 处理器
        注册表的键冲突检查同一条立场）。
        """

        if not name.strip():
            raise ModelConfigurationError("provider 名不能为空。")

        spec = ProviderSpec(
            name=name,
            adapter=adapter,
            label=label or name,
            description=description,
        )

        existing = self._providers.get(name)

        if existing is not None and existing.adapter is not adapter:
            raise ModelConfigurationError(
                f"provider {name!r} 已经登记过另一个适配器"
                f"（{type(existing.adapter).__name__} vs "
                f"{type(adapter).__name__}）。\n"
                "一个 provider 只能有一个适配器 —— 否则先登记的那个会被"
                "静默顶掉，故障表现为“有时候走的是另一个客户端”。"
            )

        self._providers[name] = spec

        return spec

    def provider(self, name: str) -> ProviderSpec:
        """取一个 provider；没登记就抛配置错。"""

        spec = self._providers.get(name)

        if spec is not None:
            return spec

        raise ModelConfigurationError(
            f"provider {name!r} 没有登记适配器。"
            f"当前已登记：{sorted(self._providers) or '（无）'}。\n"
            "两个入口：catalog 里声明（model.yaml 的 providers:）或调 "
            "ModelRegistry.register_provider()。"
        )

    def providers(self) -> tuple[ProviderSpec, ...]:
        """全部 provider，按名字排序（目录接口与测试用）。"""

        return tuple(
            self._providers[name] for name in sorted(self._providers)
        )

    def has_provider(self, name: str) -> bool:
        return name in self._providers

    # ========================================================
    # 取客户端
    # ========================================================

    def get(
            self,
            definition: ModelDefinition,
            *,
            timeout: float,
            max_retries: int | None = None,
    ) -> ModelClient:
        """
        取（必要时建）一个客户端。

        `timeout` / `max_retries` 是**建客户端时的参数**，因此进缓存键：
        同一个模型配不同超时是两个客户端 —— 这不是浪费，而是必要条件
        （超时值在客户端构造时就固定了）。

        `max_retries`
            缺省取本工厂的 `default_max_retries`（0）。要显式恢复客户端
            重试，请在模型定义的 config 里写 `max_retries`（同一个键，
            但那是配置层面看得见的动作）。
        """

        key = self._cache_key(
            definition, timeout=timeout, max_retries=max_retries
        )

        cached = self._clients.get(key.value)

        if cached is not None:
            self.stats.hits += 1
            return cached

        self.stats.misses += 1

        client = self._create(
            definition,
            credential=key.credential,
            timeout=timeout,
            max_retries=key.max_retries,
        )

        self._clients[key.value] = client
        self.stats.created += 1

        logger.debug(
            "模型客户端已创建并缓存：%s（timeout=%s, max_retries=%d, "
            "credential_ref=%s）",
            definition.id,
            timeout,
            key.max_retries,
            definition.credential_ref,
        )

        return client

    def peek(
            self,
            definition: ModelDefinition,
            *,
            timeout: float,
            max_retries: int | None = None,
    ) -> ModelClient | None:
        """
        只看缓存：命中就返回，没命中返回 `None`。

        **不建客户端、不计入 `stats`** —— 它是 `aget()` 快路径的探针，
        而 `stats` 的语义是"客户端建了几次 / 命中了几次"，探针不该
        参与计数（否则热路径上的每次调用都会多记一次 hit）。

        会抛 `ModelConfigurationError`（凭据解析失败、config 里的
        max_retries 非法）—— 与 `get()` 一致：**键算不出来本身就是
        配置错**，用 `None` 表示会让它看起来像"没缓存"。
        """

        key = self._cache_key(
            definition, timeout=timeout, max_retries=max_retries
        )

        return self._clients.get(key.value)

    async def aget(
            self,
            definition: ModelDefinition,
            *,
            timeout: float,
            max_retries: int | None = None,
    ) -> ModelClient:
        """
        `get()` 的异步版本：**只有缓存未命中时才离开事件循环**。

        为什么需要它（而不是让调用方自己包 `asyncio.to_thread`）
        -------------------------------------------------------
        建客户端的路径上挂着一串惰性 import：

            langchain.chat_models -> langchain_openai -> openai.types...

        实测（loop_watchdog 抓到的现场）能把事件循环按住 7 秒，而那是
        SSE 的 Redis 读超时被谎报成
        `TimeoutError: Timeout reading from localhost:6379` 的**根因**。
        详见 `app/common/llm.py::warmup_chat_models`。

        启动预热（A1）已经付掉了大部分 import 成本，但"别在循环上做
        同步的重活"不能只靠一处预热记着 —— 这里是那条约定的执行点。

        命中缓存**不**进线程池
        ---------------------
        那只是一次字典查询，本就不阻塞；为它付一次线程往返是纯开销，
        而热路径（每次模型调用）**全是**命中。

        ⚠️ 未命中路径上有一把锁（这是新增的约束）
        ----------------------------------------
        上面类 docstring 说 `get()` "不需要锁"，前提是它**同步执行**：
        中间没有让出点，两个并发调用不可能真的同时走到建客户端那一步。
        这里不同 —— `await asyncio.to_thread(...)` 会让出事件循环，于是
        同一个定义的两处并发首调会各建一个客户端，其中一个的连接池
        从此没人关（`_clients` 里那一格被后到的覆盖了，先建那个再没
        任何引用）。所以锁是 `to_thread` **引入**的必要条件，不是
        预防性措施。

        锁只覆盖未命中路径，且锁内**再查一次**缓存（双检）：后到的
        那个直接拿到先到者建好的客户端。

        代价：不同定义的**首次**创建会串行。建客户端是毫秒级的（import
        已由启动预热付掉），拿它换"绝不漏掉一个连接池"是划算的。

        ⚠️ 它挡不住**同步** `get()` 的并发
        ---------------------------------
        锁是 `asyncio.Lock`，只有走 `aget()` 的人参与。若某个调用方
        仍用同步 `get()`（例如 tools/ 下的脚本、冒烟测试），那条路
        不受此锁保护 —— 它本来也没有让出点，与改动前一致。
        """

        client = self.peek(
            definition, timeout=timeout, max_retries=max_retries
        )

        if client is not None:
            self.stats.hits += 1
            return client

        async with self._lock_for_running_loop():

            client = self.peek(
                definition, timeout=timeout, max_retries=max_retries
            )

            if client is not None:
                self.stats.hits += 1
                return client

            # 命中不了的这一次**交给 get() 自己**去建 —— 键的计算、异常
            # 归一化、stats 计数、缓存写入全都在那一处，这里不复制。
            # 键因此被算了两次（凭据解析也就跑两次），代价是两次
            # getenv + 一次哈希；换来的是一份实现而不是两份会分叉的。
            return await asyncio.to_thread(
                self.get,
                definition,
                timeout=timeout,
                max_retries=max_retries,
            )

    # ========================================================
    # 生命周期
    # ========================================================

    async def aclose(self) -> None:
        """
        关闭全部缓存客户端。

        只在进程关闭时调（与 `dispatcher.aclose()` / `model_factory.aclose()`
        并排）。单个客户端失败不影响其余：关闭阶段的错误不该再产生
        新的错误。
        """

        clients = list(self._clients.values())

        self._clients.clear()

        for client in clients:

            try:
                await client.adapter.aclose(client)

            except Exception:

                logger.warning(
                    "关闭模型客户端 %s 时出错（忽略）：",
                    client.model_id,
                    exc_info=True,
                )

            else:
                self.stats.closed += 1

    def clear(self) -> None:
        """
        丢弃缓存但**不关闭**客户端 —— 只给测试用。

        生产上丢弃缓存没有正当理由：那些客户端持有的连接池还在被
        并发调用引用（与 `reset_factories()` 同一条立场）。
        """

        self._clients.clear()

    # ========================================================
    # internals
    # ========================================================

    def _cache_key(
            self,
            definition: ModelDefinition,
            *,
            timeout: float,
            max_retries: int | None,
    ) -> _CacheKey:
        """
        算出缓存键，并把建客户端要用的两个输入一并带出来。

        抽出来是为了让 `get()`（同步）、`peek()` / `aget()`（异步快
        路径）用**同一份**键的计算。两处各写一遍的失效方式很隐蔽：
        它在某个字段上分叉，症状是"缓存永远不命中"—— 每次调用都重建
        客户端，不报错、只是慢，且没有任何测试会失败。
        """

        effective_retries = (
            max_retries
            if max_retries is not None
            else _config_max_retries(
                definition, self._default_max_retries
            )
        )

        credential = self._resolve_credential(definition)

        return _CacheKey(
            value=(
                definition.id,
                definition.provider,
                definition.model,
                definition.credential_ref,
                _secret_fingerprint(credential),
                timeout,
                effective_retries,
                _config_fingerprint(definition.config),
            ),
            credential=credential,
            max_retries=effective_retries,
        )

    def _lock_for_running_loop(self) -> asyncio.Lock:
        """
        取 `aget()` 未命中路径上的锁，必要时按**当前事件循环**重建。

        `asyncio.Lock` 会绑定首次用到的事件循环。本工厂是**进程级**
        的（Registry 持有它），而测试与冒烟脚本会用多次
        `asyncio.run()` —— 每次都换一个循环。锁要是死盯着第一个循环，
        第二次 `asyncio.run()` 里的 `aget()` 会在**有竞争时**抛
        "attached to a different loop"，而那是个只在测试里发作、
        生产上永远看不到的报错（生产只有一个循环）。
        """

        loop = asyncio.get_running_loop()

        if self._create_lock is None or self._create_lock_loop is not loop:
            self._create_lock = asyncio.Lock()
            self._create_lock_loop = loop

        return self._create_lock

    def _resolve_credential(
            self,
            definition: ModelDefinition,
    ) -> str | None:
        """
        解析凭据引用。

        没有 ref（绝大多数模型）时**什么都不做**：让 provider 用它自己
        配置里的凭据（providers.yaml 的 llm 段）。这就是"不重复造配置系统"
        的实际含义 —— 默认路径上本模块不认识任何密钥。
        """

        if not definition.credential_ref:
            return None

        return self._credentials.resolve(definition.credential_ref)

    def _create(
            self,
            definition: ModelDefinition,
            *,
            credential: str | None,
            timeout: float,
            max_retries: int,
    ) -> ModelClient:

        spec = self.provider(definition.provider)

        try:
            client = spec.adapter.create_client(
                definition,
                credential=credential,
                timeout=timeout,
                max_retries=max_retries,
            )

        except ModelError:
            # 适配器已经给出 Model 层异常（含配置错），原样向上。
            raise

        except Exception as exc:

            # 兜底网：适配器漏了归一化时，也**不允许** provider SDK 的
            # 异常冒到上层。建客户端阶段的失败一律按配置错处理（见
            # adapters.py 里同一段说明）。
            error = normalize_error(
                exc,
                provider=definition.provider,
                model=definition.model,
            )

            raise ModelConfigurationError(
                f"为模型 {definition.id!r} 创建客户端失败：{error}",
                provider=definition.provider,
                model=definition.model,
                details={
                    "model_id": definition.id,
                    "provider_error_type": type(exc).__name__,
                },
            ) from exc

        return ModelClient(
            definition=definition,
            client=client,
            adapter=spec.adapter,
            credential_ref=definition.credential_ref,
            timeout=timeout,
            max_retries=max_retries,
        )


# ============================================================
# internals
# ============================================================


class _CacheKey(NamedTuple):
    """
    `_clients` 的键，外加建客户端要用的两个输入。

    为什么打包而不是让调用方自己算 `effective_retries`：那两个值
    （凭据、生效的重试次数）都是**键的组成部分**，分开算就等于把
    "键算了一次、参数算了另一次"这件事留在两处，而它们一旦不一致，
    表现是缓存分键错误（同一份客户端被两种参数共用）。
    """

    value: tuple[Any, ...]

    credential: str | None

    max_retries: int


def _config_max_retries(
        definition: ModelDefinition,
        default: int,
) -> int:
    """
    从模型定义的 config 里读客户端重试次数（**显式覆盖**才生效）。

    这个键是"我知道我在干什么"的显式出口，理由见模块 docstring。
    它不是可靠性策略的一部分（那个在 ModelRuntimePolicy.max_retries）。
    """

    value = definition.config.get("max_retries")

    if value is None:
        return default

    if isinstance(value, bool) or not isinstance(value, int):
        raise ModelConfigurationError(
            f"模型 {definition.id!r} 的 config.max_retries 必须是整数，"
            f"得到 {value!r}。",
            provider=definition.provider,
            model=definition.model,
        )

    if value < 0:
        raise ModelConfigurationError(
            f"模型 {definition.id!r} 的 config.max_retries 不能为负"
            f"（得到 {value!r}）。",
            provider=definition.provider,
            model=definition.model,
        )

    return value


def _config_fingerprint(
        config: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    """
    把构造参数压成可哈希、可比较的键。

    值不可哈希时退化成 `repr`：它只用于**分键**（决定要不要复用同一个
    客户端）。两个 repr 相同的参数会共用客户端 —— 对这个用途来说正确
    足够了，而且比深比较便宜得多。
    """

    return tuple(
        (str(key), repr(config[key]))
        for key in sorted(config, key=str)
    )


def _secret_fingerprint(credential: str | None) -> str | None:
    """
    凭据的**摘要**（不是凭据本身）。

    ⚠️ 为什么不用凭据原文当缓存键
    ----------------------------
    缓存键会进调试器、可能进异常日志、也可能被人 `print(factory._clients)`。
    把密钥放进字典的键里，等于给它多加一条泄漏路径，而它本来只需要
    "区分两把不同的 key"这一个性质。

    用摘要还有个好处：**轮换 key 会让缓存自动失效** —— 同一把 ref 换了
    新 key，摘要变了，于是新 key 不会被旧的客户端吃掉。
    """

    if not credential:
        return None

    return hashlib.sha256(
        credential.encode("utf-8")
    ).hexdigest()[:16]
