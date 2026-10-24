"""
L2/L3 画像的跨请求缓存。

为什么需要
----------
`MemoryProfileProcessor`（processors/profile.py）**每一次上下文构建**都会
读一遍 L2 场景索引 + L3 画像。这两层变化很慢（L2 由后台整合任务写、L3
由 L2 举手后重写），但读取发生在热路径上：压测实测 p50 334ms。

读路径：查询前 get -> miss 则查库 -> 查询后 put。
写路径：L2/L3 抽取任务成功后，在 **post_commit** 里删掉（事务提交后才删，
避免"删了缓存、事务又回滚，读者把旧值写回去"）。

键
--
    L2: profile_cache.l2:{agent_id}:{scope.key}
    L3: profile_cache.l3:{agent_id}:{scope.key}

⚠️ `SceneScope.key` 只是 `user:<id>` / `org:<id>`，**不含 agent_id**
（见 domain/entities/scene.py 的 key 属性）。而 persona/scene 的 WHERE
条件里 `agent_id` 是第一个条件（persona_impl / scene_impl 的
`_scope_conditions`）。所以键里必须显式带上 agent_id，否则同一个用户
在不同 agent（duty_key）下的画像会互相串。

⚠️ L2 / L3 **分开缓存**：两者的更新时机不同（L2 每次整合、L3 只在
L2 举手或首次创建时），合成一个键会让 L2 的一次变化白丢 L3 的缓存。
读路径与写路径都用 `profile_cache_key(scope, layer)`，两处不会漂。

实现复用
--------
复用 `InProcessToolCache` / `RedisToolCache` —— 它们只是带 TTL 的 KV，
与"工具"没有耦合（工具处理器只是其中一个调用方）。Redis 键前缀由
`RedisToolCache(key_prefix=PROFILE_CACHE_PREFIX)` 加。

失败策略
--------
fail-open：缓存不可用（没装配 / Redis 抖动 / 反序列化失败）一律按未命中
处理，查询照常走库。缓存是优化，不能让它把一次正常的上下文构建拖失败。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.tools.processor.runtime.cache import (
    InProcessToolCache,
    RedisToolCache,
    ToolCache,
)
from app.services.memory.domain.entities.scene import SceneScope

__all__ = [
    "PROFILE_CACHE_LAYERS",
    "PROFILE_CACHE_MAX_ENTRIES",
    "PROFILE_CACHE_PREFIX",
    "PROFILE_CACHE_TTL_SECONDS",
    "configure_profile_cache",
    "get_cached_profile_blocks",
    "get_profile_cache",
    "invalidate_profile_cache",
    "profile_cache_invalidation_hook",
    "profile_cache_key",
    "put_cached_profile_blocks",
    "reset_profile_cache",
]


logger = logging.getLogger(__name__)


#: 键前缀。最终键：`profile_cache.{layer}:{agent_id}:{scope.key}`。
PROFILE_CACHE_PREFIX = "profile_cache."

#: 允许的层。分开缓存，失效也按层来。
PROFILE_CACHE_LAYERS = ("l2", "l3")

#: TTL（秒）。写路径会显式失效，TTL 只是兜底 —— 漏掉某个写路径时最多
#: 脏这么久。可用环境变量覆盖。
PROFILE_CACHE_TTL_SECONDS = int(
    os.getenv("MEMORY_PROFILE_CACHE_TTL_SECONDS", "600")
)

#: 进程内实现的容量（没有 Redis 时使用）。由 bootstrap 装配时读取。
PROFILE_CACHE_MAX_ENTRIES = int(
    os.getenv("MEMORY_PROFILE_CACHE_MAX_ENTRIES", "2048")
)


#: 进程级单例。None = 尚未装配。
_cache: ToolCache | None = None

#: 是否已经尝试过装配。False 时下一次 get_profile_cache() 会去
#: RedisClientFactory 取客户端；取不到就退回进程内实现（只试一次）。
_cache_built = False


def configure_profile_cache(cache: ToolCache | None) -> None:
    """直接安装一个实现（测试用；生产走 get_profile_cache 的懒装配）。"""

    global _cache, _cache_built

    _cache = cache
    _cache_built = True


def get_profile_cache() -> ToolCache:
    """
    取进程级实现。**懒装配**：

    - 首次调用时从 `get_redis_factory().optional_client()` 取客户端；
      有 Redis -> `RedisToolCache`，没有 -> `InProcessToolCache`。
    - 应用级 Redis 工厂由 lifespan 装配（与 session_factory 同形），
      所以这里不需要额外的启动初始化步骤。
    - 取不到工厂（测试 / 冒烟脚本没走 lifespan）同样退回进程内实现。
    """

    global _cache, _cache_built

    if _cache_built:

        # 极端情况下 configure(None) 过：仍然返回一个可用实现，
        # 而不是让调用方分支判 None。
        if _cache is None:
            _cache = InProcessToolCache(
                max_entries=PROFILE_CACHE_MAX_ENTRIES
            )
        return _cache

    _cache_built = True

    client = None

    try:
        from app.common.factories import get_redis_factory

        client = get_redis_factory().optional_client()

    except Exception:  # noqa: BLE001 —— 未装配 / 无 Redis 都退回进程内
        client = None

    if client is None:
        logger.warning(
            "画像缓存：Redis 不可用，退回进程内实现。"
            "多实例之间不共享、重启清零 —— 缓存仍然可用。"
        )

        _cache = InProcessToolCache(
            max_entries=PROFILE_CACHE_MAX_ENTRIES
        )

    else:
        # 前缀已经拼进 profile_cache_key()，这里不再加一层。
        _cache = RedisToolCache(client, key_prefix="")

    return _cache


def reset_profile_cache() -> None:
    """清掉单例与懒装配标记（测试用）。"""

    global _cache, _cache_built

    _cache = None
    _cache_built = False


def profile_cache_key(
        scope: SceneScope,
        layer: str,
) -> str:
    """
    scope + 层 -> 缓存键：

        profile_cache.l2:{agent_id}:{scope.key}
        profile_cache.l3:{agent_id}:{scope.key}

    ⚠️ `scope.key` 不含 agent_id（见模块 docstring），必须显式拼上。
    读路径与写路径都用这一个函数，两处不会漂。
    """

    if layer not in PROFILE_CACHE_LAYERS:
        raise ValueError(
            f"未知的画像缓存层 {layer!r}；只允许 {PROFILE_CACHE_LAYERS}。"
        )

    return f"{PROFILE_CACHE_PREFIX}{layer}:{scope.agent_id}:{scope.key}"


async def get_cached_profile_blocks(
        scope: SceneScope,
        layer: str,
) -> tuple[bool, list[dict[str, Any]] | None]:
    """
    读某一层的缓存。返回 `(是否命中, blocks)`。

    命中时 blocks 可能是**空列表**（这个 scope 还没有画像 / 场景）——
    那也是有效结果，应当直接返回而不查库；写路径会在之后建出画像时
    删掉这一条。
    """

    cache = get_profile_cache()

    try:
        hit, value = await cache.get(profile_cache_key(scope, layer))

    except Exception:  # noqa: BLE001 —— 见模块 docstring
        logger.warning(
            "画像缓存读取异常，按未命中处理（scope=%s，layer=%s）。",
            scope.key,
            layer,
            exc_info=True,
        )
        return False, None

    if not hit:
        return False, None

    if not isinstance(value, list):
        # 形状不对（旧版本 / 别的东西写进来的）。当作未命中，让本次查询
        # 覆盖它 —— 不要在这里抛。
        return False, None

    blocks = [
        dict(item)
        for item in value
        if isinstance(item, Mapping)
    ]

    return True, blocks


async def put_cached_profile_blocks(
        scope: SceneScope,
        layer: str,
        blocks: Sequence[Mapping[str, Any]],
) -> None:
    """写某一层的缓存。fail-open。"""

    cache = get_profile_cache()

    try:
        await cache.put(
            profile_cache_key(scope, layer),
            [dict(item) for item in blocks],
            ttl_seconds=PROFILE_CACHE_TTL_SECONDS,
        )

    except Exception:  # noqa: BLE001 —— 见模块 docstring
        logger.warning(
            "画像缓存写入异常，本次不缓存（scope=%s，layer=%s）。",
            scope.key,
            layer,
            exc_info=True,
        )


async def invalidate_profile_cache(
        scope: SceneScope,
        layer: str,
) -> None:
    """删某一层的缓存。fail-open。"""

    cache = get_profile_cache()

    try:
        await cache.invalidate(profile_cache_key(scope, layer))

    except Exception:  # noqa: BLE001 —— 见模块 docstring
        logger.warning(
            "画像缓存删除异常（scope=%s，layer=%s）。TTL 会兜底。",
            scope.key,
            layer,
            exc_info=True,
        )


def profile_cache_invalidation_hook(
        scope: SceneScope,
        layer: str,
) -> Any:
    """
    返回一个 **post-commit** 钩子（async callable）。

    用在 L2/L3 抽取服务里：

        self.post_commit.append(
            profile_cache_invalidation_hook(scope, "l2")
        )

    必须在事务**提交之后**才删：提前删的话，读者可能在事务提交前把旧值
    重新查出来写回缓存，而那个旧值随后才被这次事务更新 —— 缓存就永久
    （直到 TTL）停在旧值上。
    """

    async def _invalidate() -> None:
        await invalidate_profile_cache(scope, layer)

    return _invalidate
