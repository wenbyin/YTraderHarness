"""
应用级工厂的装配与取用。

为什么是应用级（而不是请求级）
------------------------------
工厂里装的是"这一份运行时依赖"：session_factory 从哪个数据源来、
调度器 / 能力注册表 / 图注册表 / 存储各用哪个实例。这些全是
**进程级单例**，整个应用一份，建一次就该处处可用 —— 包括
**手上没有 FastAPI Request 的地方**：

    请求路径        Service[T] 依赖
    后台 / 流水线    任务里自己开 scope
    Agent 工具      被模型调用时，根本没有 Request

原来工厂挂在 app.state 上、靠 `service_factory_of(request)` 取，
第三条拿不到。所以改成应用级：lifespan 里装配一次，
之后任何地方 `get_service_factory()` 即可。

app.common 保持纯叶子
---------------------
本模块**不认识任何具体服务或仓储**，只做两件事：

    存住调用方交来的工厂
    按 session_factory + 池子构造它们

池子里装什么完全由调用方（main.py）决定，因此这里不需要
import app.services.* 或 app.infra.* —— 那条"app.common 不得反向
依赖"的纪律因此仍然成立（见 common/exceptions.py）。

与 model_factory 的区别
-----------------------
model_factory 也是进程级单例，但它是**自己管自己**的（模型由 key
登记，注册表是它的一部分）。这里相反：工厂的内容物来自外面，
本模块只负责"放哪儿"和"从哪儿拿"。所以是 holder，不是 registry。
"""

from __future__ import annotations

from typing import Any, Mapping

from .redis_factory import RedisClientFactory, RedisClients
from .repository.base import SessionFactory
from .repository.factory import RepositoryFactory
from .service.factory import ServiceFactory

__all__ = [
    "build_factories",
    "build_redis_factory",
    "get_redis_factory",
    "get_repository_factory",
    "get_service_factory",
    "reset_factories",
    "reset_redis_factory",
]


#: 应用级单例。lifespan 装配时写入，之后只读。
_service_factory: ServiceFactory | None = None
_repository_factory: RepositoryFactory | None = None
_redis_factory: RedisClientFactory | None = None


def build_factories(
        session_factory: SessionFactory,
        *,
        pool: Mapping[str, Any] | None = None,
) -> ServiceFactory:
    """
    装配两个工厂并装到应用级。

    由 lifespan 调用一次，必须在数据源、调度器、注册表、存储都
    就绪**之后** —— 池子里的实例会被 ServiceFactory 抓走，
    之后不再重新读取，放早了会得到一个缺依赖的工厂。

    pool 的键既是 ctx.extras 里的取用名，也是构造服务时**按参数名**
    匹配的依据（见 ServiceFactory._build_kwargs）。装什么由调用方
    决定：进程级单例（dispatcher / pipeline_registry / …）与
    模块专属固定值（图名 / 输出目录…）都放这里。

    返回 ServiceFactory 只是为了方便调用方，取用请用
    get_service_factory() —— 那才是"应用级"的意义。
    """

    global _service_factory, _repository_factory

    _repository_factory = RepositoryFactory(
        session_factory=session_factory,
    )

    _service_factory = ServiceFactory(
        _repository_factory,
        pool=dict(pool or {}),
    )

    return _service_factory


def get_service_factory() -> ServiceFactory:
    """
    取应用级服务工厂。

    没有时**明确报错**而不是现场造一个 —— 现场造出来的会缺
    池子里的全部单例（调度器、注册表、存储），失败点会跑到
    某个服务构造时才出现，离原因很远。真正的原因只有一个：
    应用没跑过 lifespan。
    """

    if _service_factory is None:
        raise RuntimeError(
            "应用级 ServiceFactory 还没装配。"
            "它由 lifespan 调用 build_factories() 建立 —— "
            "请确认应用正常启动，或在测试里显式装配一次。"
        )

    return _service_factory


def get_repository_factory() -> RepositoryFactory:
    """取应用级仓储工厂。语义与 get_service_factory 相同。"""

    if _repository_factory is None:
        raise RuntimeError(
            "应用级 RepositoryFactory 还没装配。"
            "它由 lifespan 调用 build_factories() 建立。"
        )

    return _repository_factory


def build_redis_factory(
        source: RedisClients | None,
) -> RedisClientFactory:
    """
    装配应用级 Redis 客户端工厂。

    由 lifespan 在数据源就绪之后调用一次。`source` 为 None（本地没起
    Redis，`required: false`）是合法状态 —— 工厂的 `available` 为 False，
    调用方据此退回进程内实现。

    与 `build_factories` 同一条纪律：本模块只保存，不建连、不关池。
    """

    global _redis_factory

    _redis_factory = RedisClientFactory(source)

    return _redis_factory


def get_redis_factory() -> RedisClientFactory:
    """取应用级 Redis 工厂。未装配时抛 —— 与 get_service_factory 同形。"""

    if _redis_factory is None:
        raise RuntimeError(
            "应用级 RedisClientFactory 还没装配。"
            "它由 lifespan 调用 build_redis_factory() 建立 —— "
            "请确认应用正常启动，或在测试里显式装配一次。"
        )

    return _redis_factory


def reset_redis_factory() -> None:
    """清掉 Redis 工厂单例（测试用）。"""

    global _redis_factory

    _redis_factory = None


def reset_factories() -> None:
    """
    丢弃已装配的工厂。

    **只给测试用**：让每个用例能装配自己的假 session_factory，
    而不必忍受上一个用例留下的全局状态。生产上丢弃工厂没有
    正当理由 —— 它持有的连接池还在被别处引用。
    """

    global _service_factory, _repository_factory

    _service_factory = None
    _repository_factory = None

    reset_redis_factory()
