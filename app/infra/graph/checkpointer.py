"""
PostgreSQL checkpointer，作为一个标准 DataSource。

为什么它也是 DataSource
-----------------------
它需要连接、需要探活、需要随应用关闭释放 —— 与 ES / Milvus
完全同构，没有理由自己再写一套生命周期。

实现 DataSource 之后，它自动获得 DataSourceStartup 的全套语义：

    create -> health_check -> register
    enabled / required / startup_timeout_seconds
    关闭时按注册逆序统一释放

于是 main.py 里只剩 `manager.get("checkpointer").saver` 一行，
不再有第二个连接池要手工管理。

为什么是独立的一套连接池
------------------------
checkpointer 走 **psycopg**，业务侧走 **asyncpg/SQLAlchemy**：

    datasource/postgresql.py   asyncpg      -> 业务读写
    graph/checkpointer.py      psycopg      -> checkpoint

两者**绝不能共用**连接，也绝不能试图合并事务：

    - checkpointer 在自己事务里写 checkpoints / checkpoint_blobs /
      checkpoint_writes，与业务事务没有原子性关系
    - 它要求 autocommit=True 且 prepare_threshold=0，与业务连接
      的参数完全不同
    - 一次运行期间会反复写，混进业务池会拖垮请求处理

因此 DSN 也不同：checkpointer 要裸 postgresql://。
配置里用 `derive_from: postgresql` 从业务数据源推导，
避免密码写两处（见 datasource/factory.py）。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from app.infra.datasource.base import DataSource

logger = logging.getLogger(__name__)

__all__ = ["GraphCheckpointerProvider"]


class GraphCheckpointerProvider(DataSource):
    """
    持有 checkpointer 的连接池与 saver。

    生命周期由 DataSourceStartup 驱动：

        create()          仅构造，不连接
        health_check()    首次调用时建池 + setup()，随后 SELECT 1
        close()           关闭连接池

    把"建池"放进 health_check 是刻意的：DataSourceStartup 的契约
    就是 create -> health_check -> register，而 checkpointer 确实
    需要一个连接步骤。这样不用给启动器加特例。
    """

    def __init__(
            self,
            name: str,
            *,
            dsn: str,
            pool_min_size: int = 1,
            pool_max_size: int = 5,
    ) -> None:

        super().__init__(name)

        if "+" in dsn.split("://", 1)[0]:
            raise ValueError(
                "GraphCheckpointerProvider requires a plain psycopg "
                "DSN without a SQLAlchemy dialect suffix, got: "
                f"{dsn.split('://', 1)[0]}://... . "
                "Set config.dsn to a postgresql:// DSN, or use "
                "derive_from to derive it from the postgresql "
                "datasource."
            )

        self._dsn = dsn
        self._min_size = pool_min_size
        self._max_size = pool_max_size

        self._pool: AsyncConnectionPool | None = None
        self._saver: AsyncPostgresSaver | None = None

    # ========================================================
    # DataSource
    # ========================================================

    async def health_check(self) -> bool:
        """
        首次调用负责建池并建表；之后只做 SELECT 1 探活。

        setup() 是幂等的，重复调用安全。
        """

        try:
            await self._ensure_open()

            assert self._pool is not None

            async with self._pool.connection() as conn:
                await conn.execute("SELECT 1")

        except Exception:
            logger.exception(
                "Checkpointer health check failed (%s).",
                self.name,
            )
            return False

        return True

    async def close(self) -> None:
        """关闭连接池。由 DataSourceManager.close_all() 调用。"""

        pool = self._pool

        if pool is None:
            return

        self._pool = None
        self._saver = None

        await pool.close()

        logger.info(
            "Checkpointer datasource closed: %s", self.name
        )

    def resource(self) -> Any:
        """返回 saver，便于统一按 resource() 取用。"""
        return self.saver

    # ========================================================
    # DataSourceStartup 契约之外
    # ========================================================

    #: DataSourceManager.close_all() 调的是 close()，
    #: 保留 aclose 别名是为了让调用点读起来更明确。
    aclose = close

    async def _ensure_open(self) -> None:
        """幂等地建池、建 saver、建表。"""

        if self._pool is not None:
            return

        # open=False 后手动 open()：构造函数里直接开池是阻塞的，
        # 在事件循环里会卡住启动。
        pool = AsyncConnectionPool(
            conninfo=self._dsn,
            min_size=self._min_size,
            max_size=self._max_size,
            open=False,
            kwargs={
                # 这三个参数与 AsyncPostgresSaver.from_conn_string
                # 内部使用的完全一致，缺一个都会让 saver 行为异常：
                #   autocommit            每条语句自成事务
                #   prepare_threshold=0   禁用预备语句缓存，避免
                #                         连接复用时拿到失效的 plan
                #   row_factory           saver 按 dict 取值
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )

        await pool.open(wait=True)

        self._pool = pool
        self._saver = AsyncPostgresSaver(pool)

        # setup() 的签名看着像同步方法，实际是协程，必须 await。
        await self._saver.setup()

        logger.info(
            "LangGraph checkpointer ready: %s (pool %d-%d).",
            self.name,
            self._min_size,
            self._max_size,
        )

    @property
    def saver(self) -> AsyncPostgresSaver:
        """
        编译图时传给 graph.compile(checkpointer=...)。

        未 open 就取用会抛错，而不是返回 None 让问题推迟到
        图运行时才以难查的方式爆出来。
        """

        if self._saver is None:
            raise RuntimeError(
                "GraphCheckpointerProvider is not open. "
                "DataSourceStartup opens it during health_check(); "
                "if you constructed it manually, "
                "await provider.health_check() first."
            )

        return self._saver

    @property
    def is_open(self) -> bool:
        return self._saver is not None
