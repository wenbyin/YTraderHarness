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

它现在还持有 **store**（LangGraph 的跨轮存储）
--------------------------------------------
同一个池上还建了一个 `AsyncPostgresStore`，两者一起 open、一起 close。
目前 store 唯一的消费者是审批拦截器的批准账本
（app/agent/tools/processor/interceptor/approval.py）—— 那张账本必须
跨重启，否则重启后停在 `waiting_input` 的运行 resume 时，之前批过的
会**再问一遍**，而 warning 整个进程只喊一次。

⚠️ 为什么不另开一个数据源
------------------------
`AsyncPostgresStore` 需要的连接参数与 checkpointer **逐字相同** ——
autocommit=True、prepare_threshold=0、row_factory=dict_row，正是它自己的
`from_conn_string` 建池时用的那三个（见 `_ensure_open` 里的注释）。
再开一个池只是多占一份连接额度，而 store 一次只借出一条连接。

代价说清楚：本类的名字因此比它的职责**窄了**。改名的成本落在
`DataSourceType` 与配置里的 `type:` 字符串上（那是对外配置契约），
所以留着 —— 但读到这个名字时要知道它管两样东西。

⚠️ store 的 TTL 有两个开关必须显式打开
------------------------------------
`aput(ttl=…)` 只负责写 `expires_at`，**不代表读的时候会过滤**：

    omit_expired          默认 False —— 过期但未清扫的行照样被 aget 读回来
    start_ttl_sweeper()   不会自动启动；ttl_config 为空时它直接返回一个
                          `asyncio.sleep(0)` 空任务，看起来像启动了

两个都缺，"有效期"就只是个装饰。见 `_ensure_open`。
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from .checkpoint_concurrency import disable_saver_lock
from langgraph.store.base import TTLConfig
from langgraph.store.postgres.aio import AsyncPostgresStore
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
            parallel_cursor_operations: bool = True,
            store_default_ttl_minutes: int | None = 10_080,
            store_sweep_interval_minutes: int = 60,
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

        #: 是否允许 saver 的 cursor 操作并行。连接池模式下应为 True ——
        #: 理由见 checkpoint_concurrency 的模块 docstring。设成 False 只是留一条
        #: 可一键退回旧行为的开关。
        self._parallel_cursor_operations = parallel_cursor_operations

        # ⚠️ 单位是**分钟** —— langgraph 的 BaseStore 就按分钟收
        # （见 `_ensure_open` 里 TTLConfig 那段的说明）。配置侧
        # 已经按分钟写，这里不再折算。
        self._store_default_ttl = (
            None
            if store_default_ttl_minutes is None
            else float(store_default_ttl_minutes)
        )
        self._store_sweep_interval = store_sweep_interval_minutes

        self._pool: AsyncConnectionPool | None = None
        self._saver: AsyncPostgresSaver | None = None
        self._store: AsyncPostgresStore | None = None

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
        """
        停掉 store 的清扫任务，再关闭连接池。

        由 DataSourceManager.close_all() 调用 —— 那是**按注册逆序**，
        而本数据源在 DataSourceStartup 里被排到最后注册，因此是第一个
        被关的。共用同一个池的好处在这里：一个对象、一个池，不存在
        "谁先把池关了"的问题。
        """

        pool = self._pool

        if pool is None:
            return

        store = self._store

        self._pool = None
        self._saver = None
        self._store = None

        # ⚠️ 顺序不能反：sweeper 每一轮都要从池里借连接，先关池会让它
        # 下一轮以一个"池已关闭"的异常收场。那个异常被它自己的循环吞掉，
        # 只留一条没人看的 log —— 不值得为它留个隐患。
        #
        # timeout 给有限值而不是 None：它可能正卡在一次 sweep_ttl() 的
        # 查询上，无上限地等下去会让整个应用的关停挂在这里。
        if store is not None:
            await store.stop_ttl_sweeper(timeout=5)

        await pool.close()

        logger.info(
            "Checkpointer datasource closed: %s", self.name
        )

    def resource(self) -> Any:
        """
        返回 saver，便于统一按 resource() 取用。

        ⚠️ 只有 saver —— 本数据源现在也持有 store，但两者不是"同一个资源
        的两种说法"：`compile(checkpointer=…, store=…)` 要的是两个具名
        参数，所以 store 走 `store` 属性，不在这里捏成一个返回值。

        （`resource()` 目前的唯一调用点是 main.py 的索引后端那条泛化路径，
        它按名字取 ES / Milvus / Neo4j —— checkpointer 不在其中。）
        """
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

        # ⚠️ AsyncPostgresSaver 用一个 asyncio.Lock 串行化**全部** cursor
        # 操作（aio.py 的 `_cursor`）。连接池下每次操作各借各的连接，那把
        # 锁没有存在理由，却会把整个进程的 checkpoint 读写排成一条队。
        # 压测实测：322 个 run 时空转两次 checkpoint 读要 1.6s / 4.7s。
        if self._parallel_cursor_operations:
            disable_saver_lock(self._saver)
            logger.info(
                "Checkpointer: saver 的全局串行锁已关闭，cursor 操作按连接池并发。"
            )

        # setup() 的签名看着像同步方法，实际是协程，必须 await。
        await self._saver.setup()

        # ---- 同一个池上的 store（跨轮存储）------------------------
        #
        # 共用这个池是**可以**的，不是凑合：store 要求的连接参数与 saver
        # 逐字相同（autocommit / prepare_threshold=0 / row_factory=dict_row
        # —— 正是 AsyncPostgresStore.from_conn_string 自己建池时写的那三个），
        # 而且 pool 版的 store 每次操作都自己从池里 checkout 一条连接，
        # 不与 saver 抢同一条。
        #
        # ⚠️ TTLConfig 里没有一个是"顺手配的"
        # -----------------------------------
        #   omit_expired=True       读路径才会过滤掉"已过期但还没被清扫"的行。
        #                           默认是 False —— 那时 `aget` 照样把过期
        #                           记录还给你，有效期形同虚设。
        #   default_ttl             让**没显式传 ttl** 的写入也有 expires_at。
        #                           没有它，那些行的 expires_at 是 NULL，而
        #                           sweep_ttl() 的 SQL 是
        #                           `WHERE expires_at IS NOT NULL AND
        #                            expires_at < NOW()` —— 永远清不掉，
        #                           表只增不减。审批账本里"单次批准"那一笔
        #                           正是这种写入（approval.py 刻意不给它 ttl）。
        #   sweep_interval_minutes  后台清扫的间隔。
        store = AsyncPostgresStore(
            pool,
            ttl=TTLConfig(
                omit_expired=True,
                default_ttl=self._store_default_ttl,
                sweep_interval_minutes=self._store_sweep_interval,
            ),
        )

        # 与 saver 的 setup() 同理：签名看着同步，实际是协程。它建的是
        # store / store_migrations（+ 索引时才有的 vector_migrations），
        # 与 checkpointer 那几张表互不相干。幂等，重复调用安全。
        await store.setup()

        self._store = store

        # ⚠️ 清扫任务**不会自动启动**，必须显式拉起 —— ttl_config 为空时
        # 它直接返回一个 `asyncio.sleep(0)` 空任务，看起来像启动了。这也是
        # 上面那个 TTLConfig 不能省的另一半理由。
        await store.start_ttl_sweeper()

        logger.info(
            "LangGraph checkpointer ready: %s (pool %d-%d, "
            "store default_ttl=%s min, sweep every %s min).",
            self.name,
            self._min_size,
            self._max_size,
            self._store_default_ttl,
            self._store_sweep_interval,
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
    def store(self) -> AsyncPostgresStore:
        """
        编译图时传给 `graph.compile(store=...)`。

        目前唯一的消费者是审批拦截器的批准账本（见模块 docstring）。
        与 `saver` 同一条立场：未 open 就取用会抛错，而不是返回 None
        让问题推迟到图运行时 —— 那时它会静默退化成"进程内账本"，
        只有一条 warning 提示（approval.py 的 `_ledger`）。
        """

        if self._store is None:
            raise RuntimeError(
                "GraphCheckpointerProvider is not open. "
                "DataSourceStartup opens it during health_check(); "
                "if you constructed it manually, "
                "await provider.health_check() first."
            )

        return self._store

    @property
    def is_open(self) -> bool:
        return self._saver is not None
