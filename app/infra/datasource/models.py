from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DataSourceType(str, Enum):
    POSTGRESQL = "postgresql"
    ELASTICSEARCH = "elasticsearch"
    MILVUS = "milvus"
    NEO4J = "neo4j"
    REDIS = "redis"

    #: LangGraph checkpointer 的 PostgreSQL 连接。
    #:
    #: 与 POSTGRESQL 指向同一个库，但**是另一套连接**：
    #:   - 驱动不同：psycopg（checkpointer 只吃 psycopg）
    #:              而非 asyncpg / SQLAlchemy
    #:   - 连接参数不同：autocommit=True、prepare_threshold=0
    #:   - 生命周期不同：图运行的全程持有，不能和请求级事务混用
    #:   - DSN 形式不同：裸 postgresql://，不带 +asyncpg 方言后缀
    POSTGRESQL_CHECKPOINTER = "postgresql_checkpointer"


class BaseDataSourceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    type: DataSourceType

    enabled: bool = True

    required: bool = True

    startup_timeout_seconds: float = Field(
        default=5,
        gt=0,
    )


class PostgreSQLConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dsn: str

    dsn_psycopg: str | None = Field(
        default=None,
        description=(
            "psycopg 直连 DSN，供 LangGraph checkpointer 使用。"
            "checkpointer 走 psycopg 而非 SQLAlchemy，"
            "无法复用带 +asyncpg 方言后缀的 dsn。"
            "留空时由 resolved_dsn_psycopg 自动推导。"
        ),
    )

    checkpointer_pool_size: int = Field(
        default=5,
        ge=1,
        description=(
            "checkpointer 连接池大小。"
            "该池与 SQLAlchemy 的连接池相互独立，"
            "因为两者生命周期和事务边界完全不同。"
        ),
    )

    pool_size: int = Field(
        default=10,
        ge=1,
    )

    max_overflow: int = Field(
        default=20,
        ge=0,
    )

    pool_timeout: float = Field(
        default=30,
        gt=0,
    )

    pool_recycle: int = Field(
        default=1800,
        ge=-1,
    )

    echo: bool = False

    @property
    def resolved_dsn_psycopg(self) -> str:
        """
        返回可直接交给 psycopg / psycopg_pool 的 DSN。

        显式配置了 dsn_psycopg 时原样返回；
        否则从 dsn 推导，把方言后缀去掉：

            postgresql+asyncpg://...  ->  postgresql://...

        推导而非要求重复配置，是为了避免密码在两处出现、
        出现改动只改一处的漏配。
        """

        if self.dsn_psycopg:
            return self.dsn_psycopg

        from sqlalchemy.engine import make_url

        url = make_url(self.dsn)

        return url.set(
            drivername="postgresql"
        ).render_as_string(hide_password=False)


class ElasticsearchConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hosts: list[str]

    username: str | None = None

    password: str | None = None

    verify_certs: bool = True

    request_timeout: float = Field(
        default=5,
        gt=0,
    )


class MilvusConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    uri: str

    token: str | None = None

    db_name: str | None = "default"


class Neo4jConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: bolt://host:7687 或 neo4j://host:7687（集群走 neo4j:// 路由）。
    uri: str

    username: str | None = None

    password: str | None = None

    database: str | None = Field(
        default=None,
        description=(
            "目标数据库名。留空用服务端默认库。"
            "Neo4j 4 及更早是单库模型，那时必须留空。"
        ),
    )

    max_connection_pool_size: int = Field(
        default=100,
        ge=1,
        description=(
            "驱动内部连接池上限。注意这是**每进程**的："
            "多 worker 部署时总连接数是本值 × worker 数。"
        ),
    )

    connection_timeout: float = Field(
        default=5,
        gt=0,
        description=(
            "驱动建连超时（驱动内部）。"
            "与 startup_timeout_seconds 是两层："
            "后者是 health_check 的外层总超时，"
            "前者让建连本身更快失败。"
        ),
    )

    max_connection_lifetime: float = Field(
        default=3600,
        gt=0,
        description=(
            "连接最长存活秒数，到点后被回收重建。"
            "用于避开服务端/中间网络设备静默断开连接后的"
            "'半开连接'——那种连接上发查询会一直挂住。"
        ),
    )


class RedisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    #: redis://[[用户名:]密码@]主机:端口/库号
    #: 加密连接把 scheme 换成 rediss://
    #:
    #: 用整条 URL 而不是把 host / port / password 拆成几个字段：
    #: redis-py 本来就以 URL 为主入口（from_url），拆开再自己拼
    #: 只会多一层出错空间 —— 密码里的 `@` `:` `/` 需要百分号转义，
    #: 拼接实现极容易漏掉，而 from_url 里是标准库在做这件事。
    url: str

    max_connections: int = Field(
        default=10,
        ge=1,
        description=(
            "**短命令**池的上限（业务读写、health_check）。"
            "Redis 的命令执行本身是单线程的，把连接数开大并不会更快，"
            "只会占住服务端的连接额度，并让真正瓶颈（单线程执行）被掩盖。"
        ),
    )

    stream_max_connections: int = Field(
        default=20,
        ge=1,
        description=(
            "**长连接**池的上限（事件流订阅专用，见 RedisDataSource 的"
            "_stream_client）。\n\n"
            "为什么必须与 max_connections 分开\n"
            "-------------------------------\n"
            "两者的占用形态根本不同：短命令池里一条连接是「借一下还回去」，"
            "而事件流订阅者在整个订阅期间**一直占着**它那条（见 "
            "app/agent/event/run_stream.py）。共用一个池的后果已经实测过："
            "并发订阅数一到池上限，全进程的 Redis 访问就一起失败"
            "—— 包括与事件流毫无关系的调用。\n\n"
            "该开多大：**同时在看的 run 数**，不是订阅者数 —— 新设计下一个"
            "run 的所有订阅者共用一条连接。\n\n"
            "⚠️ 必须**严格大于** event_stream.yaml 里的 max_active_run_streams。"
            "redis-py 的池满时是**立刻抛** MaxConnectionsError，不会排队等；"
            "两者取等号时，一次重试或一次补发就能把它撞出来。留一点余量。"
        ),
    )

    socket_timeout: float | None = Field(
        default=5,
        gt=0,
        description=(
            "单条命令的读写超时（秒）。"
            "不设时 redis-py 会**无限等下去**，一个卡住的 Redis "
            "会把调用方的协程永久挂起 —— 这比报错难查得多。\n"
            "⚠️ 阻塞类命令（BLPOP / BRPOP / XREAD BLOCK / SUBSCRIBE）"
            "自身的阻塞时长必须小于本值，否则会被超时打断。"
            "那些场景请把它显式调大，或设为 null。"
        ),
    )

    socket_connect_timeout: float | None = Field(
        default=5,
        gt=0,
        description=(
            "建连超时（秒）。与 socket_timeout 是两层："
            "这个让'连不上'快速失败，那个管'连上了但没响应'。"
        ),
    )

    health_check_interval: float = Field(
        default=30,
        ge=0,
        description=(
            "连接空闲超过这么多秒后，下次取用前先 PING 一次。\n"
            "用途是踢掉被服务端或中间网络设备静默断开的'半开连接'"
            "—— 那种连接上发命令会一直挂住，直到 socket_timeout 才报错。\n"
            "0 表示关闭该检查（redis-py 的默认值）。"
        ),
    )

    decode_responses: bool = Field(
        default=False,
        description=(
            "是否把响应解码成 str。默认 False（返回 bytes）是刻意的：\n"
            "本数据源对上层是通用的，而 bytes 是唯一没有信息损失的形态"
            "—— 解成 str 之后再想拿回原始字节（pickle / 压缩 / "
            "protobuf 编码的值）就已经晚了。需要文本的调用方自己 decode。"
        ),
    )

    client_name: str | None = Field(
        default=None,
        description=(
            "连接名。会通过 CLIENT SETNAME 设到服务端，"
            "在 CLIENT LIST / 慢查询日志里能认出是谁连的。"
        ),
    )


class StoreSettings(BaseModel):
    """
    同一个池上的 LangGraph store（跨轮存储）的 TTL 设置。

    它挂在 checkpointer 数据源下，因为两者**共用同一个连接池**：
    store 要求的连接参数（autocommit / prepare_threshold=0 /
    row_factory=dict_row）与 checkpointer 逐字相同，另开一个池只是多占
    一份连接额度。见 `app/infra/graph/checkpointer.py` 的模块 docstring。

    ⚠️ 这里的单位是**分钟**，不是秒
    ------------------------------
    langgraph 的 `BaseStore.aput(ttl=…)` 以分钟计（其 docstring 原文：
    *"Time to live in minutes"*），PG 实现里直接
    `ttl_minutes = float(op.ttl)` 再 `NOW() + ttl_minutes * 60 seconds`。

    ⚠️ 本仓库另有一个**按秒**的对外契约：审批配置里的
    `ApprovalConfig.always_ttl_seconds`（见
    services/tool 的处理器 schema）。两者不要互相折算 —— 秒→分钟的换算
    收在 `StoreApprovalLedger` 那一处（approval.py），这里按分钟原样填。
    """

    model_config = ConfigDict(extra="forbid")

    default_ttl_minutes: int | None = Field(
        default=10_080,
        ge=1,
        description=(
            "**没显式传 ttl** 的写入项的有效期（分钟）。默认 7 天 ——\n"
            "远超任何人工审批的等待，又不会留成永不过期的死数据。\n\n"
            "⚠️ 它不是「顺手给个清理时间」，少了它有具体后果：那些写入的\n"
            "`expires_at` 会是 NULL，而清扫的 SQL 是\n"
            "`WHERE expires_at IS NOT NULL AND expires_at < NOW()` ——\n"
            "等于**永远清不掉，store 表只增不减**。审批账本里「单次批准」\n"
            "那一笔（approval.py 刻意不给它 ttl）正是这种写入。\n\n"
            "设成 None 就退回「永不过期」：只适合确实不需要清理的场景。"
        ),
    )

    sweep_interval_minutes: int = Field(
        default=60,
        ge=1,
        description=(
            "后台清扫的间隔（分钟）。⚠️ 清扫任务**不会自动启动**，"
            "由 GraphCheckpointerProvider 在 open 时显式拉起"
            "（`start_ttl_sweeper()`）、在 close 时停掉。"
        ),
    )


class CheckpointerConfig(BaseModel):
    """
    LangGraph checkpointer 的连接配置。

    dsn 与 derive_from 二选一：
        dsn            直接给裸 psycopg DSN
        derive_from    引用另一个已注册的 postgresql 数据源名，
                       从它的配置里推导（去掉 +asyncpg 方言后缀）

    推荐用 derive_from —— 密码只写一处，改的时候不会漏配。

    它还带一个 `store` 段：**同一个数据源、同一个池**上会再建一个
    LangGraph store（跨轮存储）。目前 store 唯一的消费者是审批拦截器的
    批准账本，而那张账本必须跨重启 —— 否则重启后停在 `waiting_input`
    的运行 resume 时，之前批过的会再问一遍。
    """

    model_config = ConfigDict(extra="forbid")

    dsn: str | None = None

    derive_from: str | None = "postgresql"

    pool_min_size: int = Field(
        default=1,
        ge=1,
    )

    pool_max_size: int = Field(
        default=5,
        ge=1,
    )

    store: StoreSettings = Field(
        default_factory=StoreSettings,
        description=(
            "同一个池上的 LangGraph store。见 StoreSettings 的说明。"
        ),
    )


class PostgreSQLDataSourceDefinition(BaseDataSourceConfig):
    type: Literal[DataSourceType.POSTGRESQL] = (
        DataSourceType.POSTGRESQL
    )

    config: PostgreSQLConfig


class ElasticsearchDataSourceDefinition(BaseDataSourceConfig):
    type: Literal[DataSourceType.ELASTICSEARCH] = (
        DataSourceType.ELASTICSEARCH
    )

    config: ElasticsearchConfig


class MilvusDataSourceDefinition(BaseDataSourceConfig):
    type: Literal[DataSourceType.MILVUS] = (
        DataSourceType.MILVUS
    )

    config: MilvusConfig


class Neo4jDataSourceDefinition(BaseDataSourceConfig):
    type: Literal[DataSourceType.NEO4J] = (
        DataSourceType.NEO4J
    )

    config: Neo4jConfig


class RedisDataSourceDefinition(BaseDataSourceConfig):
    type: Literal[DataSourceType.REDIS] = (
        DataSourceType.REDIS
    )

    config: RedisConfig


class CheckpointerDataSourceDefinition(BaseDataSourceConfig):
    type: Literal[DataSourceType.POSTGRESQL_CHECKPOINTER] = (
        DataSourceType.POSTGRESQL_CHECKPOINTER
    )

    config: CheckpointerConfig


DataSourceDefinition = Annotated[
    (
            PostgreSQLDataSourceDefinition
            | ElasticsearchDataSourceDefinition
            | MilvusDataSourceDefinition
            | Neo4jDataSourceDefinition
            | RedisDataSourceDefinition
            | CheckpointerDataSourceDefinition
    ),
    Field(discriminator="type"),
]
