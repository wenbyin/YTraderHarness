from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class DataSourceType(str, Enum):
    POSTGRESQL = "postgresql"
    ELASTICSEARCH = "elasticsearch"
    MILVUS = "milvus"
    NEO4J = "neo4j"

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


class CheckpointerConfig(BaseModel):
    """
    LangGraph checkpointer 的连接配置。

    dsn 与 derive_from 二选一：
        dsn            直接给裸 psycopg DSN
        derive_from    引用另一个已注册的 postgresql 数据源名，
                       从它的配置里推导（去掉 +asyncpg 方言后缀）

    推荐用 derive_from —— 密码只写一处，改的时候不会漏配。
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
            | CheckpointerDataSourceDefinition
    ),
    Field(discriminator="type"),
]
