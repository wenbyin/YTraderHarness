from __future__ import annotations

from .base import DataSource
from .elasticsearch import (
    ElasticsearchDataSource,
)
from .milvus import MilvusDataSource
from .manager import DataSourceManager
from .models import (
    CheckpointerDataSourceDefinition,
    DataSourceDefinition,
    DataSourceType,
    ElasticsearchDataSourceDefinition,
    MilvusDataSourceDefinition,
    Neo4jDataSourceDefinition,
    PostgreSQLDataSourceDefinition,
)
from .neo4j import Neo4jDataSource
from .postgresql import PostgreSQLDataSource


class DataSourceFactory:

    @staticmethod
    def create(
            definition: DataSourceDefinition,
            *,
            manager: DataSourceManager | None = None,
    ) -> DataSource:
        """
        Parameters
        ----------
        manager
            checkpointer 数据源配置了 derive_from 时，需要用它
            去取被引用的 postgresql 数据源来推导 DSN。
            其余类型不需要。
        """

        if isinstance(
                definition,
                CheckpointerDataSourceDefinition,
        ):
            # 延迟 import：app.infra.graph 依赖 langgraph / psycopg，
            # 而不跑流水线的部署（例如只提供查询接口）不该被迫装上它们。
            from app.infra.graph.checkpointer import (
                GraphCheckpointerProvider,
            )

            return GraphCheckpointerProvider(
                name=definition.name,
                dsn=DataSourceFactory._resolve_checkpointer_dsn(
                    definition, manager
                ),
                pool_min_size=(
                    definition.config.pool_min_size
                ),
                pool_max_size=(
                    definition.config.pool_max_size
                ),
            )

        if isinstance(
                definition,
                PostgreSQLDataSourceDefinition,
        ):
            return PostgreSQLDataSource(
                definition
            )

        if isinstance(
                definition,
                ElasticsearchDataSourceDefinition,
        ):
            return ElasticsearchDataSource(
                definition
            )

        if isinstance(
                definition,
                MilvusDataSourceDefinition,
        ):
            return MilvusDataSource(
                definition
            )

        if isinstance(
                definition,
                Neo4jDataSourceDefinition,
        ):
            return Neo4jDataSource(
                definition
            )

        raise ValueError(
            f"Unsupported datasource type: "
            f"{definition.type}"
        )

    # ========================================================

    @staticmethod
    def _resolve_checkpointer_dsn(
            definition: CheckpointerDataSourceDefinition,
            manager: DataSourceManager | None,
    ) -> str:
        """
        得到 checkpointer 用的裸 psycopg DSN。

        显式配了 dsn 就用它；否则从 derive_from 指向的
        postgresql 数据源推导 —— 这样密码只写一处。

        推导失败时**显式报错**而不是回退到某个默认值：
        checkpointer 连错库的后果是 checkpoint 落到别处，
        而运行本身看起来一切正常，极难发现。
        """

        config = definition.config

        if config.dsn:
            return config.dsn

        if not config.derive_from:
            raise ValueError(
                f"Checkpointer datasource {definition.name!r} "
                "must set either config.dsn or config.derive_from."
            )

        if manager is None:
            raise ValueError(
                f"Checkpointer datasource {definition.name!r} uses "
                f"derive_from={config.derive_from!r}, but no "
                "DataSourceManager was provided to resolve it."
            )

        source = manager.get_optional(config.derive_from)

        if source is None:
            raise ValueError(
                f"Checkpointer datasource {definition.name!r} "
                f"derives from {config.derive_from!r}, which is not "
                "registered. Make sure that datasource initializes "
                "before this one."
            )

        resolver = getattr(
            source, "resolved_dsn_psycopg", None
        )

        if resolver is None:
            raise ValueError(
                f"Datasource {config.derive_from!r} does not expose "
                "resolved_dsn_psycopg; it cannot be used as a "
                "checkpointer DSN source."
            )

        return resolver
