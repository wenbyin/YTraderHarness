from __future__ import annotations

from .base import DataSource


class DataSourceManager:
    """
    Application-level DataSource Registry。

    业务代码不直接创建 DataSource。

    通过：
        manager.get("postgresql")
        manager.get("elasticsearch")
        manager.get("milvus")
    获取。
    """

    def __init__(self) -> None:
        self._sources: dict[
            str,
            DataSource,
        ] = {}

    def register(
            self,
            source: DataSource,
    ) -> None:

        if source.name in self._sources:
            raise ValueError(
                f"Duplicate datasource: "
                f"{source.name}"
            )

        self._sources[
            source.name
        ] = source

    def get(
            self,
            name: str,
    ) -> DataSource:

        source = self._sources.get(
            name
        )

        if source is None:
            raise KeyError(
                f"Datasource not found: {name}"
            )

        return source

    def get_optional(
            self,
            name: str,
    ) -> DataSource | None:

        return self._sources.get(
            name
        )

    async def close_all(
            self,
    ) -> None:

        for source in reversed(
                list(self._sources.values())
        ):
            try:
                await source.close()
            except Exception:
                # Production version should log.
                pass

        self._sources.clear()
