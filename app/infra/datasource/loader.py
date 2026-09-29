from __future__ import annotations

import os
import re
import logging
from pathlib import Path
from typing import Any

import yaml

from .models import DataSourceDefinition

_ENV_PATTERN = re.compile(
    r"\$\{([A-Z0-9_]+)(?::-([^}]*))?\}"
)

logger = logging.getLogger(__name__)


class DataSourceConfigLoader:
    """
    从配置目录自动发现 DataSource 定义。

    配置目录：

        config/datasources/*.yaml

    每一个 yaml 文件代表一个 DataSource。
    """

    def __init__(
            self,
            directory: str | Path,
    ) -> None:
        self._directory = Path(
            directory
        )

    def load_all(
            self,
    ) -> list[DataSourceDefinition]:

        if not self._directory.exists():
            raise FileNotFoundError(
                f"Datasource config directory "
                f"does not exist: {self._directory}"
            )

        from pydantic import TypeAdapter

        adapter = TypeAdapter(
            DataSourceDefinition
        )

        definitions = []

        for path in sorted(
                self._directory.glob("*.yaml")
        ):
            raw = self._load_yaml(path)

            datasource_type = raw.get("type")

            if not datasource_type:
                raise ValueError(
                    f"Datasource type is missing: {path}"
                )

            try:
                expanded = self._expand_env(raw)

                parsed = adapter.validate_python(
                    expanded
                )

                definitions.append(parsed)

            except ValueError as exc:

                # PostgreSQL is the only mandatory datasource.
                #
                # Any configuration problem involving PostgreSQL
                # should abort application startup.
                if datasource_type == "postgresql":
                    raise

                # All other datasource failures are optional.
                #
                # Examples:
                #   ES_HOST is not configured
                #   MILVUS_URI is not configured
                #   malformed optional config
                logger.warning(
                    f"[Datasource] "
                    f"Ignore optional datasource "
                    f"{datasource_type}: "
                    f"{path} - {exc}"
                )

                continue

        return definitions

    @staticmethod
    def _load_yaml(
            path: Path,
    ) -> dict[str, Any]:

        with path.open(
                "r",
                encoding="utf-8",
        ) as file:
            data = yaml.safe_load(
                file
            )

        if not isinstance(
                data,
                dict,
        ):
            raise ValueError(
                f"Datasource config must be "
                f"a mapping: {path}"
            )

        return data

    @classmethod
    def _expand_env(
            cls,
            value: Any,
    ) -> Any:
        """
        展开配置里的环境变量占位符。

            ${NAME}             必须设置，否则该数据源配置被跳过
            ${NAME:-default}    未设置（或为空）时回退到 default
            ${NAME:-}           未设置（或为空）时回退到空串，
                                用于"可选值"（如无鉴权的 ES）

        `:-` 按 POSIX 语义：未设置**和**已设置为空串都算缺省。
        否则 `ES_USERNAME=` 会展开成空串并覆盖掉默认值，
        而这种写法通常是想表达"用默认值"。
        """

        if isinstance(
                value,
                str,
        ):

            def replace(match):
                name = match.group(1)
                default = match.group(2)

                result = os.getenv(name)

                if result:
                    return result

                if default is not None:
                    return default

                raise ValueError(
                    f"Environment variable "
                    f"{name} is not set"
                )

            return _ENV_PATTERN.sub(
                replace,
                value,
            )

            return _ENV_PATTERN.sub(
                replace,
                value,
            )

        if isinstance(
                value,
                dict,
        ):
            return {
                key: cls._expand_env(
                    item
                )
                for key, item in value.items()
            }

        if isinstance(
                value,
                list,
        ):
            return [
                cls._expand_env(item)
                for item in value
            ]

        return value
