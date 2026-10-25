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

        app/config/datasource/*.yaml

    目录里**只有带顶层 `type` 的那些**才是数据源定义，其余
    （部署参数、密钥、集成配置）原样跳过 —— 判据与理由见 load_all。
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

            # ========================================================
            # 本目录**不只**放数据源配置
            # ========================================================
            #
            # 判定规则只有一条：**顶层有没有 `type` 键**。有 = 数据源，
            # 没有 = 别的配置（如 casdoor.yaml），跳过即可。
            #
            # 历史：这里曾经 `raise ValueError`，等于宣布"这个目录只许
            # 放数据源"，非数据源配置于是被挪到了各自的模块目录下。
            # 现在那些配置统一住在 app/config/ 顶层（本加载器只扫
            # datasource/ 子目录），这里的 skip 只用于兜住"目录里混进
            # 了无 type 文件"这种情况。
            #
            # 跳过时**打 warning 列出文件名**，因为宽松化的代价是：
            # `tpye: postgresql` 这种键名拼错也会被跳过，之后就只在
            # 取 `postgresql` 时报一句更差的错。日志是这里唯一的线索。
            #
            # 注意这个判断在 try **之外** —— 它聊的不是"配置写得对不对"，
            # 而是"这个文件归不归我管"。归我管的，下面照旧校验。
            if not isinstance(raw, dict) or "type" not in raw:
                logger.warning(
                    "[Datasource] Ignore non-datasource config "
                    "(no top-level `type`): %s",
                    path,
                )
                continue

            datasource_type = raw["type"]

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
    ) -> Any:
        """
        读一个 yaml 文件，**只负责读，不判断形态**。

        以前这里对"不是 mapping"直接抛错，而调用点在 try 之外，
        因此一个**空文件**（safe_load 得 None）就能让整个应用起不来。
        而"这个文件不是数据源"并不是错误 —— 形态判断因此上移到
        load_all，与"缺 type"共用同一条跳过路径，`isinstance(raw, dict)`
        也顺带挡掉了 None。
        """

        with path.open(
                "r",
                encoding="utf-8",
        ) as file:
            return yaml.safe_load(file)

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
