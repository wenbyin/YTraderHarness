"""
`app/config/app.yaml` —— 部署事实（模型路径、落盘目录）。

为什么单独一个文件、不进别处
--------------------------
这些值描述的是"这台机器上东西放在哪"，不是"这一轮怎么跑"：

    config/ 的业务配置     会被冻结进 PipelineConfigSnapshot；换机器后
                          历史快照还指着旧路径（见 document/const.py 的
                          模块 docstring）
    config/datasource/     那个目录只收带顶层 `type` 的数据源
    代码里的常量            容器与开发机的路径不同，改一次要发版

折中：放 `app/config/app.yaml`，读取时用 `${VAR:-default}` 展开 ——
与 datasource 配置同一套语法、同一份 `expand_env`。

与 `PipelineConfigSnapshot` 的关系
--------------------------------
**不进去**。这里的值是进程启动时读的部署参数，不是某次运行的业务
配置。换机器只需要改环境变量，不需要重建历史快照。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.common.utils import expand_env

__all__ = [
    "AppConfig",
    "AppModels",
    "AppPaths",
    "DEFAULT_CONFIG_PATH",
    "EmbeddingModel",
    "load_app_config",
]


#: 默认配置文件位置 —— app/config/app.yaml。
#:
#: 与 app/agent/event/config.py 的 DEFAULT_CONFIG_PATH 同形：相对模块
#: 定位，不依赖进程 CWD。
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "app.yaml"
)


class AppPaths(BaseModel):
    """整份 app.yaml 的 `paths` 段。"""

    model_config = ConfigDict(extra="forbid")

    embedding_model: str = Field(
        description="稠密向量模型目录（写入期与查询期必须一致）。",
    )

    reranker_model: str = Field(
        description="精排模型目录（cross-encoder）。",
    )

    qa_output_dir: str = Field(
        description="QA 数据集的落盘目录。",
    )

    file_storage: str = Field(
        description="本地文件存储根目录。",
    )

    spec_dir: str = Field(
        description="OpenAPI 规格自检扫描的目录（开发用）。",
    )


class EmbeddingModel(BaseModel):
    """`models.embedding` 段。"""

    model_config = ConfigDict(extra="forbid")

    dimensions: int = Field(
        ge=1,
        description=(
            "稠密向量维度。建 Milvus collection 的 schema 需要它，"
            "因此必须在任何一次 embed 之前就可得。"
        ),
    )


class AppModels(BaseModel):
    """整份 app.yaml 的 `models` 段。"""

    model_config = ConfigDict(extra="forbid")

    embedding: EmbeddingModel


class AppConfig(BaseModel):
    """整份 app.yaml。"""

    model_config = ConfigDict(extra="forbid")

    paths: AppPaths

    models: AppModels


@lru_cache(maxsize=4)
def load_app_config(
        path: str | None = None,
) -> AppConfig:
    """
    读配置并展开环境变量占位符。

    按路径缓存 —— 同一份配置在一次进程里只解析一次。改了 yaml 要重启
    才生效，与项目里其它配置（数据源、LLM、模型目录）的立场一致。
    """

    file_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not file_path.exists():
        raise ValueError(
            f"应用配置文件不存在：{file_path}。"
            "该文件随代码一起提供，缺失说明部署不完整。"
        )

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}

    return AppConfig.model_validate(expand_env(raw))
