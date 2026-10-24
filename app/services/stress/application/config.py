"""
Stress Driver 的进程级配置：``app/config/stress_eval.yaml``。

为什么单独一份
--------------
这些参数描述的是 **Driver 怎么跑**（线程、连接池、token、指标刷新、
安全并发上限），不是某一次压测的业务参数。某一次压测的
session_rate / total_sessions / N 等仍然由 ``POST /stress-runs`` 的
``StressRunCreate`` 传入。

放在 ``app/config/`` 而不是 ``app/config/datasource/``：数据源加载器只
扫描 datasource 子目录，这份文件不会参与数据源解析。

环境变量展开沿用项目既有的 ``${VAR:-default}`` 语法。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.common.utils import expand_env

__all__ = [
    "DEFAULT_STRESS_EVAL_CONFIG_PATH",
    "StressDriverConfig",
    "StressEvalConfig",
    "load_stress_eval_config",
]

#: app/config/stress_eval.yaml
DEFAULT_STRESS_EVAL_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "config" / "stress_eval.yaml"
)


class StressDriverConfig(BaseModel):
    """Driver 线程与 HTTP/SSE 连接的运行参数。"""

    model_config = ConfigDict(extra="forbid")

    target_allowlist: list[str] = Field(
        default_factory=lambda: [
            "localhost",
            "127.0.0.1",
            "::1",
            "0.0.0.0",
        ],
        description="允许压测的目标 host 白名单。",
    )

    default_token_env: str = Field(
        default="STRESS_RUNNER_TOKEN",
        description="请求未指定 auth.token_env 时读取的环境变量名。",
    )

    token_file: str = Field(
        default="~/.ytrader/stress_runner_token",
        description=(
            "环境变量没有 token 时的本地文件兜底。"
            "只用于开发；生产用环境变量/密钥管理。"
        ),
    )

    http_max_connections: int = Field(
        default=2000,
        ge=1,
        description="Driver aiohttp 连接池上限。",
    )

    connect_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description="连接被测服务的超时。",
    )

    session_create_concurrency: int = Field(
        default=16,
        ge=1,
        description=(
            "同时向被测服务发起 create session 的上限。\n\n"
            "后端每个请求会占用一条业务 DB 连接（默认 pool_size=10 + "
            "max_overflow=20）。Driver 的会话并发（可达数百）远大于这个池，"
            "不限流会让 create session 在拿连接时排队 30 秒，最终返回 500。"
            "建会话只是压测的准备动作，不需要跟正式轮次同并发。"
        ),
    )

    session_create_max_attempts: int = Field(
        default=4,
        ge=1,
        description=(
            "create session 遇到 5xx / 429 / 网络错误时的最大尝试次数。"
            "重试只针对可恢复错误；4xx 直接失败。"
        ),
    )

    session_create_backoff_seconds: float = Field(
        default=0.5,
        gt=0,
        description="create session 重试的初始退避秒数（指数退避 + 抖动）。",
    )

    cancel_concurrency: int = Field(
        default=8,
        ge=1,
        description=(
            "同时向被测服务发起 run cancel 的上限。\n\n"
            "Campaign 结束时可能同时有数百个在途 run 需要取消；如果全部并发，"
            "会把后端业务 DB 池和 HTTP 连接一起打满，并产生大量 500 / "
            "WinError 10054。取消是尽力而为的动作，串行度低一点完全可以接受。"
        ),
    )

    metrics_flush_interval_seconds: float = Field(
        default=2.0,
        gt=0,
        description="运行中 collecting 快照的刷新间隔。",
    )

    trace_flush_delay_seconds: float = Field(
        default=1.0,
        ge=0,
        description="Campaign 结束后等待 Trace 落 Redis 的时间。",
    )

    sample_limit: int = Field(
        default=200_000,
        ge=1000,
        description="每类延迟分布保留的精确样本上限。",
    )

    max_hitl_resumes: int = Field(
        default=3,
        ge=0,
        description="同一轮最多自动 resume 的次数。",
    )

    event_transport: Literal["sse", "redis"] = Field(
        default="redis",
        description=(
            "Driver 从哪读运行事件。\n\n"
            "    sse     HTTP SSE（每条在途 run 占一条长连接）\n"
            "    redis   直接轮询 Redis Stream（短查询，不占长连接）\n\n"
            "高并发 / Windows 下建议 redis：SSE 会给服务端每个在途 run "
            "各占一条 HTTP + 一条 Redis 长连接，容易撞 select() 512 上限。"
        ),
    )

    event_poll_interval_ms: int = Field(
        default=1000,
        ge=50,
        description="redis 模式下两次 XRANGE 之间的间隔（毫秒）。",
    )

    event_poll_max_connections: int = Field(
        default=64,
        ge=1,
        description=(
            "redis 模式下 Driver 自己那个 Redis 客户端的连接池上限。"
            "每次 XRANGE 借一条、查完还回去，不长期占用。"
        ),
    )

    max_concurrency: int = Field(
        default=102,
        ge=1,
        description=(
            "Driver 自己允许的最大活跃会话/SSE 数。"
            "实际值还会被 event_stream.yaml 的 max_active_run_streams "
            "乘 safety_ratio 截断。"
        ),
    )

    concurrency_safety_ratio: float = Field(
        default=0.8,
        gt=0,
        le=1.0,
        description=(
            "相对后端 max_active_run_streams 留出的余量。"
            "Windows SelectorEventLoop 受 512 socket 限制，不能贴满。"
        ),
    )

    trace_aggregation_enabled: bool = Field(
        default=True,
        description="Campaign 结束后是否读取 Redis Trace 并聚合。",
    )

    @field_validator("token_file")
    @classmethod
    def _default_token_file(cls, value: str) -> str:
        return value.strip() or "~/.ytrader/stress_runner_token"


class StressEvalConfig(BaseModel):
    """``app/config/stress_eval.yaml`` 的顶层。"""

    model_config = ConfigDict(extra="forbid")

    driver: StressDriverConfig = Field(
        default_factory=StressDriverConfig,
        description="Stress Driver 运行参数。",
    )


@lru_cache(maxsize=4)
def load_stress_eval_config(
        path: str | None = None,
) -> StressEvalConfig:
    """读并校验 stress_eval.yaml（按路径缓存）。"""

    file_path = Path(path) if path else DEFAULT_STRESS_EVAL_CONFIG_PATH

    if not file_path.exists():
        raise ValueError(
            f"Stress Driver 配置文件不存在：{file_path}。"
            "请随代码提供 app/config/stress_eval.yaml。"
        )

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}

    return StressEvalConfig.model_validate(expand_env(raw))
