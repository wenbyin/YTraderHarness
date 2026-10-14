"""
运行事件流的配置（读 app/config/event_stream.yaml）。

配置统一住在 `app/config/` 下，与读它的本模块不同目录。事件流参数
不是数据源定义，因此它放在 `app/config/` 的**根**，不在
`app/config/datasource/` 里 —— 后者是 `app/main.py` 交给数据源加载器
的唯一目录，加载器只 glob 它（见 infra/datasource/loader.py）。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.common.utils import expand_env

__all__ = [
    "StreamConfig",
    "load_stream_config",
]


#: 默认配置文件位置 —— app/config/event_stream.yaml。
#:
#: 不再用 `Path(__file__).with_name(...)`：配置已经不住在本模块旁边了
#: （与 app/common/llm.py 的 DEFAULT_CONFIG_PATH 同形）。
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "event_stream.yaml"
)


class StreamConfig(BaseModel):
    """整份 event_stream.yaml。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        description="关掉后 publish() 一条 Redis 命令都不发。",
    )

    key_prefix: str = Field(
        default="agent_run",
        description="事件流的 topic。最终键为 '<key_prefix>:<topic_id>'。",
    )

    maxlen: int | None = Field(
        default=10_000,
        description=(
            "单条 stream 的长度上限（XADD ... MAXLEN ~）。"
            "None = 不裁剪。"
        ),
    )

    ttl_seconds: int | None = Field(
        default=86_400,
        description="键的存活秒数，每次写入刷新（滑动）。None = 不过期。",
    )

    block_ms: int = Field(
        default=1_000,
        ge=0,
        description=(
            "读取端 XREAD BLOCK 的阻塞毫秒数。"
            "**必须明显小于 redis.yaml 的 socket_timeout**，"
            "否则会被客户端读超时打断（见 yaml 里的说明）。"
        ),
    )

    idle_probe_after_reads: int = Field(
        default=5,
        ge=1,
        description=(
            "**连续**这么多次空读之后，去探活一次：这个 run 是不是已经"
            "不会再有事件了（见 run_stream.py 的关流规则）。\n\n"
            "⚠️ 用「空读**次数**」而不是毫秒表达"
            "--------------------------------\n"
            "一次空读 = 一次 XREAD BLOCK 到点、没有新记录，默认耗时 "
            "block_ms（1000ms），所以 5 约等于 5 秒。这么写是为了让测试"
            "注入一个小数字就能触发探活，**不必 sleep**。\n"
            "改 block_ms 时这个数字的**含义**跟着变：要的是固定的秒数，"
            "就按 block_ms 折算着改。"
        ),
    )

    subscriber_queue_size: int = Field(
        default=2_000,
        # ge=2：实现里刻意留一格给收尾信号（见 run_stream.py 的 `capacity`），
        # 取 1 会让有效容量变成 0 —— 每投一条都判"跟不上"。
        #
        # ⚠️ 这里**刻意没有**"队列必须大于单批投递量"那条下界
        # ------------------------------------------------
        # 批大小是**从本值推出来的**（run_stream.py 的 `_chunk_for`），所以
        # 两者不可能打架。把它写成配置约束的话，把队列调小的人会收到一条
        # 与他的意图毫无关系的校验错误；正确做法是让代码自己保持一致。
        ge=2,
        description=(
            "单个订阅者的待发队列上限（条）。满了就**断开这一个订阅者**"
            "—— 不阻塞共享循环（那会拖住这个 run 的所有订阅者），也不丢最旧"
            "（那会让正文静默缺洞）。取舍见 run_stream.py。\n\n"
            "取值要让一次完整补发放得下：一条流的长度上限是 maxlen"
            "（默认 10000）。2000 是折中 —— 覆盖绝大多数补发，同时把"
            "「一个卡住的浏览器标签」能占住的内存封住。\n\n"
            "单轮从 Redis 取多少条**跟着本值走**：取 min(500, 本值 - 1)。"
            "一轮取回来的那批是**同步**投给所有订阅者的（投递过程里没有 "
            "await，消费方拿不到执行权），批比队列大的话，一个完全正常的"
            "订阅者只要没来得及被调度就会被判为跟不上 —— 那是误杀。"
        ),
    )

    max_active_run_streams: int = Field(
        default=16,
        ge=1,
        description=(
            "同时在订阅的 **run** 数上限。达到之后新订阅拿到一个说明原因的"
            "503，而不是等连接池到 redis-py 内部才炸出 MaxConnectionsError。\n\n"
            "⚠️ 必须**严格小于** redis.yaml 的 stream_max_connections —— "
            "池满时 redis-py 是立刻抛异常、不排队（见那里的说明）。"
        ),
    )

    read_max_consecutive_errors: int = Field(
        default=3,
        ge=1,
        description=(
            "读取端连续读失败**达到**这个次数就把异常抛给订阅方"
            "（含第一次，也就是最多重试 N-1 次）。"
            "没到就重发一个心跳节拍继续读（见 reader.py）。"
            "设成 1 = 不重试，退回改造前的行为。"
        ),
    )

    write_queue_size: int = Field(
        default=10_000,
        ge=1,
        description=(
            "发布端写队列的上限（条）。满了**拒绝最新**那条，"
            "已接收的照常按序写出 —— 理由见 publisher.py 模块 docstring。"
        ),
    )


@lru_cache(maxsize=4)
def load_stream_config(
        path: str | None = None,
) -> StreamConfig:
    """
    读配置并展开环境变量占位符。

    按路径缓存 —— 同一份配置在一次进程里只解析一次。改了 yaml 要重启
    才生效，与项目里其它配置（数据源、LLM、模型目录）的立场一致。
    """

    file_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not file_path.exists():
        raise ValueError(
            f"事件流配置文件不存在：{file_path}。"
            "该文件随代码一起提供，缺失说明部署不完整。"
        )

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}

    return StreamConfig.model_validate(expand_env(raw))
