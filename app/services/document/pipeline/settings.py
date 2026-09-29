"""
流水线运行参数。

为什么不放 app/config/
----------------------
那个目录下的 `*.yaml` 会被 DataSourceConfigLoader 当成
DataSourceDefinition 扫描，非数据源的文件会验证失败并打出
"Ignore optional datasource" 警告。流水线参数不是数据源，
不该混进去。

因此走环境变量 + 默认值 —— 与 datasource 的 `${VAR}` 展开
是同一个思路，只是不必为三个参数引入一套配置文件格式。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Final, Literal

__all__ = ["CheckpointLevel", "PipelineSettings"]

CheckpointLevel = Literal["node", "run"]

#: 每个 super-step 落一次 checkpoint，进程被杀后可从断点续跑。
NODE_LEVEL: Final[CheckpointLevel] = "node"

#: 只在运行结束时落一次。写入量最小，但**没有中间 checkpoint
#: 就没有断点续跑** —— 中途崩溃只能整版 reprocess 重来。
RUN_LEVEL: Final[CheckpointLevel] = "run"

_ENV_PREFIX: Final[str] = "PIPELINE_"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True, slots=True)
class PipelineSettings:
    """
    一次应用启动生效的流水线参数。

    可按次覆盖的项（例如 checkpoint_level）在 dispatch payload
    里单独传，不走这里 —— 见 runtime.build_dispatch_handler。
    """

    #: 默认 run（每个 super-step 落盘，保留续跑能力）。
    #:
    #: 改成 run 会显著减少 checkpoint 写入（实测 65 KB -> 12 KB），
    #: 代价是失去断点续跑。目前没有证据表明写入是瓶颈，
    #: 因此不建议默认改成 run。
    checkpoint_level: CheckpointLevel = NODE_LEVEL

    #: 每个节点后断言 state 体积。开发/冒烟时开，生产关掉
    #: （每次节点后多一次序列化）。
    assert_state_size: bool = False

    #: 进程启动时把卡在 running 的运行标记为 failed。
    reconcile_orphans: bool = True

    @property
    def durability(self) -> str:
        """
        映射到 LangGraph 的 durability 参数。

            node -> "sync"  每个 super-step 落一次
            run  -> "exit"  只在结束时落一次
        """
        return "sync" if self.checkpoint_level == NODE_LEVEL else "exit"

    @classmethod
    def from_env(cls) -> "PipelineSettings":
        # 默认值必须给。少了它，环境变量未设置时 os.environ.get
        # 返回 None，紧接着的 .strip() 会抛 AttributeError ——
        # 而它在 lifespan 里，症状是**整个服务起不来**，
        # 报错却指向 settings.py 而不是"配置缺失"。
        # 实测踩过：服务静默启动失败，还以为是产物没写进库。
        raw_level = (
            os.environ.get(
                _ENV_PREFIX + "CHECKPOINT_LEVEL", NODE_LEVEL
            )
            or NODE_LEVEL
        ).strip().lower()

        if raw_level not in (NODE_LEVEL, RUN_LEVEL):
            # 拼错时退回默认并保持流程可用 —— 而不是让整个服务起不来。
            # 这里不打印警告是因为本模块不该依赖日志配置；
            # 调用方（main.py）会把手动传的值记进启动日志。
            raw_level = NODE_LEVEL

        return cls(
            checkpoint_level=raw_level,  # type: ignore[arg-type]
            assert_state_size=_env_flag(
                "ASSERT_STATE_SIZE", False
            ),
            reconcile_orphans=_env_flag(
                "RECONCILE_ORPHANS", True
            ),
        )

    def resolve_level(
            self,
            override: str | None,
    ) -> CheckpointLevel:
        """
        按次覆盖。

        reprocess 可以在 payload 里指定 checkpoint_level，
        用于"这一次我想能续跑"或"这一次我要最少的写入"。

        ⚠️ 级别必须在运行**开始前**定。一次以 run 级启动的运行
        中途崩溃后，没有任何中间 checkpoint 可恢复 ——
        事后改成 node 也救不回来。
        """
        if override in (NODE_LEVEL, RUN_LEVEL):
            return override  # type: ignore[return-value]
        return self.checkpoint_level
