"""
当前运行环境（dev / staging / prod）。

它服务什么
----------
`run_level` —— thread_id 的第一段（见 app/infra/graph/ids.py 的
`new_session_thread_id`）。用途是**让不同环境不去共用同一套 checkpoint**：
同一个 checkpointer 库里，dev 的会话与 prod 的会话如果 thread_id 撞上，
prod 会从 dev 的 checkpoint 续跑 —— 那是最坏的一种串味。

放在 app.common 而不是某个服务的 settings
----------------------------------------
它描述的是"整个应用跑在哪"，不属于任何单个模块 —— document 有
`PipelineSettings`（PIPELINE_* 前缀，只谈流水线参数），agent 有
model.yaml，两者都不该各自定义一遍环境。app.common 是纯叶子，
谁都能 import，不产生反向依赖。

为什么不缓存
------------
`os.environ.get` 就是一次字典查找，加上一次 strip 与比较而已。
缓存要额外提供 `reset_cache()` 给测试用，而"忘了 reset 导致测试
之间互相污染"是这类小缓存最典型的坑 —— 收益为零、代价是真的。
"""

from __future__ import annotations

import logging
import os
from enum import Enum
from typing import Final

__all__ = [
    "ENV_VAR",
    "AppEnvironment",
    "current_environment",
]

logger = logging.getLogger(__name__)

#: 读哪个环境变量。
#:
#: 刻意**不加前缀**：`PIPELINE_` 那一套是"某条流水线的参数"，
#: 而这个是"整个应用跑在哪"。混进前缀会让它看起来像是可以按模块
#: 各配一份的东西 —— 它不是。
ENV_VAR: Final[str] = "APP_ENV"


class AppEnvironment(str, Enum):
    """
    部署环境。

    成员值是**小写**的，因为它会直接进 thread_id
    （`dev:react:<uuid>`）—— 那里用大写会显得突兀，而多一次
    `.lower()` 转换就多一处能忘的地方。
    """

    DEV = "dev"
    STAGING = "staging"
    PROD = "prod"


#: 读不到 / 读不懂时用哪个。
#:
#: 选 dev 而不是 prod 是**保守**取向：默认值意味着"最坏情况是把
#: 生产数据写进 dev 的 thread 命名空间"（可发现、可清理），
#: 而不是反过来（把本地调试写进生产）。两个方向的代价不对称。
_DEFAULT: Final[AppEnvironment] = AppEnvironment.DEV


def current_environment() -> AppEnvironment:
    """
    当前运行环境。读 `APP_ENV`，未设置或认不出来时回落到 dev。

    值做 strip + lower 归一（`APP_ENV=PROD` 与 `APP_ENV=prod ` 都认），
    这样在 .env / docker-compose / k8s 里写法不一致时不会静默错档。

    ⚠️ 认不出来时**打一条 warning**
    ------------------------------
    与 `PipelineSettings.from_env()` 的静默回退刻意不同
    （那边注释说不打印是为了不依赖日志配置）。这里打的理由：
    APP_ENV 拼错（`APP_ENV=prodcution`）不是"没配"，是**配错了** ——
    静默当成 dev 会让生产流量落进 dev 的 thread 命名空间，而两边
    看起来一切正常。一条 warning 是这件事最小的可见代价。
    """

    raw = os.environ.get(ENV_VAR)

    if raw is None:
        return _DEFAULT

    normalized = raw.strip().lower()

    if not normalized:
        return _DEFAULT

    try:
        return AppEnvironment(normalized)

    except ValueError:

        logger.warning(
            "%s=%r 不是已知环境（可选：%s），按 %s 处理。"
            "它会影响 thread_id 的命名空间，配错会让不同环境共用 checkpoint。",
            ENV_VAR,
            raw,
            "/".join(e.value for e in AppEnvironment),
            _DEFAULT.value,
        )

        return _DEFAULT
