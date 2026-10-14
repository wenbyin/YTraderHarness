"""
Casdoor 鉴权组件的启动自检。

为什么值得前置构造
----------------
它是每个 /api/v1 请求的必经之路（挂载点见 app/api/v1/router.py 的
protected_router）。配置缺东西时如果拖到第一个请求才炸，症状是
"所有接口都 401" —— 那会被当成客户端或 Casdoor 的问题去查，而真正的
原因只是 yaml 里少了一行。宁可起不来。

这一段不碰网络也不碰库，所以在 lifespan 里排得很靠前：失败得越早，
日志越好读。

为什么没有 get_xxx
----------------
本模块只做一次校验并把组件交回去，调用方（组合根）把它挂到 app.state。
**不需要**进程级访问器：需要"当前鉴权组件"的地方直接调
app/common/auth 的 get_casdoor_component()，那本来就是进程级单例。
再包一层 get_xxx 只会多出一个同义的名字。
"""

from __future__ import annotations

import logging

from app.common.auth import get_casdoor_component
from app.common.auth.component import CasdoorAuthComponent
from app.common.environment import (
    AppEnvironment,
    current_environment,
)

__all__ = ["initialize_auth"]

logger = logging.getLogger(__name__)


def initialize_auth() -> CasdoorAuthComponent:
    """
    构造鉴权组件、自检、并把结果写进日志。

    Returns:
        已构造的鉴权组件。调用方负责挂到 app.state（app.state 是
        FastAPI 的事，本模块不碰）。

    Raises:
        RuntimeError: APP_ENV=prod 却把鉴权关掉了。
    """

    component = get_casdoor_component()

    if component.enabled:

        logger.info(
            "Casdoor 鉴权已启用：endpoint=%s，application=%s，"
            "要求群组=%s，可查群组的 claim=%s",
            component.config.resolved_endpoint,
            component.config.application_name,
            "、".join(component.config.required_groups),
            component.config.groups_claim,
        )

        return component

    # 生产环境不许关鉴权。
    #
    # 关掉之后所有接口就是裸奔，而且**没有任何运行期迹象** —— 请求照常
    # 成功，日志照常安静。这种开关不能靠"记得别开"，必须响亮地失败。
    # 用 APP_ENV 判断（app/common/environment.py），它刻意不缓存，
    # 所以这里读到的就是当下的值。
    if current_environment() is AppEnvironment.PROD:

        raise RuntimeError(
            "APP_ENV=prod，但 app/config/casdoor.yaml 里 "
            "enabled: false —— 拒绝以「不校验任何令牌」的姿态启动。"
            "生产环境必须打开 Casdoor 鉴权。"
        )

    logger.warning(
        "⚠️ Casdoor 鉴权已关闭（casdoor.yaml 的 enabled: false）："
        "所有 /api/v1 接口都不校验令牌，current_user 会返回一个"
        "合成用户。**仅限本地开发**。"
    )

    return component
