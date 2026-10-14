"""
FastAPI 依赖：`current_user`。

它是整个前置校验的**唯一入口** —— app/api/v1/router.py 把
`Depends(current_user)` 挂在聚合路由上，于是所有业务接口都被覆盖。

⚠️ `current_user` 必须是**模块级的单一函数对象**
-----------------------------------------------
FastAPI 的依赖缓存按"可调用对象本身"做键（fastapi/dependencies/
models.py 里的 `(dependant.call, scopes, ...)`），use_cache 默认开。
聚合路由上的那个实例与端点里 `Depends(current_user)` 的那个实例是
**同一个对象**，因此共用一份缓存 —— 一次请求只验一次 JWT。

有三种写法会静默地让它跑两遍（都不报错，只是白验一次）：
    1. 任何地方用 `Security(current_user, scopes=[...])` —— 带 scopes
       时缓存键里的 scopes 非空，两个键就不等了
    2. 任何地方写 `Depends(current_user, use_cache=False)`
    3. 把 current_user 包进工厂函数、每个路由返回一个新闭包

所以：**不要**重构它、不要给它加参数、不要包装它。
"""

from __future__ import annotations

import fnmatch
import logging

from fastapi import HTTPException, Request, status

from app.common.auth.component import get_casdoor_component
from app.common.auth.config import CasdoorConfig
from app.common.auth.models import AuthenticatedUser
from app.common.exceptions import (
    AuthenticationError,
    PermissionDeniedError,
)

__all__ = [
    "current_user",
    "extract_bearer_token",
    "require_role",
]

logger = logging.getLogger(__name__)


def extract_bearer_token(
        request: Request,
) -> str | None:
    """
    从 `Authorization: Bearer <token>` 取出 token。

    大小写不敏感地认 scheme（RFC 7235 说 scheme 不区分大小写，
    而各家客户端大小写并不统一）。取不到返回 None。
    """

    header = request.headers.get("Authorization")

    if not header:
        return None

    scheme, separator, value = header.partition(" ")

    if not separator or scheme.lower() != "bearer":
        return None

    return value.strip() or None


def _extract_query_token(
        request: Request,
        config: CasdoorConfig,
) -> str | None:
    """
    从 query string 取 token —— **只在白名单路径上**。

    唯一的用途是 SSE：浏览器的 EventSource 不能设置请求头，而
    `GET /api/v1/runs/{run_id}/events` 正是被 EventSource 消费的。

    ⚠️ query 里的 token 会进访问日志、代理日志、浏览器历史，所以
    默认只对 casdoor.yaml 里 query_token_paths 列出的路径生效。
    不要为了省事扩大这个列表。
    """

    if not config.query_token_paths:
        return None

    path = request.url.path

    if not any(
            fnmatch.fnmatch(path, pattern)
            for pattern in config.query_token_paths
    ):
        return None

    token = request.query_params.get(config.query_token_param)

    if not token:
        return None

    # 值得留一条记录：这是一条**例外通道**，出问题时要知道它被用上了。
    logger.info(
        "从 query string 取了访问令牌（path=%s，参数=%s）—— "
        "仅 EventSource 这类发不了请求头的客户端该走这条路。",
        path,
        config.query_token_param,
    )

    return token


async def current_user(
        request: Request,
) -> AuthenticatedUser:
    """
    校验调用方并返回其身份。

    这是被 `Depends` 调用的那个函数 —— 关于"为什么必须是单个模块级
    函数对象"，见模块 docstring。

    为什么是 `async def` 而不是 `def`：FastAPI 会把同步依赖丢进线程池
    跑。一次 RSA 验签是几十微秒的 CPU 活，进线程池的调度开销比它本身
    还大，而且没理由为它占一个线程。

    Raises（HTTPException 直接抛，不抛领域异常）
    -------------------------------------------
    这是刻意的：本函数跑在**路由级依赖**的位置上，而
    `to_http_error` 只在路由函数体里被调用 —— 在这里抛
    AuthenticationError 会一路穿到 ASGI 异常中间件，结果是
    **500 而不是 401**。所以 HTTP 的翻译在这一层做掉。
    """

    component = get_casdoor_component()

    # 关掉鉴权时给个替身，让 `Depends(current_user)` 的使用方一行都不用改。
    # 生产环境这个开关打不开 —— app/main.py 会拒绝启动。
    if not component.enabled:
        return component.synthetic_user()

    raw_token = (
        extract_bearer_token(request)
        or _extract_query_token(request, component.config)
    )

    if raw_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=(
                "缺少访问令牌。请在请求头带上 "
                "`Authorization: Bearer <access_token>`。"
            ),
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        return component.verify_token(raw_token)

    except AuthenticationError as exc:
        # RFC 6750：401 必须指明认证方案，客户端才知道拿什么去换。
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    except PermissionDeniedError as exc:
        # 403 而不是 401：凭据是好的，换一份也没用。
        # 给 401 会让前端陷入"反复重新登录"的循环。
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        ) from exc


def require_role(
        user: AuthenticatedUser,
        role: str,
) -> None:
    """
    调用者必须持有 `role`，否则 403。**角色门槛的唯一入口。**

    用在"只有管理员能调"的端点上，写法是在路由模块里包一个依赖：

        async def require_admin(
                user: AuthenticatedUser = Depends(current_user),
        ) -> None:
            require_role(user, get_casdoor_component().config.admin_role)

    为什么是**函数**（user 传进来），而不是一个自带 current_user 的
    依赖工厂：模块 docstring 第 3 条说的那件事 —— 把 current_user 包进
    工厂再挂上去，缓存键就换了对象，每请求白验一次签名。让调用方自己
    `Depends(current_user)`，缓存按原对象命中，验签仍然只有一次。

    Raises:
        HTTPException: 403。**直接抛**，不抛领域异常 —— 调用它的位置是
            路由级依赖，那里抛 PermissionDeniedError 会穿到 ASGI 异常
            中间件，结果是 500（与 current_user 同一段理由）。

    ⚠️ 鉴权关掉时（enabled: false）必定 403：current_user 给的那个
    合成用户**刻意不带角色**（见 CasdoorAuthComponent.synthetic_user）。
    本地要调这类端点就得用真 token —— 让 bypass 凭空有角色，会造成
    "本地能跑、生产拒绝"的差异，那正是那个开关要避免的。
    """

    component = get_casdoor_component()

    if component.has_role(user, role):
        return

    # 配错角色名的排查体验全靠这一条：与群组不匹配时那条日志同一形状
    # ——把"要求的"与"实际持有的"都打出来，照着抄进配置即可。角色名
    # 不是秘密（它就在 token 里），而少了这条就只剩一个 403。
    logger.warning(
        "角色不足：要求 %r，调用者（%s）实际持有 %r，org_name=%r。",
        role,
        user.name or user.subject or "（无名字）",
        list(user.roles),
        component.config.org_name,
    )

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail=(
            f"当前用户没有访问该接口所需的角色（需要：{role}）。"
        ),
    )
