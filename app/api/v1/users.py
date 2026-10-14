"""
用户与登录接口（Casdoor OAuth + 目录检索）。

对应 user.yaml 的 tag：Auth（登录）与 Users（目录检索）。规格与实现
都在 app/services/user/ 下，本文件只做 HTTP 层的转发 —— 与 tools.py
/ capabilities.py 等其余路由同一套分层。

⚠️ 这些接口**不在**同一层校验上
------------------------------
    POST /auth/login      公开 —— 还没登录，哪来的 token
    POST /auth/callback   公开 —— 此刻才在换 token
    POST /auth/refresh    公开 —— 它的用处正是 access token 已失效
    GET  /auth/me         **受保护**
    GET  /users/search    **受保护 + 要求角色**

三棵路由树，**注册在哪棵上就是什么待遇**（与 router.py 里那两棵聚合
路由同一套办法，只是这里多一层）：

    public_router        公开
    protected_router     受保护（/auth/me）
    admin_router         受保护 + 必须持有 casdoor.yaml: admin_role

为什么 /users/search 单开一棵、而不是在函数签名里加一行
`Depends(require_admin)`：这仓库的规矩是"门槛写在注册的那一行上"
（见 app/api/v1/router.py 的模块 docstring）。签名里那行依赖会被后来
的人当成"取值的代码"顺手删掉，而删掉之后接口**看起来**只是少了个
参数 —— 这个接口恰好是最不该出这种事的那个。

refresh 为什么必须公开：它靠 refresh_token 认证，而 access_token
此刻**应该**已经过期了 —— 要求它有效，这个接口就没有存在意义了。

本文件唯一与"HTTP 之外"有关的东西是 cookie：发起登录时下发一个
一次性 state cookie，回调时读回来。它是**传输层**的细节，因此留在
这里，不下沉到服务层（服务层不认识 cookie）。
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Depends,
    Query,
    Request,
    Response,
    status,
)

from app.common.auth import get_casdoor_component, require_role
from app.common.auth.models import AuthenticatedUser
from app.common.auth.dependencies import current_user
from app.common.environment import (
    AppEnvironment,
    current_environment,
)
from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.user.application._assembly import build_login_url
from app.services.user.application.user_service import UserService
from app.services.user.domain.schemas import (
    CurrentUserResponse,
    LoginCallbackRequest,
    LoginUrlResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserOption,
)

#: 公开端点：登录、回调、刷新。
#:
#: 它们必须在校验**之前**可用（理由见模块 docstring）。
#: 注册到 app/api/v1/router.py 的 public_router 上。
public_router = APIRouter()


#: 受保护端点，目前只有一个 /auth/me。
#:
#: ⚠️ 为什么单独一棵树，而不是混在上面那棵、靠端点签名里那行
#: `Depends(current_user)` 兜着：**那行的目的是取到用户信息，不是
#: 当防线**。混在一起的话，将来有人为了别的原因把它删掉（比如改成
#: 从 request.state 取），这个接口就**静默地**变成公开接口了 ——
#: 而它看起来只是"删了一行取值的代码"。注册在受保护的聚合路由上，
#: "它受保护"这件事就写死在 include 那一行里，不依赖谁记得加依赖。
protected_router = APIRouter()


async def require_admin(
        user: AuthenticatedUser = Depends(current_user),
) -> None:
    """
    admin_router 的门槛：调用者必须持有 casdoor.yaml: admin_role。

    角色名**每次请求**从配置取，不在 import 期读 —— 那会让"配置写坏了"
    表现成"这个模块 import 失败"，症状离原因很远。

    ⚠️ 依赖 `current_user` 不会让它验两遍签名：缓存按可调用对象去重，
    而 current_user 始终是同一个模块级函数对象（见
    app/common/auth/dependencies.py 的模块 docstring）。**别**在这里
    再加一个 `Depends(require_admin)` 之类的包装层，那就会了。
    """

    require_role(user, get_casdoor_component().config.admin_role)


#: 管理端端点：受保护 **且** 要求 admin_role。
#:
#: 挂在 app/api/v1/router.py 的 protected_router 上（于是它同时继承了
#: 登录门槛）—— 这里这棵树再加一道角色门槛。
admin_router = APIRouter(
    dependencies=[Depends(require_admin)],
)


def _set_state_cookie(
        response: Response,
        nonce: str,
) -> None:
    """
    下发防 CSRF 的 state cookie。

    它与响应体里的 state 是**同一个随机数**，但通道不同：state 走
    URL（会被 Casdoor 原样带回），nonce 走 HttpOnly cookie（攻击者
    构造的链接带不上它）。回调时要求两者一致，"这个 state 是不是
    本浏览器发起的"才成立 —— 只验签名挡不住登录 CSRF，理由见
    CasdoorAuthComponent.verify_state。
    """

    config = get_casdoor_component().config

    response.set_cookie(
        key=config.state_cookie_name,
        value=nonce,
        max_age=config.state_ttl_seconds,
        # HttpOnly：前端脚本读不到它，XSS 也偷不走。
        httponly=True,
        # Lax 而不是 Strict：登录回调是**从 Casdoor 跳回来的顶层导航**，
        # Strict 会让浏览器不带这个 cookie，回调必然失败。
        samesite="lax",
        # 生产必须 Secure（HTTPS 才发）。用 APP_ENV 推导而不是加一个
        # 配置项：多一个"记得改"的开关就多一次忘记改。
        # 本地是 http，Secure 会让浏览器直接不收 —— 所以 DEV 下必须是
        # False，不能图省事写死 True。
        secure=current_environment() is AppEnvironment.PROD,
        path="/",
    )


@public_router.post(
    "/auth/login",
    response_model=LoginUrlResponse,
    tags=["Auth"],
    status_code=status.HTTP_200_OK,
)
async def begin_login(
        response: Response,
        service: UserService = Service[UserService],
) -> LoginUrlResponse:
    """
    发起登录：返回 Casdoor 授权页地址，并下发 state cookie。

    **不直接 302**：返回地址由前端决定怎么用（跳转或弹窗），接口因此
    可以脱离浏览器测。前端拿到后 `window.location.assign(authorize_url)`
    即可 —— 注意必须**用收到本响应的同一个浏览器**打开，否则回调时
    找不到那个 cookie。
    """

    try:
        ticket = await service.begin_login()

    except ValueError as exc:
        raise to_http_error(exc) from exc

    _set_state_cookie(response, ticket.state_nonce)

    return build_login_url(ticket)


@public_router.post(
    "/auth/callback",
    response_model=TokenResponse,
    tags=["Auth"],
    status_code=status.HTTP_200_OK,
)
async def complete_login(
        payload: LoginCallbackRequest,
        request: Request,
        response: Response,
        service: UserService = Service[UserService],
) -> TokenResponse:
    """
    用授权码换令牌。

    前端从 redirect_uri 上取到 `?code=&state=` 后调本接口。
    cookie 由浏览器自动带上，不需要前端操心。
    """

    config = get_casdoor_component().config

    try:
        tokens = await service.complete_login(
            code=payload.code,
            state=payload.state,
            state_nonce=request.cookies.get(
                config.state_cookie_name
            ),
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc

    # 用掉就清掉：state 是一次性的，留着只会让"过期了为什么还有
    # cookie"这种问题变得可疑。path 要与下发时一致，否则删不掉。
    response.delete_cookie(
        config.state_cookie_name,
        path="/",
    )

    return tokens


@public_router.post(
    "/auth/refresh",
    response_model=TokenResponse,
    tags=["Auth"],
    status_code=status.HTTP_200_OK,
)
async def refresh_tokens(
        payload: RefreshTokenRequest,
        service: UserService = Service[UserService],
) -> TokenResponse:
    """
    用 refresh token 换一套新令牌。

    公开接口（理由见模块 docstring）。⚠️ 它会在**未经认证的输入**上
    触发一次对 Casdoor 的出站调用 —— 目前没有限流，将来要暴露到公网
    时应先补上。
    """

    try:
        return await service.refresh(payload)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@protected_router.get(
    "/auth/me",
    response_model=CurrentUserResponse,
    tags=["Auth"],
)
async def describe_current_user(
        user: AuthenticatedUser = Depends(current_user),
        service: UserService = Service[UserService],
) -> CurrentUserResponse:
    """
    当前调用者。

    依赖 `current_user` 已经验过签、也验过群组 —— 本函数不做任何
    判断，只把结果翻译成对外契约。

    这里那行 `Depends(current_user)` 是为了**取到值**，不是防线：
    防线是它注册在 protected_router 上（见模块 docstring）。两者都
    在也不会验两遍 —— FastAPI 的依赖缓存按可调用对象去重。
    """

    return await service.describe_current_user(user)


@admin_router.get(
    "/users/search",
    response_model=list[UserOption],
    tags=["Users"],
)
async def search_users(
        keyword: str = Query(
            default="",
            max_length=128,
            description=(
                "按 用户名 / 邮箱 / 用户 id 近似匹配（子串，大小写敏感）。"
                "留空表示不过滤，返回组织内前 limit 个。"
            ),
        ),
        limit: int = Query(
            default=20,
            ge=1,
            le=50,
            description="最多返回几个选项。给挑选控件用，所以封顶 50。",
        ),
        service: UserService = Service[UserService],
):
    """
    查用户选项（给前端的挑选控件）。

    ⚠️ **管理端接口**：门槛挂在 admin_router 那一行上（见模块
    docstring），本函数里那行 `Depends(service)` 只管取服务。

    结果不是完整用户对象，是四个字段的选项 —— 理由见 user.yaml 的
    `/users/search`。没匹配到就返回空列表，不是 404：这是"搜索结果
    为空"，不是"资源不存在"。
    """

    try:
        return await service.search_users(
            keyword=keyword,
            limit=limit,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
