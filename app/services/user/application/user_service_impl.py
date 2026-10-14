"""
UserService 的默认实现。

⚠️ 没有仓储参数 —— 这是刻意的
-----------------------------
`ServiceFactory.needs_session` 靠"__init__ 里有没有仓储契约注解"来
决定要不要开事务。本模块不落库（Casdoor 是用户唯一真相），因此这里
一个仓储都不声明，每个请求也就不会白占一条数据库连接。

**别为了"结构完整"给它加一个空仓储** —— 那会让每个 /auth/* 请求都
开一段什么都不做的事务。
"""

from __future__ import annotations

from app.common.auth.component import get_casdoor_component
from app.common.auth.models import AuthenticatedUser
from app.common.service.base import ServiceContext

from ..domain.schemas.auth import (
    CurrentUserResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserOption,
)
from ._assembly import (
    build_current_user,
    build_login_url,
    build_token_response,
    build_user_options,
)
from .user_service import LoginTicket, UserService

__all__ = ["UserServiceImpl"]


class UserServiceImpl(UserService):

    def __init__(
            self,
            *,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        # 进程级单例，取它不花钱也不碰 I/O。
        # 不用 ServiceFactory 的 pool 注入：那要往 main.py 的 pool 里
        # 加一个键，而这是"机制"不是"依赖"—— 与 get_service_factory、
        # get_model_registry 同一形状。
        self._auth = get_casdoor_component()

    # ========================================================
    # 登录
    # ========================================================

    async def begin_login(self) -> LoginTicket:

        self._require_enabled("登录相关接口")

        state, nonce = self._auth.issue_state()

        return LoginTicket(
            authorize_url=self._auth.authorize_url(
                # redirect_uri 来自配置，不接收调用方传入
                # （开放重定向，见 CasdoorConfig.redirect_uri）。
                redirect_uri=self._auth.config.redirect_uri,
                state=state,
            ),
            state=state,
            state_nonce=nonce,
        )

    async def complete_login(
            self,
            *,
            code: str,
            state: str,
            state_nonce: str | None,
    ) -> TokenResponse:

        self._require_enabled("登录相关接口")

        # 先验 state 再换 token：顺序反了就等于用一次可能被伪造的回调
        # 去消耗一个真的授权码（那个码是一次性的，被吃掉用户就得重来）。
        self._auth.verify_state(
            state=state,
            cookie_nonce=state_nonce,
        )

        raw = await self._auth.exchange_code(code)

        return build_token_response(raw)

    async def refresh(
            self,
            request: RefreshTokenRequest,
    ) -> TokenResponse:

        self._require_enabled("登录相关接口")

        raw = await self._auth.refresh(request.refresh_token)

        return build_token_response(raw)

    # ========================================================
    # 当前调用者
    # ========================================================

    async def describe_current_user(
            self,
            user: AuthenticatedUser,
    ) -> CurrentUserResponse:

        # 纯映射：user 已经由 current_user 依赖验过签了，
        # 这里不再碰 token、不再碰网络。
        return build_current_user(user)

    # ========================================================
    # 用户目录（管理端）
    # ========================================================

    async def search_users(
            self,
            *,
            keyword: str,
            limit: int,
    ) -> list[UserOption]:

        self._require_enabled("用户检索")

        rows = await self._auth.search_users(
            keyword=keyword,
            limit=limit,
        )

        # 字段裁剪（白名单）在组装函数里，不在组件里 —— 组件只认识
        # Casdoor 的原始对象，由它决定"哪几个字段能出去"等于把契约
        # 埋进更底层。
        return build_user_options(rows)

    # ========================================================

    def _require_enabled(
            self,
            what: str,
    ) -> None:
        """
        关掉鉴权时这些接口没法进行，且**不该装作能**。

        这时候 `current_user` 会给所有人一个合成用户（本地开发的便利），
        但"发起登录/换取令牌/查目录"都是要真去问 Casdoor 的，没有配置
        就是没有。报 500 而不是 401：这不是调用方的凭据问题。

        `what` 只是把话说清楚（"登录相关接口" / "用户检索"），不影响
        行为。
        """

        if self._auth.enabled:
            return

        raise RuntimeError(
            "Casdoor 鉴权已关闭（casdoor.yaml 的 enabled: false），"
            f"{what}不可用。要联调请填好凭据并把它打开。"
        )
