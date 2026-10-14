"""
用户/登录的应用服务契约。

对应 user.yaml：
    POST /auth/login      发起登录，拿授权地址
    POST /auth/callback   用授权码换令牌
    GET  /auth/me         当前调用者
    POST /auth/refresh    刷新令牌
    GET  /users/search    按关键词查用户选项（管理端）

⚠️ 前四个的路径是 `/auth/*` 而不是 `/users/*`：它们的主语是**一次
登录动作**，不是"用户这个资源"（本模块没有用户资源 —— 用户住在
Casdoor，见 domain/schemas/auth.py 的说明）。tag 因此是 `Auth`。
最后一个的主语**是**用户，所以它老老实实挂在 `/users/*` 下。

它为什么**不**做鉴权判断
------------------------
"这个 token 是不是有效、这个人够不够格"由 app/common/auth 里的组件
回答，那与业务无关。这里只负责把 OAuth 那几步串起来、把目录查询的
结果翻译成对外契约。**别把群组 / 角色判断搬进来** —— 那样别的模块
要用同样的判断时，就得反向依赖用户服务。`/users/search` 的
Y-Trader-Admin 门槛因此挂在路由上（见 app/api/v1/users.py），不在
这里。
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from app.common.auth.models import AuthenticatedUser
from app.common.service.base import BaseService

from ..domain.schemas.auth import (
    CurrentUserResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserOption,
)

__all__ = [
    "LoginTicket",
    "UserService",
]


@dataclass(frozen=True, slots=True)
class LoginTicket:
    """
    发起登录的结果。

    它**不是**对外契约（那是 LoginUrlResponse）—— 多出来的
    `state_nonce` 是给**接口层写 cookie** 用的，不该出现在响应体里。

    为什么要单独一个类型：发起登录这件事有两个去处（浏览器拿地址、
    本服务设 cookie），返回值里两样都得有；塞进 DTO 就等于把"cookie
    里装什么"泄漏成 API 的一部分。
    """

    #: 让浏览器打开的地址。
    authorize_url: str

    #: 回调时要原样带回的状态值（签名 + 自带过期时间）。
    state: str

    #: state 里的随机数，要写进 HttpOnly cookie 与本浏览器绑定。
    state_nonce: str


class UserService(BaseService):

    # ========================================================
    # 登录
    # ========================================================

    @abstractmethod
    async def begin_login(self) -> LoginTicket:
        """
        生成一次登录的授权地址与防 CSRF 状态。

        使用配置里的 redirect_uri —— **不接受调用方传入**，
        见 CasdoorConfig.redirect_uri 的说明（开放重定向）。

        Raises:
            RuntimeError: 鉴权被关掉（enabled: false）时无法发起登录。
        """

        raise NotImplementedError

    @abstractmethod
    async def complete_login(
            self,
            *,
            code: str,
            state: str,
            state_nonce: str | None,
    ) -> TokenResponse:
        """
        用授权码换一整套令牌。

        Args:
            state_nonce: 请求 cookie 里那个值。**必须**与 state 里
                一致 —— 只验签名挡不住登录 CSRF，理由见
                CasdoorAuthComponent.verify_state。

        Raises:
            AuthenticationError: state 无效/过期/与浏览器不符；
                                 或授权码已被用过、已过期。
            RuntimeError:        连不上 Casdoor。
        """

        raise NotImplementedError

    @abstractmethod
    async def refresh(
            self,
            request: RefreshTokenRequest,
    ) -> TokenResponse:
        """
        用 refresh token 换一套新令牌。

        刻意**不**要求 access token 还有效 —— 这个接口的用处正是在
        它失效之后。因此它挂在公开路由上（见 app/api/v1/router.py）。

        Raises:
            AuthenticationError: refresh token 无效或已过期。
            RuntimeError:        连不上 Casdoor。
        """

        raise NotImplementedError

    # ========================================================
    # 当前调用者
    # ========================================================

    @abstractmethod
    async def describe_current_user(
            self,
            user: AuthenticatedUser,
    ) -> CurrentUserResponse:
        """
        把校验过的调用者翻译成对外契约。

        纯映射、无 I/O：`user` 是**已经验过签**的
        （由 app/common/auth 的 current_user 依赖产出），这里不再碰
        网络也不再碰 token。
        """

        raise NotImplementedError

    # ========================================================
    # 用户目录（管理端）
    # ========================================================

    @abstractmethod
    async def search_users(
            self,
            *,
            keyword: str,
            limit: int,
    ) -> list[UserOption]:
        """
        按 用户名 / 邮箱 / 用户 id 近似匹配 Casdoor 里的用户。

        给前端挑选控件用：返回**选项**（四个字段），不是完整用户对象。

        Args:
            keyword: 关键词。空串 = 不过滤，返回前 limit 个。
            limit:   最多返回几个（上限由接口层把关）。

        Raises:
            RuntimeError: Casdoor 连不上，或它拒绝了这次查询（本应用
                          在那边没有读用户的权限）。这是**服务端**的
                          问题，所以不是 4xx。
            RuntimeError: 鉴权被关掉（enabled: false）时也没有凭据可用。

        权限不在这里判：调用者够不够格由路由上的角色门槛回答（见模块
        docstring）。
        """

        raise NotImplementedError
