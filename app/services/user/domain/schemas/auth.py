"""
用户/登录的对外契约。

对应 user.yaml 的 components/schemas：
    LoginUrlResponse / TokenResponse / RefreshTokenRequest /
    CurrentUserResponse / UserOption

全部结构逐字段对应那份 spec；字段名与取值都以它为准。要改就改
user.yaml，两边一起。

⚠️ **这里没有 User 实体**
------------------------
本模块**不落库** —— Casdoor 是用户的唯一真相。"当前用户长什么样"
这个问题的答案由 `app/common/auth/models.py::AuthenticatedUser`
持有（从 JWT 的 claim 直接来），本文件只定义**它怎么对外呈现**。
所以 domain 下没有 entities/：没有表，就没有需要与 DTO 分开的
持久化形态。

UserOption 是同一个取舍的另一面：它看起来像"用户的一部分"，但它是
**另一条链路**上的东西（Casdoor 目录检索的结果裁剪），来源是 Casdoor
的用户对象而不是 token。两者都不落库，所以都只是 DTO。
"""

from __future__ import annotations

from pydantic import Field

from app.common.schemas import APIModel

__all__ = [
    "CurrentUserResponse",
    "LoginCallbackRequest",
    "LoginUrlResponse",
    "RefreshTokenRequest",
    "TokenResponse",
    "UserOption",
]


class LoginUrlResponse(APIModel):
    """
    发起登录的结果。

    刻意**不**直接 302 跳转：后端返回地址、由前端决定怎么用它
    （`window.location.assign` 或弹窗），这样接口能脱离浏览器测，
    前端也拿得到 state 自己留档。
    """

    authorize_url: str = Field(
        description=(
            "Casdoor 授权页地址。浏览器打开它（带上 cookie）即可开始"
            "登录。\n\n"
            "**必须由收到本响应的那个浏览器打开** —— 请求同时下发了一个"
            " HttpOnly cookie，回调时要拿它校验 state 是不是同一个"
            "浏览器发起的。"
        ),
    )

    state: str = Field(
        description=(
            "防 CSRF 的状态值，回调时原样带回。签名 + 自带过期时间，"
            "不需要服务端存储。"
        ),
    )


class TokenResponse(APIModel):
    """
    Casdoor 签发的一整套令牌，原样透传。

    字段与 OAuth 2.0 的 token 响应一致，因此这里不做裁剪、不改名 ——
    多一层映射只会多一处失真。
    """

    access_token: str = Field(
        description=(
            "访问令牌。后续请求带 `Authorization: Bearer <它>`。\n\n"
            "JWT，可以用它去调所有 /api/v1 接口。"
        ),
    )

    token_type: str = Field(
        default="Bearer",
        description="固定是 Bearer。",
    )

    refresh_token: str | None = Field(
        default=None,
        description=(
            "刷新令牌。access_token 过期后用它换新的"
            "（POST /auth/refresh），有效期远长于 access_token。\n\n"
            "⚠️ 它的敏感度等同密码：拿到就能持续换出新 token。"
            "当前接口把它放在响应体里（前端自己保管），如果前端是"
            "浏览器，更稳妥的做法是改由后端写进 HttpOnly cookie —— "
            "那需要后端同时持有会话，本次未做。"
        ),
    )

    expires_in: int | None = Field(
        default=None,
        description="access_token 还有多少秒过期。由 Casdoor 给出。",
    )

    id_token: str | None = Field(
        default=None,
        description="OIDC 的 id_token（Casdoor 下与 access_token 内容基本相同）。",
    )

    scope: str | None = Field(
        default=None,
        description="本次授权到的 scope。",
    )


class RefreshTokenRequest(APIModel):
    """刷新令牌的请求体。"""

    refresh_token: str = Field(
        min_length=1,
        description="登录时拿到的 refresh_token。",
    )


class LoginCallbackRequest(APIModel):
    """
    回调请求体。

    Casdoor 验完会把浏览器跳到 redirect_uri 并带上 `?code=&state=`，
    由**前端**把这两个值原样转给本接口。另有一个 HttpOnly cookie
    随请求自动带上 —— 它不是请求体的一部分，但同样是必需的，
    见 TokenResponse / begin_login 的说明。
    """

    code: str = Field(
        min_length=1,
        description="授权码。**一次性** —— 换过一次就作废。",
    )

    state: str = Field(
        min_length=1,
        description="发起登录时拿到的 state，必须原样带回。",
    )


class UserOption(APIModel):
    """
    用户选项 —— 给前端的挑选控件的一行人。

    ⚠️ **不是**用户的完整记录。Casdoor 的用户对象有一百多个字段
    （`accessToken`、各类第三方凭据都在里面），本服务任何地方都不
    把它整份带出去；这里只有挑选控件要用的那四个。

    另一个刻意的取舍：**只有读**。没有 UserCreate / UserUpdate ——
    用户增删改在 Casdoor 控制台做（见模块 docstring）。
    """

    id: str = Field(
        description=(
            "Casdoor 用户 ID（UUID）。\n\n"
            "别的接口要指定一个用户时给的就是它 —— 同一个人登录后，"
            "`/auth/me` 的 `subject` 也是这个值。"
        ),
    )

    name: str = Field(
        default="",
        description="登录名。",
    )

    display_name: str = Field(
        default="",
        description="展示名（Casdoor 的 `displayName`）。可能为空。",
    )

    email: str = Field(
        default="",
        description="邮箱。可能为空 —— 不是所有组织都要求用户填邮箱。",
    )


class CurrentUserResponse(APIModel):
    """
    当前调用者。

    全部来自 access_token 的 claim，**一次库都不查** ——
    没有本地用户表（见模块 docstring）。

    ⚠️ 刻意**不**包含原始 claim。`AuthenticatedUser.raw_claims` 里有
    Casdoor 签发的全部内容（角色、权限等），排查问题时很好用，但那是
    服务端的东西，不该顺着这个接口流出去。
    """

    subject: str = Field(description="Casdoor 用户 ID。")

    name: str = Field(default="", description="登录名。")

    display_name: str = Field(default="", description="展示名。")

    email: str = Field(default="", description="邮箱。")

    groups: list[str] = Field(
        default_factory=list,
        description=(
            "归一化之后的群组名（拆数组 / 拆对象 / 去 owner 前缀之后"
            "的形态）。用于前端隐藏没有权限的入口。\n\n"
            "它反映的是 **token 里写着的** 群组，不是「实时」的："
            "管理员刚把某人移出群组，要等他的 token 过期（或他重新"
            "登录）才会反映出来。"
        ),
    )
