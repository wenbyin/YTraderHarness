"""
内部对象 -> 对外契约的纯函数组装。

与 capability 模块的 _assembly.py 同一个角色：**"这个东西对外长
什么样"只有一份实现**。放这里而不是散在 impl 与 router 里，是因为
两边都要用到（impl 产出 DTO 给 router，router 还要单独处理 cookie），
两处各写一遍迟早会不一致。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from app.common.auth.models import AuthenticatedUser

from ..domain.schemas.auth import (
    CurrentUserResponse,
    LoginUrlResponse,
    TokenResponse,
    UserOption,
)
from .user_service import LoginTicket

__all__ = [
    "build_current_user",
    "build_login_url",
    "build_token_response",
    "build_user_options",
]


def build_login_url(
        ticket: LoginTicket,
) -> LoginUrlResponse:
    """
    发起登录的结果 -> 对外契约。

    `state_nonce` **刻意丢掉**：它只该进 cookie，不该出现在响应体里
    （它是 state 的组成部分，但把它单列出来等于告诉前端"这个东西有
    意义"，而前端不该碰它）。
    """

    return LoginUrlResponse(
        authorize_url=ticket.authorize_url,
        state=ticket.state,
    )


def build_token_response(
        raw: dict[str, Any],
) -> TokenResponse:
    """
    Casdoor 的原始 token 响应 -> 对外契约。

    `access_token` 由上一步（组件里的 _require_access_token）保证非空，
    因此这里可以直接取。

    其余字段一律 `.get` —— Casdoor 给不给 refresh_token / id_token
    取决于应用上有没有开对应的 grant，缺了就是 None，不是错误。
    """

    return TokenResponse(
        access_token=raw["access_token"],
        token_type=raw.get("token_type") or "Bearer",
        refresh_token=raw.get("refresh_token"),
        expires_in=raw.get("expires_in"),
        id_token=raw.get("id_token"),
        scope=raw.get("scope"),
    )


def build_current_user(
        user: AuthenticatedUser,
) -> CurrentUserResponse:
    """
    校验过的调用者 -> 对外契约。

    ⚠️ 只取白名单里的字段：`user.raw_claims` **不要**带出去
    （原因见 CurrentUserResponse 的 docstring）。
    """

    return CurrentUserResponse(
        subject=user.subject,
        name=user.name,
        display_name=user.display_name,
        email=user.email,
        groups=list(user.groups),
    )


def build_user_options(
        rows: Sequence[Mapping[str, Any]],
) -> list[UserOption]:
    """
    Casdoor 的用户对象 -> 用户选项。

    ⚠️ 白名单是**这里的**职责，不是组件的：组件只负责"怎么问
    Casdoor"，问回来的原始对象整份交给本函数，由它决定哪几个字段
    可以出去。Casdoor 的用户对象有一百多个字段（`accessToken`、
    `password`、各类第三方凭据都在里面）—— 顺手 `**row` 展开就等于
    把目录连底裤一起发给前端。

    缺失字段一律当空串：Casdoor 对没填的项有时给 `None`、有时干脆
    不给这个键，两种都不该变成 `None`（契约里那几项是 string）。
    """

    return [
        UserOption(
            id=str(row.get("id") or ""),
            name=str(row.get("name") or ""),
            display_name=str(row.get("displayName") or ""),
            email=str(row.get("email") or ""),
        )
        for row in rows
    ]
