"""
用户模块的对外契约。

domain/schemas/
└── auth.py
    └── LoginUrlResponse / TokenResponse / RefreshTokenRequest /
        CurrentUserResponse / UserOption

全部结构逐字段对应 user.yaml 的 components/schemas；字段名以那份
spec 为准。要改就改 user.yaml，两边一起。

⚠️ 这里没有 `User` 这个 schema。"用户"不是一个能被本服务读写的资源
（它住在 Casdoor）。对外只有两个视角：**当前调用者**
（CurrentUserResponse，来自 token）与**目录里的候选人**
（UserOption，来自 Casdoor 的目录查询，管理端用）。
"""

from .auth import (
    CurrentUserResponse,
    LoginCallbackRequest,
    LoginUrlResponse,
    RefreshTokenRequest,
    TokenResponse,
    UserOption,
)

__all__ = [
    "CurrentUserResponse",
    "LoginCallbackRequest",
    "LoginUrlResponse",
    "RefreshTokenRequest",
    "TokenResponse",
    "UserOption",
]
