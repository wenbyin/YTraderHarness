from __future__ import annotations
from fastapi import HTTPException, status

from app.common.exceptions import (
    AuthenticationError,
    ConflictError,
    NotFoundError,
    PermissionDeniedError,
    ServiceUnavailableError,
)

__all__ = [
    "to_http_error"
]


def to_http_error(
        exc: ValueError,
) -> HTTPException:
    """
    把 Service 层异常映射为 HTTP 状态码。

        NotFoundError          -> 404
        ConflictError          -> 409
        AuthenticationError    -> 401（带 WWW-Authenticate: Bearer）
        PermissionDeniedError  -> 403
        ServiceUnavailableError -> 503
        其他 ValueError        -> 400

    ⚠️ 这里**接不到**路由级依赖里抛的异常
    -------------------------------------
    本函数只在路由函数体里被 `except ValueError` 调用。而挂在
    `APIRouter(dependencies=[...])` 上的依赖跑在函数体**之外**，
    它抛的东西直接穿到 ASGI 异常中间件 —— 于是在那里抛
    AuthenticationError 的后果是 **500，不是 401**。

    所以鉴权依赖（app/common/auth/dependencies.py）自己翻译成
    HTTPException；本函数负责的是"端点体内被 Service 抛出来"那条路径。
    两条路都要有，因为两条都真的会发生。
    """

    if isinstance(exc, AuthenticationError):
        return HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        )

    if isinstance(exc, PermissionDeniedError):
        return HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(exc),
        )

    if isinstance(exc, NotFoundError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )

    if isinstance(exc, ConflictError):
        return HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )

    if isinstance(exc, ServiceUnavailableError):
        return HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        )

    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc),
    )
