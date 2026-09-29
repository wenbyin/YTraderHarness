from __future__ import annotations
from fastapi import HTTPException, status

from app.common.exceptions import (
    ConflictError,
    NotFoundError,
)

__all__ = [
    "to_http_error"
]


def to_http_error(
        exc: ValueError,
) -> HTTPException:
    """
    把 Service 层异常映射为 HTTP 状态码。

        NotFoundError   -> 404
        ConflictError   -> 409
        其他 ValueError -> 400
    """

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

    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=str(exc),
    )
