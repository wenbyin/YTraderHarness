"""
跨模块共享的应用层异常。

放在 app.common 而不是某个 service 模块下，
是为了让 app.common.utils 这类工具函数能够使用它们，
而不必反向依赖 app.services.*。

app.common 必须保持为纯叶子：
不允许出现 from app.services... 的导入。
"""

from __future__ import annotations


class ApplicationError(Exception):
    """
    应用层异常基类。

    同时继承 ValueError，
    使既有的 `except ValueError` 调用方继续可用。
    """


class NotFoundError(ApplicationError, ValueError):
    """
    请求的资源不存在。

    Controller 映射为 404。
    """


class ConflictError(ApplicationError, ValueError):
    """
    请求与现有数据冲突。

    包括：
        - 唯一键重复
        - 资源已被逻辑删除
        - 状态不允许当前操作

    Controller 映射为 409。
    """
