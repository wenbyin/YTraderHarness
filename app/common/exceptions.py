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


class AuthenticationError(ApplicationError, ValueError):
    """
    没有凭据，或凭据不成立。

    包括：
        - 完全没带 token
        - 签名不符 / 已过期 / audience 不对
        - token 结构无法解析

    与 PermissionDeniedError 的分界是**"你是谁"与"你能干什么"**：
    这一条回答不出"你是谁"，所以客户端该换一份凭据重试。

    Controller 映射为 401，并附带 `WWW-Authenticate: Bearer`
    （RFC 6750 要求 401 指明认证方案）。
    """


class PermissionDeniedError(ApplicationError, ValueError):
    """
    凭据成立，但这个人不许做这件事。

    鉴权组件目前只用它表达"token 有效但不在要求的群组里"。
    客户端**换凭据也没用** —— 所以要给 403 而不是 401，
    否则前端会陷入"反复重新登录"的循环。

    Controller 映射为 403。
    """


class ServiceUnavailableError(ApplicationError, ValueError):
    """
    依赖的外部服务没有启用或不可达，接口此刻无法完成请求。

    与 500 的分界：这是**可解释的、部署/配置层面的**不可用
    （比如本地开发关掉了 Casdoor，而登录接口必须去问它），
    不是代码 bug。客户端重试不会好，换凭据也不会。

    Controller 映射为 503。
    """
