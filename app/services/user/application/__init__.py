"""
用户模块的应用服务。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现
（同包同名的 X_impl.py）。

⚠️ 实现（UserServiceImpl）的 __init__ **只有 ctx 一个参数**，刻意
不声明任何仓储 —— 于是 ServiceFactory.needs_session 判定为"不需要
事务"，/auth/* 的每个请求都不会白占一条数据库连接。本模块不落库
（Casdoor 是用户唯一真相）。

_assembly.py 不是服务，是内部对象 -> 对外契约的纯函数组装。
"""

from .user_service import LoginTicket, UserService
from .user_service_impl import UserServiceImpl

__all__ = [
    "LoginTicket",
    "UserService",
    "UserServiceImpl",
]
