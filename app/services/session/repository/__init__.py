"""
Session 模块的持久化契约与实现。

契约 -> 实现的对应**没有任何登记表**：
工厂靠 common/introspect.py 的 resolve_impl 反射发现
（契约的子类，或同包同名的 X_impl.py）。

三个契约（会话 / 运行 / 消息）而不是一个 —— 理由见 agent_session.py 与
message.py 的说明：三者的读写形状与生命周期都不同，合成一个只会互相牵制。
"""

from .agent_run import AgentRunRepository
from .agent_run_impl import AgentRunRepositoryImpl
from .agent_session import AgentSessionRepository
from .agent_session_impl import AgentSessionRepositoryImpl
from .message import MessageRepository
from .message_impl import MessageRepositoryImpl

__all__ = [
    "AgentRunRepository",
    "AgentRunRepositoryImpl",
    "AgentSessionRepository",
    "AgentSessionRepositoryImpl",
    "MessageRepository",
    "MessageRepositoryImpl",
]
