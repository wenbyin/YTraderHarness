"""
Session 模块的应用服务。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现
（同包同名的 X_impl.py）。服务的 __init__ 参数用仓储契约注解，
由 ServiceFactory 自动注入。

三个服务（会话 / 运行 / 消息），各自只依赖自己需要的仓储 ——
AgentRunService 与 MessageService 都额外注入会话仓储，因为"会话必须存在"
这条校验要读它（而不是自己直接查表）。AgentRunService 还注入消息仓储：
用户消息是在它建运行的那一步写下的（见那个文件的写入点）。

⚠️ 还有两个**不是服务**的模块
---------------------------
    agent_run_dispatch.py   产出 dispatcher 的 handler（组合根构造一次、
                            交给调度器）
    message_recorder.py     把图产出的消息翻成实体并落库，被 handler 调用

两者都**不经工厂**、因此不在下面的 import 里。写在这里只是因为
"派发逻辑 / 落库逻辑在哪"是读这个包时一定会问的问题。
"""

from .agent_run_service import AgentRunService
from .agent_run_service_impl import AgentRunServiceImpl
from .agent_session_service import AgentSessionService
from .agent_session_service_impl import AgentSessionServiceImpl
from .message_service import MessageService
from .message_service_impl import MessageServiceImpl

__all__ = [
    "AgentRunService",
    "AgentRunServiceImpl",
    "AgentSessionService",
    "AgentSessionServiceImpl",
    "MessageService",
    "MessageServiceImpl",
]
