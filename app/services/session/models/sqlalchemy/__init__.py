"""
Session 模块的 SQLAlchemy 模型。

这里逐个 import 全部模型（与 tool 模块同一理由）：模型之间是字符串形式的
relationship 目标时，SQLAlchemy 要到 mapper 配置才解析 —— 少加载一个模块
就会以一句离原因很远的 "failed to locate a name" 失败。

本模块的三张表之间**没有** relationship（运行单向指向会话、消息单向指向
两者，都不反向聚合），因此这里的作用主要是让 initializer 的
import_all_models() 一次注册完。

⚠️ 新增模型**必须在这里 import 一行**，否则
`Base.metadata.create_all` 根本不知道有那张表 —— 它静默跳过，不报错：
启动看起来一切正常，直到第一次查询报 "relation does not exist"。
"""

from .agent_message import AgentMessage
from .agent_run import AgentRun
from .agent_session import AgentSession

__all__ = [
    "AgentMessage",
    "AgentRun",
    "AgentSession",
]
