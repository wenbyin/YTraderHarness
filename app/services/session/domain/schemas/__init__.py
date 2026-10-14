"""
Session 模块的对外契约。

domain/schemas/
├── enum.py
│   └── SessionStatus / RunStatus / MessageRole / MessageKind / MessageStatus
├── session.py
│   └── CreateSessionRequest / Session
├── run.py
│   └── RunInput / CreateRunRequest / Run / ResumeAnswer / ResumeRequest
└── agent_message.py
    └── ContentBlock / AgentMessage / Pagination / MessagePage

⚠️ 上表里的 AgentMessage 在规格（messages.yaml）里叫 Message —— 类名与表名
一起加了 agent_ 前缀，生成的 OpenAPI 组件名由 title 加 app/common/openapi.py
的改名表钉回 Message，因此对外契约没变。见 agent_message.py 的说明。

全部结构逐字段对应 components/schemas；字段名与枚举取值都以 spec 为准。
⚠️ 消息那一组在**另一份 spec**（messages.yaml）里，不是 session.yaml ——
规格分了两份文件，实现上不分家（同包同路由）。

`ResumeRequest` 是对 `POST /runs/{run_id}/resume` 的补充定义 —— 规格里
早有它（interrupt_id + data），端点现在也实现了。

⚠️ agent_id **仍然没有校验**，但这不再是空洞
------------------------------------------
以前这里写着"一个已知的契约空洞"：agent_id 是个自由字符串，而"有哪些
Agent"没有目录，因此它无法映射到一张图。

接派发（③）时把这件事定了：**从这张 API 进来的只有一种图**，前端不选图、
也不感知 agent_id。因此**不建目录** —— agent_id 照旧是规格的必填字段
（契约先行，不单方面改它），照旧落库，但运行一律走 `agent_react`
（见 application/agent_run_service_impl.py）。

"任意字符串都能建会话"因此不会炸在任何地方，它只是一个记录下来的标签。
将来真的出现第二种图时，"多个 agent_id 各自对应哪张图"才需要一块目录 ——
那时补，而不是现在先摆一张只有一行的表。
"""

from .enum import (
    MessageKind,
    MessageRole,
    MessageStatus,
    RunStatus,
    SessionStatus,
)
from .agent_message import (
    AgentMessage,
    ContentBlock,
    MessagePage,
    Pagination,
    decode_cursor,
    encode_cursor,
)
from .run import (
    CreateRunRequest,
    ResumeAnswer,
    ResumeRequest,
    Run,
    RunInput,
)
from .session import CreateSessionRequest, Session

__all__ = [
    "AgentMessage",
    "ContentBlock",
    "CreateRunRequest",
    "CreateSessionRequest",
    "MessageKind",
    "MessagePage",
    "MessageRole",
    "ResumeAnswer",
    "ResumeRequest",
    "MessageStatus",
    "Pagination",
    "Run",
    "RunInput",
    "RunStatus",
    "Session",
    "SessionStatus",
    "decode_cursor",
    "encode_cursor",
]
