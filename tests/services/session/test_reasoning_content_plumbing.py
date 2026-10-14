"""
`reasoning_content` 在两个**逐字段手抄**的关口上不能掉。

为什么这两条测试比别处更值
------------------------
从实体到库里那一行、再从实体到接口那个 JSON，中间是**逐字段**写的
（`add_many` 里一个一个命名参数、`build_message` 里一个一个命名参数），
不是 `model_dump()` 那种照单全收。漏一行的症状是**静默**的：

    add_many 漏了        -> 库里那一列恒为 NULL（接口、前端都看不到）
    build_message 漏了    -> 库里有值，接口永远不返回（更隐蔽）

两处都不会报错、都不会让任何既有用例变红 —— 所以各钉一条。
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace
from uuid import uuid4

from app.services.session.application._assembly import build_message
from app.services.session.domain.entities import AgentMessage as AgentMessageEntity
from app.services.session.repository.message_impl import MessageRepositoryImpl

_THINKING = "先看要不要查工具，再答。"


class _StubSession:
    """
    只记下 `add_all` 收到什么。

    `add_many` 在真路径上还会 flush（由调用方的事务收尾），这里不需要 ——
    本测试问的是"交给 ORM 的那个对象上有没有那个字段"。
    """

    def __init__(self) -> None:
        self.added: list[object] = []

    def add_all(self, models: object) -> None:
        self.added.extend(models)  # type: ignore[arg-type]

    async def flush(self) -> None:
        return None


def _entity(**overrides: object) -> AgentMessageEntity:
    now = datetime.now(timezone.utc)

    payload: dict[str, object] = {
        "message_id": uuid4(),
        "session_id": uuid4(),
        "run_id": uuid4(),
        "seq": 1,
        "role": "assistant",
        "kind": "chat",
        "content": [{"type": "text", "text": "答复"}],
        "reasoning_content": _THINKING,
        "status": "completed",
        "created_at": now,
        "updated_at": now,
    }

    payload.update(overrides)

    return AgentMessageEntity(**payload)  # type: ignore[arg-type]


async def test_add_many_copies_reasoning_into_the_orm_row() -> None:
    """实体 -> ORM 那一跳带上思维链（漏了就是库里恒 NULL）。"""

    session = _StubSession()
    repository = MessageRepositoryImpl(session=session)  # type: ignore[arg-type]

    await repository.add_many([_entity()])

    assert len(session.added) == 1

    row = session.added[0]

    assert getattr(row, "reasoning_content") == _THINKING
    # 正文那一列不受影响：推理不进内容块数组（见 message_recorder）。
    assert getattr(row, "content") == [{"type": "text", "text": "答复"}]


def test_build_message_returns_reasoning() -> None:
    """实体 -> 契约那一跳带上思维链（漏了就是接口永远不返回它）。"""

    dto = build_message(_entity())

    assert dto.reasoning_content == _THINKING
    # 名字与契约（messages.yaml 的 Message.reasoning_content）一致 ——
    # 与 meta -> metadata 那种改名不同，这一个不改名。
    assert dto.model_dump()["reasoning_content"] == _THINKING


def test_user_message_has_no_reasoning() -> None:
    """
    用户消息恒为 NULL —— "人说的话"没有草稿纸。

    实体默认值给的是 None，所以两处写入点（建运行写用户消息、取消公告）
    不传它就自动是对的。这条钉住那个默认值。
    """

    entity = _entity(role="user", reasoning_content=None)

    assert entity.reasoning_content is None
    assert build_message(entity).reasoning_content is None

    # 默认值本身也是 None（不是空串）：空串会让前端渲染一个永远空着的
    # 推理框，见 tests/agent/context/test_message_reasoning.py。
    assert AgentMessageEntity(
        message_id=uuid4(),
        session_id=uuid4(),
        seq=2,
        role="assistant",
        kind="chat",
        content=[],
        status="completed",
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
    ).reasoning_content is None


def test_contract_field_is_optional() -> None:
    """契约里它不是必填（与 business_agent_id 同一个形状）。"""

    dto = build_message(_entity(reasoning_content=None))

    assert dto.model_dump()["reasoning_content"] is None

    # 顺带一条：SimpleNamespace 不是它的形状 —— 契约的 extra=forbid 只拦
    # 构造时的多余关键字，这里只是提醒别用鸭子对象冒充 DTO。
    assert not isinstance(dto, SimpleNamespace)
