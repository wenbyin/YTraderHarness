"""
`AgentSessionService.delete_session` 与 `create_run` 归档门禁。

不连数据库：假仓储 / 假调度器 / 假 thread 锁。验的是删除的执行顺序：

    严格本人 -> 列未终态运行 -> 按“有没有进程内任务”分流
             -> 归档 -> post-commit 里请求取消 / 放锁
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

import pytest

import app.services.session.application.agent_session_service_impl as session_impl_module
from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import ServiceContext
from app.services.session.application.agent_run_service_impl import (
    AgentRunServiceImpl,
)
from app.services.session.application.agent_session_service_impl import (
    AgentSessionServiceImpl,
)
from app.services.session.domain.entities import (
    AgentRun as AgentRunEntity,
)
from app.services.session.domain.entities import (
    AgentSession as AgentSessionEntity,
)
from app.services.session.domain.schemas import (
    CreateRunRequest,
    RunInput,
    RunStatus,
)

_NOW = datetime(2026, 10, 18, 12, 0, tzinfo=timezone.utc)

_UNFINISHED = {
    RunStatus.QUEUED.value,
    RunStatus.RUNNING.value,
    RunStatus.WAITING_INPUT.value,
}


def _session(
        *,
        user_id: str | None = "user-1",
        status: str = "active",
        thread_id: str | None = "thread-1",
        updated_at: datetime | None = None,
) -> AgentSessionEntity:

    return AgentSessionEntity(
        session_id=uuid4(),
        agent_id="stock_research",
        status=status,
        thread_id=thread_id,
        user_id=user_id,
        created_at=_NOW,
        updated_at=updated_at or _NOW,
    )


def _run(
        *,
        session_id: UUID,
        status: str,
) -> AgentRunEntity:

    return AgentRunEntity(
        run_id=uuid4(),
        session_id=session_id,
        status=status,
        created_at=_NOW,
        updated_at=_NOW,
    )


class _FakeSessionRepository:

    def __init__(self, sessions: list[AgentSessionEntity]) -> None:
        self._sessions = {
            session.session_id: session for session in sessions
        }

        self.archive_calls: list[UUID] = []

    async def get_owned_for_update(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSessionEntity | None:

        session = self._sessions.get(session_id)

        if session is None or session.user_id != user_id:
            return None

        return session

    async def get_for_update(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> AgentSessionEntity | None:

        session = self._sessions.get(session_id)

        if session is None:
            return None

        if session.user_id not in (user_id, None):
            return None

        return session

    async def archive(
            self,
            session_id: UUID,
            *,
            user_id: str,
    ) -> bool:

        session = self._sessions.get(session_id)

        if session is None or session.user_id != user_id:
            return False

        self.archive_calls.append(session_id)
        session.status = "archived"

        return True


class _FakeRunRepository:

    def __init__(self, runs: list[AgentRunEntity]) -> None:
        self._runs = {run.run_id: run for run in runs}
        self.add_calls: list[AgentRunEntity] = []

    async def get(
            self,
            run_id: UUID,
    ) -> AgentRunEntity | None:
        return self._runs.get(run_id)

    async def add(self, run: AgentRunEntity) -> AgentRunEntity:
        self.add_calls.append(run)
        self._runs[run.run_id] = run
        return run

    async def list_unfinished_for_session(
            self,
            session_id: UUID,
    ) -> list[AgentRunEntity]:

        return [
            run
            for run in self._runs.values()
            if run.session_id == session_id and run.status in _UNFINISHED
        ]

    async def cancel_if_unfinished(self, run_id: UUID) -> bool:

        run = self._runs.get(run_id)

        if run is None or run.status not in _UNFINISHED:
            return False

        run.status = RunStatus.CANCELLED.value
        return True


class _FakeDispatcher:

    def __init__(
            self,
            *,
            active: set[UUID] | None = None,
            accepted: bool = True,
    ) -> None:
        self._active = active or set()
        self._accepted = accepted

        self.is_active_calls: list[UUID] = []
        self.cancel_calls: list[UUID] = []

    async def is_active(self, run_id: UUID) -> bool:
        self.is_active_calls.append(run_id)
        return run_id in self._active

    async def cancel(self, run_id: UUID) -> bool:
        self.cancel_calls.append(run_id)
        return self._accepted


class _FakeThreadLock:

    def __init__(self) -> None:
        self.released: list[tuple[str | None, UUID]] = []

    def release(
            self,
            thread_id: str | None,
            *,
            run_id: UUID,
    ) -> None:
        self.released.append((thread_id, run_id))


def _service(
        sessions: _FakeSessionRepository,
        runs: _FakeRunRepository,
        *,
        dispatcher: _FakeDispatcher | None = None,
) -> tuple[AgentSessionServiceImpl, ServiceContext]:

    ctx = ServiceContext()

    service = AgentSessionServiceImpl(
        agent_session_repository=sessions,  # type: ignore[arg-type]
        agent_run_repository=runs,  # type: ignore[arg-type]
        dispatcher=dispatcher,  # type: ignore[arg-type]
        ctx=ctx,
    )

    return service, ctx


async def _flush_post_commit(ctx: ServiceContext) -> None:
    for hook in list(ctx.post_commit):
        await hook()


async def test_other_users_session_is_404() -> None:
    session = _session(user_id="user-2")
    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([])

    service, ctx = _service(sessions, runs)

    with pytest.raises(NotFoundError):
        await service.delete_session(session.session_id, user_id="user-1")

    assert sessions.archive_calls == []
    assert session.status == "active"


async def test_legacy_null_session_is_not_deletable() -> None:
    session = _session(user_id=None)
    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([])

    service, _ = _service(sessions, runs)

    with pytest.raises(NotFoundError):
        await service.delete_session(session.session_id, user_id="user-1")

    assert session.status == "active"


async def test_waiting_input_is_cancelled_in_transaction_and_lock_released(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    waiting_input 没有任务可 cancel：事务内直接置 cancelled，
    提交后补放 handler 刻意留着的 thread 锁。
    """

    lock = _FakeThreadLock()
    monkeypatch.setattr(
        session_impl_module, "get_thread_run_lock", lambda: lock,
    )

    session = _session()
    run = _run(session_id=session.session_id, status=RunStatus.WAITING_INPUT.value)

    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([run])
    dispatcher = _FakeDispatcher()

    service, ctx = _service(sessions, runs, dispatcher=dispatcher)

    await service.delete_session(session.session_id, user_id="user-1")

    # 事务内已经取消，归档也已发生。
    assert run.status == RunStatus.CANCELLED.value
    assert session.status == "archived"

    # waiting_input 不该去问调度器“你手里有没有它”—— 它本来就没有任务。
    assert dispatcher.is_active_calls == []

    await _flush_post_commit(ctx)

    assert dispatcher.cancel_calls == []
    assert lock.released == [(session.thread_id, run.run_id)]


async def test_active_running_is_cancelled_via_dispatcher_after_commit(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = _FakeThreadLock()
    monkeypatch.setattr(
        session_impl_module, "get_thread_run_lock", lambda: lock,
    )

    session = _session()
    run = _run(session_id=session.session_id, status=RunStatus.RUNNING.value)

    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([run])
    dispatcher = _FakeDispatcher(active={run.run_id})

    service, ctx = _service(sessions, runs, dispatcher=dispatcher)

    await service.delete_session(session.session_id, user_id="user-1")

    # 有任务的运行不直接改库：handler 还要回滚 checkpoint、写中断记录。
    assert run.status == RunStatus.RUNNING.value
    assert session.status == "archived"

    # 提交前不请求取消：事务还可能回滚。
    assert dispatcher.cancel_calls == []

    await _flush_post_commit(ctx)

    assert dispatcher.cancel_calls == [run.run_id]

    # 锁由 handler 在取消收尾时放，删除路径不抢着放。
    assert lock.released == []


async def test_queued_without_dispatcher_task_is_hard_cancelled(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    调度器还没受理的 queued（post-commit 尚未执行 / 进程重启的孤儿）：
    事务内直接取消，提交后放锁。
    """

    lock = _FakeThreadLock()
    monkeypatch.setattr(
        session_impl_module, "get_thread_run_lock", lambda: lock,
    )

    session = _session()
    run = _run(session_id=session.session_id, status=RunStatus.QUEUED.value)

    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([run])
    dispatcher = _FakeDispatcher(active=set())

    service, ctx = _service(sessions, runs, dispatcher=dispatcher)

    await service.delete_session(session.session_id, user_id="user-1")

    assert run.status == RunStatus.CANCELLED.value
    assert dispatcher.is_active_calls == [run.run_id]

    await _flush_post_commit(ctx)

    assert dispatcher.cancel_calls == []
    assert lock.released == [(session.thread_id, run.run_id)]


async def test_no_dispatcher_hard_cancels_everything() -> None:
    session = _session()
    run = _run(session_id=session.session_id, status=RunStatus.RUNNING.value)

    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([run])

    service, _ = _service(sessions, runs, dispatcher=None)

    await service.delete_session(session.session_id, user_id="user-1")

    assert run.status == RunStatus.CANCELLED.value
    assert session.status == "archived"


async def test_already_archived_is_idempotent() -> None:
    earlier = _NOW.replace(day=10)
    session = _session(status="archived", updated_at=earlier)

    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([])

    service, ctx = _service(sessions, runs)

    await service.delete_session(session.session_id, user_id="user-1")

    # 不写库、不推进 updated_at、也没有钩子要跑。
    assert sessions.archive_calls == []
    assert session.updated_at == earlier
    assert ctx.post_commit == []


async def test_archived_session_cannot_create_run() -> None:
    session = _session(status="archived")

    sessions = _FakeSessionRepository([session])
    runs = _FakeRunRepository([])

    service = AgentRunServiceImpl(
        agent_run_repository=runs,  # type: ignore[arg-type]
        agent_session_repository=sessions,  # type: ignore[arg-type]
        message_repository=object(),  # type: ignore[arg-type]
    )

    with pytest.raises(ConflictError, match="归档"):

        await service.create_run(
            session.session_id,
            CreateRunRequest(input=RunInput(text="你好")),
            user_id="user-1",
        )

    assert runs.add_calls == []
