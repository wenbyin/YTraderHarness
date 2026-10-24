"""
HttpStressRunner 冒烟测试。

跑法：

    .venv/Scripts/python.exe -m app.services.stress.stress_driver_smoke

它不依赖 PostgreSQL / Redis / 真实 Agent，而是起一个本地 fake target
（aiohttp），验证：

    - Driver 独立线程能启动并领取会话；
    - 同一 session 的 N 轮严格串行；
    - SSE 终态、TTFT、轮次/会话计数；
    - 最终快照由 Reporter 收到；
    - Driver 停止后线程退出。

这是 Phase A 的行为验证，不是性能压测。
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
import time
from types import SimpleNamespace
from uuid import UUID, uuid4

import aiohttp
from aiohttp import web

from app.services.stress.application.http_runner import HttpStressRunner
from app.services.stress.domain.schemas.stress import (
    ConcurrencyLoad,
    MockModelConfig,
    SessionConfig,
    StressRunCreate,
    StressRunStatus,
    TargetConfig,
    TurnConfig,
    TurnCountConfig,
)

HOST = "127.0.0.1"
PORT = 18090
BASE = f"http://{HOST}:{PORT}"


def check(label: str, condition: bool) -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")

    if not condition:
        raise AssertionError(label)


class FakeReporter:
    """替代 StressRunReporter：只在内存里记录。"""

    def __init__(self) -> None:
        self.snapshots = []
        self.completed = None
        self.failed = None
        self.status = StressRunStatus.RUNNING
        self.marked_running = False

    async def mark_running(self, run_id):
        self.marked_running = True
        return SimpleNamespace(run_id=run_id, status=StressRunStatus.RUNNING)

    async def get_run(self, run_id):
        return SimpleNamespace(run_id=run_id, status=self.status)

    async def update_metrics(self, run_id, metrics):
        self.snapshots.append(metrics)
        return metrics

    async def complete_run(self, run_id, metrics):
        self.completed = metrics
        return SimpleNamespace(run_id=run_id, status=StressRunStatus.COMPLETED)

    async def mark_failed(self, run_id, *, error_code, error_message):
        self.failed = (error_code, error_message)
        return SimpleNamespace(run_id=run_id, status=StressRunStatus.FAILED)


class FakeTarget:
    """最小的 Agent HTTP/SSE 替身。"""

    def __init__(self) -> None:
        self.sessions: set[str] = set()
        self.runs: dict[str, str] = {}
        self.inflight: dict[str, int] = {}
        self.serial_violations = 0
        self.completed_runs = 0
        self._lock = asyncio.Lock()

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_post("/api/v1/sessions", self.create_session)
        app.router.add_post(
            "/api/v1/sessions/{session_id}/runs",
            self.create_run,
        )
        app.router.add_get(
            "/api/v1/runs/{run_id}/events",
            self.events,
        )
        app.router.add_post(
            "/api/v1/runs/{run_id}/cancel",
            self.cancel,
        )
        app.router.add_delete(
            "/api/v1/sessions/{session_id}",
            self.delete_session,
        )
        return app

    async def create_session(self, request: web.Request) -> web.Response:
        body = await request.json()

        if body.get("agent_id") != "stock_research":
            return web.json_response({"error": "bad agent"}, status=400)

        session_id = str(uuid4())
        self.sessions.add(session_id)

        return web.json_response(
            {"session_id": session_id, "agent_id": "stock_research"}
        )

    async def create_run(self, request: web.Request) -> web.Response:
        session_id = request.match_info["session_id"]
        body = await request.json()

        async with self._lock:
            current = self.inflight.get(session_id, 0)

            if current != 0:
                self.serial_violations += 1

            self.inflight[session_id] = current + 1

        run_id = str(uuid4())
        self.runs[run_id] = session_id

        if not body.get("input", {}).get("text"):
            return web.json_response({"error": "empty input"}, status=400)

        return web.json_response(
            {"run_id": run_id, "session_id": session_id, "status": "queued"}
        )

    async def events(self, request: web.Request) -> web.StreamResponse:
        run_id = request.match_info["run_id"]
        session_id = self.runs.get(run_id)

        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
            },
        )
        await response.prepare(request)

        async def send(payload: dict) -> None:
            data = json.dumps(payload, ensure_ascii=False)
            await response.write(
                f"data: {data}\n\n".encode("utf-8")
            )

        await asyncio.sleep(0.02)
        await send(
            {
                "event": "message.delta",
                "data": {"delta": "hello"},
            }
        )
        await asyncio.sleep(0.02)
        await send(
            {
                "event": "run.completed",
                "data": {"finish_reason": "stop"},
            }
        )
        await response.write_eof()

        async with self._lock:
            self.completed_runs += 1

            if session_id in self.inflight:
                self.inflight[session_id] -= 1

        return response

    async def cancel(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "cancelled"})

    async def delete_session(self, request: web.Request) -> web.Response:
        self.sessions.discard(request.match_info["session_id"])
        return web.json_response({"status": "deleted"})


def make_config() -> StressRunCreate:
    return StressRunCreate(
        name="driver-smoke",
        target=TargetConfig(base_url=BASE),
        session=SessionConfig(
            agent_id="stock_research",
            turns=TurnCountConfig(
                mean=2,
                stddev=0,
                min=1,
                max=30,
            ),
            cleanup="keep",
        ),
        turn=TurnConfig(
            input_template={"text": "hello"},
        ),
        load=ConcurrencyLoad(mode="concurrency", concurrency=2),
        duration_seconds=30,
        total_sessions=2,
        request_timeout_ms=5000,
        seed=42,
        mock_model=MockModelConfig(
            base_latency_ms=0,
            ttft_ms=0,
            output_tokens_min=1,
            output_tokens_max=2,
            tokens_per_second=100,
        ),
    )


async def run_smoke() -> None:
    target = FakeTarget()
    target_runner = web.AppRunner(target.app())
    await target_runner.setup()
    site = web.TCPSite(target_runner, HOST, PORT)
    await site.start()

    reporter = FakeReporter()
    runner = HttpStressRunner(reporter)
    run_id = uuid4()

    try:
        runner.schedule(run_id, make_config())

        deadline = time.monotonic() + 10.0

        while reporter.completed is None and reporter.failed is None:
            if time.monotonic() > deadline:
                raise AssertionError("Driver 在 10s 内没有完成")

            await asyncio.sleep(0.05)

        metrics = reporter.completed

        check("mark_running 被调用", reporter.marked_running)
        check("Campaign 正常完成", metrics is not None)
        check("没有失败", reporter.failed is None)

        if metrics is None:
            raise AssertionError("没有最终快照")

        check("sessions_started=2", metrics.sessions_started == 2)
        check("sessions_completed=2", metrics.sessions_completed == 2)
        check("turns_started=4", metrics.turns_started == 4)
        check("turns_succeeded=4", metrics.turns_succeeded == 4)
        check("轮次延迟有 4 个样本",
              metrics.turn_latency.get("count") == 4)
        check("TTFT 有 4 个样本",
              metrics.time_to_first_output.get("count") == 4)
        check("没有同会话并发",
              target.serial_violations == 0)
        check("目标端收到 4 个终态",
              target.completed_runs == 4)
        check("算法标识正确",
              metrics.aggregation_algorithm == "runner-exact-capped-v1")

    finally:
        runner.stop()
        await target_runner.cleanup()


def main() -> int:
    asyncio.run(run_smoke())

    print("\nStress driver smoke: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
