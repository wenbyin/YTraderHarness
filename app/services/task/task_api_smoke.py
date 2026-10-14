"""
任务查询/控制接口的冒烟：**真打 HTTP**（ASGI），不是直接调服务。

跑法：
    python -m app.services.task.task_api_smoke

⚠️ 为什么要走 HTTP，不能只调服务
------------------------------
直接 `await service.list_tasks(...)` 会绕过 FastAPI 的 response_model
校验与序列化。那样即使装配函数把**实体**当 DTO 返回、字段名拼错、
枚举没转成字符串，也一律全绿 —— 而那些问题在真实请求上就是 500 或
前端拿到 undefined。

所以这里用 httpx 的 ASGITransport 真发请求（不启真端口），让 FastAPI
把响应按 task.yaml 的契约校验一遍。断言也据此写：**先断类型，再断取值**
（"attempt 是 int"比"attempt 等于 1"更接近这个文件要验的东西）。

⚠️ 挂的是 tasks_router 本身，不是 protected_router
-----------------------------------------------
protected_router 上挂着鉴权依赖，而本冒烟要验的是任务接口本身的行为，
不是鉴权（那有专门的用例，见 tests/api/test_auth_enforcement.py）。
绕开鉴权是这个文件的目的，不是疏忽。
"""

from __future__ import annotations

import asyncio
import logging
from abc import abstractmethod
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

import httpx
from fastapi import FastAPI
from pydantic import BaseModel
from sqlalchemy import delete, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.api.v1.tasks import router as tasks_router
from app.common.factories import build_factories, reset_factories
from app.common.repository.factory import RepositoryFactory
from app.common.service.base import BaseService
from app.common.task import Task, TaskStatus
from app.infra.task import (
    DbPollingTaskDispatcher,
    OutboxRelay,
    RetryPolicy,
    TaskHandlerRegistryImpl,
    Worker,
)
from app.services.task.application import TaskService
from app.services.task.models.sqlalchemy.task import Task as TaskModel
from app.services.task.models.sqlalchemy.task_outbox import (
    TaskOutbox as TaskOutboxModel,
)
from app.services.task.repository import TaskOutboxStore, TaskStore

DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

PROBE = "task_api_smoke"

WAIT_SECONDS = 10.0

FAILS: list[str] = []


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        FAILS.append(label)


def section(title: str) -> None:
    print(f"\n== {title} ==")


class ApiParams(BaseModel):
    label: str = ""


class ApiService(BaseService):

    @abstractmethod
    async def ok(self, params: ApiParams, ctx: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    async def boom(self, params: ApiParams, ctx: Any) -> None:
        raise NotImplementedError


class ApiServiceImpl(ApiService):

    async def ok(self, params: ApiParams, ctx: Any) -> None:
        return None

    async def boom(self, params: ApiParams, ctx: Any) -> None:
        raise ValueError("这条就是会失败")


def build_registry() -> TaskHandlerRegistryImpl:

    registry = TaskHandlerRegistryImpl()

    for method in ("ok", "boom"):
        registry.register(
            handler=f"{PROBE}.{method}",
            service_contract=ApiService,
            method=method,
            params_type=ApiParams,
        )

    return registry


async def main() -> int:

    logging.basicConfig(level=logging.CRITICAL)

    engine = create_async_engine(DSN)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    from app.infra.datasource.initializer import (
        import_all_models,
        initialize_database,
    )

    import_all_models()
    await initialize_database(engine)

    repositories = RepositoryFactory(session_factory=session_factory)
    registry = build_registry()

    services = build_factories(
        session_factory,
        pool={"handler_registry": registry},
    )

    worker = Worker(
        repositories=repositories,
        services=services,
        registry=registry,
        policy=RetryPolicy(jitter=0.0),
        lease_seconds=60.0,
        identity="api-smoke",
    )
    dispatcher = DbPollingTaskDispatcher(
        worker=worker,
        repositories=repositories,
        concurrency=2,
        sweep_interval=0.15,
    )
    relay = OutboxRelay(
        dispatcher=dispatcher,
        repositories=repositories,
        interval=0.15,
    )

    # TaskService 的提交后投递钩子要靠这个键（见 TaskServiceImpl 的说明）。
    services._pool["task_dispatcher"] = dispatcher

    await dispatcher.start()

    app = FastAPI()
    app.include_router(tasks_router, prefix="/api/v1")

    created: list[UUID] = []

    async def submit(handler: str, label: str, **kw) -> UUID:

        async with services.scope(TaskService) as service:

            task_id = await service.submit(
                Task(
                    handler=f"{PROBE}.{handler}",
                    params=ApiParams(label=label),
                    **kw,
                )
            )

        created.append(task_id)

        return task_id

    async def wait_for(task_id: UUID, *statuses: TaskStatus):
        deadline = asyncio.get_running_loop().time() + WAIT_SECONDS

        while asyncio.get_running_loop().time() < deadline:

            async with repositories.scope(begin=False) as repos:
                row = await repos.create(TaskStore).get(task_id)

            if row is not None and row.status in statuses:
                return row

            await asyncio.sleep(0.025)

        return None

    try:

        transport = httpx.ASGITransport(app=app)

        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://task-smoke",
        ) as client:

            # =================================================
            section("造几条不同状态的任务")

            succeeded = await submit("ok", "will-succeed")
            failed = await submit("boom", "will-fail", max_attempts=1)

            await wait_for(succeeded, TaskStatus.SUCCESS)
            await wait_for(failed, TaskStatus.FAILED)

            # 一条直接落库、没人投递的任务，用来验取消
            pending_id = uuid4()

            async with repositories.scope() as repos:

                await repos.create(TaskStore).add(
                    Task(
                        handler=f"{PROBE}.ok",
                        params=ApiParams(label="to-cancel"),
                        task_id=pending_id,
                    ).seal_root()
                )

            created.append(pending_id)

            check("三条任务各自到位",
                  succeeded and failed and pending_id)

            # =================================================
            section("GET /tasks（列表）")

            response = await client.get("/api/v1/tasks", params={"limit": 50})

            check("200", response.status_code == 200,
                  f"实际 {response.status_code} {response.text[:200]}")

            body = response.json()

            check("有 items（列表）与 total（数字）",
                  isinstance(body.get("items"), list)
                  and isinstance(body.get("total"), int),
                  f"实际 {type(body.get('items')).__name__}"
                  f"/{type(body.get('total')).__name__}")

            mine = [
                it for it in body["items"]
                if it.get("handler", "").startswith(PROBE)
            ]

            check("能查到本次造的任务", len(mine) >= 3,
                  f"实际 {len(mine)}")

            if mine:
                item = mine[0]

                # 先断类型：FastAPI 的 response_model 会在这里暴露装配问题
                check("task_id 是字符串",
                      isinstance(item.get("task_id"), str))
                check("status 是字符串（枚举序列化过了，不是对象）",
                      isinstance(item.get("status"), str),
                      f"实际 {type(item.get('status')).__name__}")
                check("attempt / max_attempts 是数字",
                      isinstance(item.get("attempt"), int)
                      and isinstance(item.get("max_attempts"), int))
                check("created_at 是 ISO 字符串",
                      isinstance(item.get("created_at"), str))
                # 列表**不含** params —— Summary 与 Detail 的划分必须在
                # HTTP 边界上成立，否则"列表少发点字段"就白设计了。
                check("列表不含 params（只有详情才给）",
                      "params" not in item)

            # 状态筛选（同时证明枚举取值与契约一致）
            filtered = await client.get(
                "/api/v1/tasks", params={"status": "failed"}
            )
            check("按 status 过滤返回 200", filtered.status_code == 200)

            check("过滤结果全部是 failed",
                  all(
                      it["status"] == "failed"
                      for it in filtered.json()["items"]
                  ))

            # 非法枚举值必须被 FastAPI 挡在前面的 422，
            # 而不是一路走到仓储变成一个空结果或 500。
            bogus = await client.get(
                "/api/v1/tasks", params={"status": "no-such-status"}
            )
            check("非法 status 是 422（枚举接上了契约）",
                  bogus.status_code == 422,
                  f"实际 {bogus.status_code}")

            # =================================================
            section("GET /tasks/{task_id}（详情）")

            detail = await client.get(f"/api/v1/tasks/{failed}")

            check("200", detail.status_code == 200)

            payload = detail.json()

            check("详情**有** params，且是对象",
                  isinstance(payload.get("params"), dict),
                  f"实际 {type(payload.get('params')).__name__}")
            check("params 的值原样送达",
                  payload.get("params", {}).get("label") == "will-fail",
                  f"实际 {payload.get('params')}")
            check("详情带 error（失败原因能查到）",
                  isinstance(payload.get("error"), str)
                  and "ValueError" in payload["error"],
                  f"实际 {payload.get('error')!r}")
            check("详情带 causation_id 字段（可为 null）",
                  "causation_id" in payload)
            check("详情带租约字段（可为 null）",
                  "lease_expires_at" in payload
                  and "leased_by" in payload)

            missing = await client.get(f"/api/v1/tasks/{uuid4()}")
            check("不存在的任务 -> 404", missing.status_code == 404,
                  f"实际 {missing.status_code}")

            bad_uuid = await client.get("/api/v1/tasks/not-a-uuid")
            check("非法 UUID -> 422", bad_uuid.status_code == 422,
                  f"实际 {bad_uuid.status_code}")

            # =================================================
            section("GET /tasks/{task_id}/lineage（执行链）")

            lineage = await client.get(f"/api/v1/tasks/{succeeded}/lineage")

            check("200", lineage.status_code == 200)

            chain = lineage.json()

            check("root_task_id 是字符串",
                  isinstance(chain.get("root_task_id"), str))
            check("tasks 是列表", isinstance(chain.get("tasks"), list))
            check("链里有自己那一条",
                  any(
                      t["task_id"] == str(succeeded)
                      for t in chain["tasks"]
                  ))

            # 传链上任意一条都该能推出整条链 —— 传子任务也应当可以。
            # 这里只有一条，所以等价于传自己，但仍然验到了"反推根"的分支
            # 不会 500。
            check("root_task_id 等于任务自己（它是这条链的起点）",
                  chain.get("root_task_id") == str(succeeded))

            lineage_missing = await client.get(
                f"/api/v1/tasks/{uuid4()}/lineage"
            )
            check("不存在的任务 -> 404",
                  lineage_missing.status_code == 404,
                  f"实际 {lineage_missing.status_code}")

            # =================================================
            section("POST /tasks/{task_id}/cancel")

            cancel = await client.post(
                f"/api/v1/tasks/{pending_id}/cancel"
            )

            check("200", cancel.status_code == 200,
                  f"实际 {cancel.status_code} {cancel.text[:200]}")
            check("回真实状态 cancelled（不是无脑的成功）",
                  cancel.json().get("status") == "cancelled",
                  f"实际 {cancel.json().get('status')}")

            # 已结束的任务：取消来晚了不是错误
            late = await client.post(f"/api/v1/tasks/{succeeded}/cancel")
            check("取消一条已成功的是 200 且回它的终态（幂等，不报 409）",
                  late.status_code == 200
                  and late.json().get("status") == "success",
                  f"实际 {late.status_code} {late.json().get('status')}")

            cancel_missing = await client.post(
                f"/api/v1/tasks/{uuid4()}/cancel"
            )
            check("不存在 -> 404", cancel_missing.status_code == 404)

            # =================================================
            section("POST /tasks/{task_id}/retry")

            retry = await client.post(f"/api/v1/tasks/{failed}/retry")

            check("200", retry.status_code == 200,
                  f"实际 {retry.status_code} {retry.text[:200]}")

            revived = retry.json()

            check("回到 pending",
                  revived.get("status") == "pending",
                  f"实际 {revived.get('status')}")
            check("attempt 清零（否则这个接口在它最该用的场景里没用）",
                  revived.get("attempt") == 0,
                  f"实际 {revived.get('attempt')}")
            check("保留上一次的 error（重跑前正要看它）",
                  isinstance(revived.get("error"), str)
                  and "ValueError" in revived["error"],
                  f"实际 {revived.get('error')!r}")

            # 重跑之后它会被真的再跑一次（max_attempts=1 -> 再失败即终态）
            refailed = await wait_for(failed, TaskStatus.FAILED)
            check("重跑之后确实被执行了（又失败，因为那条逻辑就是坏的）",
                  refailed is not None,
                  f"实际 {refailed.status if refailed else None}")

            # 已成功的不许重跑：会产生第二次外部副作用
            retry_success = await client.post(
                f"/api/v1/tasks/{succeeded}/retry"
            )
            check("重跑一条已成功的是 409（会产生第二次副作用）",
                  retry_success.status_code == 409,
                  f"实际 {retry_success.status_code}")

            retry_missing = await client.post(
                f"/api/v1/tasks/{uuid4()}/retry"
            )
            check("不存在 -> 404", retry_missing.status_code == 404)

            # =================================================
            section("POST /tasks/{task_id}/reschedule（防抖）")

            # 一条还没到点的延迟任务：改它的时间不该让它跑起来
            delayed_id = uuid4()

            async with repositories.scope() as repos:

                await repos.create(TaskStore).add(
                    Task(
                        handler=f"{PROBE}.ok",
                        params=ApiParams(label="delayed"),
                        task_id=delayed_id,
                        execute_at=(
                            datetime.now(timezone.utc)
                            + timedelta(hours=1)
                        ),
                    ).seal_root()
                )

            created.append(delayed_id)

            resched = await client.post(
                f"/api/v1/tasks/{delayed_id}/reschedule",
                json={
                    "execute_at": (
                        datetime.now(timezone.utc)
                        + timedelta(hours=2)
                    ).isoformat()
                },
            )

            check("200", resched.status_code == 200,
                  f"实际 {resched.status_code} {resched.text[:200]}")

            moved = resched.json()

            check("generation 从 1 变成 2",
                  moved.get("generation") == 2,
                  f"实际 {moved.get('generation')}")
            check("execute_at 是 ISO 字符串（时区值被序列化）",
                  isinstance(moved.get("execute_at"), str))
            check("状态仍是 pending（改时间不会让它提前跑）",
                  moved.get("status") == "pending",
                  f"实际 {moved.get('status')}")

            # 反复重排不会堆出多条任务 —— 防抖的核心
            for _ in range(2):
                await client.post(
                    f"/api/v1/tasks/{delayed_id}/reschedule",
                    json={
                        "execute_at": (
                            datetime.now(timezone.utc)
                            + timedelta(hours=3)
                        ).isoformat()
                    },
                )

            listed = await client.get(
                "/api/v1/tasks",
                params={"handler": f"{PROBE}.ok", "limit": 200},
            )

            same = [
                it for it in listed.json()["items"]
                if it["task_id"] == str(delayed_id)
            ]

            check("重排三次之后仍然只有这一条任务（没堆出新任务）",
                  len(same) == 1, f"实际 {len(same)} 条")
            check("列表里的 generation 也是 4（1 + 3 次重排）",
                  bool(same) and same[0]["generation"] == 4,
                  f"实际 {same[0]['generation'] if same else None}")

            # 已终结的不许重排
            resched_done = await client.post(
                f"/api/v1/tasks/{succeeded}/reschedule",
                json={"execute_at": None},
            )
            check("重排一条已成功的是 409",
                  resched_done.status_code == 409,
                  f"实际 {resched_done.status_code}")

            resched_missing = await client.post(
                f"/api/v1/tasks/{uuid4()}/reschedule",
                json={"execute_at": None},
            )
            check("不存在 -> 404", resched_missing.status_code == 404)

    finally:

        await relay.aclose()
        await dispatcher.aclose()

        if created:
            async with session_factory() as session:
                await session.execute(
                    delete(TaskOutboxModel).where(
                        TaskOutboxModel.task_id.in_(created)
                    )
                )
                await session.execute(
                    update(TaskModel)
                    .where(TaskModel.task_id.in_(created))
                    .values(parent_task_id=None)
                )
                await session.execute(
                    delete(TaskModel).where(
                        TaskModel.task_id.in_(created)
                    )
                )
                await session.commit()

            print(f"\n  已清理 {len(created)} 条探针任务")

        reset_factories()
        await engine.dispose()

    if FAILS:
        print(f"\n失败 {len(FAILS)} 项：")
        for f in FAILS:
            print(f"  - {f}")
        return 1

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
