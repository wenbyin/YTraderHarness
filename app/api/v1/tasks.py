"""
异步任务的查询与控制接口。

对应 task.yaml 的 tag：Tasks。规格与实现都在 app/services/task/ 下，
本文件只做 HTTP 层的转发 —— 与 tools.py / capabilities.py 等其余路由
同一套分层。

⚠️ 这里**没有**提交任务的接口，这是刻意的
---------------------------------------
任务由业务代码在服务端通过 TaskService.submit() 创建，不对外暴露
POST /tasks。理由见 task.yaml 的 info.description：一个公开的提交接口
等于把"任意已注册 Handler 都能被任意已认证调用方触发"打开成一条通路，
而那条通路绕过了业务接口自己的授权与限流。

所以本文件只做三件事：看（列表 / 详情）、追（执行链）、干预（取消 / 重试）。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.task.application._assembly import (
    build_task_detail,
    build_task_summary,
)
from app.services.task.application.task_service import TaskService
from app.services.task.domain.schemas import (
    TaskCancelResponse,
    TaskDetail,
    TaskLineage,
    TaskPage,
    TaskReschedule,
    TaskSummary,
)
from app.common.task.types import TaskStatus

router = APIRouter()


# ============================================================
# 看
# ============================================================


@router.get(
    "/tasks",
    response_model=TaskPage,
    tags=["Tasks"],
)
async def list_tasks(
        status: TaskStatus | None = Query(
            default=None,
            description="按状态过滤。不传表示不过滤。",
        ),
        handler: str | None = Query(
            default=None,
            description="按 handler 精确过滤，如 run.execute。",
        ),
        root_task_id: UUID | None = Query(
            default=None,
            description="按执行链的根任务过滤，用于取整条链上的任务。",
        ),
        parent_task_id: UUID | None = Query(
            default=None,
            description="按直接父任务过滤，用于取某个任务的子任务。",
        ),
        limit: int = Query(
            default=50,
            ge=1,
            le=200,
            description="本页条数上限。",
        ),
        offset: int = Query(
            default=0,
            ge=0,
            description="偏移量。",
        ),
        service: TaskService = Service[TaskService],
):
    """
    列出任务，最近创建的在前。

    结果**不含 params 与 error 正文**（见 TaskSummary）：一页可能几百条，
    渲染列表用不到它们。要那些字段请取单条。
    """

    try:

        tasks, total = await service.list_tasks(
            status=status,
            handler=handler,
            root_task_id=root_task_id,
            parent_task_id=parent_task_id,
            limit=limit,
            offset=offset,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc

    return TaskPage(
        items=[build_task_summary(t) for t in tasks],
        total=total,
    )


@router.get(
    "/tasks/{task_id}",
    response_model=TaskDetail,
    tags=["Tasks"],
)
async def get_task(
        task_id: UUID,
        service: TaskService = Service[TaskService],
):
    """
    取单条任务的完整视图：参数、失败原因、租约。

    排查"它为什么还没跑"看三处：
        status            pending 且 execute_at 未到 -> 延迟任务，正常
        error            上一次失败的原因
        lease_expires_at running 却已过期 -> 执行者崩了，等着被收回
    """

    try:
        task = await service.get(task_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc

    return build_task_detail(task)


# ============================================================
# 追
# ============================================================


@router.get(
    "/tasks/{task_id}/lineage",
    response_model=TaskLineage,
    tags=["Tasks"],
)
async def get_task_lineage(
        task_id: UUID,
        service: TaskService = Service[TaskService],
):
    """
    取这条任务所属的**整条执行链**，按产生顺序（老的在前）。

    传链上任意一条都可以，不必自己找根：根由这条任务反推出来
    （它自己的 root_task_id 为空时，它就是根）。这样前端在详情页拿到
    一条任务时不必先判断"我是不是根"。
    """

    try:

        task = await service.get(task_id)

        root_task_id = (
            task.root_task_id
            if task.root_task_id is not None
            else task.task_id
        )

        chain = await service.get_lineage(root_task_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc

    return TaskLineage(
        root_task_id=root_task_id,
        tasks=[build_task_summary(t) for t in chain],
    )


# ============================================================
# 干预
# ============================================================


@router.post(
    "/tasks/{task_id}/cancel",
    response_model=TaskCancelResponse,
    tags=["Tasks"],
)
async def cancel_task(
        task_id: UUID,
        service: TaskService = Service[TaskService],
):
    """
    取消任务。**返回的是取消之后它真实所处的位置**，不是一个无脑的成功。

    取消是协作式的：已经跑起来的 Handler 不会被强杀，执行者下次写回
    结果时会被拒绝，于是它最终停在 cancelled。所以状态是 running 表示
    "正在收尾"，不是"没取消成功"。

    这里**不报错**：已经结束的任务（取消来晚了）如实回它的终态。

    Raises:
        NotFoundError: 任务不存在 -> 404。
    """

    try:
        status = await service.cancel(task_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc

    return TaskCancelResponse(
        task_id=task_id,
        status=status,
    )


@router.post(
    "/tasks/{task_id}/reschedule",
    response_model=TaskDetail,
    tags=["Tasks"],
)
async def reschedule_task(
        task_id: UUID,
        request: TaskReschedule,
        service: TaskService = Service[TaskService],
):
    """
    改一条**还没开始跑**的任务的执行时间（防抖）。

    典型用法是审批超时：每来一次新动静就把截止时间往后推一次。
    反复调它不会堆出多条任务 —— 改的是同一条，并且每次都会让上一个
    已生成的定时失效（generation +1），最后那一次的时间赢。

    与"取消旧的、再提交一条新的"的差别就在这里：后者在"取消还没生效、
    新定时已经到点"那个窗口里会让旧定时跑掉，而这个接口没有那个窗口。

    两种情况会 409：
        正在跑          它马上就写回结果了，改时间没有意义
        已终结         改了会让「已完成」变成「还会再跑」
    """

    try:

        await service.reschedule(task_id, execute_at=request.execute_at)

        # 回读：重排之后 generation 与 execute_at 都变了，
        # 只有从库里读才是真的。
        task = await service.get(task_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc

    return build_task_detail(task)


@router.post(
    "/tasks/{task_id}/retry",
    response_model=TaskDetail,
    tags=["Tasks"],
)
async def retry_task(
        task_id: UUID,
        service: TaskService = Service[TaskService],
):
    """
    人工重跑一条**已失败或已取消**的任务。

    这是"再跑一轮"：attempt 清零、立即执行，上一条 error 保留
    （重跑之前正要看它）。

    两种情况会 409：
        还在排队 / 正在跑   重跑会与执行中的那次冲突
        已经成功            重跑会产生第二次外部副作用
                            （真通知了两次、真扣了两次款）——
                            那该提交一条新任务，由调用方显式承担语义。
    """

    try:

        await service.retry(task_id)

        # 回读一次而不是把 submit 时的样子凑出来：重跑之后任务的状态、
        # attempt、execute_at 都变了，只有从库里读才是真的。
        task = await service.get(task_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc

    return build_task_detail(task)
