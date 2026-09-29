"""
工具处理器链的挂载接口。

对应 tool_processor.yaml 的 Tool Processors 与 Tool Processor Chain
两组：
    GET    /tools/{tool_id}/processors
    POST   /tools/{tool_id}/processors
    PATCH  /tools/{tool_id}/processors/{processor_id}
    DELETE /tools/{tool_id}/processors/{processor_id}
    PUT    /tools/{tool_id}/processors
    PUT    /tools/{tool_id}/processors/order

⚠️ 路由顺序：/tools/{tool_id}/processors/order 必须注册在
/tools/{tool_id}/processors/{processor_id} **之前**。
今天两者方法不同（PUT vs PATCH/DELETE），看似不冲突，但保留这个
顺序是因为同类问题的成本极不对称 —— 万一以后有人给宽路径加上
PUT，或者加一条 GET /tools/{tool_id}/processors/order，
"order" 就会被当成一个非法 UUID 拿去解析，报错离原因很远。
document 模块的 /documents/retrieve vs /documents/{id} 踩过同一个坑。

用户写的处理器链**不会被同步覆盖**
----------------------------------
与 tool_definition 不同：那张表归代码，同步会重写它；这张表归人，
没有任何同步任务会碰它。处理器实现改名只影响"读链时还能不能
把 processor 字段填出来"，不影响链上已有的行。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.tool.application.tool_processor_service import (
    ToolProcessorService,
)
from app.services.tool.domain.schemas import (
    CreateToolProcessorRequest,
    ReorderToolProcessorsRequest,
    ReplaceToolProcessorChainRequest,
    ToolProcessor,
    ToolProcessorChain,
    UpdateToolProcessorRequest,
)

router = APIRouter()


# ============================================================
# Chain
# ============================================================
#
# /order 必须排在最前面，理由见模块 docstring。


@router.put(
    "/tools/{tool_id}/processors/order",
    response_model=ToolProcessorChain,
    tags=["Tool Processor Chain"],
)
async def reorder_tool_processors(
        tool_id: UUID,
        request: ReorderToolProcessorsRequest,
        service: ToolProcessorService = Service[
            ToolProcessorService
        ],
):
    """
    按给定顺序重排整条链。

    必须提交该 Tool **全部**挂载记录的 ID —— 少给、多给、或混进
    别的 Tool 的 ID 都会 409。部分重排没有明确的语义（剩下那些
    排哪？），接受它只会让那件事变成实现细节。
    """

    try:
        return await service.reorder(tool_id, request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/tools/{tool_id}/processors",
    response_model=ToolProcessorChain,
    tags=["Tool Processor Chain"],
)
async def replace_tool_processor_chain(
        tool_id: UUID,
        request: ReplaceToolProcessorChainRequest,
        service: ToolProcessorService = Service[
            ToolProcessorService
        ],
):
    """
    整体替换一条链。数组顺序即执行顺序，传空数组即全部摘掉。

    **替换语义**：没列出的会被删除。行是全删重建的 ——
    processor_id 会全部重新生成，前端提交后需要重新 GET。
    """

    try:
        return await service.replace_chain(tool_id, request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 单条
# ============================================================


@router.get(
    "/tools/{tool_id}/processors",
    response_model=ToolProcessorChain,
    tags=["Tool Processors"],
)
async def list_tool_processors(
        tool_id: UUID,
        enabled_only: bool = Query(
            default=False,
            description=(
                "只返回启用的处理器。管理面一般用 false —— "
                "关掉的行也要看得见，否则会以为它被删了。"
            ),
        ),
        service: ToolProcessorService = Service[
            ToolProcessorService
        ],
):
    """
    取一个 Tool 的处理器链（按 sequence 升序）。

    链上引用了**代码里已经不存在的实现**时，那一行照样返回，
    只是没有 processor 字段 —— 前端据此提示"请重挂或删除"，
    而不是让整条链读不出来。
    """

    try:
        return await service.list_chain(
            tool_id, enabled_only=enabled_only
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/tools/{tool_id}/processors",
    response_model=ToolProcessor,
    status_code=201,
    tags=["Tool Processors"],
)
async def add_tool_processor(
        tool_id: UUID,
        request: CreateToolProcessorRequest,
        service: ToolProcessorService = Service[
            ToolProcessorService
        ],
):
    """
    往链上挂一条记录，返回它。

    request.sequence 只是**位置提示**（插到所有 sequence 小于它的
    行之后），落库前整条链会重编号成 1..N —— 因此传 99 而链只有
    3 条不报错，结果就是插到末尾。

    ⚠️ AROUND 处理器在链上是**两条记录**（先进 + 后出，各带自己
    那份 config）。这个接口一次只建一条，所以挂 AROUND 需要调用
    两次 —— 中间那个"只有一半"的状态是允许的，读链时按一套
    AROUND 校验（先进后出）也只对**已成对**的区间生效。
    """

    try:
        return await service.add_processor(tool_id, request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.patch(
    "/tools/{tool_id}/processors/{processor_id}",
    response_model=ToolProcessor,
    tags=["Tool Processors"],
)
async def update_tool_processor(
        tool_id: UUID,
        processor_id: UUID,
        request: UpdateToolProcessorRequest,
        service: ToolProcessorService = Service[
            ToolProcessorService
        ],
):
    """
    改一条挂载记录：execution_mode / hook / config / enabled。

    **改不了 processor_key 与位置** —— 前者是"换一个实现"，
    后者是"换位置"，两件事各有专门的操作（删了重挂 / PUT order）。
    混进 PATCH 会让"我只想改个参数"变成一次隐式的链重排。
    """

    try:
        return await service.update_processor(
            tool_id, processor_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.delete(
    "/tools/{tool_id}/processors/{processor_id}",
    status_code=204,
    tags=["Tool Processors"],
)
async def delete_tool_processor(
        tool_id: UUID,
        processor_id: UUID,
        service: ToolProcessorService = Service[
            ToolProcessorService
        ],
):
    """
    摘掉一条记录。剩下的会重新编号成 1..N（补洞）。
    """

    try:
        await service.delete_processor(tool_id, processor_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc
