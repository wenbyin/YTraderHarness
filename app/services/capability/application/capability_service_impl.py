"""
CapabilityService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import ServiceContext

from ..domain.entities import Capability as CapabilityEntity
from ..domain.schemas import (
    Capability,
    CapabilityCreate,
    CapabilityReplace,
    CapabilityUpdate,
)
from ..repository.capability import CapabilityRepository
from ._assembly import build_capability
from .capability_service import CapabilityService
from ...tool.registry import tool

#: 沿 parent_id 往上走的步数上限。
#:
#: 库里不该有环（写入时挡了），但万一有（历史数据、人工改库），
#: 没有这个上限就是一个死循环 —— 一个查询接口把进程挂死，
#: 比返回一个错误严重得多。
_MAX_TREE_DEPTH = 64


class CapabilityServiceImpl(CapabilityService):

    def __init__(
            self,
            *,
            capability_repository: CapabilityRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._capabilities = capability_repository

    # ========================================================
    # 能力树
    # ========================================================
    @tool
    async def list_capabilities(
            self,
            *,
            parent_id: UUID | None = None,
            keyword: str | None = None,
            is_active: bool | None = None,
    ) -> list[Capability]:

        entities = await self._capabilities.list(
            parent_id=parent_id,
            keyword=keyword,
            is_active=is_active,
        )

        return [build_capability(e) for e in entities]

    async def create_capability(
            self,
            request: CapabilityCreate,
    ) -> Capability:

        existing = await self._capabilities.get_by_code(
            request.code
        )

        if existing is not None:
            raise ConflictError(
                f"能力标识已存在：{request.code}"
            )

        await self._require_parent(request.parent_id)

        now = datetime.now(timezone.utc)

        entity = await self._capabilities.add(
            CapabilityEntity(
                capability_id=uuid4(),
                code=request.code,
                name=request.name,
                description=request.description,
                parent_id=request.parent_id,
                sort_order=request.sort_order,
                is_active=True,
                created_at=now,
                updated_at=now,
            )
        )

        return build_capability(entity)

    async def get_capability(
            self,
            capability_id: UUID,
    ) -> Capability:

        return build_capability(
            await self._require_capability(capability_id)
        )

    async def update_capability(
            self,
            capability_id: UUID,
            request: CapabilityUpdate,
    ) -> Capability:

        current = await self._require_capability(capability_id)

        # 局部更新：没传的字段保持原值。
        # 用 model_fields_set 而不是"判 None" —— 否则没法把
        # description 显式改成 null（清空说明）。
        provided = request.model_fields_set

        parent_id = (
            request.parent_id
            if "parent_id" in provided
            else current.parent_id
        )

        if "parent_id" in provided:
            await self._require_parent(parent_id)
            await self._assert_no_cycle(
                capability_id, parent_id
            )

        entity = CapabilityEntity(
            capability_id=current.capability_id,
            code=current.code,
            name=(
                request.name if "name" in provided
                else current.name
            ),
            description=(
                request.description if "description" in provided
                else current.description
            ),
            parent_id=parent_id,
            sort_order=(
                request.sort_order if "sort_order" in provided
                else current.sort_order
            ),
            is_active=(
                request.is_active if "is_active" in provided
                else current.is_active
            ),
            created_at=current.created_at,
            updated_at=datetime.now(timezone.utc),
        )

        updated = await self._capabilities.update(entity)

        return build_capability(updated)

    async def replace_capability(
            self,
            capability_id: UUID,
            request: CapabilityReplace,
    ) -> Capability:

        current = await self._require_capability(capability_id)

        # 替换语义：请求里没写的可空字段就是 None。
        # 不像 PATCH 那样看 model_fields_set —— 那是局部更新才需要的
        # 区分，这里"没传"与"传了 null"是同一个意思。
        await self._require_parent(request.parent_id)
        await self._assert_no_cycle(
            capability_id, request.parent_id
        )

        entity = CapabilityEntity(
            capability_id=current.capability_id,
            code=current.code,
            name=request.name,
            description=request.description,
            parent_id=request.parent_id,
            sort_order=request.sort_order,
            is_active=request.is_active,
            created_at=current.created_at,
            updated_at=datetime.now(timezone.utc),
        )

        updated = await self._capabilities.update(entity)

        return build_capability(updated)

    async def delete_capability(
            self,
            capability_id: UUID,
            *,
            force: bool = False,
    ) -> None:

        current = await self._require_capability(capability_id)

        # 子能力先查：这是唯一 force 也解不开的一关。撞到外键再报错的
        # 话，错误信息离原因太远，说不清"到底该先做什么"。
        children = await self._capabilities.count_children(
            capability_id
        )

        if children:
            raise ConflictError(
                f"能力 {current.code} 下还有 {children} 个子能力，"
                "先把它们删掉或挪到别的父能力下，再删本节点。"
                "只想让它不再被使用的话，改用 is_active=false 停用。"
            )

        # 工具引用这一关 force 可以放行，但必须由调用方显式确认 ——
        # 关联行随 CASCADE 无声消失，不提示就等于悄悄把工具变成未关联。
        if not force:
            links = await self._capabilities.count_links(
                [capability_id]
            )

            if links:
                raise ConflictError(
                    f"能力 {current.code} 正被 {links} 个工具引用；"
                    "删除会一并解除这些关联，"
                    "确认请带 force=true 重试。"
                )

        await self._capabilities.delete(capability_id)

    # ========================================================
    # internals
    # ========================================================

    async def _require_capability(
            self,
            capability_id: UUID,
    ) -> CapabilityEntity:

        entity = await self._capabilities.get(capability_id)

        if entity is None:
            raise NotFoundError(
                f"能力不存在：{capability_id}"
            )

        return entity

    async def _require_parent(
            self,
            parent_id: UUID | None,
    ) -> None:
        """父节点必须存在。不存在时按冲突处理（比 404 更贴切：
        是这次请求引用的对象不对，不是请求的 URL 不存在）。"""

        if parent_id is None:
            return

        if await self._capabilities.get(parent_id) is None:
            raise ConflictError(f"父能力不存在：{parent_id}")

    async def _assert_no_cycle(
            self,
            capability_id: UUID,
            parent_id: UUID | None,
    ) -> None:
        """
        把 node 挂到 parent 下面会不会成环。

        只需检查 parent 是不是 node 自己、或 node 的后代 ——
        沿 parent 往上走，撞见 node 就是成环。
        """

        if parent_id is None:
            return

        if parent_id == capability_id:
            raise ConflictError("能力不能把自己作为父能力。")

        cursor = parent_id

        for _ in range(_MAX_TREE_DEPTH):

            if cursor is None:
                return

            if cursor == capability_id:
                raise ConflictError(
                    "不能把能力挂到它自己的后代下面（会成环）。"
                )

            ancestor = await self._capabilities.get(cursor)

            if ancestor is None:
                return

            cursor = ancestor.parent_id

        raise ConflictError(
            "能力树层级过深或存在环，已中止。"
            "请检查能力数据。"
        )
