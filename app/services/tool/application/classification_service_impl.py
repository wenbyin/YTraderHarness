"""
ClassificationService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。
"""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID, uuid4

from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import ServiceContext

from ..domain.entities import ToolClassification
from ..domain.schemas import (
    Classification,
    ClassificationCreate,
    ClassificationUpdate,
    ToolClassificationUpdate,
)
from ..repository.tool_classification import (
    ToolClassificationRepository,
)
from ..repository.tool_definition import ToolDefinitionRepository
from ._assembly import build_classification
from .classification_service import ClassificationService


#: 沿 parent_id 往上走的步数上限。
#:
#: 库里不该有环（写入时挡了），但万一有（历史数据、人工改库），
#: 没有这个上限就是一个死循环 —— 一个查询接口把进程挂死，
#: 比返回一个错误严重得多。
_MAX_TREE_DEPTH = 64


class ClassificationServiceImpl(ClassificationService):

    def __init__(
            self,
            *,
            tool_classification_repository: (
                ToolClassificationRepository
            ),
            tool_definition_repository: ToolDefinitionRepository,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._classifications = tool_classification_repository
        self._definitions = tool_definition_repository

    # ========================================================
    # 分类树
    # ========================================================

    async def list_classifications(
            self,
            *,
            parent_id: UUID | None = None,
            keyword: str | None = None,
            is_active: bool | None = None,
    ) -> list[Classification]:

        entities = await self._classifications.list(
            parent_id=parent_id,
            keyword=keyword,
            is_active=is_active,
        )

        return [build_classification(e) for e in entities]

    async def create_classification(
            self,
            request: ClassificationCreate,
    ) -> Classification:

        existing = await self._classifications.get_by_code(
            request.code
        )

        if existing is not None:
            raise ConflictError(
                f"分类标识已存在：{request.code}"
            )

        await self._require_parent(request.parent_id)

        now = datetime.now(timezone.utc)

        entity = await self._classifications.add(
            ToolClassification(
                classification_id=uuid4(),
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

        return build_classification(entity)

    async def get_classification(
            self,
            classification_id: UUID,
    ) -> Classification:

        return build_classification(
            await self._require_classification(classification_id)
        )

    async def update_classification(
            self,
            classification_id: UUID,
            request: ClassificationUpdate,
    ) -> Classification:

        current = await self._require_classification(
            classification_id
        )

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
                classification_id, parent_id
            )

        entity = ToolClassification(
            classification_id=current.classification_id,
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

        updated = await self._classifications.update(entity)

        return build_classification(updated)

    # ========================================================
    # 工具 <-> 分类
    # ========================================================

    async def get_tool_classifications(
            self,
            tool_id: UUID,
    ) -> list[Classification]:

        await self._require_tool(tool_id)

        entities = await self._classifications.list_for_tool(
            tool_id
        )

        return [build_classification(e) for e in entities]

    async def replace_tool_classifications(
            self,
            tool_id: UUID,
            request: ToolClassificationUpdate,
    ) -> list[Classification]:

        await self._require_tool(tool_id)

        # 一次查回来比逐个 get 好，而且能一次报出**全部**无效 id，
        # 而不是让调用方改一个再撞一个。
        wanted = list(dict.fromkeys(request.classification_ids))

        if wanted:
            found = await self._classifications.list()

            known = {c.classification_id for c in found}

            unknown = [i for i in wanted if i not in known]

            if unknown:
                raise NotFoundError(
                    "分类不存在："
                    + ", ".join(str(i) for i in unknown)
                )

        await self._classifications.replace_tool_classifications(
            tool_id, wanted
        )

        entities = await self._classifications.list_for_tool(
            tool_id
        )

        return [build_classification(e) for e in entities]

    # ========================================================
    # internals
    # ========================================================

    async def _require_classification(
            self,
            classification_id: UUID,
    ) -> ToolClassification:

        entity = await self._classifications.get(
            classification_id
        )

        if entity is None:
            raise NotFoundError(
                f"分类不存在：{classification_id}"
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

        if await self._classifications.get(parent_id) is None:
            raise ConflictError(f"父分类不存在：{parent_id}")

    async def _require_tool(self, tool_id: UUID) -> None:
        if await self._definitions.get(tool_id) is None:
            raise NotFoundError(f"Tool 定义不存在：{tool_id}")

    async def _assert_no_cycle(
            self,
            classification_id: UUID,
            parent_id: UUID | None,
    ) -> None:
        """
        把 node 挂到 parent 下面会不会成环。

        只需检查 parent 是不是 node 自己、或 node 的后代 ——
        沿 parent 往上走，撞见 node 就是成环。
        """

        if parent_id is None:
            return

        if parent_id == classification_id:
            raise ConflictError("分类不能把自己作为父分类。")

        cursor = parent_id

        for _ in range(_MAX_TREE_DEPTH):

            if cursor is None:
                return

            if cursor == classification_id:
                raise ConflictError(
                    "不能把分类挂到它自己的后代下面（会成环）。"
                )

            ancestor = await self._classifications.get(cursor)

            if ancestor is None:
                return

            cursor = ancestor.parent_id

        raise ConflictError(
            "分类树层级过深或存在环，已中止。"
            "请检查分类数据。"
        )
