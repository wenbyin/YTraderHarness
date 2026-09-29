"""
ToolProcessorService 的默认实现。

事务边界由调用方（ServiceFactory.scope()）决定，本服务内部不
commit / rollback。一次链替换必须整体成功或整体回滚 ——
"链只写了一半"比报错难查得多。

写之前的校验顺序是**刻意的**
------------------------------
    _validate_mount   先查实现与参数（404 / 409 / 400）
    _validate_onion   再查 AROUND 的嵌套
    最后才落库

洋葱校验在**落库之前**做，靠的是"把最终顺序先在内存里拼出来"：
写完再回滚也能得到同样的结果，但那要多一次数据库往返，
而且失败时依赖事务边界正确 —— 能静态发现的问题不要留给运行期。
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.agent.tools.processor.base import BaseToolProcessor
from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import ServiceContext

from ..domain.entities import ToolProcessor as ToolProcessorEntity
from ..domain.schemas import (
    CreateToolProcessorRequest,
    ExecutionMode,
    ProcessorHook,
    ReorderToolProcessorsRequest,
    ReplaceToolProcessorChainRequest,
    ToolProcessor,
    ToolProcessorChain,
    UpdateToolProcessorRequest,
)
from ..registry.processor_registry import (
    ToolProcessorRegistry,
    get_processor_registry,
)
from ..repository.tool_definition import ToolDefinitionRepository
from ..repository.tool_processor import ToolProcessorRepository
from ._assembly import build_chain, build_tool_processor
from .tool_processor_service import ToolProcessorService

__all__ = ["ToolProcessorServiceImpl"]

logger = logging.getLogger(__name__)


def validate_onion(
        chain: list[tuple[str, ProcessorHook]],
) -> None:
    """
    AROUND 之间必须**先进后出**（洋葱模型）。

    chain 按最终顺序给出 (processor_key, hook)。

    为什么只有 AROUND 受约束
    ------------------------
    AROUND 包住整次调用，两个 AROUND 就构成嵌套：先进的那个必须
    后出，否则"包住"这件事讲不通。BEFORE / AFTER 各自只跑一次、
    不嵌套，排在哪都不影响别人，因此完全不参与这条校验。

    为什么用栈
    ----------
    一条 AROUND 处理器在链上是**两条记录**（先进 + 后出），构成一个
    区间；校验就是"这些区间必须正确嵌套，不得交叉"：

        AAAAA BBBBB BBBBB AAAAA     ✅ 嵌套
        AAAAA BBBBB AAAAA BBBBB     ❌ 交叉

    按左端点升序扫一遍，栈里留着"还没结束的区间"，且它们构成一条
    嵌套链（每个都包在下一个里面）。新区间必须落在栈顶之内，
    越过栈顶右端就是交叉。

    ⚠️ 只比较相邻两个区间是**不够**的：A 包住 B、C 从 A 内部开始
    却伸出 A 外面时，A-B 与 B-C 两对都不交叉，但 A-C 交叉。

    只有一条记录的 key **放行**
    ---------------------------
    POST 一次只能建一条，因此"还缺后出那一半"是挂载过程的合法
    中间态。按"恰好两条"强校验会让 POST 被自己拒掉。
    """

    spans: dict[str, list[int]] = {}

    for index, (key, hook) in enumerate(chain, start=1):

        if hook is not ProcessorHook.AROUND:
            continue

        if key in spans:
            spans[key][1] = index
        else:
            spans[key] = [index, index]

    intervals = sorted(
        (low, high, key)
        for key, (low, high) in spans.items()
        if high > low
    )

    stack: list[tuple[int, int, str]] = []

    for low, high, key in intervals:

        # 已经结束的区间出栈：它们的右端在当前区间左端之前。
        while stack and stack[-1][1] < low:
            stack.pop()

        if stack and high > stack[-1][1]:

            outer = stack[-1][2]

            raise ConflictError(
                f"AROUND 处理器 {key!r} 与 {outer!r} 的位置交叉了，"
                "违反洋葱模型（先进后出）。"
                "AROUND 在链上是两条记录（先进 + 后出），"
                "两条记录构成的区间必须完整地包在别的 AROUND 之内，"
                "或在它之外 —— 不能一个伸进另一个里面又冒出头来。"
            )

        stack.append((low, high, key))


class ToolProcessorServiceImpl(ToolProcessorService):

    def __init__(
            self,
            *,
            tool_definition_repository: ToolDefinitionRepository,
            tool_processor_repository: ToolProcessorRepository,
            registry: ToolProcessorRegistry | None = None,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._definitions = tool_definition_repository
        self._processors = tool_processor_repository

        # 与 ProcessorDefinitionServiceImpl 同一个注入口：
        # 用例要能把目录收窄到假实现上。生产走默认值。
        self._registry = (
            registry if registry is not None
            else get_processor_registry()
        )

    # ========================================================
    # 读
    # ========================================================

    async def list_chain(
            self,
            tool_id: UUID,
            *,
            enabled_only: bool = False,
    ) -> ToolProcessorChain:

        await self._require_tool(tool_id)

        rows = await self._processors.list_for_tool(
            tool_id, enabled_only=enabled_only
        )

        return build_chain(tool_id, rows, self._impls(rows))

    # ========================================================
    # 写
    # ========================================================

    async def add_processor(
            self,
            tool_id: UUID,
            request: CreateToolProcessorRequest,
    ) -> ToolProcessor:

        await self._require_tool(tool_id)

        now = datetime.now(timezone.utc)

        config = self._validate_mount(
            request.processor_key,
            request.hook,
            request.execution_mode,
            request.config,
        )

        current = await self._processors.list_for_tool(tool_id)

        index = self._insertion_index(current, request.sequence)

        # 最终顺序先在内存里拼出来，洋葱校验才有东西可查 ——
        # 见模块 docstring。
        planned = [
            (row.processor_key, row.hook) for row in current
        ]
        planned.insert(
            index, (request.processor_key, request.hook)
        )

        validate_onion(planned)

        row = await self._processors.insert(
            ToolProcessorEntity(
                processor_id=uuid4(),
                tool_id=tool_id,
                processor_key=request.processor_key,
                sequence=index + 1,
                execution_mode=request.execution_mode,
                hook=request.hook,
                config=config,
                enabled=request.enabled,
                created_at=now,
                updated_at=now,
            ),
            position=index + 1,
        )

        return build_tool_processor(
            row, self._registry.get(row.processor_key)
        )

    async def update_processor(
            self,
            tool_id: UUID,
            processor_id: UUID,
            request: UpdateToolProcessorRequest,
    ) -> ToolProcessor:

        await self._require_tool(tool_id)

        current = await self._processors.get(
            tool_id, processor_id
        )

        if current is None:
            raise NotFoundError(
                f"挂载记录不存在：{processor_id}"
            )

        # 用 model_fields_set 判"传没传"，不用 `is not None`：
        # `config={}` 与 `enabled=false` 都是合法修改，
        # 按 None 判会把它们当成"什么都没改"。
        #
        # 对 hook / execution_mode 而言，显式传 null 与不传等价
        # （"留空表示不改"），因此这两处仍按 None 判。
        given = request.model_fields_set

        hook = (
            request.hook
            if request.hook is not None
            else current.hook
        )

        execution_mode = (
            request.execution_mode
            if request.execution_mode is not None
            else current.execution_mode
        )

        config = (
            request.config
            if "config" in given and request.config is not None
            else current.config
        )

        enabled = (
            request.enabled
            if "enabled" in given and request.enabled is not None
            else current.enabled
        )

        # 改完必须**重跑**相容性校验：不改的话，一条合法的
        # AROUND+SYNC 改成 ASYNC 就绕过了挂载校验，然后在执行期
        # 撞上 DB 的 CHECK —— 而 CHECK 抛的是 IntegrityError，
        # 兜不成 409。
        config = self._validate_mount(
            current.processor_key,
            hook,
            execution_mode,
            config,
        )

        rows = await self._processors.list_for_tool(tool_id)

        validate_onion([
            (
                row.processor_key,
                hook if row.processor_id == processor_id
                else row.hook,
            )
            for row in rows
        ])

        updated = await self._processors.update(
            current.model_copy(
                update={
                    "hook": hook,
                    "execution_mode": execution_mode,
                    "config": config,
                    "enabled": enabled,
                    "updated_at": datetime.now(timezone.utc),
                }
            )
        )

        return build_tool_processor(
            updated, self._registry.get(updated.processor_key)
        )

    async def delete_processor(
            self,
            tool_id: UUID,
            processor_id: UUID,
    ) -> None:

        await self._require_tool(tool_id)

        # 摘掉一条不会让 AROUND 的区间交叉（只会让配对的变成单条，
        # 那是允许的中间态），因此这里不需要重跑洋葱校验。
        deleted = await self._processors.delete(
            tool_id, processor_id
        )

        if not deleted:
            raise NotFoundError(
                f"挂载记录不存在：{processor_id}"
            )

    async def replace_chain(
            self,
            tool_id: UUID,
            request: ReplaceToolProcessorChainRequest,
    ) -> ToolProcessorChain:

        await self._require_tool(tool_id)

        now = datetime.now(timezone.utc)

        # 先把整批校验并物化完，再动数据库：某个 item 不合法时
        # 一行都不该写进去（虽然同一事务里抛异常也会回滚，
        # 但"先校验后写"让意图明确，也不依赖事务边界）。
        entities: list[ToolProcessorEntity] = []

        for index, item in enumerate(request.items, start=1):

            entities.append(
                ToolProcessorEntity(
                    processor_id=uuid4(),
                    tool_id=tool_id,
                    processor_key=item.processor_key,
                    sequence=index,
                    execution_mode=item.execution_mode,
                    hook=item.hook,
                    config=self._validate_mount(
                        item.processor_key,
                        item.hook,
                        item.execution_mode,
                        item.config,
                    ),
                    enabled=item.enabled,
                    created_at=now,
                    updated_at=now,
                )
            )

        validate_onion([
            (item.processor_key, item.hook)
            for item in request.items
        ])

        rows = await self._processors.replace_all(
            tool_id, entities
        )

        return build_chain(tool_id, rows, self._impls(rows))

    async def reorder(
            self,
            tool_id: UUID,
            request: ReorderToolProcessorsRequest,
    ) -> ToolProcessorChain:

        await self._require_tool(tool_id)

        rows = await self._processors.list_for_tool(tool_id)

        by_id = {row.processor_id: row for row in rows}

        if set(request.processor_ids) != set(by_id):

            missing = sorted(
                str(i) for i in set(by_id) - set(request.processor_ids)
            )
            unknown = sorted(
                str(i) for i in set(request.processor_ids) - set(by_id)
            )

            raise ConflictError(
                "链已过期：给的 processor_ids 与当前链不一致。"
                + (f" 少了：{missing}。" if missing else "")
                + (f" 多了：{unknown}。" if unknown else "")
                + " 请重新 GET 一次链再提交 —— 部分重排没有明确的"
                "语义（剩下那些排哪？），因此不接受子集。"
            )

        ordered = [by_id[pid] for pid in request.processor_ids]

        # 重排不改变每行的 hook，但会改变 AROUND 记录的先后 ——
        # 那正是洋葱模型关心的，所以要按新顺序再校验一次。
        validate_onion([
            (row.processor_key, row.hook) for row in ordered
        ])

        await self._processors.reorder(
            tool_id, request.processor_ids
        )

        fresh = await self._processors.list_for_tool(tool_id)

        return build_chain(tool_id, fresh, self._impls(fresh))

    # ========================================================
    # internals
    # ========================================================

    def _validate_mount(
            self,
            processor_key: str,
            hook: ProcessorHook,
            execution_mode: ExecutionMode,
            config: dict[str, Any],
    ) -> dict[str, Any]:
        """
        校验"能不能这么挂"，返回**归一化后的 config**。

        归一化指的是过一遍实现的 config_model：默认值落库、
        未知键按模型的策略处理、类型不对直接拒。存原始字典会让
        "库里存的是 A、读出时按 B 解释"这种漂移无从发现。
        """

        # 未注册 -> NotFoundError（→404）。与 GET /processors/{key}
        # 对未注册 key 的响应一致：那个定义本身就是一个资源。
        processor = self._registry.require(processor_key)

        if not processor.supports_hook(hook):
            raise ConflictError(
                f"{processor_key} 不支持 hook={hook.value}。"
                f"它支持的是 "
                f"{sorted(h.value for h in processor.supported_hooks)}。"
            )

        if not processor.supports_execution_mode(execution_mode):
            raise ConflictError(
                f"{processor_key} 不支持 execution_mode="
                f"{execution_mode.value}。它支持的是 "
                f"{sorted(m.value for m in processor.supported_execution_modes)}。"
            )

        if (
                hook is ProcessorHook.AROUND
                and execution_mode is not ExecutionMode.SYNC
        ):
            # 在服务层先拦一道。放到 DB 的 CHECK 上会让这条错误变成
            # IntegrityError —— 它不是 ValueError，to_http_error
            # 兜不住，最终是 500 而不是规格要求的 409。
            raise ConflictError(
                "AROUND 必须配 SYNC：它包住整次调用，一个不被 await "
                "的包装器无法决定何时、是否调用工具。"
            )

        if processor.config_model is None:
            return dict(config)

        # 校验失败抛 pydantic 的 ValidationError —— 它是 ValueError
        # 的子类，因此 to_http_error 会落 400，与规格一致。
        return processor.config_model.model_validate(
            config
        ).model_dump(mode="json")

    def _impls(
            self,
            rows: list[ToolProcessorEntity],
    ) -> dict[str, BaseToolProcessor | None]:
        """
        本次读到的行用到了哪些实现。

        值可能是 None（实现已被删掉）—— 组装层据此把 processor
        字段整个省掉，而不是编一个假定义出来。
        """

        return {
            row.processor_key: self._registry.get(row.processor_key)
            for row in rows
        }

    @staticmethod
    def _insertion_index(
            current: list[ToolProcessorEntity],
            sequence: int | None,
    ) -> int:
        """
        算出新行的最终位置（0-based 下标）。

        sequence 只是**位置提示**：插到"所有 sequence 小于它"的行
        之后。因此传 99 而链只有 3 条不报错，结果就是插到末尾 ——
        落库前整条链会重编号成 1..N（见仓储的说明）。
        """

        if sequence is None:
            return len(current)

        return sum(
            1 for row in current if row.sequence < sequence
        )

    async def _require_tool(self, tool_id: UUID) -> None:

        if await self._definitions.get(tool_id) is None:
            raise NotFoundError(f"Tool 定义不存在：{tool_id}")
