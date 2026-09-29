"""
实体 -> 对外契约的组装。

纯函数、不碰数据库 —— 有两个服务要用同一套组装逻辑：

    ToolQueryService          列表与详情
    ToolDocumentationService  只取/只写说明

放在这里而不是各自实现一遍，是因为"生效值怎么算"必须**只有一份**：
列表页显示的描述、详情页显示的描述、写完说明后回读的描述，
三者不一致的话，前端会看到"改完没生效"或"两处显示不一样"。
"""

from __future__ import annotations

from uuid import UUID

from app.agent.tools.processor.base import BaseToolProcessor

from ..domain.entities import (
    ToolClassification,
    ToolDefinition,
    ToolDocumentation as ToolDocumentationEntity,
)
from ..domain.entities import (
    ToolProcessor as ToolProcessorEntity,
)
from ..domain.schemas import (
    Classification,
    ParameterDocumentation,
    ProcessorDefinition,
    ToolDefinitionDetail,
    ToolDocumentation,
    ToolManifest,
    ToolProcessor,
    ToolProcessorChain,
    ToolSummary,
)
from ..domain.schemas.enum import ExecutionMode, ProcessorHook
from ..registry.schema import output_description

__all__ = [
    "build_chain",
    "build_classification",
    "build_definition_detail",
    "build_documentation",
    "build_manifest",
    "build_processor_definition",
    "build_summary",
    "build_tool_processor",
]


def build_manifest(definition: ToolDefinition) -> ToolManifest:
    """领域实体 -> ToolManifest。字段一一对应，没有计算。"""

    return ToolManifest(
        id=definition.tool_id,
        tool_key=definition.tool_key,
        name=definition.name,
        source_type=definition.source_type,
        source_ref=definition.source_ref,
        entrypoint=definition.entrypoint,
        input_schema=definition.input_schema,
        output_schema=definition.output_schema,
        source_docstring=definition.source_docstring,
        manifest_hash=definition.manifest_hash,
        source_state=definition.source_state,
        discovered_at=definition.discovered_at,
        last_synced_at=definition.last_synced_at,
    )


def build_classification(
        entity: ToolClassification,
) -> Classification:
    return Classification(
        id=entity.classification_id,
        code=entity.code,
        name=entity.name,
        description=entity.description,
        parent_id=entity.parent_id,
        sort_order=entity.sort_order,
        is_active=entity.is_active,
    )


def _source_descriptions(
        definition: ToolDefinition,
) -> dict[str, str]:
    """代码里的参数说明：{"/query": "..."}。没有则空。"""

    docstring = definition.source_docstring or {}

    parameters = docstring.get("parameters")

    return dict(parameters) if isinstance(parameters, dict) else {}


def _source_return_description(
        definition: ToolDefinition,
) -> str | None:
    """
    返回值描述的**代码侧来源**：输出模型自己的 docstring。

    pydantic 会把模型类的 docstring 放进 schema 的 description，
    所以这里能白拿一份 —— 不必让 service 方法额外写一遍。

    规则与扫描期填 source_docstring["returns"] 时是**同一条**
    （DTO 优先，回落才是方法里的 Returns 段），因此共用
    registry.schema.output_description，不各写一份 —— 两份迟早会
    漂移成两个说法，那正是本模块开头说的"生效值必须只有一份"。
    """

    return output_description(definition.output_schema)


def build_documentation(
        definition: ToolDefinition,
        stored: ToolDocumentationEntity | None,
) -> ToolDocumentation:
    """
    组装说明：代码来源 + 人工覆盖 + 生效值。

    三者都给出来，前端才能标出"这一条被人工改过" ——
    只给一个分不清来源的字符串，人就无从判断该不该撤回去。

    stored 为 None（从没人工改过）时全部回退到代码来源。
    """

    sources = _source_descriptions(definition)

    overrides: dict[str, str | None] = {}

    if stored is not None:
        overrides = {
            item.parameter_path: item.description_override
            for item in stored.parameter_documentations
        }

    # 两边的路径都要列：代码里有说明但没被覆盖的、以及
    # 覆盖了但代码里已经没有的参数（方法改了签名之后的残留）。
    # 后者不能丢 —— 用户得看到它并决定要不要清掉。
    paths = sorted(set(sources) | set(overrides))

    parameters = [
        ParameterDocumentation(
            parameter_path=path,
            source_description=sources.get(path),
            description_override=overrides.get(path),
            effective_description=(
                overrides.get(path) or sources.get(path)
            ),
        )
        for path in paths
    ]

    source_raw = (definition.source_docstring or {}).get("raw")
    source_return = _source_return_description(definition)

    description_override = (
        stored.description_override if stored is not None else None
    )
    return_override = (
        stored.return_description_override
        if stored is not None
        else None
    )

    return ToolDocumentation(
        tool_id=definition.tool_id,
        description_override=description_override,
        return_description_override=return_override,
        parameter_descriptions=parameters,
        effective_description=description_override or source_raw,
        effective_return_description=(
            return_override or source_return
        ),
        updated_by=stored.updated_by if stored is not None else None,
        updated_at=stored.updated_at if stored is not None else None,
    )


def build_summary(
        definition: ToolDefinition,
        *,
        description_override: str | None,
        classifications: list[ToolClassification],
) -> ToolSummary:
    """
    列表项。

    description 用**生效值**而不是原始 docstring —— 人在列表页
    看到的应该与他配置的一致，否则会以为改动没保存。
    """

    source_raw = (definition.source_docstring or {}).get("raw")

    return ToolSummary(
        id=definition.tool_id,
        tool_key=definition.tool_key,
        name=definition.name,
        source_type=definition.source_type,
        source_state=definition.source_state,
        description=description_override or source_raw,
        classifications=[
            build_classification(item) for item in classifications
        ],
    )


def build_definition_detail(
        definition: ToolDefinition,
        stored: ToolDocumentationEntity | None,
        classifications: list[ToolClassification],
) -> ToolDefinitionDetail:
    """详情：事实 + 说明 + 分类，一次组装完。"""

    return ToolDefinitionDetail(
        manifest=build_manifest(definition),
        documentation=build_documentation(definition, stored),
        classifications=[
            build_classification(item) for item in classifications
        ],
    )


# ============================================================
# Processor
# ============================================================


def build_processor_definition(
        processor: BaseToolProcessor,
) -> ProcessorDefinition:
    """
    实现实例 -> ProcessorDefinition。

    supported_hooks / supported_execution_modes 按**枚举声明顺序**
    输出，不按字母序：BEFORE / AFTER / AROUND 这个顺序本身就是
    调用发生的顺序，前端照着渲染比 AFTER / AROUND / BEFORE 好读。
    集合是 frozenset（无序），所以这里必须显式排序 —— 让它随
    Python 的哈希顺序漂，同一个实现在两次请求里会给出不同的数组。
    """

    return ProcessorDefinition(
        key=processor.get_key(),
        name=processor.display_name or processor.get_key(),
        description=processor.description,
        processor_type=type(processor).processor_type,
        supported_hooks=[
            hook
            for hook in ProcessorHook
            if hook in processor.supported_hooks
        ],
        supported_execution_modes=[
            mode
            for mode in ExecutionMode
            if mode in processor.supported_execution_modes
        ],
        config_schema=type(processor).get_config_schema(),
        # 列出来的必然来自注册表，因此恒为 true。
        # 这个字段存在的意义是让前端对"挂载记录里的 processor"
        # 与"目录里的条目"用同一套判断（见 schemas/processor.py）。
        registry_available=True,
    )


def build_tool_processor(
        entity: ToolProcessorEntity,
        processor: BaseToolProcessor | None,
) -> ToolProcessor:
    """
    挂载记录 -> DTO。

    processor 为 None 表示这个 key 在代码里已经不存在了 ——
    DTO 的 model_serializer 会把这个字段整个去掉（不是置 null），
    行本身照常返回。
    """

    return ToolProcessor(
        processor_id=entity.processor_id,
        tool_id=entity.tool_id,
        processor_key=entity.processor_key,
        sequence=entity.sequence,
        execution_mode=entity.execution_mode,
        hook=entity.hook,
        config=dict(entity.config),
        enabled=entity.enabled,
        created_at=entity.created_at,
        updated_at=entity.updated_at,
        processor=(
            build_processor_definition(processor)
            if processor is not None
            else None
        ),
    )


def build_chain(
        tool_id: UUID,
        rows: list[ToolProcessorEntity],
        processors: dict[str, BaseToolProcessor | None],
) -> ToolProcessorChain:
    """
    整条链 -> DTO。

    processors 是"本次用到的那些实现"的映射，由调用方从注册表取好传
    进来（本模块是纯函数，不碰注册表也不碰数据库）。

    按 sequence 再排一次：仓储已经 order_by 了，但**组装层不该依赖
    别人的排序**。链的顺序就是语义本身，这里少写一个 sort，
    将来任何一条换掉的查询都会静默改变执行顺序。
    """

    return ToolProcessorChain(
        tool_id=tool_id,
        processors=[
            build_tool_processor(
                row, processors.get(row.processor_key)
            )
            for row in sorted(rows, key=lambda r: r.sequence)
        ],
    )
