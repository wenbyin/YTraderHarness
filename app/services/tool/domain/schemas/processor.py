"""
处理器挂载的对外契约。

对应 tool_processor.yaml 的 components/schemas：
    ProcessorDefinition / ToolProcessor / ToolProcessorChain
    Create- / Update- / Replace- / Reorder- 四个请求
    PagedProcessorDefinitionResponse

两组东西分得很开
----------------
    ProcessorDefinition   **代码侧的实现**（只读，来自注册表）
    ToolProcessor         **挂载记录**（可写，来自数据库）

前者永远不会被接口创建或修改 —— 处理器实现归代码。管理面能做的是
"把已注册的实现挂到某个 Tool 上"，不是"造一个处理器"。

枚举从 enum.py 来，不在这里并列定义第二份
------------------------------------------
ProcessorType / ProcessorHook / ExecutionMode 住在
domain/schemas/enum.py（本模块 re-export，方便调用方一次 import）。
与 tool.py 对 SourceType / SourceState 的处理一致。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import (
    Field,
    field_validator,
    model_serializer,
    model_validator,
)

from app.common.schemas import APIModel

from .enum import ExecutionMode, ProcessorHook, ProcessorType

__all__ = [
    "CreateToolProcessorRequest",
    "ExecutionMode",
    "PagedProcessorDefinitionResponse",
    "ProcessorDefinition",
    "ProcessorHook",
    "ProcessorType",
    "ReorderToolProcessorsRequest",
    "ReplaceToolProcessorChainRequest",
    "ReplaceToolProcessorItem",
    "ToolProcessor",
    "ToolProcessorChain",
    "UpdateToolProcessorRequest",
]


# ============================================================
# 代码侧：一个已注册的实现
# ============================================================


class ProcessorDefinition(APIModel):
    """
    一个**已注册的处理器实现**。

    全部字段只读 —— 它们由实现类上的类属性决定（name /
    display_name / description / processor_type / supported_hooks /
    supported_execution_modes / config_model），接口层改不了。

    这张表是**代码事实的快照**，不是数据库里的行。与
    tool_definition 的区别就在这里：那个是"同步进库的代码事实"
    （要能按它筛选、关联、留历史），而处理器实现数量少、变化随发版，
    读时现算即可，没有入库的必要。
    """

    key: str = Field(
        description=(
            "注册键，也是挂载时 processor_key 的取值。"
            "⚠️ 一旦被挂载就**永不复用** —— 复用它等于让历史配置"
            "指向另一个实现。"
        ),
    )

    name: str = Field(
        description="给人看的名字。",
    )

    description: str | None = Field(
        default=None,
        description="一句话说明它做什么。",
    )

    processor_type: ProcessorType = Field(
        description=(
            "INTERCEPTOR 改控制流 / TRANSFORMER 改数据 / OBSERVER 只读旁路。"
            "由实现继承的 type 基类决定，不是作者声明的。"
        ),
    )

    supported_hooks: list[ProcessorHook] = Field(
        description=(
            "它能挂的段。挂载时 hook 不在这个集合里会被拒（409）。"
        ),
    )

    supported_execution_modes: list[ExecutionMode] = Field(
        description=(
            "它支持的执行方式。AROUND 必然只支持 SYNC。"
        ),
    )

    config_schema: dict[str, Any] = Field(
        description=(
            "参数结构的 JSON Schema（draft 2020-12），由实现的 "
            "config_model 原生输出。前端据此生成表单并做本地校验；"
            "服务端写挂载时用**同一个模型**校验（见 "
            "application/tool_processor_service_impl.py）。"
            "无参数时是 properties 为空的 object —— "
            "**始终存在**，不会缺席。"
        ),
    )

    registry_available: bool = Field(
        description=(
            "该实现当前在进程注册表里吗。"
            "这个接口列出的条目**必然为 true**（列的就是注册表）；"
            "它出现在这里是为了让前端用同一套判断处理挂载记录 —— "
            "那边的 processor 字段可能整个缺席。"
        ),
    )


class PagedProcessorDefinitionResponse(APIModel):
    """注册表分页。"""

    items: list[ProcessorDefinition] = Field(
        description="本次过滤后的实现条目。",
    )

    page: int = Field(ge=1, description="页码，从 1 开始。")

    page_size: int = Field(ge=1, description="每页条数。")

    total: int = Field(
        ge=0,
        description="**过滤之后**的总数，不是本页条数。",
    )


# ============================================================
# 数据库侧：一条挂载记录
# ============================================================


class ToolProcessor(APIModel):
    """
    某个 Tool 的链上的一行。

    字段名与 ORM 列名对齐（processor_id ↔ 主键列 id 由组装层映射），
    对外一律用 processor_id —— 与 tool_id / classification_id 的
    命名习惯一致。
    """

    processor_id: UUID = Field(
        description="挂载记录 ID。",
    )

    tool_id: UUID = Field(
        description="所属 Tool 定义 ID。",
    )

    processor_key: str = Field(
        description="指向的处理器实现键。",
    )

    sequence: int = Field(
        ge=1,
        description=(
            "在链里的位置，从 1 开始**连续**。服务端每次写完都会"
            "重编号 1..N，因此它不一定等于写请求里传来的值。"
        ),
    )

    execution_mode: ExecutionMode = Field(
        description="SYNC=参与结果；ASYNC=旁路不等。",
    )

    hook: ProcessorHook = Field(
        description="BEFORE / AFTER / AROUND。",
    )

    config: dict[str, Any] = Field(
        description=(
            "该实现的业务参数。写入时会过一遍实现的 config_model，"
            "因此这里看到的是**归一化后**的值（默认值已落库、"
            "未知键已被剔除或拒绝）。"
        ),
    )

    enabled: bool = Field(
        description="false = 留在链里占位，执行时跳过。",
    )

    created_at: datetime = Field(description="创建时间。")

    updated_at: datetime = Field(description="最后更新时间。")

    processor: ProcessorDefinition | None = Field(
        default=None,
        description=(
            "指向的实现。**实现已从代码里删掉时这个字段整个缺席**"
            "（不是 null）—— 那种行照样留在链里，前端据此提示"
            "『请重挂或删除』，而不是让整条链读不出来。"
        ),
    )

    @model_serializer(mode="wrap")
    def _omit_absent_processor(self, handler: Any) -> dict[str, Any]:
        """
        `processor` 为 None 时**去掉这个键**，而不是输出 null。

        规格里 processor 不在 required 里，语义是"可以没有" ——
        而 `null` 传达的是"有这个东西，它的值未知"，两者对前端
        是不同的分支。

        为什么用 model_serializer 而不是路由层的
        `response_model_exclude_none=True`：后者会把**所有** None
        一起吃掉，包括 ProcessorDefinition.description 那种
        "就是没有描述"的合法 null。
        """

        data = handler(self)

        if self.processor is None:
            data.pop("processor", None)

        return data


class ToolProcessorChain(APIModel):
    """
    一个 Tool 的完整处理器链。

    不分页 —— 链的长度是人配出来的（几条到十几条），
    与能力目录不分页是同一个取舍。
    """

    tool_id: UUID = Field(description="所属 Tool 定义 ID。")

    processors: list[ToolProcessor] = Field(
        description=(
            "按 sequence 升序。**未注册的实现对应的行也在里面**，"
            "只是那些行没有 processor 字段。"
        ),
    )


# ============================================================
# 写请求
# ============================================================


class CreateToolProcessorRequest(APIModel):
    """往链上挂一条记录。"""

    processor_key: str = Field(
        min_length=1,
        max_length=255,
        description="要挂的处理器实现键。不在注册表里 → 404。",
    )

    sequence: int | None = Field(
        default=None,
        ge=1,
        description=(
            "插入位置。**只是位置提示**：服务端按它决定插在哪一行"
            "之后，然后把整条链重编号成 1..N，因此传 5 而链只有 3 条"
            "不会报错，结果就是插在末尾。不传 = 追加到末尾。"
        ),
    )

    execution_mode: ExecutionMode = Field(
        description="SYNC / ASYNC。AROUND 只能配 SYNC，否则 409。",
    )

    hook: ProcessorHook = Field(
        description="BEFORE / AFTER / AROUND。实现不支持则 409。",
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "该实现的业务参数。会过一遍实现的 config_model，"
            "不合法 → 400。"
        ),
    )

    enabled: bool = Field(
        default=True,
        description="是否启用。",
    )


class UpdateToolProcessorRequest(APIModel):
    """
    改一条挂载记录。

    ⚠️ **改不了 processor_key 与 sequence**
    ------------------------------
    前者是"换一个实现"，后者是"换位置" —— 两件事各有专门的操作
    （删了重挂 / PUT order），混进 PATCH 会让"我只想改个参数"
    变成一次隐式的链重排。
    """

    execution_mode: ExecutionMode | None = Field(
        default=None,
        description="留空表示不改。",
    )

    hook: ProcessorHook | None = Field(
        default=None,
        description="留空表示不改。改完会重新校验与实现的相容性。",
    )

    config: dict[str, Any] | None = Field(
        default=None,
        description=(
            "留空表示不改。**整体替换**，不是合并 —— "
            "要清空就传 {}。"
        ),
    )

    enabled: bool | None = Field(
        default=None,
        description="留空表示不改。",
    )

    @model_validator(mode="after")
    def _at_least_one_field(self) -> "UpdateToolProcessorRequest":
        """
        对应规格里的 minProperties: 1。

        pydantic 没有原生表达，只能自己查。判 `model_fields_set`
        而不是"四个字段里有几个不是 None"：`config={}` 与
        `enabled=false` 都是合法修改，按 None 判会把它们当成
        "什么都没改"。
        """

        if not self.model_fields_set:
            raise ValueError(
                "至少要给一个字段（execution_mode / hook / "
                "config / enabled）。"
            )

        return self


class ReplaceToolProcessorItem(APIModel):
    """
    整体替换时的一项。

    刻意**没有 sequence** —— 数组下标就是顺序。让调用方同时给
    下标和 sequence 只会制造"两者不一致时听谁的"这种问题。
    """

    processor_key: str = Field(
        min_length=1,
        max_length=255,
        description="要挂的处理器实现键。",
    )

    execution_mode: ExecutionMode = Field(
        description="SYNC / ASYNC。",
    )

    hook: ProcessorHook = Field(
        description="BEFORE / AFTER / AROUND。",
    )

    config: dict[str, Any] = Field(
        default_factory=dict,
        description="该实现的业务参数。",
    )

    enabled: bool = Field(
        default=True,
        description="是否启用。",
    )


class ReplaceToolProcessorChainRequest(APIModel):
    """
    整体替换一条链。

    数组顺序 = 执行顺序。**允许空数组**（规格没写 minItems）——
    空数组的语义是"把这个 Tool 的处理器全摘掉"，是个正当操作。
    """

    items: list[ReplaceToolProcessorItem] = Field(
        description=(
            "完整的有序链。**替换语义**：没列出的会被删除，"
            "因此这是「整条链长什么样」的一次性提交，不是增量。"
            "⚠️ 行会被**全删重建**（processor_id 全部重新生成）—— "
            "同一个 processor_key 可以在一条链里出现多次，"
            "按 key 复用 id 本质上是歧义的。"
        ),
    )


class ReorderToolProcessorsRequest(APIModel):
    """重排一条链。"""

    processor_ids: list[UUID] = Field(
        min_length=1,
        description=(
            "该 Tool **全部**挂载记录的 ID，按期望的执行顺序排列。"
            "少给、多给、或混进别的 Tool 的 ID 都会 409（链已过期）—— "
            "部分重排没有明确的语义，接受它只会让「剩下的排哪」"
            "变成实现细节。"
        ),
    )

    @field_validator("processor_ids")
    @classmethod
    def _unique(cls, value: list[UUID]) -> list[UUID]:
        """对应规格里的 uniqueItems: true。pydantic 不查这个。"""

        if len(set(value)) != len(value):
            raise ValueError("processor_ids 不能有重复项。")

        return value
