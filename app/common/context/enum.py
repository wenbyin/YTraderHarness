"""
Context Builder 的枚举定义。

取值以 `app/services/context/context_builder.yaml` 为**唯一准绳**
----------------------------------------------------------------
那份规格是**对外契约**（前端按字面值渲染流程图、按 `stage_id`
归位观测数据），因此这里的每个值都必须与它逐字一致，包括
`StageId` 的**顺序**。

两边漂移的后果不是报错而是静默错位：前端拿到一个它不认识的
stage_id 会画不出节点，而服务端一切正常。改动任一处的取值都
意味着规格要同步改。

与 tool / document 两个模块的关系
---------------------------------
它们各自的枚举住在自己的 `domain/schemas/enum.py`，因为那些枚举
只被那一个模块消费。Context Builder 的枚举被契约层（app/agent）、
调度层（app/agent）、注册表（app/agent）三方共用，且将来还要被
`/context-builders` 接口消费 —— 放共享内核比放任何一方都合适。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "BuilderExecutionMode",
    "ContextBuilderStatus",
    "ContextBuilderType",
    "StageId",
    "StageKind",
]


# ============================================================
# Context Builder 的分类
# ============================================================


class ContextBuilderType(str, Enum):
    """
    一个 Context Builder 的**执行语义**。

    它不是"随便起个名字的流程"，而是一个**服务器定义的类型标识**：
    规格里明说这是 type identifier，不是 arbitrary workflow name。

    与 tool 模块把 `processor_type` 由 type 基类定死同一手法：
    这里也是**继承来的**，具体实现不能自己声明（见 base.py 的六个
    type 基类）。声明与实际能力不符时，没有第二处可以撒谎。

        AGENT_INVOKE    主 Agent / ReAct 调用。通常在主模型调用前
                        即时构建上下文，可能包含 history / memory /
                        knowledge / capabilities / budgeting / compaction。
        PLANNER         独立的规划模型调用。通常只用任务范围内的
                        history 与规划指令，不需要完整执行历史。
        CLASSIFIER      路由 / 分类用的窄输入。默认没有对话历史、
                        memory、knowledge 与工具执行上下文。
        COMPACTION      辅助 LLM 调用的上下文，目的是把既有上下文
                        摘要压缩。通常由别的 builder 在触发预算或
                        策略时**嵌套**调用。
        MEMORY          memory 抽取 / 反思用的上下文。可以由独立后台
                        工作流执行，只应包含该 scope 与策略下允许
                        抽取的信息。
        SUB_AGENT       隔离到子任务 scope 的上下文。通常只看到父
                        Agent 历史的简化视图与部分能力。
    """

    AGENT_INVOKE = "agent_invoke"
    PLANNER = "planner"
    CLASSIFIER = "classifier"
    COMPACTION = "compaction"
    MEMORY = "memory"
    SUB_AGENT = "sub_agent"


class ContextBuilderStatus(str, Enum):
    """
    一个 Context Builder 定义的状态。

        ACTIVE      正常可用
        DISABLED    停用。**只是元数据**，不是删除 —— 定义仍然
                    可被 `include_disabled=true` 列出，注册表也照常
                    注册它（见 registry.py）。跳过它会让能力目录与
                    实际能取到的东西对不上。

    ⚠️ 取值是**小写**，与 `SourceState` 的大写不一致 —— 各自照
    各自的规格来（tool_processor.yaml 写的是大写，本规格写的是
    小写）。两者都是对外契约，都不算笔误。
    """

    ACTIVE = "active"
    DISABLED = "disabled"


class BuilderExecutionMode(str, Enum):
    """
    一个 Context Builder 通常**跑在哪**。

        EMBEDDED_MODEL_INVOCATION   作为准备父 LLM 调用的一部分运行
        INDEPENDENT_WORKFLOW        通常作为自己的工作流被调用
        EITHER                      两种都可能

    ⚠️ 刻意不叫 `ExecutionMode`
    ---------------------------
    `app/services/tool/domain/schemas/enum.py` 已经有一个同名的
    `ExecutionMode`（SYNC / ASYNC，说的是"处理器与工具调用是否
    同步"）。两个枚举**值域完全不同**，而下面这种 import 在同一
    个文件里都很自然（比如一个既列 tool processor 又列 context
    builder 的能力目录服务）：

        from app.services.tool.domain.schemas.enum import ExecutionMode
        from app.common.context import ExecutionMode      # ← 会拿错

    拿错之后在调用点**看不出来**（都是枚举成员，比较不报错，只是
    永远不相等）。所以符号名在这里避开，**取值保持与规格一致** ——
    规格名是 JSON 线上的名字，Python 的类名不出现在响应里，改名
    零代价，拿错人却有代价。
    """

    EMBEDDED_MODEL_INVOCATION = "embedded_model_invocation"
    INDEPENDENT_WORKFLOW = "independent_workflow"
    EITHER = "either"


# ============================================================
# 固定流水线的阶段
# ============================================================


class StageId(str, Enum):
    """
    Context Builder 的**固定流水线阶段**。

    ⚠️ 声明顺序 = 流水线顺序，且它是**冻结契约**
    --------------------------------------------
    规格里写明 stage order 由服务器定义、客户端必须当作不可变。
    这里用枚举的声明顺序承载那个顺序，因此：

        默认    在末尾追加新阶段（老实现不受影响）
        例外    需要插到 PROMPT_ASSEMBLER 之前的阶段 —— 必须同时更新
                PIPELINE_TEMPLATE、所有 builder 与测试，并抬
                ContextBuilderDefinition.version。1.1.0 的
                TASK_RESOLVER / SKILL_LOADER 就是这种显式插入。

    真正驱动执行的是 `contracts.PIPELINE_TEMPLATE` —— 一份**显式
    写死**的元组，不是 `tuple(StageId)`。理由见那边的说明：从枚举
    推导会让"有人重排枚举"静默变成"重排流水线"，而显式元组至少
    让改动落在一个需要 review 的地方。

    十三个阶段各自的职责
    --------------------
        CONTEXT_RESOLVER     读 builder_type + state，输出 skip 列表
                             与各阶段参数。**流水线的入口**。
        TASK_RESOLVER        如果当前有 task，把任务信息注入 system
                             并作为“当前输入”；无 task 时跳过。
        SKILL_LOADER         按 task.metadata.skill_key 加载 skill，
                             用统一 SkillSnapshot 渲染 system prompt，
                             并把 skill + task 能力解析成 tool_keys。
        PROFILE_LOADER       加载用户 / 租户画像
        SUMMARY_LOADER       生成并注入对话摘要。**唯一会自己调 LLM 的
                             阶段**（见 contracts.LLM_INVOCATION_STAGES）
        HISTORY_SELECTOR     取最近若干轮历史（参数：取几条、怎么排）
        MEMORY_RETRIEVER     长期记忆检索
        KNOWLEDGE_RETRIEVER  知识库检索
        CAPABILITY_LOADER    加载本次可用的工具（参数：加载哪些）
        PROMPT_ASSEMBLER     按 state 决定层序，产出 ModelRequest。
                             **出口阶段**。
        TOKEN_BUDGETER       按预算裁剪（参数：预算多少、裁剪优先级）
        CONTEXT_COMPACTOR    上下文压缩：超预算时把历史窗口与摘要的
                             重叠压成 0（纯切片，**不调 LLM**）
        CONTEXT_VALIDATOR    校验最终上下文是否可用
    """

    CONTEXT_RESOLVER = "context_resolver"
    TASK_RESOLVER = "task_resolver"
    SKILL_LOADER = "skill_loader"
    PROFILE_LOADER = "profile_loader"
    SUMMARY_LOADER = "summary_loader"
    HISTORY_SELECTOR = "history_selector"
    MEMORY_RETRIEVER = "memory_retriever"
    KNOWLEDGE_RETRIEVER = "knowledge_retriever"
    CAPABILITY_LOADER = "capability_loader"
    PROMPT_ASSEMBLER = "prompt_assembler"
    TOKEN_BUDGETER = "token_budgeter"
    CONTEXT_COMPACTOR = "context_compactor"
    CONTEXT_VALIDATOR = "context_validator"


class StageKind(str, Enum):
    """
    阶段的**种类**。

    纯粹是给前端分组的标签（流程图里给同类阶段上同一种样式），
    运行时**不读它** —— 调度只认 `StageId`。

    单独列出来而不是从 StageId 的名字推，是因为"某阶段属于哪一类"
    是规格里的显式声明，推导出来的规则（按下划线后缀猜）会在第一个
    不符合命名的阶段上悄悄失效。
    """

    RESOLVER = "resolver"
    LOADER = "loader"
    SELECTOR = "selector"
    RETRIEVER = "retriever"
    ASSEMBLER = "assembler"
    BUDGETER = "budgeter"
    COMPACTOR = "compactor"
    VALIDATOR = "validator"
