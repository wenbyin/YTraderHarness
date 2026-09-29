"""
处理能力目录的对外契约。

对应 app/api/v1/processor_capabilities.py 暴露的接口，
规格见 app/services/document/processor_capability.yaml。

与 observability.py 的节点目录的分工
------------------------------------
    observability.py  运行观测：这一次跑得怎么样、卡在哪
    capability.py     能力目录：平台有哪些能力、参数怎么填

节点目录回答"这个 node_id 是什么意思"（历史运行里的旧 id 也要能解释），
能力目录回答"有哪些能力可选、选中之后参数怎么填"。

前端契约（必读）
----------------
1. **processor_key 是稳定契约键**，会被冻结进
   PipelineConfigSnapshot，一旦发布**永不复用**。
   遇到不认识的 key **不要丢弃这条配置** —— 它可能来自历史版本，
   回退到显示原始 key 即可。

2. **modality_type 每个条目都有**，它回答"这个能力处理什么内容的"。
   文档也有 modality_type（DocumentTypeDefinition），两者对接：
   audio 类型的文档只该选到 audio 的能力。

   因此"按模态筛能力"对所有 role 都成立，不限于 loader。

3. **selector 只在 role=loader 时出现**，装的是文件后缀自选谓词。
   其他 role 由配置显式点名（chunker_name / processor_name / …），
   不做自选，所以没有这个字段。

4. **config_schema 是 JSON Schema（draft 2020-12）**，
   由后端 pydantic 的 model_json_schema() 原生输出，与 OpenAPI 3.1
   的 schema 版本天然一致。

   前端据此生成表单并做本地校验；服务端在写入 definition 时用
   同一份结构校验 —— 两边规则不会漂移。

   表单生成器普遍消费的键：title / description / default / enum /
   minimum / maximum / format / anyOf。

   注意：可空字段（str | None）产出的是
       {"anyOf": [{"type": "string"}, {"type": "null"}]}
   而不是 {"type": ["string", "null"]}，解析器要认前者。

   没有参数的能力返回空 object，此时前端应隐藏表单，
   而不是渲染一个空面板。

5. **能力目录不是"这个文件能不能传"。** 某个
   DocumentTypeDefinition 收不收某后缀，看它的 analysis_mode 与
   DocumentProcessorDefinition —— 那是另一个接口的事。
"""

from __future__ import annotations

from typing import Any

from pydantic import Field

from app.common.schemas import APIModel

from .enum import ModalityType, ProcessorRole


class ProcessorCapabilitySelector(APIModel):
    """
    **仅 role=loader 出现。**

    装的是"文件刚进来、还没查任何配置时，按什么把 loader 选出来"
    的两个谓词，对应 DocumentParser 的 supports() 与 is_default()。

    为什么只有 loader 有：文件带着后缀进来，在任何人查配置**之前**，
    系统必须先决定"谁能读它"。其余 role 都是被配置显式点名的
    （chunker_name / processor_name / …），不做自选。

    注意 modality_type **不在**这里 —— 它是"这个能力处理什么内容"，
    是所有 role 都有的适用域，因此挂在 entry 顶层。
    """

    support_suffix: list[str] = Field(
        description=(
            "该 loader 认的文件后缀，**服务端已归一化**："
            "小写 + 带前导点，例如 [\".txt\"]。"
            "可以增删，不是冻结值。"
        ),
    )

    is_default: bool = Field(
        description=(
            "同一后缀有多个候选 loader 时，是否为默认选中的那个。"
        ),
    )


class ProcessorCapabilityEntry(APIModel):
    """
    目录中的一项 —— 一个已注册的处理能力。

    processor_key 与 role 直取实现类，没有额外加工。
    """

    processor_key: str = Field(
        description=(
            "稳定契约键，例如 TEXT_TXT_COMMON_PARSER。"
            "它也是 GET /document-processors 响应里 loader_name / "
            "processor_name 等字段的取值。"
        ),
    )

    role: ProcessorRole = Field(
        description="该能力在写入流水线里承担的阶段。",
    )

    modality_type: ModalityType = Field(
        description=(
            "该能力处理什么模态的内容 —— 所有 role 都有，不是 loader 专属。"
            "文档本身也有 modality_type（DocumentTypeDefinition），"
            "两者对接：audio 类型的文档只该选到 audio 的能力。"
        ),
    )

    selector: ProcessorCapabilitySelector | None = Field(
        default=None,
        description=(
            "仅 role=loader 出现 —— 装的是文件后缀自选谓词。"
            "其他 role 由配置显式点名，不做自选。"
        ),
    )

    config_schema: dict[str, Any] = Field(
        description=(
            "参数结构的 JSON Schema（draft 2020-12）。"
            "前端据此生成表单并做本地校验。"
            "无参数时是 properties 为空的 object —— "
            "**始终存在**，不会缺席。"
        ),
    )


class ProcessorCapabilityCatalogResponse(APIModel):
    """
    处理能力目录。

    不分页 —— 条目数量级等于已注册的实现数，
    与节点目录接口的取舍一致。
    """

    items: list[ProcessorCapabilityEntry] = Field(
        description="本次过滤后的能力条目。",
    )

    support_suffix: list[str] = Field(
        description=(
            "派生值：本次返回的 items 中 **role=loader** 那些条目的 "
            "support_suffix 去重并集，已排序。"
            "**受查询过滤影响** —— 传了 role / modality_type / "
            "file_extension 就只统计过滤后的集合。"
            "过滤到没有 loader 条目时是空数组，不是 null。"
        ),
    )

    total: int = Field(
        description="items 的长度。",
    )
