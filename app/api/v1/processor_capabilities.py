"""
处理能力目录接口。

规格见 app/services/document/processor_capability.yaml。

前端消费面
----------
    GET /document-processor-capabilities                能力目录
    GET /document-processor-capabilities/{processor_key} 单个能力

与 /document-processors 的区别（必读）
--------------------------------------
    /document-processors                配置资源：
                                        某类型下 .pdf 走哪个 loader，
                                        可增删改的数据库行。

    /document-processor-capabilities    能力目录：
                                        代码里已注册的实现，只读。

后者的 processor_key，正是前者 loader_name / processor_name
等字段的取值。两者是"配置引用能力"的关系。

前端契约
--------
1. **processor_key 是稳定契约键**，会被冻结进 PipelineConfigSnapshot，
   一旦发布永不复用。遇到不认识的 key **不要丢弃该条配置** ——
   它可能来自历史版本，回退到显示原始 key 即可。

2. **config_schema 是 JSON Schema（draft 2020-12）**，
   前端据此生成表单并做本地校验。注意可空字段产出的是
   {"anyOf": [{"type": "string"}, {"type": "null"}]}。

3. **selector 只在 role=loader 出现。**

4. 本目录回答"平台有哪些能力"，**不回答"这个文件能不能传"** ——
   后者看 DocumentTypeDefinition 的 analysis_mode 与
   DocumentProcessorDefinition。
"""

from __future__ import annotations

from fastapi import (
    APIRouter,
    Query,
)

from app.common.utils import to_http_error

from app.services.document.application.capability_service import (
    ProcessorCapabilityService,
)
from app.services.dependencies import Service
from app.services.document.domain.schemas.capability import (
    ProcessorCapabilityCatalogResponse,
    ProcessorCapabilityEntry,
)
from app.services.document.domain.schemas.enum import (
    ModalityType,
    ProcessorRole,
)

router = APIRouter(
    tags=["Document Processor Capabilities"],
)


# ============================================================
# 能力目录
# ============================================================


@router.get(
    "/document-processor-capabilities",
    response_model=ProcessorCapabilityCatalogResponse,
)
async def list_processor_capabilities(
        role: ProcessorRole | None = Query(
            default=None,
            description=(
                "只看某个阶段的能力。前端做「选解析器 / 选切分器」"
                "这类下拉时用它。"
            ),
        ),
        modality_type: ModalityType | None = Query(
            default=None,
            description=(
                "按模态过滤，对**所有 role** 都成立 —— "
                "modality 是能力的适用域，不是 loader 专属。"
                "「列出 audio 相关的全部能力」就用它。"
            ),
        ),
        file_extension: str | None = Query(
            default=None,
            max_length=32,
            description=(
                "只作用于 role=loader 的条目，例如 .txt 或 txt，"
                "大小写不敏感。"
            ),
        ),
        service: ProcessorCapabilityService = Service[ProcessorCapabilityService],
):
    """
    平台当前的处理器能力目录。

    不分页 —— 条目数量级等于已注册的实现数。

    响应里的 support_suffix 是**过滤后** loader 条目的后缀并集：
    要"全平台可解析后缀"的完整清单，不带任何参数调用一次。

    某个 role 下返回空列表是正常状态（该阶段还没实现），
    不是接口出错。
    """

    try:
        return await service.list_capabilities(
            role=role,
            modality_type=modality_type,
            file_extension=file_extension,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/document-processor-capabilities/{processor_key}",
    response_model=ProcessorCapabilityEntry,
)
async def get_processor_capability(
        processor_key: str,
        service: ProcessorCapabilityService = Service[ProcessorCapabilityService],
):
    """
    按 processor_key 查单个能力。

    给前端解析一个"不认识的 key"用 —— 例如从历史
    PipelineConfigSnapshot 里读到的 key，不必为了它拉整个目录。

    404 时前端应回退到显示原始 key，而不是丢弃这条配置。
    """

    try:
        return await service.get_capability(processor_key)

    except ValueError as exc:
        raise to_http_error(exc) from exc
