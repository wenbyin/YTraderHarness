"""
Processor 类阶段的共享执行逻辑。

阶段 4 / 5 / 6（归一化 / 清洗 / 元数据抽取）**形状完全一样**：

    读上一阶段的文本产物
      -> 按名字取 DocumentProcessor
      -> process()
      -> 写新的产物
      -> 返回新的 ArtifactRef

区别只有三处：processor 名字、输出产物的 kind、写回 state 的键。
把差异抽成参数、共性留在这里，避免三份几乎相同的节点代码
各自漂移（尤其是产物 kind 与 state 键这种容易写错的地方）。

ProcessorMode 的行为
--------------------
    none    不执行，上游产物原样透传
    common  用系统默认链：Normalize → Clean → Enrich Metadata
    config  用 DocumentProcessorDefinition.processor_config 里的链

配置来源是 version.meta_data.pipeline_config（运行开始时冻结的
快照），不是当前的 DocumentProcessorDefinition —— 否则改了定义
会让历史版本的重跑行为发生漂移，provenance 就断了。
"""

from __future__ import annotations

from typing import Any, Final

from langgraph.runtime import Runtime

from app.infra.graph import NodeSpan

from app.services.document.domain.schemas import (
    ProcessorRequest,
)
from app.services.document.domain.schemas.enum import ProcessorRole

from ..context import DocumentPipelineContext
from ..state import ArtifactRef, DocumentPipelineState
from ._common import file_id_of, tenant_of

__all__ = [
    "PROCESSOR_STAGE_OUTPUT",
    "processor_config_of",
    "processor_name_of",
    "run_processor_stage",
]


def processor_name_of(
        state: DocumentPipelineState,
) -> str | None:
    """
    从 pipeline_config 快照里安全地取出 processor 名。

    ⚠️ 不能写成 `.get("processor", {}).get("name")`：
    create_document 构造快照时，未匹配到前处理定义的情况下
    `processor` 这个键是**存在且值为 None** 的（见
    document_service_impl._build_version）。dict.get 的默认值
    只在**键不存在**时生效，键存在而值为 None 时它照样返回
    None，接着 .get 就炸了。

    实测踩过一次，因此统一走这里。
    """

    config = state.get("pipeline_config") or {}
    processor = config.get("processor") or {}

    return processor.get("name")


def processor_config_of(
        state: DocumentPipelineState,
) -> dict[str, Any]:
    """
    从 pipeline_config 快照里安全地取出 processor 的配置字典。

    与 processor_name_of 是同一个 None 陷阱（见上），因此同样不能写成
    `.get("processor", {}).get("config")`。

    为什么阶段 4/5/6 共用这一份配置
    -------------------------------
    快照里 `processor` 只有**一个** name / config，而阶段 4/5/6 是三个
    不同 role、各自有自己的 config_model。在配置表把 processor_name
    拆成按 role 的键之前，三边只能读同一份扁平字典：

        {"encoding": true, ..., "page_number": {...}, ...}
         ↑ normalizer 认这些        ↑ cleaner 认这些

    好在三个 config_model 的键名**互不重叠**，各取所需即可；pydantic
    默认忽略未知键，所以互相喂对方那半份不会报错。

    ⚠️ 代价是**拼错键名会被静默忽略**。等配置表按 role 拆开、每份配置
    只喂给一个阶段之后，各 config_model 应当改用 extra="forbid"，
    让拼写错误在写入配置时就暴露。
    """

    config = state.get("pipeline_config") or {}
    processor = config.get("processor") or {}

    return dict(processor.get("config") or {})


#: node_id -> (上游 ref 键, 输出 ref 键, 产物 kind, 该阶段的 role)
#:
#: role 同时承担两件事：按它去注册表取能力，以及校验取到的能力
#: 是不是用在了对的阶段（错级会在这里被拦下）。
PROCESSOR_STAGE_OUTPUT: Final[
    dict[str, tuple[str, str, str, ProcessorRole]]
] = {
    "normalization": (
        "parse_ref", "normalize_ref", "normalized_text",
        ProcessorRole.NORMALIZE,
    ),
    "cleaning": (
        "normalize_ref", "clean_ref", "clean_text",
        ProcessorRole.CLEAN,
    ),
    "metadata_extraction": (
        "clean_ref", "extract_ref", "extracted_metadata",
        ProcessorRole.ENRICH_METADATA,
    ),
}


async def run_processor_stage(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
        *,
        span: NodeSpan,
        spec: Any,
        input_key: str,
        output_kind: str,
        processor_name: str | None,
) -> dict[str, Any]:
    """
    执行一个 Processor 阶段。

    Parameters
    ----------
    input_key
        上游 ArtifactRef 在 state 里的键。
    output_kind
        本阶段产物的类型，会作为 File 的 file_name 落库。
    processor_name
        从 pipeline_config 快照里取出的 Processor 名。
        None 表示该阶段不配置自定义 Processor。

    Returns:
        {输出 ref 键: ArtifactRef}
    """

    from uuid import UUID

    context = runtime.context

    stage = PROCESSOR_STAGE_OUTPUT.get(spec.node_id)

    if stage is None:
        raise ValueError(
            f"No processor stage registered for node "
            f"{spec.node_id!r}."
        )

    expected_input, output_key, _kind, role = stage

    if expected_input != input_key or _kind != output_kind:
        # 调用方传错键是很容易犯又很难查的错（产物会静默写错位置），
        # 因此在入口直接拦住。
        raise ValueError(
            f"{spec.node_id}: stage mapping mismatch. "
            f"expected input={expected_input!r} kind={_kind!r}, "
            f"got input={input_key!r} kind={output_kind!r}."
        )

    upstream: ArtifactRef | None = state.get(input_key)  # type: ignore[literal-required]

    if upstream is None:
        raise ValueError(
            f"{spec.node_id}: upstream ref {input_key!r} is missing. "
            "上游阶段没有产出内容，说明图被改坏了。"
        )

    document_version_id = UUID(
        state["document_version_id"]
    )

    # --------------------------------------------------------
    # 取上游正文
    # --------------------------------------------------------

    text = await context.artifact_store.get_text(
        file_id_of(upstream)
    )

    if text is None:
        raise ValueError(
            f"{spec.node_id}: cannot read upstream artifact "
            f"file_id={upstream['file_id']!r}."
        )

    # 同 processor_name_of 的理由：processor 键可能是 None。
    mode = (
        (state.get("pipeline_config") or {})
        .get("processor") or {}
    ).get("mode", "none")

    # --------------------------------------------------------
    # 按 role 取能力并执行
    # --------------------------------------------------------
    #
    # 为什么按 role 取，而不是按快照里的 processor.name
    # ---------------------------------------------------
    # 快照里的 processor.name 只有**一个**值，而阶段 4/5/6 是三个
    # 不同的 role —— 一个名字没法同时指对三个阶段（拿它去查，清洗
    # 阶段会捞到归一化器）。role 才是正确的维度。
    #
    # 等配置表把 processor_name 拆成按 role 的键之后，再在这里叠加
    # "配置点名优先"的优先级。
    #
    # mode 目前**不参与**判断：ProcessorMode 这个枚举在配置表里没有
    # 对应列、快照也没写，所以 processor_name_of 读到的 mode 恒为
    # "none"。按它短路会让整条 Processor 链变成死代码。等 mode 真正
    # 落库后，再把"none 就透传"的门禁加回来。
    #
    # 各阶段实现的语义差别（真实实现就位后）：
    #   normalization      改变文本（编码 / 空白 / 全半角）
    #   cleaning           缩短文本（去页眉页脚 / 水印）
    #   metadata_extraction 产出的其实是**元数据**而不是正文，
    #       这里为复用同一条产物通道而写成 JSON 文本，
    #       将来应考虑换成独立的 metadata 产物类型。
    # --------------------------------------------------------

    capability = context.registry.processor_by_role(role)

    if capability is None:
        # 该阶段还没有实现 —— 透传，与各节点原有的占位行为一致。
        processed = text
        capability_key = None
    else:
        result = await capability.process(
            ProcessorRequest(
                document_id=state["document_id"],
                document_version_id=state["document_version_id"],
                text=text,
                # 不传 config 的话，各能力只会用自己的**默认值**跑 ——
                # 用户在前端配的那一堆开关全是死的，而且不会有任何报错。
                config=processor_config_of(state),
                # 文档级元数据要到阶段 6 才产出，此处还没有可传的。
                metadata={},
            )
        )

        # text 允许为 None（ProcessorResult 的字段是可选的），
        # 真拿到 None 时退回上游正文，避免把下游产物写成空。
        processed = text if result.text is None else result.text

        capability_key = capability.get_key()

    # --------------------------------------------------------
    # 内容未变 -> 不写新 File，直接复用上游 ref
    # --------------------------------------------------------
    #
    # 这一步在真实实现下也会经常命中：某份文档可能本来就很干净，
    # cleaning 无事可做。此时再写一份内容完全相同的 File 是纯浪费。
    #
    # 注意透传后 ref["kind"] 会显示上游的类型（不是本阶段的
    # output_kind）—— 这是**刻意的物理语义**：ref 描述的是
    # 「实际存着的那条 File」，不是「本阶段的名字」。
    # 阶段身份由 node record 的 node_id 承担，不会丢。
    # --------------------------------------------------------

    if processed == text:
        span.set_metrics({
            "processor_role": role.value,
            # 实际用的是谁（None = 该阶段还没实现，走的透传）
            "capability_key": capability_key,
            # 配置里点名的那个 —— 目前不参与取值，仅作 provenance
            "configured_processor_name": processor_name,
            "processor_mode": mode,
            "input_char_count": len(text),
            "char_count": len(processed),
            "passthrough": True,
        })

        span.set_output_summary({
            "ref": {
                "file_id": upstream["file_id"],
                "checksum": upstream["checksum"],
                "kind": upstream["kind"],
            },
            "note": "内容未变，复用上游产物（未写新文件）。",
        })

        return {output_key: upstream}

    ref = await context.artifact_store.put_text(
        kind=output_kind,
        text=processed,
        tenant_id=tenant_of(state),
        content_type="text/plain",
    )

    span.set_metrics({
        "processor_role": role.value,
        "capability_key": capability_key,
        "configured_processor_name": processor_name,
        "processor_mode": mode,
        "input_char_count": len(text),
        "char_count": len(processed),
    })

    span.set_output_summary({
        "ref": {
            "file_id": ref["file_id"],
            "checksum": ref["checksum"],
            "char_count": ref["char_count"],
            "preview": ref["preview"],
        },
    })

    return {output_key: ref}
