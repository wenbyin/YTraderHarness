"""
阶段 6：Metadata Extraction。

归属：Processor（ProcessorMode.COMMON 默认链的第 3 步）。

职责
----
抽出文档级元数据：标题、作者、日期、章节结构、语言、
业务标签等。这些会：

    - 随每个 chunk 落到 document_chunk.metadata
    - 进入切分/增强的判断依据（例如按章节切分）

本节点是**扇出前的最后一个线性节点** ——
它跑完，clean_ref 与 extract_ref 就都齐了，fanout.py 据此
构造 Send payload。

这也是最后一个能拿到完整主 state 的地方；一旦进入扇出，
分支节点只能看到 Send payload（见 state.py 的 BranchInput）。
"""

from __future__ import annotations

import json
from typing import Any

from langgraph.runtime import Runtime

from app.infra.graph import node_span

from ..catalog import NODE_SPECS
from ..registry import PipelineRegistry
from app.services.document.domain.schemas.enum import (
    RETRIEVAL_EMBEDDING_KIND,
)
from app.services.document.chunker.identity import (
    chunker_identity,
)

from ..context import DocumentPipelineContext
from ..state import DocumentPipelineState
from ._common import tenant_of
from ._processor import (
    processor_name_of,
    run_processor_stage,
)

_SPEC = next(
    s for s in NODE_SPECS if s.node_id == "metadata_extraction"
)

__all__ = ["metadata_extraction_node"]


async def metadata_extraction_node(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """抽取文档级元数据。返回 {"extract_ref": ArtifactRef}。"""

    async with node_span(
            runtime,
            node_id=_SPEC.node_id,
            label=_SPEC.label,
            group=_SPEC.group,
            sequence=_SPEC.sequence,
            input_summary={
                "clean_ref": state.get("clean_ref"),
            },
    ) as span:

        # ----------------------------------------------------
        # 1. 产出元数据产物（当前为透传占位）
        # ----------------------------------------------------

        result = await run_processor_stage(
            state,
            runtime,
            span=span,
            spec=_SPEC,
            input_key="clean_ref",
            output_kind="extracted_metadata",
            processor_name=processor_name_of(state),
        )

        # ----------------------------------------------------
        # 1b. 覆盖 extract_ref：真正产出**元数据**
        # ----------------------------------------------------
        #
        # ⚠️ 上面那步目前**产不出元数据**。ProcessorRole.ENRICH_METADATA
        # 没有任何实现，_processor.py 于是透传正文、并因为"内容未变"
        # 直接复用了 clean_ref —— 也就是说 result["extract_ref"] 指向的是
        # **清洗后的正文**，不是元数据字典。
        #
        # 后果很隐蔽：下游（阶段 8 的增强）把它 json.loads 一下，正文不是
        # JSON、解析失败，于是拿到空 dict，增强静默地什么都不做。
        #
        # 而且这件事**本来就不该做成 capability**：要产出的 title / 业务
        # 标识来自数据库（Document / DTD），不来自正文 —— 文本 processor
        # 拿不到它们。所以修在节点层，不修注册表。
        # ----------------------------------------------------

        metadata = await _document_metadata(state, runtime)

        if metadata:

            context = runtime.context

            ref = await context.artifact_store.put_text(
                kind="extracted_metadata",
                text=json.dumps(metadata, ensure_ascii=False),
                tenant_id=tenant_of(state),
                content_type="application/json",
            )

            result["extract_ref"] = ref

        span.set_metrics({
            "document_metadata_keys": sorted(metadata),
        })

        # ----------------------------------------------------
        # 2. 决定扇出哪些 retrieval —— 本节点是扇出前的最后
        #    一个线性节点，扇出计划必须在这里定下来。
        # ----------------------------------------------------

        branches = await _resolve_branches(state, runtime)

        span.set_metrics({
            "branch_count": len(branches),
        })

        span.set_output_summary({
            **(span.output_summary or {}),
            "branches": [
                {
                    "retrieval_type": b["retrieval_type"],
                    "chunker": b["chunker_name"],
                    "storage_backend": b["storage_backend"],
                    "index": b["index_identifier"],
                }
                for b in branches
            ],
        })

        result["branches"] = branches

        return result


#: 从 Document.metadata 里挑成"文档级事实"的键。
#:
#: 与 enricher 的 _INHERITED_KEYS 是**同一批语义**：这些是"这条 chunk
#: 属于谁"，而不是"这一阶段算出了什么"。挑出来单独放，下游（阶段 8）
#: 才知道哪些该继承。
_BUSINESS_KEYS: tuple[str, ...] = (
    "company_code",
    "company_name",
)


async def _document_metadata(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> dict[str, Any]:
    """
    组装文档级元数据：源文件事实 + 业务归属。

    两个来源，分工不同：

        File（经 state["file_id"]）  源文件**本身**的事实 ——
                                    file_name / mime_type / file_size /
                                    origin / update。这些是"这份数据是什么"。
        Document / DTD               我们的**业务归属** ——
                                    title / document_key / source /
                                    dtd_name / 业务标识。

    其中两个键是**专为阶段 8 的检索前缀**产的（见 enricher/common.py 的
    _EMBED_PREFIX_KEYS）：`update` 与 `dtd_name`。它们是复合/格式化过的
    值，不是原始字段的直接拷贝。

    `file_name` 尤其关键：它是阶段 8 格式识别的**第一依据**（后缀名优先
    于内容嗅探）。拿不到它，markdown / json / yaml 的识别就只能靠猜。

    读不到任何一项都不报错，返回已拿到的部分 —— 缺一项只是增强少加
    一条，不该让整条写入链挂掉。
    """

    from uuid import UUID

    from app.common.repository.factory import RepositoryFactory
    from app.services.file.repository.file import FileRepository

    from ...repository.document import DocumentRepository
    from ...repository.type_definition import (
        DocumentTypeDefinitionRepository,
    )

    context = runtime.context

    if context.session_factory is None:
        return {}

    document_id = UUID(state["document_id"])
    type_id = UUID(state["document_type_definition_id"])
    file_id = UUID(state["file_id"])

    # 只读决策：开一段短会话，不开显式事务（begin=False）。
    repositories = RepositoryFactory(
        session_factory=context.session_factory
    )

    async with repositories.scope(begin=False) as repos:

        # 源文件。它可能已被清理（软删）—— 拿不到就跳过这一组，
        # 不影响下面从 Document 拿的部分。
        file = await repos.create(FileRepository).get(file_id)

        document = await repos.create(DocumentRepository).get(
            document_id
        )

        if document is None:
            return {}

        definition = (
            await repos.create(DocumentTypeDefinitionRepository)
            .get(type_id)
        )

    metadata: dict[str, Any] = {}

    # ---- 源文件事实 ----

    if file is not None:

        if file.file_name:
            metadata["file_name"] = file.file_name

        # 叫 mime_type 而不是 content_type：后者在这个项目里已经被
        # "正文"占用了（chunk 的 content），同名会让读的人以为是正文类型。
        if file.content_type:
            metadata["mime_type"] = file.content_type

        metadata["file_size"] = file.file_size

        if file.origin:
            metadata["origin"] = file.origin.value

        # 上传时间。**必须转成字符串** —— 这个 dict 会被 json.dumps
        # 写进产物，datetime 对象进去就是 TypeError。
        #
        # 取日期粒度而不是 .isoformat()：它进的是检索前缀，
        # "2026-04-25T14:30:00+08:00" 里的时分秒与 +08:00 只会给
        # 稀疏那一路增加无意义的 token。
        if file.created_at is not None:
            metadata["update"] = file.created_at.strftime("%Y-%m-%d")

    # ---- 业务归属 ----

    # 标题优先用 Document 的（业务名），没有就退回文件名 ——
    # 文件名至少是个能看的东西，比没有强。
    title = document.title or (
        file.file_name if file is not None else None
    )

    if title:
        metadata["title"] = title

    if document.document_key:
        metadata["document_key"] = document.document_key

    if definition is not None:
        # source 保持原样 —— 检索侧与过滤条件还在用它，改名波及面
        # 远大于收益。
        metadata["source"] = definition.type_key

        # dtd_name 是给检索前缀用的**复合串**：中文名读得懂、也更容易
        # 在用户查询的字面里出现，编码则保证唯一。
        metadata["dtd_name"] = (
            f"{definition.name}({definition.type_key})"
        )

    # 业务标识：先看 Document（文档级事实），再看 File.metadata
    # （源系统自带的话优先信它 —— 它是外部系统给的原始标识）。
    document_meta = dict(document.metadata or {})
    file_meta = dict(file.metadata or {}) if file else {}

    for key in _BUSINESS_KEYS:

        value = file_meta.get(key) or document_meta.get(key)

        if value not in (None, ""):
            metadata[key] = value

    return metadata


async def _resolve_branches(
        state: DocumentPipelineState,
        runtime: Runtime[DocumentPipelineContext],
) -> list[dict[str, Any]]:
    """
    解析出本次要扇出的 RetrievalDefinition 列表。

    这是**只读**决策，因此允许节点开一段短会话 ——
    见 context.py 的 session_factory 使用纪律。

    选取规则本身是纯函数（selection.select_retrievals），
    这里只负责把它需要的输入从库里读出来。
    """

    from uuid import UUID

    from app.common.repository.factory import RepositoryFactory

    from ...repository.document_index_binding import (
        DocumentIndexBindingRepository,
    )
    from ...repository.document_version import (
        DocumentVersionRepository,
    )
    from ...repository.processor_retrieval_binding import (
        DocumentProcessorRetrievalBindingRepository,
    )
    from ...repository.retrieval_definition import (
        DocumentRetrievalDefinitionRepository,
    )
    from ..selection import (
        RetrievalCandidate,
        select_retrievals,
    )

    context = runtime.context

    if context.session_factory is None:
        raise RuntimeError(
            "metadata_extraction needs session_factory to resolve "
            "retrieval definitions, but the pipeline context has none."
        )

    version_id = UUID(state["document_version_id"])
    type_id = UUID(state["document_type_definition_id"])

    restrict = [
        UUID(x)
        for x in (
            state["pipeline_config"].get(
                "document_processor_definition_ids"
            )
            or []
        )
    ]

    # 只读决策：开一段短会话，不开显式事务（begin=False）。
    repositories = RepositoryFactory(
        session_factory=context.session_factory
    )

    async with repositories.scope(begin=False) as repos:
        version = await repos.create(
            DocumentVersionRepository
        ).get(version_id)

        if version is None:
            raise ValueError(
                f"DocumentVersion not found: {version_id}"
            )

        retrievals = (
            await repos.create(DocumentRetrievalDefinitionRepository)
            .list_all_by_type_definition(type_id)
        )

        bindings = (
            await repos.create(
                DocumentProcessorRetrievalBindingRepository
            ).list_by_type_definition(type_id)
        )

        index_binding_repo: DocumentIndexBindingRepository = (
            repos.create(DocumentIndexBindingRepository)
        )

        # retrieval definition -> 它的 index 实例（可能没有，
        # 例如 PostgreSQL document retrieval 不需要独立索引）
        index_by_retrieval: dict[UUID, Any] = {}

        for retrieval in retrievals:
            instances = (
                await index_binding_repo
                .list_by_retrieval_definition(
                    retrieval.retrieval_definition_id
                )
            )
            if instances:
                index_by_retrieval[
                    retrieval.retrieval_definition_id
                ] = instances[0]

    bound_by_processor: dict[UUID, list[UUID]] = {}

    for binding in bindings:
        bound_by_processor.setdefault(
            binding.document_processor_definition_id, []
        ).append(binding.document_retrieval_definition_id)

    selected = select_retrievals(
        candidates=[
            RetrievalCandidate(
                document_retrieval_definition_id=(
                    r.retrieval_definition_id
                ),
                retrieval_type=r.retrieval_type.value,
                chunker_name=r.chunker_name,
                chunker_config=dict(r.chunker_config),
                retriever_name=r.retriever_name,
                retriever_config=dict(r.retriever_config),
                storage_backend=r.storage_backend.value,
                storage_config=dict(r.storage_config),
            )
            for r in retrievals
        ],
        version_processor_definition_id=(
            version.document_processor_definition_id
        ),
        bound_retrieval_ids_by_processor=bound_by_processor,
        restrict_to_processor_ids=restrict,
    )

    branches = []

    for c in selected:

        config = dict(c.chunker_config)

        # length_function="model"（按**向量模型自己的词表**量长度）
        # 需要知道用哪个模型的词表，这里从本分支的 embedder 推出来。
        #
        # 为什么补在计划阶段而不是 chunker 节点：身份**在这一段**就算
        # 好了（见下），chunker 拿的也正是分支里这份配置。补晚了会让
        # "算身份用的配置"与"实际切分用的配置"不是同一份 —— 那是本文件
        # 反复强调不能出现的情况。
        #
        # 填进去的是**字符串**（模型路径）不是 tokenizer 对象：这份配置
        # 会原样落进 document_chunk.chunker_config，是 chunk 身份的一部分，
        # 必须可序列化且稳定。
        if (
                config.get("length_function") == "model"
                and not config.get("length_tokenizer")
        ):
            config["length_tokenizer"] = _tokenizer_id_for(
                runtime.context.registry,
                c.retrieval_type,
            )

        # 切分产出的身份 —— 在计划阶段算好，随分支带下去：
        # 扇出（切几次）与产物归集（chunk_refs 的键）都要它，而下游的
        # 路由函数只拿得到 state，没有注册表，算不出身份。
        #
        # 配置在这里被**归一化**（默认值补齐、字段集合稳定），因为
        # chunker 会把它原样落进 document_chunk.chunker_config，而查询
        # 侧反查归属时用的是同一个 chunker_identity()。两边必须落在同一
        # 形式上，否则 retrieval 找不到自己的 chunk，且不报错。
        #
        # ⚠️ 这一点会让"chunker_name 写错"从"某个分支失败"变成"整个计划
        # 失败"：身份算不出来就没法决定切几次。chunker 名字是服务端配置、
        # 对所有同名的分支一致，不是那种"一个后端挂了其余照跑"的运行时
        # 故障，所以在计划阶段显式失败是合适的。
        identity = chunker_identity(
            runtime.context.registry,
            c.chunker_name,
            c.chunker_config,
        )

        branches.append({
            "document_retrieval_definition_id": str(
                c.document_retrieval_definition_id
            ),
            "retrieval_type": c.retrieval_type,
            "chunker_name": c.chunker_name,
            "chunker_config": identity.config,
            "chunker_digest": identity.digest,
            "retriever_name": c.retriever_name,
            "retriever_config": c.retriever_config,
            "storage_backend": c.storage_backend,
            "storage_config": c.storage_config,
            "document_index_binding_id": (
                None
                if c.document_retrieval_definition_id
                not in index_by_retrieval
                else str(
                    index_by_retrieval[
                        c.document_retrieval_definition_id
                    ].document_index_binding_id
                )
            ),
            "index_identifier": (
                None
                if c.document_retrieval_definition_id
                not in index_by_retrieval
                else index_by_retrieval[
                    c.document_retrieval_definition_id
                ].index_identifier
            ),
        })

    return branches


def _tokenizer_id_for(
        registry: PipelineRegistry,
        retrieval_type: str,
) -> str:
    """
    本分支 embedder 用来度量输入长度的**词表标识**（模型名或目录）。

    取不到就**显式失败**，不回落到 tiktoken：两者在中文下差 16%~33%，
    静默回落等于让人以为"已经按模型上限量过了"，而 chunk 尾巴被模型
    静默截断正是 length_function="model" 要消除的那个问题。

    在计划阶段失败也与本文件既有的立场一致（chunker_name 写错同样
    在这里炸）：配置错了就该在计划时暴露，而不是等某个分支产出一批
    长度不对的 chunk 之后才发现。
    """

    kind = RETRIEVAL_EMBEDDING_KIND.get(retrieval_type)

    if kind is None:
        raise ValueError(
            f"retrieval_type={retrieval_type!r} 不需要向量，但它配了 "
            "length_function='model' —— 没有 embedder 就取不到词表，"
            "请改用 char 或 token，或指明 length_tokenizer。"
        )

    embedder = registry.embedder_for(kind)

    if embedder is None:
        raise ValueError(
            f"retrieval_type={retrieval_type!r} 配了 "
            "length_function='model'，但向量形态 "
            f"{kind.value} 没有已注册的 embedder，取不到词表。"
        )

    tokenizer_id = getattr(type(embedder), "tokenizer_id", None)

    if not tokenizer_id:
        raise ValueError(
            f"embedder {embedder.get_key()} 没有声明 tokenizer_id，"
            "无法按模型自己的词表度量长度。请改用 char / token，"
            "或在该 embedder 上声明 tokenizer_id。"
        )

    return tokenizer_id
