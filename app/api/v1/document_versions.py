"""
处理结果页的查询接口：按 版本 × retrieval 查 Chunk。

规格见 app/services/document/document_version.yaml。

前端消费面
----------
    GET /document-versions/{vid}/retrievals                       tab 列表
    GET /document-versions/{vid}/retrievals/{rid}/chunks          某个 tab 的内容

页面怎么用
----------
    一个 Document -> N 个 DocumentVersion
      -> 每个版本下 N 个 retrieval（= N 个 tab）
        -> 每个 tab 下：该路的 chunk 列表 + 该路运行时用的 chunker 配置

两个接口是**成对**的：先拉 retrievals 拿到 rid 与 chunk 数，再按 tab
懒加载 chunks。

不分页 vs 分页
--------------
retrievals 不分页（数量级是个位数）；chunks 分页（可能上千条）。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import (
    APIRouter,
    Query,
)

from app.common.utils import to_http_error

from app.services.dependencies import Service
from app.services.document.application.version_chunk_service import (
    VersionChunkService,
)
from app.services.document.domain.schemas.version_chunks import (
    ChunkProjectionListResponse,
    ChunkProjectionRequest,
    DocumentChunkListResponse,
    DocumentVersionRetrievalListResponse,
)

router = APIRouter(
    tags=["Document Version Chunks"],
)


# ============================================================
# tab 列表
# ============================================================


@router.get(
    "/document-versions/{document_version_id}/retrievals",
    response_model=DocumentVersionRetrievalListResponse,
)
async def list_document_version_retrievals(
        document_version_id: UUID,
        service: VersionChunkService = Service[VersionChunkService],
):
    """
    某个版本实际参与处理的 retrieval 列表 —— 前端据此生成 N 个 tab。

    只返回**这个版本跑过的** retrieval，而不是该类型下当前配置的全部。

    每个条目给出 retrieval_type / storage_backend（tab 标签）、chunk_count
    （决定是否懒加载）、processing_status（这一路成功还是失败）。

    失败的 retrieval 也会出现 —— 否则前端会以为这一路不存在，
    而不是「它失败了」。

    ⚠️ chunker 当前恒为 null
    ------------------------
    流水线还没把"那时用的 chunker 配置"记下来。按契约前端应显示「未知」，
    **不要**回退去读当前定义 —— 那会把今天的配置冒充成当时的。
    """

    try:
        return await service.list_retrievals(document_version_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 某个 tab 的内容
# ============================================================


@router.get(
    "/document-versions/{document_version_id}/retrievals"
    "/{document_retrieval_definition_id}/chunks",
    response_model=DocumentChunkListResponse,
)
async def list_document_version_retrieval_chunks(
        document_version_id: UUID,
        document_retrieval_definition_id: UUID,
        preview_chars: int | None = Query(
            default=None,
            ge=1,
            le=4096,
            description=(
                "只返回每条 content 的前 N 个字符，并用 "
                "content_truncated 标记。留空返回完整内容。"
            ),
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=200),
        service: VersionChunkService = Service[VersionChunkService],
):
    """
    某个 tab（= 某版本下的某一路 retrieval）的 chunk 列表。

    (document_version_id, document_retrieval_definition_id) 已经**唯一
    定位**一组 chunk —— 表上的唯一约束就是这两个加 chunk_index ——
    所以不需要别的过滤条件。

    排序固定为 chunk_index 升序，翻页不会重复或漏块。

    preview_chars 是给「不分块」那个 chunker 留的防护：选中它时整篇
    正文就是一个 chunk，单条可能几十万字。

    该 retrieval 若没参与这个版本，返回 404 —— 而不是空列表，
    那样前端分不清「没跑」和「跑了但没切出东西」。
    """

    try:
        return await service.list_chunks(
            document_version_id,
            document_retrieval_definition_id,
            preview_chars=preview_chars,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 投影：这一条 chunk 当时写给索引的是什么
# ============================================================


@router.post(
    "/document-chunk-projections",
    response_model=ChunkProjectionListResponse,
)
async def get_chunk_projections(
        request: ChunkProjectionRequest,
        include_vector: bool = Query(
            default=False,
            description=(
                "是否返回完整向量。默认只给摘要（维度 / 非零项数 / 模长 / "
                "前几个值）—— 稠密向量 512 个 float，N 条一起返回体积不小。"
            ),
        ),
        service: VersionChunkService = Service[VersionChunkService],
):
    """
    按 chunk_id 批量查看**投影** —— 这些 chunk 当时写给索引的是什么。

    与 `/document-versions/.../chunks` 的区别
    ----------------------------------------
        那个给**原件**：这一条是什么（从 PG 的 document_chunk 读）
        这个给**投影**：它被变成什么样子写进索引（从 embedding 产物读）

    投影不落库（阶段 10 现算现发），唯一的痕迹是阶段 9 写出的产物，
    因此这里读的是产物而不是索引本身 —— 索引挂着也能看。代价是它
    反映"当时发出去的"，不是"索引里现在存着的"。

    ⚠️ 只有需要向量化的 retrieval 有投影产物
    ----------------------------------------
    keyword / document 那两路在阶段 9 是**透传**的，没有单独的投影产物。
    这种情况下 kind / fields 为 null，unavailable_reason 会说明原因。

    结果与入参**一一对应且同序**，取不到的那些带 unavailable_reason ——
    不抛错、也不悄悄丢掉，因为前端点了一条却什么都拿不到时，
    需要知道为什么。

    路径用 POST 是因为 chunk_id 可能一次传上百个；语义上它是只读的。
    """

    try:
        return await service.get_projections(
            request.document_chunk_ids,
            document_retrieval_definition_id=(
                request.document_retrieval_definition_id
            ),
            include_vector=include_vector,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
