"""
查询链路冒烟工具（开发用）。

放在 app/agent/ 而不是 app/services/ 下，与 pipeline_smoke.py 同理：
这里只是命令行工具，不该被应用启动流程加载。

用法
----
    python -m app.agent.retrieve_smoke

不连数据库、不连 ES：用内存假实现跑通
DocumentRetrieveServiceImpl.retrieve_documents 的完整路径，覆盖
单 DTD / 多 DTD 公共链 / override 继承 / 过滤 / 裁剪 / 校验。

它**不**验证真实召回质量 —— 那要连上 ES 与真实数据。
它验证的是编排：配置怎么解析、分数怎么归一化、回表怎么补、
哪些候选会被剔除、响应字段对不对。
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone

from app.services.document.application.document_retrieve_service_impl import (
    DocumentRetrieveServiceImpl,
)
from app.services.document.domain.entities import (
    Document,
    DocumentChunk,
    DocumentRetrievalDefinition,
    DocumentTypeDefinition,
    DocumentVersion,
)
from app.services.document.domain.schemas import (
    DocumentRetrieveRequest,
    IndexSearchResponse,
    IndexSearchResult,
)
from app.services.document.domain.schemas.enum import (
    DefinitionStatus,
    DocumentStatus,
    EmbeddingOutput,
    ModalityType,
    ProcessingStatus,
    RetrievalType,
    StorageBackend,
    VersionStatus,
)

NOW = datetime.now(timezone.utc)


# ============================================================
# 假仓储
# ============================================================


class FakeTypes:
    def __init__(self, definitions):
        self._by_id = {
            d.document_type_definition_id: d
            for d in definitions
        }

    async def get_many(self, ids):
        return [
            self._by_id[i] for i in ids if i in self._by_id
        ]


class FakeRetrievals:
    def __init__(self, retrievals):
        self._all = retrievals

    async def list_all_by_type_definition(
            self, document_type_definition_id, *,
            include_deleted=False,
    ):
        return [
            r for r in self._all
            if r.document_type_definition_id
            == document_type_definition_id
        ]


class FakeChunks:
    def __init__(self, chunks):
        self._by_id = {c.document_chunk_id: c for c in chunks}

    async def list_by_ids(self, ids):
        return [
            self._by_id[i] for i in ids if i in self._by_id
        ]


class FakeVersions:
    def __init__(self, versions, active_by_type=None):
        self._by_id = {
            v.document_version_id: v for v in versions
        }
        # {document_type_definition_id: [生效版本 id]}
        self._active = active_by_type or {}

    async def get_many(self, ids):
        return [
            self._by_id[i] for i in ids if i in self._by_id
        ]

    async def list_active_ids_by_type_definitions(self, type_ids):
        return [
            version_id
            for type_id in type_ids
            for version_id in self._active.get(type_id, [])
        ]


class FakeDocuments:
    def __init__(self, documents):
        self._by_id = {d.document_id: d for d in documents}

    async def get_many(self, ids):
        return [
            self._by_id[i] for i in ids if i in self._by_id
        ]


class FakeBackend:
    """按 index_identifier 返回预置结果，并记录收到的完整请求。"""

    def __init__(self, results_by_container, failing=()):
        self._results = results_by_container
        #: 这些容器一查就抛 —— 模拟 Milvus 在恢复 / ES 超时。
        self._failing = set(failing)
        self.searches = []
        self.requests = []

    async def search(self, request):
        self.searches.append(request.index_identifier)
        self.requests.append(request)

        if request.index_identifier in self._failing:
            raise RuntimeError(
                f"{request.index_identifier}: backend unavailable"
            )
        hits = self._results.get(
            request.index_identifier, []
        )
        return IndexSearchResponse(
            results=[
                IndexSearchResult(
                    document_chunk_id=chunk_id,
                    score=score,
                    retrieval_source=request.index_identifier,
                    metadata={"page_number": 1},
                    # 每条片段带上来源容器名，便于断言"多路的高亮都留下来了"。
                    highlights=[
                        f"<em>{request.index_identifier}</em>"
                    ],
                )
                for chunk_id, score in hits
            ]
        )


class FakeEmbedder:
    """
    假的 embedder，**按形态返回不同的形状**。

    只记录"被要求向量化什么文本"，不真的调模型 —— 本工具要验证的是
    **编排**（查询侧有没有按形态拿到正确的表示、有没有放进正确的字段），
    不是模型质量。

    形状必须区分开：稠密是 list[float]，稀疏是 dict[int, float]。
    两边都返回 list 的话，"稀疏被塞进 query_vector" 这类错误就漏掉了。
    """

    def __init__(self, kind):
        self.kind = kind
        self.queries = []

    async def embed_query(self, text, *, config):
        self.queries.append((text, config))

        if self.kind is EmbeddingOutput.SPARSE:
            return {1024: 0.31, 88: 0.7}

        return [0.5] * 8


class FakeRegistry:
    """按形态分发 embedder —— 与真实注册表的 embedder_for 同语义。"""

    def __init__(self, backend, embedders=None):
        self._backend = backend
        self._embedders = embedders or {}

    def index_backend(self, backend_type):
        return self._backend

    def embedder_for(self, kind):
        return self._embedders.get(kind)


# ============================================================
# 场景
# ============================================================


def build(
        *,
        with_embedder=True,
        with_active=True,
        failing=(),
        dense_also_hits_c2=False,
):
    dtd_a = uuid.uuid4()
    dtd_b = uuid.uuid4()
    doc_a = uuid.uuid4()
    doc_b = uuid.uuid4()
    ver_a = uuid.uuid4()
    ver_b = uuid.uuid4()
    ret_a = uuid.uuid4()
    ret_b = uuid.uuid4()
    c1, c2, c3 = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()

    types = [
        DocumentTypeDefinition(
            document_type_definition_id=dtd_a, type_key="a",
            name="A", modality_type=ModalityType.TEXT,
            created_at=NOW, updated_at=NOW,
            fusion_strategy="rrf", fusion_config={"top_k": 10},
        ),
        DocumentTypeDefinition(
            document_type_definition_id=dtd_b, type_key="b",
            name="B", modality_type=ModalityType.TEXT,
            created_at=NOW, updated_at=NOW,
        ),
    ]

    retrievals = [
        DocumentRetrievalDefinition(
            retrieval_definition_id=ret_a,
            document_type_definition_id=dtd_a,
            retrieval_type=RetrievalType.KEYWORD,
            storage_backend=StorageBackend.ELASTICSEARCH,
            status=DefinitionStatus.ACTIVE,
            created_at=NOW, updated_at=NOW,
        ),
        DocumentRetrievalDefinition(
            retrieval_definition_id=ret_b,
            document_type_definition_id=dtd_b,
            retrieval_type=RetrievalType.KEYWORD,
            storage_backend=StorageBackend.ELASTICSEARCH,
            status=DefinitionStatus.ACTIVE,
            created_at=NOW, updated_at=NOW,
        ),
        # dtd_a 再加两路向量召回：验证查询表示确实被算出来、并落到了
        # **正确的字段**上（稠密走 query_vector，稀疏走 sparse_query）。
        # 假后端对这两个容器返回空，因此不影响上面那些关于结果条数的断言。
        DocumentRetrievalDefinition(
            retrieval_definition_id=uuid.uuid4(),
            document_type_definition_id=dtd_a,
            retrieval_type=RetrievalType.DENSE_VECTOR,
            storage_backend=StorageBackend.MILVUS,
            status=DefinitionStatus.ACTIVE,
            retriever_config={"batch_size": 4},
            created_at=NOW, updated_at=NOW,
        ),
        DocumentRetrievalDefinition(
            retrieval_definition_id=uuid.uuid4(),
            document_type_definition_id=dtd_a,
            retrieval_type=RetrievalType.SPARSE_VECTOR,
            storage_backend=StorageBackend.ELASTICSEARCH,
            status=DefinitionStatus.ACTIVE,
            retriever_config={"sparse_flag": 1},
            created_at=NOW, updated_at=NOW,
        ),
    ]

    # chunk 的归属现在是 chunker 身份，不是 retrieval ——
    # 见 DocumentChunk 的说明。
    chunks = [
        DocumentChunk(
            document_chunk_id=c1, document_version_id=ver_a,
            chunker_key="TEXT_TXT_RECURSIVE_CHUNKER",
            chunker_config={"chunk_size": 512},
            chunk_index=0, content="比亚迪第一大股东是王传福。",
        ),
        DocumentChunk(
            document_chunk_id=c2, document_version_id=ver_a,
            chunker_key="TEXT_TXT_RECURSIVE_CHUNKER",
            chunker_config={"chunk_size": 512},
            chunk_index=1, content="第二大股东是香港中央结算。",
        ),
        DocumentChunk(
            document_chunk_id=c3, document_version_id=ver_b,
            chunker_key="TEXT_TXT_RECURSIVE_CHUNKER",
            chunker_config={"chunk_size": 512},
            chunk_index=0, content="王传福持有比亚迪股权。",
        ),
    ]

    versions = [
        DocumentVersion(
            document_version_id=ver_a, document_id=doc_a,
            file_id=uuid.uuid4(), version_number=1,
            version_status=VersionStatus.ACTIVE,
            processing_status=ProcessingStatus.COMPLETED,
            created_at=NOW, updated_at=NOW,
        ),
        DocumentVersion(
            document_version_id=ver_b, document_id=doc_b,
            file_id=uuid.uuid4(), version_number=1,
            version_status=VersionStatus.ACTIVE,
            processing_status=ProcessingStatus.COMPLETED,
            created_at=NOW, updated_at=NOW,
        ),
    ]

    documents = [
        Document(
            document_id=doc_a,
            document_type_definition_id=dtd_a,
            status=DocumentStatus.ACTIVE,
            created_at=NOW, updated_at=NOW,
        ),
        Document(
            document_id=doc_b,
            document_type_definition_id=dtd_b,
            status=DocumentStatus.ACTIVE,
            created_at=NOW, updated_at=NOW,
        ),
    ]

    keyword = "document_instance_keyword"

    backend = FakeBackend(
        {
            keyword: [(c1, 9.5), (c2, 4.0)],
            # 默认空：让稠密那一路不影响上面那些既有断言。
            # 打开开关时让 c2 也被稠密那一路命中，构造
            # "同一 chunk 被两路命中"的场景。
            "document_instance_dense_vector": (
                [(c2, 0.9)] if dense_also_hits_c2 else []
            ),
        },
        failing=failing,
    )

    if with_embedder:
        embedders = {
            EmbeddingOutput.DENSE: FakeEmbedder(
                EmbeddingOutput.DENSE
            ),
            EmbeddingOutput.SPARSE: FakeEmbedder(
                EmbeddingOutput.SPARSE
            ),
        }
    else:
        embedders = {}

    service = DocumentRetrieveServiceImpl(
        type_definition_repository=FakeTypes(types),
        retrieval_definition_repository=FakeRetrievals(
            retrievals
        ),
        chunk_repository=FakeChunks(chunks),
        version_repository=FakeVersions(
            versions,
            # only_active 默认 true，所以要给出"当前生效版本"，
            # 否则会被当成"这些 DTD 下没有生效版本"而短路成空结果。
            active_by_type=(
                {dtd_a: [ver_a], dtd_b: [ver_b]}
                if with_active
                else {}
            ),
        ),
        document_repository=FakeDocuments(documents),
        pipeline_registry=FakeRegistry(backend, embedders),
    )

    return service, backend, embedders, {
        "dtd_a": dtd_a, "dtd_b": dtd_b,
        "ver_a": ver_a, "ver_b": ver_b,
        "c1": c1, "c2": c2, "c3": c3,
        "doc_a": doc_a, "doc_b": doc_b,
    }


def check(label, condition):
    print(
        f"  [{'OK ' if condition else 'FAIL'}] {label}"
    )
    if not condition:
        raise AssertionError(label)


async def main() -> None:

    service, backend, embedders, ids = build()

    # ---- 单 DTD ----
    print("单 DTD：")
    response = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
        )
    )
    check("scope=single", response.retrieve_scope.value == "single")
    check("mode=dtd", response.retrieve_processor_mode.value == "dtd")
    check("source=dtd", response.retrieve_processor.source.value
          == "document_type_definition")
    check("2 条结果", len(response.results) == 2)
    check("rank 连续",
          [r.rank for r in response.results] == [1, 2])
    check("分数降序",
          response.results[0].score >= response.results[1].score)
    check("分数归一化",
          response.results[0].score == 1.0)
    check("正文已回表",
          response.results[0].content == "比亚迪第一大股东是王传福。")
    check("document_id 已回填",
          response.results[0].document_id == ids["doc_a"])
    check("pipeline_keys=[keyword]",
          response.results[0].pipeline_keys == ["keyword"])
    check("DTD 的 top_k=10 生效",
          response.retrieve_processor.fusioner.top_k == 10)

    # ---- override：只改 cutter.top_k ----
    print("单 DTD + override：")
    response = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieve_processor={"cutter": {"top_k": 1}},
        )
    )
    check("source=runtime_override",
          response.retrieve_processor.source.value
          == "runtime_override")
    check("裁剪到 1 条", len(response.results) == 1)
    check("DTD 的 fusioner.top_k 仍继承为 10",
          response.retrieve_processor.fusioner.top_k == 10)

    # ---- 多 DTD：走公共链 ----
    print("多 DTD：")
    response = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[
                ids["dtd_a"], ids["dtd_b"],
            ],
            input_text="第一大股东",
        )
    )
    check("scope=multi", response.retrieve_scope.value == "multi")
    check("mode=common_chain",
          response.retrieve_processor_mode.value == "common_chain")
    check("source=common_retrieve_processor_chain",
          response.retrieve_processor.source.value
          == "common_retrieve_processor_chain")
    check("公共链用系统默认 cutter.top_k=10",
          response.retrieve_processor.cutter.top_k == 10)
    # dtd_b 没有任何召回（假后端只配了 dtd_a 的容器结果，
    # 但两路容器名相同 -> 都会命中），因此这里只检查不报错
    check("有结果", len(response.results) >= 1)

    # ---- filter：document_version_ids 过滤 ----
    print("过滤：")
    response = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieve_processor={
                "filter": {
                    "document_version_ids": [str(uuid.uuid4())]
                }
            },
        )
    )
    check("不存在的版本 -> 结果为空",
          response.results == [])

    # ---- filter：metadata 精确匹配 ----
    response = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieve_processor={
                "filter": {"metadata": {"page_number": 1}}
            },
        )
    )
    check("metadata 命中 -> 保留", len(response.results) == 2)

    response = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieve_processor={
                "filter": {"metadata": {"page_number": 99}}
            },
        )
    )
    check("metadata 不命中 -> 过滤掉", response.results == [])

    # ---- cutter.max_content_chars ----
    response = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieve_processor={"cutter": {"max_content_chars": 5}},
        )
    )
    check("正文被截断到 5 字",
          response.results[0].content == "比亚迪第一")

    # ---- DTD 不存在 -> NotFoundError ----
    print("校验：")
    from app.common.exceptions import NotFoundError
    try:
        await service.retrieve_documents(
            DocumentRetrieveRequest(
                document_type_definition_ids=[uuid.uuid4()],
                input_text="x",
            )
        )
        check("不存在的 DTD 应报错", False)
    except NotFoundError:
        check("不存在的 DTD -> NotFoundError", True)

    check("召回确实走了 keyword 容器",
          backend.searches
          and backend.searches[0] == "document_instance_keyword")

    # ---- 向量召回：查询侧要先算查询表示 ----
    #
    # 上面已经跑过多次查询，所以这里只断言"每次都做对了"，
    # 不断言总次数（那种断言会被前面新增的场景轻易打破）。
    print("稠密向量召回：")
    dense_embedder = embedders[EmbeddingOutput.DENSE]

    check("查询文本被向量化过（每次稠密召回一次）",
          len(dense_embedder.queries) > 0)
    check(
        "向量化的是查询文本，不是别的",
        {text for text, _ in dense_embedder.queries}
        == {"第一大股东"},
    )
    check(
        "配置与写入期同源（retriever_config）",
        all(
            config == {"batch_size": 4}
            for _, config in dense_embedder.queries
        ),
    )

    dense_requests = [
        request for request in backend.requests
        if request.index_identifier
        == "document_instance_dense_vector"
    ]
    check("稠密容器被查过", len(dense_requests) > 0)
    check(
        "每次稠密召回都带上了 query_vector",
        all(
            request.query_vector == [0.5] * 8
            for request in dense_requests
        ),
    )

    # ---- 稀疏向量召回：落到 sparse_query，不是 query_vector ----
    print("稀疏向量召回：")
    sparse_embedder = embedders[EmbeddingOutput.SPARSE]

    check("查询文本被稀疏编码过",
          len(sparse_embedder.queries) > 0)
    check(
        "稀疏那路用的是它自己的配置",
        all(
            config == {"sparse_flag": 1}
            for _, config in sparse_embedder.queries
        ),
    )

    sparse_requests = [
        request for request in backend.requests
        if request.index_identifier
        == "document_instance_sparse_vector"
    ]
    check("稀疏容器被查过", len(sparse_requests) > 0)
    check(
        "稀疏表示落在 sparse_query（整数维度索引）",
        all(
            request.sparse_query == {1024: 0.31, 88: 0.7}
            for request in sparse_requests
        ),
    )
    check(
        "稀疏那路**没有**把 dict 塞进 query_vector",
        all(
            request.query_vector is None
            for request in sparse_requests
        ),
    )

    # ---- keyword：两种表示都不带 ----
    print("keyword 路：")
    keyword_requests = [
        request for request in backend.requests
        if request.index_identifier
        == "document_instance_keyword"
    ]
    check("keyword 那一路不带任何查询表示",
          all(
              request.query_vector is None
              and request.sparse_query is None
              for request in keyword_requests
          ))

    # ---- 该形态没有 embedder 时：只跳过那一路，不是整个查询失败 ----
    #
    # 这是"向量后端没配好"与"关键词那路照常能查"的边界：
    # 稠密那路应当明确跳过（并留 warn 日志），keyword 的结果照常返回。
    print("缺 embedder：")
    bare_service, bare_backend, _, bare_ids = build(
        with_embedder=False
    )

    bare_response = await bare_service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[bare_ids["dtd_a"]],
            input_text="第一大股东",
        )
    )
    check("keyword 的结果照常返回",
          len(bare_response.results) == 2)
    check(
        "稠密容器完全没被查",
        "document_instance_dense_vector"
        not in bare_backend.searches,
    )

    # ---- only_active：默认只检索当前生效版本 ----
    print("only_active：")

    # 上面所有查询都没传 only_active，也就是走的默认值。
    check(
        "默认每条召回都下推了生效版本过滤",
        all(
            request.document_version_ids is not None
            for request in backend.requests
        ),
    )
    check(
        "过滤集合就是各 DTD 的生效版本",
        {
            version_id
            for request in backend.requests
            for version_id in request.document_version_ids
        }
        == {ids["ver_a"], ids["ver_b"]},
    )

    backend.requests.clear()
    await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            only_active=False,
        )
    )
    check(
        "only_active=false 时不过滤（版本集合为 None）",
        all(
            request.document_version_ids is None
            for request in backend.requests
        ),
    )

    # 生效版本集合为空 -> 短路，**不能**把空集合当"不过滤"传给后端
    empty_service, empty_backend, _, empty_ids = build(
        with_active=False
    )
    empty_resp = await empty_service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[empty_ids["dtd_a"]],
            input_text="第一大股东",
        )
    )
    check("没有生效版本时结果为空", empty_resp.results == [])
    check("没有生效版本时**根本没查索引**",
          empty_backend.requests == [])

    # ---- 一路后端挂了：跳过该路，但绝不能连带失败或静默 ----
    print("后端故障降级：")
    down_service, down_backend, _, down_ids = build(
        failing=("document_instance_dense_vector",)
    )

    down_resp = await down_service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[down_ids["dtd_a"]],
            input_text="第一大股东",
        )
    )

    check("整个请求没有失败", down_resp is not None)
    check(
        "ES 那两路的结果照常返回",
        len(down_resp.results) == 2,
    )
    check(
        "挂掉的那一路出现在 degraded_routes",
        len(down_resp.degraded_routes) == 1
        and down_resp.degraded_routes[0].retrieval_type
        is RetrievalType.DENSE_VECTOR,
    )
    check(
        "reason 带上了后端异常信息",
        "backend unavailable"
        in down_resp.degraded_routes[0].reason,
    )
    check(
        "正常时 degraded_routes 为空",
        response.degraded_routes == [],
    )

    # ---- 多路命中同一 chunk：融合后要报出**全部**路 ----
    #
    # 这是融合唯一真正有价值的产出：各路的分数尺度不可比
    # （BM25 是几十、余弦是 0..1），但"被几路同时选中"是可比的。
    # 只报一路等于白做融合。
    print("多路命中同一 chunk：")

    multi_service, _, _, multi_ids = build(
        # 同一个 c2 既被 keyword 命中，也被 dense_vector 命中。
        dense_also_hits_c2=True,
    )

    multi_resp = await multi_service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[multi_ids["dtd_a"]],
            input_text="第一大股东",
        )
    )

    by_content = {
        result.document_chunk_id: result
        for result in multi_resp.results
    }

    check(
        "被两路命中的 chunk 仍然只出现一条（融合掉了）",
        len(multi_resp.results) == 2,
    )
    check(
        "它报出了**全部**命中它的路，而不是首条那一路",
        by_content[multi_ids["c2"]].pipeline_keys
        == ["keyword", "dense_vector"],
    )
    check(
        "只被一路命中的 chunk 仍是一路",
        by_content[multi_ids["c1"]].pipeline_keys
        == ["keyword"],
    )
    check(
        "两路命中的那条排在前面（RRF 累加了两次）",
        multi_resp.results[0].document_chunk_id
        == multi_ids["c2"],
    )

    # 高亮：同一条被两路命中时，**两路的片段都要留下** ——
    # 只保留首次命中那一路等于把其余路的标记丢了。
    print("高亮：")
    check(
        "两路命中的 chunk 带上了两路各自的高亮片段",
        by_content[multi_ids["c2"]].highlights
        == {
            "keyword": ["<em>document_instance_keyword</em>"],
            "dense_vector": [
                "<em>document_instance_dense_vector</em>"
            ],
        },
    )
    check(
        "只被一路命中的 chunk 只有那一路的片段",
        by_content[multi_ids["c1"]].highlights
        == {
            "keyword": ["<em>document_instance_keyword</em>"]
        },
    )

    # ---- retrieval_types：定向召回，取交集 ----
    print("定向召回：")

    # dtd_a 配了 keyword / dense_vector / sparse_vector 三路。
    def containers_of(resp):
        return {
            key
            for result in resp.results
            for key in result.pipeline_keys
        }

    backend.requests.clear()
    only_keyword = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieval_types=["keyword"],
        )
    )
    check(
        "点名 keyword 时只查了 keyword 容器",
        {
            request.index_identifier
            for request in backend.requests
        }
        == {"document_instance_keyword"},
    )
    check("结果只来自 keyword",
          containers_of(only_keyword) == {"keyword"})

    backend.requests.clear()
    await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieval_types=["keyword", "dense_vector"],
        )
    )
    check(
        "点名两路就查两个容器（不含 sparse）",
        {
            request.index_identifier
            for request in backend.requests
        }
        == {
            "document_instance_keyword",
            "document_instance_dense_vector",
        },
    )

    # 取交集：dtd_a 没有 graph 那一路，点名它不会报错，也不会召出东西。
    backend.requests.clear()
    none_matched = await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
            retrieval_types=["graph"],
        )
    )
    check(
        "点名文档里没有的类型 -> 没有查到任何索引",
        backend.requests == [],
    )
    check("结果是空，而不是报错", none_matched.results == [])
    check(
        "被筛掉的路不进 degraded_routes（不是'本该跑却没跑'）",
        none_matched.degraded_routes == [],
    )

    # 不传 = 全部
    backend.requests.clear()
    await service.retrieve_documents(
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="第一大股东",
        )
    )
    check(
        "不传 retrieval_types 时三路全查",
        {
            request.index_identifier
            for request in backend.requests
        }
        == {
            "document_instance_keyword",
            "document_instance_dense_vector",
            "document_instance_sparse_vector",
        },
    )

    # 空数组被拒（而不是被当成"全部"）
    try:
        DocumentRetrieveRequest(
            document_type_definition_ids=[ids["dtd_a"]],
            input_text="x",
            retrieval_types=[],
        )
        check("空数组应当被拒绝", False)
    except Exception:
        check("retrieval_types=[] 被拒绝（422）", True)

    print("\n全部通过。")


if __name__ == "__main__":
    asyncio.run(main())
