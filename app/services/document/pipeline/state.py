"""
Document 处理流水线的图状态。

用 TypedDict 而非 Pydantic
--------------------------
    - Send 的 payload 是 dict、绕过 schema 校验，Pydantic state
      会给出假的校验安全感
    - API 边界已经是 Pydantic（APIModel, extra="forbid"），
      在图层重复一次只增加每次 channel 写入的校验开销

Pydantic 只用在 API DTO 与 domain entity。

两条实测得出的硬约束
--------------------
1. **state 里未声明的键会被 LangGraph 丢弃。**
   写入未声明字段不会报错，只是静默消失。

2. **Send 的 payload 会「取代」state，不是与父 state 合并。**
   扇出分支节点看到的 state **只有** payload 里的键，
   父 state 的字段一个都看不到。
   因此扇出所需的一切必须在 fanout.py 里显式塞进 Send payload
   —— 这就是下面 BranchInput 存在的原因。

state 里绝不放正文
------------------
checkpointer 按 **channel 版本**存储：每个 super-step 只写
**发生变化**的 channel，不是整个 state 重写（实测确认）。

但承载正文的 channel 有 5 个标量 *_ref，加 3 个 dict
（chunk_refs / enrich_refs / vector_refs，各含 N 项），
而且 **Send 的 payload 是每分支一份**。正文一旦进 state，
每一个这样的 channel 都要各存一份完整正文。

实测对比（432 KB 的文档）：

    用 ArtifactRef    state 约 8 KB，与文档大小无关，写入约 65 KB
    正文进 state      约 4.8 MB，写入约 14 MB（相差约 200 倍）

所以 *_ref 全部是 ArtifactRef（只含 file_id / checksum /
计数 / preview），正文落在 File 模块里。
assert_checkpoint_safe 会在有人破坏这条规则时直接报错。
"""

from __future__ import annotations

from operator import add
from typing import Annotated, Any, TypedDict

__all__ = [
    "ArtifactRef",
    "BranchInput",
    "DocumentPipelineState",
    "RetrievalBranch",
    "RetrievalResult",
    "merge_dicts",
]


def merge_dicts(
        left: dict[str, Any] | None,
        right: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    并发分支各自往同一个 dict 写**不同键**时的合并器。

    扇出分支按 retrieval_definition_id 归集自己的产物，
    键天然不冲突，因此浅合并即可。

    不能用 operator.add —— 那是列表的 reducer。
    """
    return {**(left or {}), **(right or {})}


# ============================================================
# 引用
# ============================================================


class ArtifactRef(TypedDict):
    """
    指向一份中间产物的 checkpoint-safe 指针。

    产物就是 File 模块里的一条 File，因此身份是 **file_id**：

        file_id  ->  FileService.get_file_content()  ->  bytes

    刻意**不**把 storage_key 放进 state：它是存储层的实现细节，
    换后端（本地 → S3）时会变，而 file_id 不变。代价是取回时
    多一次主键查询，可接受。

    preview **必须**截断（上限见 instrumentation.MAX_PREVIEW_CHARS）。
    它是唯一会进 state 的内容 —— 正因为如此，state 的大小与
    文档大小解耦：正文 1 KB 还是 100 MB，state 都是同一量级。
    """

    file_id: str
    checksum: str
    kind: str
    mime_type: str | None
    byte_size: int
    char_count: int | None
    token_count: int | None
    preview: str


# ============================================================
# 扇出
# ============================================================


class RetrievalBranch(TypedDict):
    """
    一个扇出单元：一个 DocumentRetrievalDefinition 的执行配置。

    由 fanout.py 从库里读出并填好，随 Send payload 传给分支节点。
    """

    document_retrieval_definition_id: str
    retrieval_type: str

    chunker_name: str | None
    #: **归一化之后**的配置（见 chunker/identity.py）。chunker 会把它
    #: 原样落到 document_chunk.chunker_config，查询侧靠它反查归属。
    chunker_config: dict[str, Any]

    #: 本路切分产出的身份摘要 —— 由 metadata_extraction 算好带下来。
    #:
    #: 为什么在计划阶段算：扇出（决定切几次）与产物归集（chunk_refs
    #: 的键）都要它，而下游的路由函数只拿得到 state，拿不到注册表 ——
    #: 身份要过各 chunker 实现的 config_model，没注册表算不出来。
    chunker_digest: str

    retriever_name: str | None
    retriever_config: dict[str, Any]

    storage_backend: str
    storage_config: dict[str, Any]

    document_index_binding_id: str | None
    index_identifier: str | None


class BranchInput(TypedDict):
    """
    Send payload 的结构 —— 也就是扇出分支节点能看到的**全部** state。

    ⚠️ Send 的 payload **取代** state，且只作用于**那一个节点**
    （两者均已实测）：

        - 分支节点看不到父 state 的任何字段
        - 分支节点的下游边，看到的是**父状态**而不是分支状态

    后一条决定了本流水线的扇出形状：4 个扇出阶段不能靠普通边
    依次串联，必须在每两个阶段之间插入一个"路由屏障"节点 ——
    它跑一次、读到全部分支汇合后的父状态，再把下一批 Send 发出去。
    详见 fanout.py。

    因此**每个**扇出阶段调用的路由函数，都要从父状态里按
    retrieval_definition_id 把该分支的上游产物取出来，塞进
    upstream_ref —— 分支节点自己拿不到父状态。

    ⚠️ 父 state 中未声明的键会被 LangGraph 丢弃。
    漏一个字段，分支节点就会拿到 KeyError，而报错不会告诉你
    是因为 Send 的语义。
    """

    branch: RetrievalBranch

    document_id: str
    document_version_id: str
    document_type_definition_id: str

    #: 阶段 5 清洗后的正文引用 —— 阶段 7 切分的输入。
    clean_ref: ArtifactRef

    #: 阶段 6 抽出的元数据引用，会写进每个 chunk 的 metadata。
    extract_ref: ArtifactRef | None

    #: **本分支在上一阶段产出的东西**，由路由函数按 rid 注入：
    #:     阶段 7  chunking          -> None（用 clean_ref）
    #:     阶段 8  chunk_enrichment  -> chunk_refs[rid]
    #:     阶段 9  embedding         -> enrich_refs[rid]
    #:     阶段 10 indexing          -> vector_refs[rid]
    upstream_ref: ArtifactRef | None

    #: 该分支的最终结果，经父状态的 reducer 汇总（阶段 10 写）。
    retrieval_results: Annotated[
        list["RetrievalResult"], add
    ]


class RetrievalResult(TypedDict):
    """
    一个扇出分支的最终结果。

    经 operator.add reducer 汇总回主 state，finalize 节点据此
    写 DocumentVersionRetrievalBinding 并聚合出版本级状态。
    """

    document_retrieval_definition_id: str
    document_index_binding_id: str | None

    status: str

    chunk_count: int
    token_count: int
    embedding_dimensions: int | None

    error_type: str | None
    error_message: str | None


# ============================================================
# 主状态
# ============================================================


class DocumentPipelineState(TypedDict):
    """
    主图状态。

    阶段 1-6 线性跑一次，阶段 7-10 按 RetrievalDefinition 扇出。
    """

    # ---- identity，由 ingestion 写一次，此后不变 ----
    document_id: str
    document_version_id: str
    document_type_definition_id: str
    tenant_id: str | None
    file_id: str
    version_number: int

    # ---- 运行快照（不可变）----
    pipeline_config: dict[str, Any]
    analysis_mode: str

    # ---- 阶段 1-6 产物：只放引用 ----
    source_ref: ArtifactRef | None
    detected_format: dict[str, Any] | None
    parse_ref: ArtifactRef | None
    normalize_ref: ArtifactRef | None
    clean_ref: ArtifactRef | None
    extract_ref: ArtifactRef | None

    # ---- 扇出计划，由 metadata_extraction 写一次 ----
    #
    # 必须是 state 字段：每两个扇出阶段之间的路由屏障都要重新
    # 读它才能构造下一批 Send。路由函数只会拿到状态，拿不到别的。
    branches: list[RetrievalBranch]

    # ---- 各扇出阶段按 retrieval_definition_id 归集的产物 ----
    #
    # 分支节点写不了父状态（Send 的 payload 只作用于自己那一个节点），
    # 因此各阶段把产物写进这些**父状态通道**，由 reducer 合并；
    # 下一阶段的路由屏障再按 rid 取出来注入 payload。
    chunk_refs: Annotated[dict[str, ArtifactRef], merge_dicts]
    enrich_refs: Annotated[dict[str, ArtifactRef], merge_dicts]
    vector_refs: Annotated[dict[str, ArtifactRef], merge_dicts]

    # ---- 阶段 10 汇总的最终结果（并发写通道，reducer: add）----
    retrieval_results: Annotated[list[RetrievalResult], add]

    # ---- 控制 ----
    stage_errors: Annotated[list[dict[str, Any]], add]
    status: str


#: 需要交给 assert_checkpoint_safe 做结构校验的引用字段。
#: 新增 *_ref 字段时必须同步加进来，否则护栏管不到它。
CHECKPOINT_REF_KEYS: tuple[str, ...] = (
    "source_ref",
    "parse_ref",
    "normalize_ref",
    "clean_ref",
    "extract_ref",
)
