"""
流水线冒烟工具（开发用）。

放在 app/agent/ 而不是 app/services/ 下，是为了让
initializer 的 pkgutil.walk_packages **永远不 import 它** ——
它只是命令行工具，不该被应用启动流程加载。

用法
----
    python -m app.agent.pipeline_smoke topology    # 拓扑自检 + 指纹
    python -m app.agent.pipeline_smoke stub-run    # 桩节点跑通全图

stub-run 用桩替换全部节点实现，因此**不需要**任何能力实现
（Parser / Chunker / IndexBackend 目前都还是空的）就能把
扇出拓扑、观测埋点、runner 终态推导整条链路验证一遍。

这也是目前唯一能端到端验证图的方式，因为真实节点全是 TODO。
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.runtime import Runtime

from app.infra.graph import (
    GraphRunner,
    NodeRunStatus,
    RunRequest,
    RunSnapshot,
    RunStatus,
    RunSummary,
    new_thread_id,
)
from app.services.document.pipeline import (
    DocumentPipelineContext,
    build_document_pipeline_graph,
    build_graph_definition,
)
from app.services.document.pipeline import graph as graph_module
from app.services.document.pipeline.catalog import (
    CONTROL_NODE_IDS,
    FANOUT_NODE_IDS,
    GRAPH_VERSION,
    LINEAR_NODE_IDS,
    NODE_SPECS,
)
from app.services.document.pipeline.state import ArtifactRef

__all__ = ["main"]

#: 十个阶段，顺序即 stage 表。拓扑自检拿它与 NODE_SPECS 比对。
EXPECTED_STAGES = (
    "ingestion",
    "format_detection",
    "parsing",
    "normalization",
    "cleaning",
    "metadata_extraction",
    "chunking",
    "chunk_enrichment",
    "embedding",
    "indexing",
)


# ============================================================
# topology
# ============================================================


def cmd_topology() -> int:
    """把"节点与阶段表 1:1"变成机器可校验的断言。"""

    stages = tuple(
        s.node_id for s in NODE_SPECS if s.group != "control"
    )

    problems: list[str] = []

    if stages != EXPECTED_STAGES:
        problems.append(
            f"阶段节点与阶段表不一致：\n  期望 {EXPECTED_STAGES}\n"
            f"  实际 {stages}"
        )

    sequences = [s.node_id for s in
                 sorted(NODE_SPECS, key=lambda s: s.sequence)]

    if [s.sequence for s in sorted(
            NODE_SPECS, key=lambda s: s.sequence
    )] != list(range(len(NODE_SPECS))):
        problems.append(
            "sequence 不连续或重复（必须从 0 递增，且永不重编号）"
        )

    if len(set(sequences)) != len(NODE_SPECS):
        problems.append("存在重复的 node_id")

    compiled = build_document_pipeline_graph().compile()
    node_names = set(compiled.get_graph().nodes)

    for spec in NODE_SPECS:
        if spec.node_id not in node_names:
            problems.append(
                f"NODE_SPECS 里的 {spec.node_id!r} 不在编译后的图中"
            )

    definition = build_graph_definition(
        compiled=compiled, context_factory=lambda _rc: None
    )

    print(f"图名        {definition.name}")
    print(f"版本        {GRAPH_VERSION}")
    print(f"指纹        {definition.topology_fingerprint}")
    print(f"线性节点    {sorted(LINEAR_NODE_IDS)}")
    print(f"扇出节点    {sorted(FANOUT_NODE_IDS)}")
    print(f"控制节点    {sorted(CONTROL_NODE_IDS)}")
    print(f"编译后节点  {len(node_names)}"
          "（含 __start__/__end__ 与 3 个路由屏障）")

    if problems:
        print("\n自检失败：")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print("\n拓扑自检通过：10 个阶段与阶段表 1:1，sequence 连续，"
          "全部节点已装配。")
    return 0


# ============================================================
# stub-run
# ============================================================


def _ref(kind: str) -> ArtifactRef:
    return ArtifactRef(
        file_id=str(uuid.uuid4()),
        checksum="abc",
        kind=kind,
        mime_type=None,
        byte_size=10,
        char_count=10,
        token_count=None,
        preview="smoke",
    )


class _MemoryRecorder:
    """把观测写进内存，供断言用。"""

    def __init__(self) -> None:
        self.runs: dict[Any, dict[str, Any]] = {}
        self.nodes: dict[tuple[str, str, int], Any] = {}

    async def create_run(self, snapshot: RunSnapshot) -> None:
        self.runs[snapshot.pipeline_run_id] = {"snapshot": snapshot}

    async def mark_run_running(self, run_id, *, started_at) -> None:
        pass

    async def finish_run(self, run_id, outcome) -> None:
        self.runs[run_id]["outcome"] = outcome

    async def upsert_node_run(self, update) -> None:
        key = (
            update.node_id,
            update.branch_key,
            update.attempt,
        )
        self.nodes[key] = update

    async def summarize(self, run_id) -> RunSummary:
        rows = list(self.nodes.values())
        return RunSummary(
            node_count=len(rows),
            failed_node_count=sum(
                1 for u in rows
                if u.status == NodeRunStatus.FAILED
            ),
            failed_node_ids=tuple(
                u.node_id for u in rows
                if u.status == NodeRunStatus.FAILED
            ),
            completed_node_count=sum(
                1 for u in rows
                if u.status == NodeRunStatus.COMPLETED
            ),
        )

    async def mark_orphans_failed(self, *, older_than) -> int:
        return 0


def _install_stubs(retrieval_ids: list[str]) -> None:
    """
    用桩替换全部节点实现。

    ⚠️ 桩的 runtime 参数**必须**叫 `runtime` 且有类型注解 ——
    LangGraph 靠「参数名 + 注解」两者同时匹配来决定注入。
    写成 `rt` 会得到 "missing 1 required positional argument"。
    """

    RT = Runtime[DocumentPipelineContext]

    async def ingestion(state, runtime: RT) -> dict:
        return {"source_ref": _ref("source_bytes")}

    async def format_detection(state, runtime: RT) -> dict:
        return {"detected_format": {"parser": "stub"}}

    async def parsing(state, runtime: RT) -> dict:
        return {"parse_ref": _ref("parsed_text")}

    async def normalization(state, runtime: RT) -> dict:
        return {"normalize_ref": _ref("normalized_text")}

    async def cleaning(state, runtime: RT) -> dict:
        return {"clean_ref": _ref("clean_text")}

    async def metadata_extraction(state, runtime: RT) -> dict:
        return {
            "extract_ref": _ref("extracted_metadata"),
            "branches": [
                {
                    "document_retrieval_definition_id": rid,
                    "retrieval_type": "dense_vector",
                    "chunker_name": "stub_chunker",
                    "chunker_config": {},
                    "retriever_name": "stub_retriever",
                    "retriever_config": {},
                    "storage_backend": "milvus",
                    "storage_config": {},
                    "document_index_binding_id": None,
                    "index_identifier": f"idx-{rid[:6]}",
                }
                for rid in retrieval_ids
            ],
        }

    def rid_of(state) -> str:
        return state["branch"]["document_retrieval_definition_id"]

    # ⚠️ 扇出桩**必须**自己调 node_span，不能像线性桩那样直接返回。
    #
    # 线性节点的行来自 callback 兜底（callbacks 覆盖 linear + control），
    # 但扇出节点**按设计排除在 callback 之外** —— callback 拿到的
    # metadata 里 N 个 Send 分支共享同一个节点名，无法区分是哪个
    # retrieval，硬写会产生无归属的幽灵行（见 catalog.CALLBACK_NODE_IDS）。
    #
    # 因此扇出的观测完全依赖节点自己调 node_span。桩不调 = 没有行，
    # 而真实节点是调的。
    from app.infra.graph import node_span

    from app.services.document.pipeline.catalog import NODE_SPECS

    specs = {s.node_id: s for s in NODE_SPECS}

    def _fanout(node_id, channel, produce):
        spec = specs[node_id]

        async def node(state, runtime: RT) -> dict:
            rid = rid_of(state)
            async with node_span(
                    runtime,
                    node_id=spec.node_id,
                    label=spec.label,
                    group=spec.group,
                    sequence=spec.sequence,
                    branch_scope_id=uuid.UUID(rid),
            ) as span:
                span.set_metrics(produce(rid)[1])
                return {channel: {rid: _ref(produce(rid)[2])}}

        return node

    chunking = _fanout(
        "chunking", "chunk_refs",
        lambda rid: (rid, {"chunk_count": 3}, "chunks"),
    )
    chunk_enrichment = _fanout(
        "chunk_enrichment", "enrich_refs",
        lambda rid: (rid, {"chunk_count": 3}, "chunks_enriched"),
    )
    embedding = _fanout(
        "embedding", "vector_refs",
        lambda rid: (rid, {"vector_count": 3, "embedding_dimensions": 1024},
                     "vectors"),
    )

    async def indexing(state, runtime: RT) -> dict:
        spec = specs["indexing"]
        rid = rid_of(state)
        async with node_span(
                runtime,
                node_id=spec.node_id,
                label=spec.label,
                group=spec.group,
                sequence=spec.sequence,
                branch_scope_id=uuid.UUID(rid),
        ) as span:
            span.set_metrics({"upserted": 3})
            return {
                "retrieval_results": [
                    {
                        "document_retrieval_definition_id": rid,
                        "document_index_binding_id": None,
                        "status": "completed",
                        "chunk_count": 3,
                        "token_count": 30,
                        "embedding_dimensions": 1024,
                        "error_type": None,
                        "error_message": None,
                    }
                ]
            }

    async def finalize_version(state, runtime: RT) -> dict:
        return {"status": "completed"}

    graph_module.NODE_FUNCTIONS.update({
        "ingestion": ingestion,
        "format_detection": format_detection,
        "parsing": parsing,
        "normalization": normalization,
        "cleaning": cleaning,
        "metadata_extraction": metadata_extraction,
        "chunking": chunking,
        "chunk_enrichment": chunk_enrichment,
        "embedding": embedding,
        "indexing": indexing,
        "finalize_version": finalize_version,
    })


async def _run_stub(retrieval_count: int) -> int:
    retrieval_ids = [
        str(uuid.uuid4()) for _ in range(retrieval_count)
    ]

    _install_stubs(retrieval_ids)

    recorder = _MemoryRecorder()

    document_id = uuid.uuid4()
    version_id = uuid.uuid4()
    type_id = uuid.uuid4()
    file_id = uuid.uuid4()
    run_id = uuid.uuid4()
    thread_id = new_thread_id("docver", version_id, run_id)

    compiled = build_document_pipeline_graph().compile(
        checkpointer=InMemorySaver()
    )

    definition = build_graph_definition(
        compiled=compiled,
        context_factory=lambda rc: DocumentPipelineContext(
            pipeline_run_id=rc.pipeline_run_id,
            recorder=rc.recorder,
            artifact_store=None,  # 桩不用产物存储
            registry=None,
        ),
    )

    snapshot = RunSnapshot(
        pipeline_run_id=run_id,
        thread_id=thread_id,
        graph_name=definition.name,
        graph_version=definition.version,
        topology_fingerprint=definition.topology_fingerprint,
        topology_snapshot=definition.topology_snapshot,
        subject_type="document_version",
        subject_id=version_id,
        subject_ref={
            "document_id": document_id,
            "document_type_definition_id": type_id,
            "file_id": file_id,
            "tenant_id": None,
        },
        trigger_source="smoke",
    )

    outcome = await GraphRunner(recorder=recorder).run(
        RunRequest(
            definition=definition,
            pipeline_run_id=run_id,
            thread_id=thread_id,
            input_state={
                "document_id": str(document_id),
                "document_version_id": str(version_id),
                "document_type_definition_id": str(type_id),
                "tenant_id": None,
                "file_id": str(file_id),
                "version_number": 1,
                "pipeline_config": {},
                "analysis_mode": "default",
                "source_ref": None,
                "detected_format": None,
                "parse_ref": None,
                "normalize_ref": None,
                "clean_ref": None,
                "extract_ref": None,
                "branches": [],
                "chunk_refs": {},
                "enrich_refs": {},
                "vector_refs": {},
                "retrieval_results": [],
                "stage_errors": [],
                "status": "pending",
            },
            run_snapshot=snapshot,
        )
    )

    for key, update in sorted(
            recorder.nodes.items(),
            key=lambda kv: (kv[1].sequence, kv[0][0]),
    ):
        node_id, branch_key, _attempt = key
        print(
            f"  {node_id:<20} branch={branch_key[:8]:<9} "
            f"seq={update.sequence:<3} {update.status.value}"
        )

    # 期望行数：
    #   线性 6 个 + 控制 1 个          各 1 行
    #   扇出 4 个阶段                  各 N 行（每个 retrieval 一行）
    linear_and_control = len(NODE_SPECS) - len(FANOUT_NODE_IDS)
    expected = linear_and_control + len(FANOUT_NODE_IDS) * retrieval_count

    problems: list[str] = []

    if outcome.status != RunStatus.COMPLETED:
        problems.append(
            f"运行状态应为 completed，实际 {outcome.status.value}"
            f"（错误：{outcome.error_message}）"
        )

    if len(recorder.nodes) != expected:
        problems.append(
            f"节点行数应为 {expected}，实际 {len(recorder.nodes)}"
        )

    for fanout_id in FANOUT_NODE_IDS:
        branches = {
            key[1] for key in recorder.nodes
            if key[0] == fanout_id
        }
        if len(branches) != retrieval_count or "" in branches:
            problems.append(
                f"{fanout_id} 的分支数应为 {retrieval_count}"
                f" 且都带 branch_key，实际 {branches}"
            )

    finalize_rows = [
        key for key in recorder.nodes
        if key[0] == "finalize_version"
    ]
    if len(finalize_rows) != 1:
        problems.append(
            f"finalize_version 应只跑一次，实际 {len(finalize_rows)} 次"
        )

    if problems:
        print("\n冒烟失败：")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    print(
        f"\n冒烟通过：{retrieval_count} 个 retrieval 扇出，"
        f"{len(recorder.nodes)} 个节点行，"
        "分支互不串扰，finalize 汇合后跑一次。"
    )
    return 0


def cmd_stub_run() -> int:
    return asyncio.run(_run_stub(2))


# ============================================================


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)

    command = argv[0] if argv else "topology"

    if command == "topology":
        return cmd_topology()

    if command == "stub-run":
        return cmd_stub_run()

    print(f"未知子命令：{command}")
    print("可用：topology | stub-run")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
