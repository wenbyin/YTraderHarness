"""
两张图（文档管线 + Agent 对话）的装配。

产出的是一个引用束（Pipelines），交给两个地方用：

    app/bootstrap/graph_runtime.py   建调度器、注册 handler
    main.py 的 build_factories 池子  dispatcher / react_definition /
                                     pipeline_registry / graph_registry /
                                     artifact_store / file_storage

为什么聚成一个模块
----------------
这些对象互相引用得很紧（记录器给图用、图注册进注册表、产物存储给管线
节点用、模型注册表给 React 图用），而它们**只有一起才成立**：单独看
`artifact_store` 是看不出来它必须与 `recorder` 来自同一个 session_factory
的。分散在 lifespan 的一百多行里，读的人得自己把关系拼出来。

没有 aclose
----------
本模块装配出来的东西都**没有自己的生命周期**：图注册表构造后即冻结、
产物存储与记录器是无状态的、注册流程只是把定义塞进注册表。它们脚下的
连接池归数据源管理器（见 app/infra/datasource/manager.py），
所以停摆那一侧不需要本模块出场。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from app.agent.graph.react.runtime import register_react_graph
from app.agent.model import get_model_registry
from app.infra.graph import GraphRegistry
from app.infra.graph.registry import GraphDefinition
from app.services.document.pipeline.observability.artifacts import (
    FileArtifactStore,
)
from app.services.document.pipeline.observability.recorder_impl import (
    SqlAlchemyRunRecorder,
)
from app.services.document.pipeline.registry import PipelineRegistry
from app.services.document.pipeline.runtime import (
    register_document_pipeline,
)
from app.services.document.pipeline.settings import PipelineSettings
from app.services.file.storage.local import LocalFileStorage
from app.services.session.application.agent_run_dispatch import (
    NodeLogRecorder,
)

__all__ = ["Pipelines", "build_pipelines"]

logger = logging.getLogger(__name__)


def _index_clients(
        datasource_manager: Any,
) -> dict:
    """
    从数据源管理器取出索引后端需要的客户端。

    只取**已经在跑**的那些：数据源没启起来（容器没开、连不上）时
    manager 里根本没有那一条，对应的索引后端就注册不上 —— 这是想要
    的行为，见 PipelineRegistry.found_index_backends 的说明。

    Neo4j 不在其中：图那一类的入库还没做（GraphExtractionEmbedder
    也还是空置），等它落地时在这里加一行即可。
    """

    from app.services.document.domain.schemas.enum import (
        StorageBackend,
    )

    wanted = {
        "elasticsearch": StorageBackend.ELASTICSEARCH,
        "milvus": StorageBackend.MILVUS,
    }

    clients: dict = {}

    for name, backend in wanted.items():

        source = datasource_manager.get_optional(name)

        if source is None:
            logger.warning(
                "Index backend %s is unavailable: datasource %r is not "
                "running. Retrievals pointing at it will fail explicitly.",
                backend.value,
                name,
            )
            continue

        clients[backend] = source.resource()

    return clients


@dataclass(frozen=True, slots=True)
class Pipelines:
    """装配好的两张图，以及它们共用的那些组件。"""

    settings: PipelineSettings

    #: 文档管线的 SQL 记录器（写 document_pipeline_run 那套）。
    recorder: SqlAlchemyRunRecorder

    #: Agent 这条链路的节点级观测：只打日志，不落库。
    node_recorder: NodeLogRecorder

    #: 产物即 File —— 与用户上传共用一套存储与同一张表。
    file_storage: LocalFileStorage
    artifact_store: FileArtifactStore

    #: 能力注册表（反射发现各 role 的实现）。
    capabilities: PipelineRegistry

    graph_registry: GraphRegistry

    #: 两张已编译的图定义。dispatcher 按它们的 name 注册 handler。
    document_definition: GraphDefinition
    react_definition: GraphDefinition


def build_pipelines(
        *,
        datasource_manager: Any,
        session_factory: Any,
) -> Pipelines:
    """
    装配两张图。

    Args:
        datasource_manager: 已初始化的数据源管理器。
        session_factory: 业务读写用的 async_sessionmaker
            （来自 postgresql 数据源）。
    """

    # 运行参数：checkpoint 级别（node/run）、state 体积断言、
    # 启动时收敛孤儿运行。可用 PIPELINE_* 环境变量覆盖。
    settings = PipelineSettings.from_env()

    logger.info(
        "Pipeline settings: checkpoint_level=%s (durability=%s), "
        "assert_state_size=%s, reconcile_orphans=%s",
        settings.checkpoint_level,
        settings.durability,
        settings.assert_state_size,
        settings.reconcile_orphans,
    )

    recorder = SqlAlchemyRunRecorder(session_factory)

    # 产物即 File —— 走 FileService 写入，与用户上传共用同一套存储与
    # 同一张表。pipeline **不自建存储、不自定路径**。
    #
    # 去重在 FileService.store_artifact 内部按 (tenant, checksum) 完成，
    # 因此 reprocess 同一文件不会重复落盘。
    #
    # ⚠️ 已知待办：产物与用户上传目前**没有字段区分**，因此会出现在
    # GET /files 列表里。改造 File 模块时要加 origin 字段并默认过滤掉产物。
    file_storage = LocalFileStorage()

    artifact_store = FileArtifactStore(
        session_factory=session_factory,
        storage=file_storage,
    )

    # 能力注册表：反射发现各 role 的实现（见 pipeline/registry.py）。
    #
    # 索引后端的客户端从数据源层取现成的给进去 —— 连接池、认证、超时都由
    # infra 管，业务层不另建第二套。
    #
    # 两个数据源都是 required: false（本地连不上时只警告、不拦启动），
    # 所以这里必须用 get_optional：拿不到就不注册该后端，之后配置里用到它
    # 会在查表时明确失败，而不是让某个分支悄悄跳过写入。
    capabilities = PipelineRegistry(
        index_clients=_index_clients(datasource_manager),
    )

    graph_registry = GraphRegistry()

    checkpointer_provider = datasource_manager.get("checkpointer")

    definition = register_document_pipeline(
        registry=graph_registry,
        checkpointer=checkpointer_provider.saver,
        recorder=recorder,
        artifact_store=artifact_store,
        capabilities=capabilities,
        session_factory=session_factory,
        settings=settings,
    )

    # --------------------------------------------------------
    # 图二：Agent 对话（ReAct）。
    #
    # 图本身早就装配好了（app/agent/graph/react/），这里只补它一直缺的
    # 两样东西：注册进 GraphRegistry，以及给 dispatcher 一个 handler。
    # 与 document 并排而不是各写一套 —— 两者共用的正是 GraphRegistry
    # 与 InProcessGraphDispatcher。
    #
    # `get_model_registry()` 必须在**启动期**调：它读 model.yaml 并装配
    # 注册表，"目录里少一个模型"这类配置错因此会让应用起不来，而不是等
    # 某个用户踩到（见 react/runtime.py 的说明）。
    # --------------------------------------------------------

    # 节点级观测：只打日志，不落库（逐节点进度由将来的 SSE 承担，见
    # agent_run_dispatch.py）。同一实例也交给 handler —— RunContext 里的
    # recorder 与 ReactContext 里的应当是同一次运行的同一个。
    node_recorder = NodeLogRecorder()

    react_definition = register_react_graph(
        registry=graph_registry,
        checkpointer=checkpointer_provider.saver,
        recorder=node_recorder,
        model_registry=get_model_registry(),
        # store 与 checkpointer 来自**同一个数据源、同一个池** —— 两者要的
        # 连接参数逐字相同，另开一个池只多占连接。见
        # infra/graph/checkpointer.py 的模块 docstring。
        #
        # 它在图里的角色是审批拦截器的**批准账本**（approval.py）。不传的
        # 后果很具体：账本退回**进程内**，重启后停在 `waiting_input` 的
        # 运行 resume 时，之前已经批过的调用会**再被问一遍** —— 而那条
        # warning 整个进程只喊一次，是最不容易被联想到"存储"的一条线索。
        store=checkpointer_provider.store,
    )

    return Pipelines(
        settings=settings,
        recorder=recorder,
        node_recorder=node_recorder,
        file_storage=file_storage,
        artifact_store=artifact_store,
        capabilities=capabilities,
        graph_registry=graph_registry,
        document_definition=definition,
        react_definition=react_definition,
    )
