from __future__ import annotations

import asyncio
import logging
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone

# ============================================================
# 直接运行时的路径引导
# ============================================================
#
# `python app/main.py` 时 sys.path[0] 是 app/ 而不是项目根，
# 下面的 `from app.*` 会直接 ModuleNotFoundError。
#
# PyCharm 点"运行"时它会自动把 content root 加进 PYTHONPATH，
# 所以那里不会暴露这个问题 —— 但命令行会，而且报错指向的是
# "No module named 'app'"，离原因（启动方式）很远。
#
# 只在 __main__ 下补，正常 import 本模块的场景不受影响。
# 必须放在所有 `from app.*` **之前**。
# ============================================================

if __name__ == "__main__":
    from pathlib import Path

    _project_root = str(
        Path(__file__).resolve().parent.parent
    )

    if _project_root not in sys.path:
        sys.path.insert(0, _project_root)


from fastapi import FastAPI

from app.api.v1.router import router as api_router
from app.common.model_factory import get_model_factory
from app.services.document.embedder.bge_small_zh_local import (
    register_preload as register_embedder_preload,
)
from app.services.document.retrieve.rerank import (
    register_preload as register_reranker_preload,
)

from app.infra.datasource.startup import (
    DataSourceStartup,
)

from app.infra.datasource.initializer import initialize_database

from app.infra.graph import GraphRegistry
from app.infra.graph.dispatcher_impl import (
    InProcessGraphDispatcher,
)

from app.common.factories import build_factories
from app.services.document.pipeline.catalog import (
    GRAPH_NAME as DOCUMENT_GRAPH_NAME,
)
from app.services.document.dependencies import (
    make_pipeline_state_loader,
)
from app.services.document.pipeline.observability.artifacts import (
    FileArtifactStore,
)
from app.services.document.pipeline.observability.recorder_impl import (
    SqlAlchemyRunRecorder,
)
from app.services.document.pipeline.registry import (
    PipelineRegistry,
)
from app.services.document.pipeline.settings import (
    PipelineSettings,
)
from app.services.document.pipeline.runtime import (
    build_dispatch_handler,
    register_document_pipeline,
)
from app.services.file.storage.local import LocalFileStorage


logger = logging.getLogger(__name__)


# ============================================================
# Windows 事件循环
# ============================================================
#
# psycopg 的异步模式**无法**在 Windows 默认的 ProactorEventLoop
# 上工作，连接时会直接报：
#
#     Psycopg cannot use the 'ProactorEventLoop' to run in async mode.
#
# 而 LangGraph 的 Postgres checkpointer 走的就是 psycopg，
# 因此必须切到 SelectorEventLoop。
#
# 这里设置 policy 只覆盖"app.main 先于事件循环被导入"的场景
# （主要是测试：pytest-asyncio 按 policy 建循环）。
#
# **它救不了 uvicorn**：uvicorn 从 0.36 起改用 loop_factory，
# 根本不看 policy，且 uvicorn CLI 是先建循环再 import app。
# 因此生产/开发入口都必须走 app/serve.py，那里显式传
# loop_factory=SelectorEventLoop。
# ============================================================

if sys.platform == "win32":
    import warnings

    with warnings.catch_warnings():
        # 这个 deprecation 是已知且已记录的，不必每次启动都刷屏。
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(
            asyncio.WindowsSelectorEventLoopPolicy()
        )


def _index_clients(
        datasource_manager,
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


def _assert_compatible_event_loop() -> None:
    """
    在连库之前确认事件循环能跑 psycopg。

    不做这道检查的话，失败方式是连接池空转 15 秒后超时，
    报"Required datasource checkpointer failed to initialize" ——
    完全指不到真正的原因。实测踩过一次，所以这里显式拦下。
    """

    if sys.platform != "win32":
        return

    loop = asyncio.get_running_loop()
    loop_name = type(loop).__name__

    if "Proactor" not in loop_name:
        return

    raise RuntimeError(
        f"The running event loop is {loop_name}, but psycopg "
        "(used by the LangGraph checkpointer) cannot run on it.\n"
        "\n"
        "Use one of these entry points, both of which pass "
        "loop_factory explicitly:\n"
        "\n"
        "    python run_server.py --port 8080\n"
        "    python app/main.py\n"
        "\n"
        "Do NOT use `uvicorn app.main:app` — uvicorn's CLI creates "
        "the event loop BEFORE importing the app, and since 0.36 it "
        "uses its own loop_factory that ignores the asyncio policy "
        "(it returns ProactorEventLoop on Windows)."
    )


@asynccontextmanager
async def lifespan(
        app: FastAPI,
):
    """
    Application lifecycle.

    启动阶段：
        - 初始化数据源（含 LangGraph checkpointer）
        - 初始化 DB
        - 装配流水线图并注册
        - 启动调度器
        - 收敛上次进程残留的 running 运行

    关闭阶段：
        - 停止调度器（等在跑的任务结束）
        - 关闭全部数据源（含 checkpointer 连接池）
    """

    _assert_compatible_event_loop()

    # ========================================================
    # 1. 数据源
    # ========================================================
    #
    # checkpointer 也是一个标准数据源（见 config/checkpointer.yaml），
    # 因此它在这里被自动建池、建表、探活、注册 —— main.py 不需要
    # 单独维护第二套连接池的生命周期。

    startup = DataSourceStartup(
        config_directory=(
            "./app/config/"
        )
    )

    datasource_manager = await startup.initialize()

    app.state.datasource_manager = datasource_manager

    # ========================================================
    # 2. Initialize PostgreSQL schema
    # ========================================================

    postgresql = (
        datasource_manager.get(
            "postgresql"
        )
    )

    await initialize_database(
        postgresql.engine
    )

    # ========================================================
    # 3. 装配流水线
    # ========================================================

    session_factory = postgresql.session_factory

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

    # 产物即 File —— 走 FileService 写入，与用户上传共用同一套
    # 存储与同一张表。pipeline **不自建存储、不自定路径**。
    #
    # 去重在 FileService.store_artifact 内部按 (tenant, checksum)
    # 完成，因此 reprocess 同一文件不会重复落盘。
    #
    # ⚠️ 已知待办：产物与用户上传目前**没有字段区分**，
    # 因此会出现在 GET /files 列表里。改造 File 模块时要加
    # origin 字段并默认过滤掉产物。
    file_storage = LocalFileStorage()

    # 同一实例也交给 File 模块的依赖使用，避免两处各自 new。
    app.state.file_storage = file_storage

    artifact_store = FileArtifactStore(
        session_factory=session_factory,
        storage=file_storage,
    )

    # 同一实例也交给业务侧的工厂（见下面的 build_service_factory），
    # 否则每个请求都要重建一个 —— 它是无状态的，建一次就够。
    app.state.artifact_store = artifact_store

    # 能力注册表：反射发现各 role 的实现（见 pipeline/registry.py）。
    #
    # 挂到 app.state 供只读的能力目录接口使用
    # （GET /document-processor-capabilities）。该接口不碰数据库 ——
    # registry 构造后即冻结，因此它拿的是同一个共享实例，
    # 而不是每个请求重建一份。
    # 索引后端的客户端从数据源层取现成的给进去 —— 连接池、认证、
    # 超时都由 infra 管，业务层不另建第二套。
    #
    # 两个数据源都是 required: false（本地连不上时只警告、不拦启动），
    # 所以这里必须用 get_optional：拿不到就不注册该后端，之后配置里
    # 用到它会在查表时明确失败，而不是让某个分支悄悄跳过写入。
    capabilities = PipelineRegistry(
        index_clients=_index_clients(datasource_manager),
    )

    app.state.capabilities = capabilities

    graph_registry = GraphRegistry()

    checkpointer_provider = datasource_manager.get(
        "checkpointer"
    )

    definition = register_document_pipeline(
        registry=graph_registry,
        checkpointer=checkpointer_provider.saver,
        recorder=recorder,
        artifact_store=artifact_store,
        capabilities=capabilities,
        session_factory=session_factory,
        settings=settings,
    )

    app.state.graph_registry = graph_registry
    app.state.pipeline_definition = definition

    # ========================================================
    # 4. 调度器
    # ========================================================

    dispatcher = InProcessGraphDispatcher()

    dispatcher.register(
        definition.name,
        build_dispatch_handler(
            definition=definition,
            recorder=recorder,
            resume_state_loader=make_pipeline_state_loader(
                session_factory
            ),
            settings=settings,
        ),
    )

    await dispatcher.start()

    app.state.graph_dispatcher = dispatcher

    # ========================================================
    # 5. 服务工厂（应用级）
    # ========================================================
    #
    # 业务侧所有 Service / Repository 都从这里构造
    # （见 common/factories.py 与 common/service/factory.py）。
    #
    # 池子在这里拼：装配是**组合根**的职责，因此 main.py 知道
    # "哪个实例对应哪个键"，而 app.common 那边只负责收下与保管 ——
    # 这样 common 不必反向 import app.services / app.infra。
    #
    # 必须在上面那些单例都就绪**之后**构建：它们会被工厂抓走，
    # 之后不再重新读。放前面会得到一个缺依赖的工厂，
    # 且要到某个服务构造时才报错，离原因很远。
    #
    # 键名即构造服务时**按参数名**匹配的依据，改名要同步改各服务
    # 的构造参数名（对不上的由"按类型匹配"兜底，见
    # ServiceFactory._build_kwargs）。

    build_factories(
        session_factory,
        pool={
            "dispatcher": dispatcher,
            "pipeline_registry": capabilities,
            "graph_registry": graph_registry,
            "artifact_store": artifact_store,
            "file_storage": file_storage,
            # 模块专属固定值：文档图的注册名。
            # 它由 pipeline.catalog 定义、运行期不变，因此直接给值，
            # 而不是像上面几个那样从 app.state 上摸。
            "graph_name": DOCUMENT_GRAPH_NAME,
        },
    )

    # ========================================================
    # 6. 收敛上次进程残留的运行
    # ========================================================
    #
    # 进程中途被杀时，run 行会永远停在 running，前端一直转圈。
    # 刚启动的这一刻，任何 running 的行都必然是上次的残留。

    try:
        fixed = (
            await recorder.mark_orphans_failed(
                older_than=datetime.now(timezone.utc)
            )
            if settings.reconcile_orphans
            else 0
        )
        if fixed:
            logger.warning(
                "Marked %d orphaned pipeline run(s) as failed.",
                fixed,
            )
    except Exception:
        logger.exception(
            "Failed to reconcile orphaned pipeline runs."
        )

    # ========================================================
    # 6. 模型预热
    # ========================================================
    #
    # **后台加载，不 await** —— 让服务先就绪，两个模型在后台加载。
    # 真正用到模型的请求会在工厂里等（带日志与超时），而不是让整个
    # 启动卡十几秒。
    #
    # 登记集中在这里而不是各模块 import 时自己登记：登记只是记下
    # "key -> loader"，不触发加载；集中一处能一眼看清本进程会加载
    # 哪几个模型、各有多大。
    #
    # 注意这是**加载**而不是下载：模型必须已经在本机（见 const.py
    # 里各常量的下载记录）。离线环境不会在这里联网。

    model_factory = get_model_factory()

    register_embedder_preload(model_factory)
    register_reranker_preload(model_factory)

    model_factory.preload_all()

    # ========================================================

    try:
        yield

    finally:
        # 顺序要紧：先停调度器（等在跑的任务收尾，它们还要写库），
        # 再收模型加载任务（避免进程退出时 asyncio 报
        # "Task was destroyed but it is pending"），最后关数据源。
        try:
            await dispatcher.aclose()
        except Exception:
            logger.exception("Failed to stop graph dispatcher.")

        try:
            await model_factory.aclose()
        except Exception:
            logger.exception("Failed to close model factory.")

        await datasource_manager.close_all()


app = FastAPI(
    title="YTrader Agent",
    version="1.0.0",
    description=(
        "YTrader Agent / AI Quant Platform Backend API"
    ),
    lifespan=lifespan,
)

app.include_router(
    api_router
)

print("YTrader Agent / AI Quant Platform Backend API startup")

if __name__ == "__main__":
    # PyCharm 里点"运行"、`python app/main.py`、`python -m app.main`
    # 走的都是这里。
    #
    # 注意**不能**用 uvicorn.run(..., loop="asyncio")：那会走
    # uvicorn 自己的 loop_factory，在 Windows 上拿到
    # ProactorEventLoop，checkpointer 直接连不上。
    # 启动逻辑统一在 app/serve.py。
    from app.serve import serve

    serve(host="localhost", port=8080)
