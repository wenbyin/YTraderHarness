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

from app.api.v1.router import (
    protected_router,
    public_router,
)
from app.common.llm import warmup_chat_models
from app.common.model_factory import get_model_factory
from app.common.openapi import pin_contract_component_names
from app.mock.runtime import (
    start_mock_server_if_enabled,
    stop_mock_server,
)
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

# ============================================================
# 装配层
# ============================================================
#
# 各组件的构造细节都在 app/bootstrap/ 下，本文件只负责**排序**与
# 把结果挂到 app 上（见那个包的 __init__.py：为什么要分出这一层）。
#
# 这里能一眼读出的，正是只有组合根才知道的东西：
#
#     0    哨兵        要在任何同步活之前起来
#     0.5  鉴权        不碰网络不碰库，失败越早越好
#     1    数据源      checkpointer 也是数据源，一起建池
#     1.5  事件流      必须在图装配之前（节点第一次运行就要能发事件）
#     2    建表        数据源起来之后
#     3    两张图      依赖 2 的 session_factory 与 1 的 checkpointer
#     4    图调度器    依赖 3 的图定义
#     4.5  任务运行时  必须在 5 之前：池子要收它的单例
#     5    服务工厂    必须在 3/4/4.5 之后：池子在这里被一次抓走
#     5.1  任务启动    必须在 5 之后：Worker 执行时才取服务工厂
#     6    收敛残留    图已就绪，可以翻上次的 running 行
#     6.x  预热        最后，不阻塞服务就绪

from app.bootstrap.auth import initialize_auth
from app.bootstrap.event_stream import (
    aclose_event_stream,
    initialize_event_stream,
)
from app.bootstrap.graph_runtime import (
    aclose_graph_runtime,
    build_graph_runtime,
    start_graph_runtime,
)
from app.bootstrap.mcp import (
    initialize_mcp_providers,
    shutdown_mcp_providers,
)
from app.bootstrap.pipelines import build_pipelines
from app.bootstrap.tool_cache import initialize_tool_cache
from app.bootstrap.task import (
    aclose_task_infrastructure,
    build_task_infrastructure,
    start_task_infrastructure,
)
from app.agent.prompt import (
    DefaultPromptProvider,
    found_prompts,
    get_prompt_provider,
    set_prompt_provider,
)
from app.common.factories import (
    build_factories,
    build_redis_factory,
    get_service_factory,
)
from app.common.loop_watchdog import LoopWatchdog
from app.services.document.pipeline.catalog import (
    GRAPH_NAME as DOCUMENT_GRAPH_NAME,
)
from app.services.stress.application.http_runner import (
    HttpStressRunner,
)
from app.services.stress.application.stress_reporter import (
    StressRunReporter,
)


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
    # 0. 事件循环卡顿哨兵
    # ========================================================
    #
    # 位置刻意在**最前面**：启动阶段本身也是"同步代码跑在循环上"的重灾区
    # （建池、建表、装图都是一次性的同步活），而它要盯的正是这个。
    #
    # 为什么需要它：启动后第一次请求曾经必现一条
    # "Timeout reading from localhost:6379" —— 实测结论是某段同步代码把
    # 循环按住 ≥5 秒，asyncio 的读超时定时器在循环恢复的瞬间先到期，
    # 把"回复早就到了"谎报成了网络故障。当时是靠临时探针定位的；这个
    # 哨兵把定位变成常驻能力：卡住超过阈值就把当时的线程栈打出来。
    #
    # 它只报不治（把那段同步活搬走是另一件事）；也不会因为报不出来
    # 而影响启动 —— 见 app/common/loop_watchdog.py。
    loop_watchdog = LoopWatchdog()
    loop_watchdog.start()

    # ========================================================
    # 0.5 Casdoor 鉴权自检
    # ========================================================
    #
    # 位置刻意靠前：这一段不碰网络也不碰库，失败得越早，日志越好读。
    # 构造、自检与 prod 环境保护都在 app/bootstrap/auth.py。

    app.state.auth = initialize_auth()

    # ========================================================
    # 0.6 Mock 模型服务（开发 / 前端联调）
    # ========================================================
    #
    # 位置刻意在模型注册表装配与 warmup 之前：
    #   - warmup 会在启动期预建 llm 客户端，mock 必须先 listen；
    #   - 启动失败必须让应用起不来 —— 显式要求 mock 却悄悄退回真实模型，
    #     比启动失败更糟（前端以为在测 mock，实际在花钱调云端）。
    #
    # 它在自己的守护线程里跑，不占用主事件循环；关闭时停止线程。

    app.state.mock_model_server = start_mock_server_if_enabled()

    # ========================================================
    # 1. 数据源
    # ========================================================
    #
    # checkpointer 也是一个标准数据源（见 config/checkpointer.yaml），
    # 因此它在这里被自动建池、建表、探活、注册 —— main.py 不需要
    # 单独维护第二套连接池的生命周期。

    startup = DataSourceStartup(
        config_directory=(
            "./app/config/datasource/"
        )
    )

    datasource_manager = await startup.initialize()

    app.state.datasource_manager = datasource_manager

    # Redis 客户端工厂：与 session_factory 同一形状 —— 数据源就绪后装配
    # 一次，之后任何地方 `get_redis_factory().optional_client()` 取用。
    # Redis 是 required: false，所以 source 可能是 None；调用方据此退回
    # 进程内实现（例如 L2/L3 画像缓存）。
    build_redis_factory(datasource_manager.get_optional("redis"))

    # ========================================================
    # 1.5 运行事件流（Redis Stream）
    # ========================================================
    #
    # 位置必须在业务图装配**之前**：节点第一次运行就要能拿到发布器。
    # 装配与停摆见 app/bootstrap/event_stream.py（含"为什么发布端与读取端
    # 用两个不同的 client"、以及读取端为什么不共用业务查询池）。

    await initialize_event_stream()

    # 工具结果缓存：有 Redis 就用 Redis，没有就退回进程内。
    # 位置同样要在图装配之前 —— 节点第一次跑工具就要拿到最终实现。
    initialize_tool_cache()

    # ========================================================
    # 1.6 MCP provider（外部工具来源）
    # ========================================================
    #
    # 必须在第一次模型调用之前完成：`select_specs` 只读进程级注册表，
    # 晚一步注册，那一轮对话就看不到 MCP 工具。
    #
    # 这里只负责**连接 + 注册**，不写 tool_definition —— 那是管理接口
    # `POST /tools/discovery/sync` 的事（快照与运行时分开，本地工具
    # 也是同一套）。provider 连不上只记 ERROR，不阻断启动，理由见
    # app/bootstrap/mcp.py。

    app.state.mcp_tools = await initialize_mcp_providers()

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
    # 3. 装配两张图
    # ========================================================
    #
    # 文档管线 + Agent 对话，连同它们共用的记录器 / 产物存储 / 能力注册表。
    # 构造细节见 app/bootstrap/pipelines.py（含"能力注册表为什么要用
    # get_optional 取索引后端"、"批准账本不传 store 的后果"）。

    session_factory = postgresql.session_factory

    pipelines = build_pipelines(
        datasource_manager=datasource_manager,
        session_factory=session_factory,
    )

    # 挂到 app.state 是给**只读**的接口/探针用的（能力目录接口、
    # 图定义查询）。它们构造后即冻结，因此拿的是同一个共享实例，
    # 而不是每个请求重建一份。
    app.state.file_storage = pipelines.file_storage
    app.state.artifact_store = pipelines.artifact_store
    app.state.capabilities = pipelines.capabilities
    app.state.graph_registry = pipelines.graph_registry
    app.state.pipeline_definition = pipelines.document_definition
    app.state.react_graph_definition = pipelines.react_definition

    # ========================================================
    # 4. 图调度器
    # ========================================================
    #
    # 建 + 注册两个 handler + 启动，都在 app/bootstrap/graph_runtime.py。
    # 建与启动分成两步（与下面任务那套同形），是因为启动时要把 dispatcher
    # 登记进进程级访问器 —— 而事件流读取端排在上面更早的位置，那时它还
    # 不存在（见那个模块的说明）。

    app.state.graph_dispatcher = build_graph_runtime(
        pipelines=pipelines,
        session_factory=session_factory,
    )

    await start_graph_runtime()

    # ========================================================
    # 4.5 统一异步任务基础设施
    # ========================================================
    #
    # 装配、Handler 注册、生命周期都在 app/bootstrap/task.py ——
    # ★ 业务 Handler 的注册点也在那里（_register_handlers）。
    # 放那里的硬理由：注册要 import 业务契约，而 app/common 与
    # app/infra 都不允许 import 业务模块（依赖方向）。
    #
    # 本文件只做两件"只有组合根知道"的事：
    #
    #   1. 现在**建**它 —— 因为下面 build_factories 的池子要收它的两个
    #      单例（handler_registry / task_dispatcher）。
    #   2. 等工厂建好之后再 start()（见下面 5.1）—— Worker 执行任务时才
    #      去取应用级服务工厂，先启动的话，一条积压任务会在工厂还不存在
    #      时被执行。这个顺序是硬约束，理由写在那个模块的 docstring 里。

    task_infrastructure = build_task_infrastructure(session_factory)

    app.state.task_infrastructure = task_infrastructure

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

    # 真实压测 Driver：独立守护线程 + Reporter 回主 loop 写库。
    # 创建在这里（而不是 build_factories 内部）是为了 lifespan 关闭时
    # 能直接 stop 它，且能复用当前的 running loop。
    redis_source = datasource_manager.get_optional("redis")
    redis_url = getattr(redis_source, "url", None)

    stress_runner = HttpStressRunner(
        StressRunReporter(
            session_factory,
            asyncio.get_running_loop(),
        ),
        redis_url=redis_url,
    )
    app.state.stress_runner = stress_runner

    build_factories(
        session_factory,
        pool={
            # 图调度器。⚠️ 与下面任务运行时那个 task_dispatcher 是两个
            # 不同的东西，键名不同也是刻意的：工厂按**参数名**优先匹配，
            # 名字若撞上，注入进去的会是另一个调度器
            # （见 TaskServiceImpl 的构造参数说明）。
            "dispatcher": app.state.graph_dispatcher,
            # resume 的服务层校验要读 checkpoint 上真实的待处理中断集合
            # （见 AgentRunServiceImpl._require_complete_answers）。给的是
            # 已编译的图，与上面交给 dispatcher 的是同一个 definition。
            "react_definition": pipelines.react_definition,
            "pipeline_registry": pipelines.capabilities,
            "graph_registry": pipelines.graph_registry,
            "artifact_store": pipelines.artifact_store,
            "file_storage": pipelines.file_storage,
            # 模块专属固定值：文档图的注册名。
            # 它由 pipeline.catalog 定义、运行期不变，因此直接给值，
            # 而不是像上面几个那样从 app.state 上摸。
            "graph_name": DOCUMENT_GRAPH_NAME,
            # 真实压测 Driver：通过 HTTP/SSE 驱动被测 Agent。
            # 它在独立守护线程里跑，DB 写入由 StressRunReporter
            # 投回当前主 loop（见 services/stress/application/http_runner.py）。
            "stress_runner": stress_runner,
            # 任务运行时的两个单例（handler_registry / task_dispatcher），
            # 键名与各自的作用见 app/bootstrap/task.py::pool_entries。
            **task_infrastructure.pool_entries,
        },
    )

    # --------------------------------------------------------
    # 5.1 拉起任务执行侧
    # --------------------------------------------------------
    #
    # 必须排在工厂之后，理由见上面 4.5 的第 2 条（也是
    # app/bootstrap/task.py 的 docstring）。两条循环各自"先跑一轮再睡"，
    # 所以上一次进程崩时残留的到期任务与未投递记录在这一刻开始被捞回来
    # —— 这就是"服务重启恢复"。

    await start_task_infrastructure()

    # ========================================================
    # 5.2 提示词 Provider
    # ========================================================
    #
    # 与 model registry / policy resolver 同一形状：进程级一份，由组合根装上
    # （见 app/agent/prompt/provider.py 的 get/set/reset 三件套）。
    #
    # 职责提示词要走 ServiceFactory 造 PromptService，所以这一步必须排在
    # 上面那个工厂**之后**。
    #
    # ⚠️ 之后那次主动取用不是"顺手热身"，它挡的是两类完全不同的东西
    # ------------------------------------------------------------
    #   节点提示词目录错（重复 key / prompts 目录被删 / 模块里没有 PROMPT）
    #       是**代码错误**，让它在这里把服务拦下来 —— 比"第一次对话时
    #       分类节点抛 PromptNotFoundError"早得多，也清楚得多。
    #
    #   职责提示词取不到（服务解析不出来、表没接上）
    #       是**环境状况**，只记日志、不拦启动：取不到时主循环的行为就是
    #       本次改动之前的行为（没有 system 提示词，只有注入分节），对话
    #       完全成立。拿它拦住整个服务是不划算的。
    #       ——但那种"静默降级"在这里是唯一会被看见的地方，所以要 re-raise
    #       成一条 ERROR 日志（见下面 except）。

    set_prompt_provider(
        DefaultPromptProvider(service_factory=get_service_factory())
    )

    found_prompts()

    try:
        duty_prompt = await get_prompt_provider().duty_prompt()

    except Exception:
        logger.exception(
            "职责提示词预热失败（不阻止启动）：主模型这一进程里不会有"
            "system 提示词。请在观测的 classification.notes 与 provider 的"
            "日志里确认原因。"
        )

    else:
        logger.info("职责提示词已就绪：%d 字符。", len(duty_prompt))

    # ========================================================
    # 6. 收敛上次进程残留的运行
    # ========================================================
    #
    # 进程中途被杀时，run 行会永远停在 running，前端一直转圈。
    # 刚启动的这一刻，任何 running 的行都必然是上次的残留。

    try:
        fixed = (
            await pipelines.recorder.mark_orphans_failed(
                older_than=datetime.now(timezone.utc)
            )
            if pipelines.settings.reconcile_orphans
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
    # 6.1 云端 LLM 的 provider import 预热
    # ========================================================
    #
    # 与上面那段（本地 torch 权重，后台加载）是两件不同的事，排在这里
    # 是为了**与它们并行**：权重加载要几十秒，这段时间正好用来热 import。
    #
    # 热的是什么：进程起来后第一次模型调用会卡在一串惰性 import 上
    # （langchain -> langchain_openai -> openai.types...，挂在 PyCharm
    # 调试器下实测 7 秒）。而那是**同步**的、跑在事件循环线程上 ——
    # 它按住循环的这段时间里，SSE 那条 XREAD 的读超时定时器会在循环
    # 恢复的瞬间先到期，把"回复早就到了"谎报成
    #
    #     TimeoutError: Timeout reading from localhost:6379
    #
    # 也就是说那条 Redis 报错的根因在这里，与 Redis 无关。完整链路见
    # app/common/llm.py 的 warmup_chat_models 与 app/common/loop_watchdog.py。
    #
    # ⚠️ 必须 **await 在 yield 之前**：预热的意义就是"在开始接请求之前
    # 把这份成本付掉"。改成后台任务（像上面 preload_all 那样）就等于没做
    # —— 第一个请求照样撞上那段未完成的 import。
    #
    # 丢线程池：预热本身是同步阻塞的（那正是它要解决的属性），而这里
    # 虽然还没开始接请求，watchdog 已经在数秒了 —— 让它误报一条
    # "启动阶段卡住了"没有任何意义。

    try:
        warmup = await asyncio.to_thread(warmup_chat_models)
        logger.info("云端 LLM provider 预热：%s", warmup.describe())

    except Exception:
        # 预热是优化，不是前置条件 —— 装少一个包不该让服务起不来。
        logger.exception(
            "云端 LLM provider 预热失败（忽略，不影响启动）："
            "第一次模型调用会自己付这份成本。"
        )

    # ---- OpenAPI schema 预热 ----------------------------------
    #
    # `/openapi.json` 与 `/docs` 的 schema 是**首次请求时才生成**的 ——
    # 见本文件里覆盖 `app.openapi` 那段（那里写着"文档要等第一次请求
    # /openapi.json 才生成"）。而生成是**同步**的：它要遍历全部路由与
    # 全部 Pydantic 模型，再给每个组件重命名一次。
    #
    # 实测这一下把事件循环卡了 2.4 秒（watchdog 阈值 2.0）。那段时间里
    # 所有请求、SSE 心跳、以及 Redis 的**读超时定时器**全部停摆 —— 而
    # 后者会伪装成一条网络故障报错（`TimeoutError: Timeout reading from
    # localhost:6379`），与真正的原因隔了十万八千里。这与上面那段注释里
    # 描述的 LLM 预热是同一个坑。
    #
    # ⚠️ 同一取舍，逐条适用：**必须 await 在 yield 之前**（留到第一次
    #    请求再算，就是把这份成本转嫁给某个不幸的调用方），且**丢线程
    #    池**（它本来就是同步阻塞的，而 watchdog 此刻已经在数秒）。
    #
    # 幂等：生成完会缓存进 `app.openapi_schema`，之后每次请求只走一遍
    # `pin_contract_component_names`（它自己会早退）。
    #
    # 顺带一提：加了新端点与新 DTO 之后这个成本会涨 —— 它不是凭空变慢的，
    # 所以更不该留在请求路径上。
    try:
        await asyncio.to_thread(app.openapi)
        logger.info("OpenAPI schema 预热完成（首个请求不必再付这份成本）")

    except Exception:
        logger.exception(
            "OpenAPI schema 预热失败（忽略，不影响启动）："
            "第一次请求 /openapi.json 会自己付这份成本。"
        )

    # ========================================================

    try:
        yield

    finally:
        # ====================================================
        # 停摆：顺序是**这里唯一的知识**
        # ====================================================
        #
        # 每个组件"怎么关自己"都在各自的 bootstrap 模块里；本文件负责的是
        # 它们之间的**先后**，因为那跨模块、只有组合根看得见：
        #
        #   1. 任务运行时   收尾要写库 -> 必须早于数据源
        #   2. 图调度器     同上；且必须早于事件发布器（见第 3 条）
        #   3. 事件流       排在 2 之后：还在收尾的图任务要把
        #                   run.completed 这类终止事件发出去，先停发布器
        #                   会让订阅端永远等不到终止事件
        #   4. 模型工厂     收掉加载任务，避免退出时 asyncio 报
        #                   "Task was destroyed but it is pending"
        #   5. 卡顿哨兵     留到最后一刻：收尾阶段同样可能被同步代码按住，
        #                   而那正是"关不干净"最需要现场的时候
        #   6. 数据源       最后。3 必须早于它：反过来的话是"连接池先被关、
        #                   任务还在它上面阻塞读"，收尾日志里会刷出一串
        #                   看起来像"Redis 挂了"的噪音
        #
        # 每个 aclose_* 都是"没装配过则无操作"—— 上面的装配可能没走到
        # 它们就失败了，而那时最需要安静收场。

        # 压测 Driver 独立于主 loop，但它的 Reporter 还要写库；必须在
        # 数据源关闭之前停掉。先发取消信号，让 Campaign 尽量落
        # incomplete 快照，再 join 线程。
        try:
            stress_runner = getattr(app.state, "stress_runner", None)

            if stress_runner is not None:
                stress_runner.stop()

        except Exception:
            logger.exception("Failed to stop stress driver.")

        try:
            await aclose_task_infrastructure()
        except Exception:
            logger.exception("Failed to stop task infrastructure.")

        try:
            await aclose_graph_runtime()
        except Exception:
            logger.exception("Failed to stop graph dispatcher.")

        # MCP 是外部子进程：图停了之后它不会再有新调用，放在这里
        # 关闭。必须在数据源之前 —— 它的调用栈可能还会碰 session。
        try:
            await shutdown_mcp_providers()
        except Exception:
            logger.exception("Failed to stop MCP providers.")

        try:
            await aclose_event_stream()
        except Exception:
            logger.exception("Failed to stop event stream.")

        try:
            await model_factory.aclose()
        except Exception:
            logger.exception("Failed to close model factory.")

        # Mock 只被模型客户端使用；客户端收掉之后再停它。
        try:
            stop_mock_server(
                getattr(app.state, "mock_model_server", None)
            )
        except Exception:
            logger.exception("Failed to stop mock model server.")

        try:
            await loop_watchdog.aclose()
        except Exception:
            logger.exception("Failed to stop loop watchdog.")

        await datasource_manager.close_all()


app = FastAPI(
    title="YTrader Agent",
    version="1.0.0",
    description=(
        "YTrader Agent / AI Quant Platform Backend API"
    ),
    lifespan=lifespan,
)

# 两棵路由树，顺序无所谓（路径不重叠）。
#
# 分开注册是**鉴权方案的一部分**：public_router 上没挂校验依赖，
# protected_router 挂了。谁在哪棵树上，见 app/api/v1/router.py。
app.include_router(
    public_router
)

app.include_router(
    protected_router
)

# 生成的文档里把实现类名换回契约组件名（AgentMessage -> Message）。
#
# 为什么不是在 DTO 上声明：FastAPI 拿类的 __name__ 当组件名，Pydantic 的
# title 只改 schema 体内那个字段，管不着 $ref 的名字 —— 只能在生成之后改
# 一次，见 app/common/openapi.py。
#
# 覆盖 app.openapi 而不是 app.openapi_schema：后者此时还是 None（文档要等
# 第一次请求 /openapi.json 才生成），直接赋值会让 FastAPI 跳过生成。
_fastapi_openapi = app.openapi


def _contract_named_openapi() -> dict:

    # FastAPI 自己就会把结果缓存进 app.openapi_schema。这里**不**用
    # "刚生成才改"那种写法：缓存可能已经被别人填好了（谁手里留着原来那个
    # app.openapi 的引用、或者测试里预设过），那样改名会被跳过，而跳过的
    # 表现是文档悄悄多出一个组件名 —— 没人会去看。改名是幂等的，无条件
    # 走一次更稳，也已经改完的文档会立刻早退。
    if app.openapi_schema is None:
        _fastapi_openapi()

    pin_contract_component_names(app.openapi_schema)

    return app.openapi_schema


app.openapi = _contract_named_openapi

print("YTrader Agent / AI Quant Platform Backend API startup")

if __name__ == "__main__":
    # PyCharm 里点"运行"、`python app/main.py`、`python -m app.main`
    # 走的都是这里。
    #
    # 注意**不能**用 uvicorn.run(..., loop="asyncio")：那会走
    # uvicorn 自己的 loop_factory，在 Windows 上拿到
    # ProactorEventLoop，checkpointer 直接连不上。
    # 启动逻辑统一在 app/serve.py。
    import argparse

    from app.serve import serve

    parser = argparse.ArgumentParser(
        description="YTrader Agent backend server"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用进程内 mock 模型服务；默认跟随 mock_model.yaml。",
    )
    args = parser.parse_args()

    serve(host=args.host, port=args.port, mock=args.mock)
