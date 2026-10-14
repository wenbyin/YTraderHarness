"""
Agent 运行时会话与运行的接口。

对应 session.yaml 的前六个端点：
    POST /sessions                    建会话
    GET  /sessions/{session_id}       查会话
    POST /sessions/{session_id}/runs  建运行
    GET  /runs/{run_id}/events        事件流（SSE）
    POST /runs/{run_id}/resume        回答一次人工审批
    POST /runs/{run_id}/cancel        取消一次正在跑的运行

以及 messages.yaml 的一个端点（**另一份规格**，路由在本文件）：
    GET  /sessions/{session_id}/messages   一页会话历史

⚠️ 消息那个端点是**游标分页**，与仓库其余列表接口不同
------------------------------------------------
documents / files / tools 一律是 `page` + `page_size` + `total`，这里却是
`limit` + 不透明的 `before` / `after`。理由是聊天历史的形状：

    它在尾部不断追加。页码分页下，"翻到第 2 页"的过程中又来了一条新消息，
    第 2 页的内容就整体位移了 —— 客户端会重复渲染或直接漏掉一条。
    游标指向一条**具体的消息**，它周围的内容不会因为前面加了东西而变。

代价是给不出 total（keyset 算总数要额外一次 COUNT，而它对"还能翻多久"
本来就没有稳定答案）。契约因此没有 total，用 has_more_before /
has_more_after 代替 —— 见 domain/schemas/agent_message.py 的说明。

两个游标不能同时给（一个说往旧、一个说往新）：同时给是 400，不是
"挑一个执行"。挑一个等于静默丢掉另一个，客户端会拿到一页它没要的数据
却看不出为什么。

⚠️ `resume` 与 `cancel` 解决的是**两件不同的事**，别混用
----------------------------------------------------
    resume    图**停着**（`waiting_input`），等人回答这次审批 -> 继续跑
    cancel    图**在跑**（`queued` / `running`），不要它了 -> 中止并丢弃

最容易搞错的是"停着的那次能不能取消"：**不能**，它没有在跑的任务可取消。
它在等一个回答，而"回答 reject"就等于不要这一步（见 cancel_run 的说明
里的 409 文案）。这不是实现偷懒 —— 停在审批上的运行没有 asyncio 任务，
硬要"取消"就只能替用户凭空做一个决定。

⚠️ 事件流（SSE）的三条取舍
------------------------
1. **不做 Service 依赖，自己开一段短事务。** `Service[...]` 会把一段
   请求级事务的开合与**整个响应**绑定，而本响应是长连接（可能挂几分钟）。
   因此存在性检查用 `get_service_factory().scope(...)` 单独开、查完就放，
   流式部分不再持有任何数据库连接。

2. **先补发再跟实时。** XREAD 从 0 起，晚连的客户端也能看到全过程；
   断线重连时 `Last-Event-ID` 头会被当成起点续上（纯 SSE 规范，
   session.yaml 的声明式契约里没有游标参数，因此不新增参数）。

3. **收到终止事件就正常关流。** 终止判据来自契约层
   （`app/common/event` 的 TERMINAL_EVENT_TYPES），不是写死在这里的
   字符串。运行没跑完时流会一直开着 —— 那是对的。

⚠️ 建会话返回 **200** 而不是 201
------------------------------
仓库其他模块的创建接口用 201（如 create_capability）。这里跟规格走：
session.yaml 里 createSession 声明的是 "200"。契约先行 —— 规格决定，
实现不单方面改。要改成 201 就改 session.yaml，两边一起。
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.agent.event import (
    HEARTBEAT,
    RunStreamAttachment,
    TooManyRunStreamsError,
    get_run_stream_registry,
    sse_frame,
)
from app.agent.tools.processor.context import build_caller_context
from app.common.auth import AuthenticatedUser, current_user
from app.common.factories import get_service_factory
from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.session.application.agent_run_service import (
    AgentRunService,
)
from app.services.session.application.agent_session_service import (
    AgentSessionService,
)
from app.services.session.application.message_service import (
    DEFAULT_LIMIT,
    MessageService,
)
from app.services.session.domain.schemas import (
    CreateRunRequest,
    CreateSessionRequest,
    MessagePage,
    ResumeRequest,
    Run,
    Session,
)

router = APIRouter()


def _owner_id(
        user: AuthenticatedUser,
) -> str | None:
    """
    会话归属的**写入值**：subject 为空串时归一成 None（-> 列里 NULL）。

    只有**写**的两条路径用它（create_session / create_run）。读路径直接传
    `user.subject` —— 那边类型是 str，而空串匹配不到任何真实归属，效果正好
    是"只看得到公共数据"。两者是同一套语义的两面：

        **没有身份 = 只能碰公共数据，且自己造出来的也是公共的。**

    为什么不在 auth 层就把空 sub 挡掉：`AuthenticatedUser` 是纯机制，
    不替业务决定"空 sub 该不该放行"（见 app/common/auth 的边界说明）。
    """

    return user.subject or None


def _org_id(
        user: AuthenticatedUser,
) -> str | None:
    """
    会话**组织归属**的写入值：空串归一成 None（-> 列里 NULL）。

    与 `_owner_id` 同一形状，但语义是另一回事：这是团队 / 组织的记录，
    既不参与鉴权也不参与可见性过滤（过滤仍然只看 user_id）。

    只在建会话时用一次 —— 运行行上没有组织这一列（图里没有消费者），
    职责键是另一条链路（见 create_run 里的 duty_key）。
    """

    return user.org_id or None


@router.post(
    "/sessions",
    response_model=Session,
    tags=["Sessions"],
)
async def create_session(
        request: CreateSessionRequest,
        service: AgentSessionService = Service[
            AgentSessionService
        ],
        user: AuthenticatedUser = Depends(current_user),
):
    """建一个会话。初始状态 active。"""

    try:
        return await service.create_session(
            request,
            user_id=_owner_id(user),
            org_id=_org_id(user),
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/sessions/{session_id}",
    response_model=Session,
    tags=["Sessions"],
)
async def get_session(
        session_id: UUID,
        service: AgentSessionService = Service[
            AgentSessionService
        ],
        user: AuthenticatedUser = Depends(current_user),
):
    """取一个会话。"""

    try:
        return await service.get_session(
            session_id,
            user_id=user.subject,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/sessions/{session_id}/messages",
    response_model=MessagePage,
    tags=["Messages"],
)
async def list_session_messages(
        session_id: UUID,
        limit: int = Query(
            default=DEFAULT_LIMIT,
            ge=1,
            le=100,
        ),
        before: str | None = Query(
            default=None,
            description="不透明游标：取它之前（更旧）的一页。",
        ),
        after: str | None = Query(
            default=None,
            description="不透明游标：取它之后（更新）的一页。",
        ),
        service: MessageService = Service[MessageService],
        user: AuthenticatedUser = Depends(current_user),
):
    """
    取一页会话历史，按 seq 升序（旧 -> 新）。

    不给游标时返回**最新**的一页（规格如此）。要往两头翻就拿响应里的
    `pagination.before_cursor` / `after_cursor` 再请求一次：它们分别指向
    本页最旧与最新的那条。

    ⚠️ 游标是不透明的
    ----------------
    客户端不该解析它。实现是 `base64url("v1:<seq>")`，但契约只承诺它是一个
    能原样回传的字符串 —— 换编码不破坏契约，解析它的客户端会先坏掉。

    ⚠️ 会话不存在是 404，不是空数组
    ---------------------------
    "这个会话还没聊过"与"这个会话不存在"是两件事，前者返回空数组。混成
    同一种响应会让客户端分不清自己拿错了 session_id。

    Raises:
        NotFoundError: 会话不存在 -> 404。
        ValueError: 两个游标同时给，或游标解析不出来 -> 400。

    ⚠️ limit 越界是 **422**，不是 400
    ------------------------------
    `Query(ge=1, le=100)` 由 FastAPI 自己校验，越界回的是它标准的 422；
    而 messages.yaml 这一栏只声明了 400。这是本仓库的既有取舍（其余列表
    接口的 page_size 同样如此）：参数**形状**不对归 422，业务校验失败归
    400。按惯例保留，不为了对齐这一行而自己抛。
    """

    try:
        return await service.list_messages(
            session_id,
            user_id=user.subject,
            limit=limit,
            before=before,
            after=after,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/sessions/{session_id}/runs",
    response_model=Run,
    tags=["Runs"],
)
async def create_run(
        session_id: UUID,
        request: CreateRunRequest,
        service: AgentRunService = Service[AgentRunService],
        user: AuthenticatedUser = Depends(current_user),
):
    """
    在某个会话下建一次运行。

    返回的运行是 `queued` —— 此刻它确实还没开始跑：派发登记在请求事务提交
    之后（见 agent_run_service_impl.py）。提交后它会被交给 agent_react 图，
    状态随之走到 running / completed / failed。

    ⚠️ 但**本响应**看不到那个终态：返回时它确实还没开始跑。要看后续进展
    用 `GET /runs/{run_id}/events`（事件流），或者查库
    （agent_run.status / error_message）。

    Raises:
        NotFoundError: 会话不存在 -> 404。
        ValueError: input.text 为空或只有空白 -> 400。
    """

    try:
        return await service.create_run(
            session_id,
            request,
            user_id=_owner_id(user),
            # 身份快照：**这里是唯一拿得到 token 的地方**（派发是异步的，
            # 图跑起来时手里已经没有它了），所以角色/群组必须在建运行这
            # 一刻存下来，工具级的 checkauth 才有依据。
            caller_identity=build_caller_context(
                user_id=user.subject,
                groups=user.groups,
                roles=user.roles,
            ),
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/runs/{run_id}/resume",
    tags=["Runs"],
)
async def resume_run(
        run_id: UUID,
        request: ResumeRequest,
        service: AgentRunService = Service[AgentRunService],
        user: AuthenticatedUser = Depends(current_user),
):
    """
    回答一次人工审批，让停在 `waiting_input` 的运行继续。

    ⚠️ 参数怎么来，以及为什么必须**一次带全**
    -------------------------------------
    每个待答项的 `interrupt_id` 与可选项都来自
    `GET /runs/{run_id}/events` 上的 `hitl.requested` 事件：它的 `data` 里
    既有 `interrupt_id`（回答时原样带回来），也有 `options`（这次该画哪
    几个按钮）。

    扇出下**多个工具可以同时等审批**，所以一次运行可能挂着好几个
    `hitl.requested`。`answers` 必须凑齐**当前所有**的 `interrupt_id`，少
    一个或给多了都 409 —— 少了那个会静默留在待办里，而本次请求返回 200
    （一次"成功"的什么都没推进）。

    ⚠️ 没有响应体
    -----------
    与 session.yaml 一致（那里只写 "Resume accepted"）。受理的那一刻库里
    还是 `waiting_input` —— 把它说成 `running` 是个还没成为事实的断言。
    后续进展继续看事件流：`hitl.resolved` 表示这次回答被受理，之后图接着
    跑，最终以 `run.completed` / `run.failed` 关流。

    ⚠️ 只受理**一次**
    ------------
    恢复被派发之后运行不再处于 `waiting_input`，重复调用会 409。这与
    "同一个回答提交两次"是同一件事的两面 —— 幂等要客户端按 `interrupt_id`
    自己做，服务端不做（做了就得存"哪个回答用过"，而那是另一份状态）。

    Raises:
        NotFoundError: 运行不存在，或它所属的会话不属于当前用户 -> 404。
        ConflictError: 这次运行不在 `waiting_input` -> 409。
    """

    # ⚠️ 这层 try/except 不能省。
    #
    # 上面 docstring 承诺的是 409（ConflictError），而 ConflictError 继承自
    # ValueError —— 不转的话它会一路冒到 Starlette 的兜底处理器，变成一个
    # **500**：客户端拿到"服务器错误"，自然去重试，而真正的原因是"答案集合
    # 与当前待审批的对不上"（要么带上全部 interrupt_id，要么什么都别做）。
    # 与其余 5 个路由（create_session / get_session / list_messages /
    # create_run / stream_run_events）保持同一形状。
    try:
        await service.resume_run(
            run_id,
            request,
            user_id=_owner_id(user),
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/runs/{run_id}/cancel",
    tags=["Runs"],
)
async def cancel_run(
        run_id: UUID,
        service: AgentRunService = Service[AgentRunService],
        user: AuthenticatedUser = Depends(current_user),
):
    """
    取消一次正在跑的运行。

    ⚠️ 它取消的是**正在执行的任务**，不是"那个 run 行"
    ------------------------------------------------
    任务在下一个 await 点抛出 `CancelledError`，然后 handler 走取消收尾：

        rollback 到运行前的 checkpoint（这轮写下的半截状态丢掉）
        状态记成 `cancelled`
        放掉 thread 锁

    ⚠️ 没有响应体，也不保证**已经**取消完
    -----------------------------------
    与 session.yaml 一致（那里只写 "Cancel accepted"）。取消是异步的：
    受理之后任务还要跑到下一个 await 点才真的停。要看结果，盯事件流上的
    `run.failed`（`data.status` 是 `cancelled`）或查库。

    ⚠️ 停在审批上的运行**不能**这样取消（409）
    --------------------------------------
    它没有在跑的任务 —— 它在等一次 `resume`。见 `cancel_run` 服务契约。

    Raises:
        NotFoundError: 运行不存在，或它所属的会话不属于当前用户 -> 404。
        ConflictError: 状态不允许取消（-> 409，文案里说明该做什么）。
    """

    # 同 resume_run：docstring 承诺的 409 要靠这一层转出来，否则是 500。
    try:
        await service.cancel_run(
            run_id,
            user_id=_owner_id(user),
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/runs/{run_id}/events",
    tags=["Runs"],
    # ⚠️ response_class 必须给：不给的话 FastAPI 会按默认的 JSON 响应
    # 往 200 里再加一条 application/json，于是文档写的是
    # "json 或 text/event-stream"，而契约（session.yaml）只声明了后者。
    # 运行时其实一直是对的（StreamingResponse 自己的 media_type 优先），
    # 错的只是文档 —— 但文档错了就会有人照着它去解析 JSON。
    response_class=StreamingResponse,
    responses={
        200: {
            "description": "SSE event stream",
            "content": {
                "text/event-stream": {
                    "schema": {"type": "string"},
                },
            },
        },
        404: {"description": "运行不存在"},
        503: {
            "description": (
                "事件流不可用：Redis 没起来，或并发订阅的 run 数已达上限"
            )
        },
    },
)
async def stream_run_events(
        run_id: UUID,
        request: Request,
        user: AuthenticatedUser = Depends(current_user),
) -> StreamingResponse:
    """
    订阅一次运行的事件流（SSE）。

    事件名与事件形状见 `app/common/event`；本函数只负责"把流接到 HTTP 上"。

    帧长这样（`text/event-stream`）：

        id: 1759412273324-0
        event: message.delta
        data: {"event": "message.delta", "topic": "agent_run", ...}

    因此浏览器可以直接 `addEventListener("message.delta", ...)`，不必先
    解 JSON 再自己分派。

    流会**从头补发**，晚连的客户端也能看到全过程；断线重连时把最后收到的
    `id` 填进 `Last-Event-ID` 头即可续上。

    ⚠️ 什么时候**关**
    ----------------
    三种"没有下文了"都会正常结束响应（不报错）：

        run.completed / run.failed   运行终止了
        等审批（hitl.requested）      运行暂停了。**所有** interrupt 都写进
                                    流、且 DB 已转 waiting_input 之后才关
        run 不会再产出事件            空闲探活发现执行器没了、或行已终止
                                    （崩溃 / 重启后的残留连接）

    前两种之外，客户端可随时断开；服务端在心跳节拍上发现（最多滞后一个
    `block_ms`）并及时摘掉订阅，不会把连接挂着。

    ⚠️ 同一个 run 的多个订阅者**共用一条 Redis 连接**
    ---------------------------------------------
    见 app/agent/event/run_stream.py —— 这是"并发订阅数打满连接池"那个
    故障的根治办法。

    Raises:
        NotFoundError: 运行不存在，**或它所属的会话不属于当前用户**
                       -> 404（同一个响应，不泄露后者）。
        HTTPException: Redis 没起来、或并发订阅的 run 数到上限 -> 503
                       （而不是给一条永远空着的流）。
    """

    # 1. 运行存在吗？—— 真相在库里，不在 Redis。
    #
    # 键可能因为 TTL 过期而消失（运行还在），也可能还没建（刚建好、
    # 第一个事件还没发），所以"看键在不在"两种情况都会判错。
    #
    # ⚠️ 这里刻意**不用** Service[...] 依赖：那个依赖会把一段请求级事务
    # 开到**响应结束**，而下面是条长连接（可能挂几分钟）。单独开一段
    # 短事务，查完立刻放掉，流式部分不再持有任何数据库连接。
    factory = get_service_factory()

    try:
        async with factory.scope(AgentRunService) as service:
            await service.get_run(run_id, user_id=user.subject)

    except ValueError as exc:
        raise to_http_error(exc) from exc

    # 2. 没有广播端 = 没有 Redis。给明确的 503。
    #
    # 退化成"一条永远不会有事件的空流"是最坏的选择：客户端看不出哪里
    # 不对，只会一直等 —— 那正是 SSE 最难查的失败形态。
    registry = get_run_stream_registry()

    if registry is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "事件流不可用：Redis 数据源没有起来。"
                "检查 app/config/datasource/redis.yaml，以及启动日志里那条 warning。"
            ),
        )

    # 3. 挂上订阅者。
    #
    # ⚠️ 这一步必须在**返回响应之前**做完（因此 attach 是同步的）：
    #    上限检查一旦失败要给 503，而 StreamingResponse 一旦开始吐状态行
    #    就再也改不成 503 了。
    try:
        attachment = registry.attach(
            run_id,
            start=request.headers.get("last-event-id") or "0",
        )

    except TooManyRunStreamsError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return StreamingResponse(
        _frames(attachment, request=request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # 反向代理（nginx 等）默认会缓冲响应，那会把"实时"变成
            # "攒够一块再发" —— 正好是 SSE 要避免的。
            "X-Accel-Buffering": "no",
        },
    )


async def _frames(
        attachment: RunStreamAttachment,
        *,
        request: Request,
) -> AsyncIterator[str]:
    """
    把广播端给的记录编成 SSE 帧。

    什么时候结束，判断**不在**这里 —— 那是广播端的事（见
    app/agent/event/run_stream.py 的模块 docstring）。这里只翻译两种东西：

        一条记录   编成 SSE 帧发出去
        None      阻塞到点、没有新记录 —— 发一个注释行当心跳，顺便问一句
                  客户端还在不在（唯一能**及时**发现断线的地方）

    ⚠️ 收尾信号与错误都不是"内容"，因此循环体里判别不到
    ----------------------------------------------
    收尾时迭代**正常结束**（`async for` 退出），读不下去时**抛出**。这条
    契约由 `RunStreamAttachment._iterate` 保证，所以这里不要写"如果这一条
    是 StreamClosed 就 return"那类分支 —— 它永远不会命中，而写错的代价
    是有人以为自己在处理收尾。

    抛出那条是刻意的：与 app/agent/event/reader.py 开头那条立场一致，
    订阅方拿到一个"卡住的空流"比拿到一个明确的错误糟得多。

    ⚠️ `finally` 里一定要摘掉订阅者
    -----------------------------
    不过**不能只靠它**：Starlette 直接 `async for` 这个生成器，没有用
    `aclosing` 包一层（responses.py:250），所以客户端断开时 `finally` 要等到
    被垃圾回收才跑。真正及时的路径是心跳里的 `is_disconnected()` —— 那会让
    这个生成器**正常返回**，`finally` 当场就执行。见
    run_stream.py 的 `RunStream._iterate`。
    """

    try:
        async for item in attachment:

            if item is None:
                # 阻塞到点了、还没有新记录 —— 发一个注释行当心跳。
                if await request.is_disconnected():
                    return

                yield HEARTBEAT
                continue

            # 正常收尾不额外补发"关闭"之类的事件：终止事件本身就是那个
            # 信号，客户端收到就知道可以收尾了。
            yield sse_frame(item)

    finally:
        attachment.aclose()
