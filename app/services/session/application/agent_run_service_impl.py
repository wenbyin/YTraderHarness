"""
AgentRunService 的默认实现。

事务边界由调用方决定，本服务内部不 commit / rollback。

派发：只**登记**，不在本事务里执行
--------------------------------
建完运行后，本实现把一次调度塞进 `ctx.post_commit`，由组合根在请求事务
**提交之后**执行（与 document 的 create_document 同一套，见
app/common/service/factory.py::scope）。

为什么不能在这里直接 dispatch：

    本方法跑在请求级事务里。此刻运行行还没提交，后台任务立刻去读
    必然查不到；而且图可能跑上几分钟，压在请求上就是一次超时。

post_commit 的那段代码在 `async with session.begin()` 退出后执行，请求抛
异常回滚时会被整段跳过 —— 因此不会出现"事务回滚了但运行已经派出去"。

thread_id 按会话稳定，同一 thread 同时只跑一次
--------------------------------------------
运行不再各自拥有一条 thread：thread_id 存在会话行上（首轮生成，此后
复用），多轮因此共用一个 checkpoint —— 这正是多轮历史能累积的前提。
见 `_resolve_thread_id`。

准入控制在**请求期**：同一会话已有运行在途、或会话已归档（archived，
只读）时，`create_run` 直接抛 `ConflictError`（-> 409），而不是让它排进
调度器（那是"排队"，调用方拿到的是一个可能很久以后才跑的 run）。会话
行用 `get_for_update` 读取，与归档串行化。锁的释放点只有一个 ——
dispatch handler 的 `finally`；另有两处兜底：请求事务回滚（补偿钩子）与
调度被拒收（post-commit 钩子）。三者的完整理由见各自的注释。

⚠️ 锁是**进程内**的，因此部署必须 workers=1
------------------------------------------
多进程下每个进程各有一份锁，护栏形同虚设，而 checkpointer 是共享的
Postgres —— 后果不只是"没拦住"，是并发写同一份 checkpoint。
见 app/infra/graph/thread_lock.py。

用户消息在这里落库（消息写入的两个点之一）
----------------------------------------
`create_run` 除了建运行行，还往 agent_message 表写一条**用户消息** —— 这是
"接收时"的定义（见 `_record_user_message`）。另一个写入点在派发 handler
里（本轮产出的 assistant 消息），见 application/agent_run_dispatch.py。

两处合起来构成一条规矩：**每条消息在它自己的结局确定的那一刻落库**。
用户消息的结局在接收时就确定了（一句话说完就是完整的），assistant 消息的
结局要等图跑完。因此不是"运行结束时把所有消息一起写"。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from app.agent.event import publish_trace_event
from app.agent.graph.react.catalog import AGENT_TYPE
from app.agent.prompt import get_prompt_provider
from app.common.environment import current_environment
from app.common.event import EventType, SpanKind
from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import RollbackHook, ServiceContext
from app.infra.graph import (
    DispatchEnvelope,
    GraphDefinition,
    get_thread_run_lock,
    new_session_thread_id,
)
from app.infra.graph.dispatcher import GraphDispatcher
from app.services.prompt.catalog import DEFAULT_DUTY

from ..domain.entities import AgentMessage as AgentMessageEntity
from ..domain.entities import AgentRun as AgentRunEntity
from ..domain.entities import AgentSession as AgentSessionEntity
from ..domain.schemas import (
    CreateRunRequest,
    MessageKind,
    MessageRole,
    MessageStatus,
    ResumeRequest,
    Run,
    RunStatus,
    SessionStatus,
)
from ..repository.agent_run import AgentRunRepository
from ..repository.agent_session import AgentSessionRepository
from ..repository.message import MessageRepository
from ._assembly import build_run
from .agent_run_dispatch import REACT_GRAPH_NAME, pending_interrupt_ids
from .agent_run_service import AgentRunService

logger = logging.getLogger(__name__)

#: 组合根在事务提交后要执行的钩子（与 document 侧的约定同名同形）。
PostCommitHook = Callable[[], Awaitable[None]]


class AgentRunServiceImpl(AgentRunService):

    def __init__(
            self,
            *,
            agent_run_repository: AgentRunRepository,
            agent_session_repository: AgentSessionRepository,
            message_repository: MessageRepository,
            dispatcher: GraphDispatcher | None = None,
            react_definition: GraphDefinition | None = None,
            post_commit: list[PostCommitHook] | None = None,
            rollback: list[RollbackHook] | None = None,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._runs = agent_run_repository
        self._sessions = agent_session_repository

        #: 已编译的 react 图。`resume_run` 靠它读 checkpoint 上**真实的**
        #: 待处理中断集合，好把"少了/多了"在**请求期**同步报出来
        #: （见 `resume_run` 的校验）。
        #:
        #: 没接线（单元测试直接构造）时为 None —— 那时不校验，但那种场合
        #: 也**没有 dispatcher**，resume 本来就不会真的发生（见 _schedule_resume
        #: 的早退）。
        self._react_definition = react_definition

        # 用户消息是在建运行这一步写下的 —— 见 _record_user_message。
        self._messages = message_repository

        # 没接线（单元测试直接构造）时为 None：那时运行照旧落库，
        # 只是没人推进它 —— 与本次改动之前的行为一致。
        #
        # ⚠️ 它同时决定**要不要取 thread 锁**：没有 dispatcher 就没人会
        # 推进这次运行，也就没人会放锁 —— 取了就是死锁自己。见 create_run。
        self._dispatcher = dispatcher

        # 与 document 的同一个做法：不显式给就接到上下文那条上，
        # 经 ServiceFactory.scope() 构造时那正是提交后要遍历的列表。
        self._post_commit = (
            post_commit
            if post_commit is not None
            else self._ctx.post_commit
        )

        # 补偿钩子那条。回滚时把锁还回去 —— 见 _acquire_thread_lock。
        self._rollback = (
            rollback
            if rollback is not None
            else self._ctx.rollback
        )

    async def get_run(
            self,
            run_id: UUID,
            *,
            user_id: str,
    ) -> Run:

        entity = await self._runs.get(run_id)

        if entity is None:
            raise NotFoundError(f"运行不存在：{run_id}")

        # 归属校验落在**会话表**上，不在 run 仓储里：后者被后台派发共用
        # （agent_run_dispatch._load_run），那里没有请求上下文、绝不能带
        # 过滤。会话表是归属的唯一真相。
        #
        # 文案与上面那条**故意一模一样**：响应里不该多出"这个 run 存在
        # 但不属于你"这种信息。
        session = await self._sessions.get(
            entity.session_id, user_id=user_id,
        )

        if session is None:
            raise NotFoundError(f"运行不存在：{run_id}")

        return build_run(entity)

    async def create_run(
            self,
            session_id: UUID,
            request: CreateRunRequest,
            *,
            user_id: str | None,
            caller_identity: dict[str, Any] | None = None,
    ) -> Run:

        # ⚠️ 顺序有讲究：**便宜且无副作用**的校验排在最前，取锁排在它们
        # 之后。反过来的话，一次 404 / 400 也会先占上锁再立刻还回去 ——
        # 平白多出一条能失败的路径，而且失败方式很难看（占了锁又抛）。
        # ⚠️ 锁定读，不是普通 get
        # ----------------------
        # 删除会话（归档）也锁这一行。只读的话会出现：
        #     create_run 读到 active -> 归档提交 -> create_run 插入运行
        # 结果给一个已归档的会话开了新运行。`FOR UPDATE` 让两边在会话行上
        # 排队，后拿到锁的一方读到的是新状态。
        session = await self._sessions.get_for_update(
            session_id, user_id=user_id,
        )

        # 会话必须存在，**且对调用方可见**（归属过滤在仓储里做）。
        # 运行是派生自会话的，没有会话的运行是孤儿。
        # 按 404 处理（URL 里的那个资源不存在），而不是 409 —— 冲突是
        # "请求引用的对象不对"，而这里是"请求的路径不存在"。
        # 别人的会话走**同一条** 404，不区分 —— 免得泄露某个 id 是否存在。
        if session is None:
            raise NotFoundError(f"会话不存在：{session_id}")

        # 归档 = 只读。放在取 thread 锁之前：这是一次不产生任何副作用的
        # 拒绝，不该先占锁再还。
        if session.status == SessionStatus.ARCHIVED.value:
            raise ConflictError(
                "会话已归档，不能再开新运行。"
            )

        # 空白输入在这里就拒掉（-> 400），不放进图：图对空上下文的反应是
        # 在 prompt 组装那一步抛 "组装之后的上下文是空的"，那会变成一次
        # failed 的运行 —— 而"你没说话"本该是调用方的一次 400。
        # DTO 的 min_length=1 挡不住只有空格的字符串。
        text = request.input.text.strip()

        if not text:
            raise ValueError("input.text 是空的，没有可处理的内容。")

        # ⚠️ run_id 必须在**取锁之前**定下来
        # ----------------------------------
        # 它是锁的**持有者令牌**，而 handler 放锁时用的正是同一个值
        # （envelope.pipeline_run_id）。两者一旦不是同一个数，锁就永远
        # 放不掉 —— 那种卡死要重启进程才能恢复，且没有任何日志指向它。
        now = datetime.now(timezone.utc)
        run_id = uuid4()

        thread_id = await self._resolve_thread_id(session)

        # 这次运行用哪份**职责提示词** —— 「选职责」的落点就在这一步，
        # 此后这一轮运行里不再变（见 _resolve_duty_key）。
        #
        # 排在取锁之前：它只是一次进程内的取用（不碰数据库），而取锁之后
        # 那段里的每一步都是"要落库的东西"，不该混进一个只读动作。
        duty_key = await _resolve_duty_key()

        # Trace 的开关：客户端传了才开。空白串归一成 None，避免生成一个
        # 空 trace_id 的 Redis 键（它看起来与"没开"一模一样）。
        trace_id = (request.trace_id or "").strip() or None

        # 没有 dispatcher 就不取锁：没人推进这次运行，也就没人放锁 ——
        # 取了就是把自己锁死。条件必须与 _schedule 的早退**完全一致**。
        if self._dispatcher is not None:
            self._acquire_thread_lock(thread_id, run_id)

        entity = await self._runs.add(
            AgentRunEntity(
                run_id=run_id,
                session_id=session_id,
                status=RunStatus.QUEUED.value,
                input=request.input.model_dump(),
                meta=dict(request.metadata),
                # 记的是**发起人**（token 里的），不是 session.user_id：
                # 存量公共会话（user_id 为 NULL）上仍能记下真实执行者，
                # 图也因此拿到一个非空的 user_id（见 _initial_state）。
                user_id=user_id,
                # 调用者身份的快照（groups / roles）。**必须**在这里存下来：
                # 派发是异步的，图跑起来时手里已经没有 token 了，而工具级的
                # checkauth 拦截器要靠它判"这个人够不够格调这个工具"。
                # 不给就存空 —— 那时受保护的工具会按"没有身份"一律拒绝。
                caller_identity=dict(caller_identity or {}),
                # 这一轮用哪份职责提示词。存下来之后：派发时进图 state
                # （builder 节点据此取提示词），写消息时填进
                # agent_message.business_agent_id —— 三处因此指向同一个值。
                duty_key=duty_key,
                # Trace 的开关就是"客户端有没有传 trace_id"。
                trace_id=trace_id,
                created_at=now,
                updated_at=now,
            )
        )

        await self._record_user_message(
            session_id=session_id,
            run_id=run_id,
            text=text,
            now=now,
            # 用户消息也带上这次的职责：同一轮的两条消息（user / assistant）
            # 指向同一个职责，只写一半会让"按职责统计"变成一件要先看
            # role 的事。
            business_agent_id=duty_key,
        )

        # 「发消息」这个动作也要反映到**会话**上：消息写在 agent_message，
        # agent_session 一行都没变，所以这里显式把 updated_at 推到当下
        # （理由见仓储 touch 的说明）。同一个请求事务，与运行行、用户消息
        # 一起提交 —— 不会出现"消息在、时间没动"的半边状态。
        await self._sessions.touch(session_id)

        # run.accepted：服务已经确定 run_id 并落下了运行行。
        #
        # 放在 **post-commit** 而不是这里直接发：本方法跑在请求事务里，
        # 事务回滚时"已接受"就是一个假事实。post-commit 保证事件与
        # 运行行同时成立。
        async def _accepted() -> None:
            await publish_trace_event(
                EventType.RUN_ACCEPTED,
                trace_id=trace_id,
                run_id=str(run_id),
                thread_id=str(session_id),
                span_id=str(run_id),
                span_kind=SpanKind.RUN,
                span_name="run",
                payload={"agent_type": AGENT_TYPE},
            )

        self._post_commit.append(_accepted)

        self._schedule(
            session_id,
            run_id,
            thread_id,
            trace_id=trace_id,
        )

        return build_run(entity)

    async def resume_run(
            self,
            run_id: UUID,
            request: ResumeRequest,
            *,
            user_id: str | None,
    ) -> None:

        entity = await self._runs.get(run_id)

        if entity is None:
            raise NotFoundError(f"运行不存在：{run_id}")

        # 归属校验与 get_run 走同一条路：落在**会话表**上，别人的会话得到
        # 一模一样的 404 —— 不区分，免得泄露某个 id 是否存在。
        #
        # 锁定读：与归档串行化。真正的兜底还有 dispatch handler 的终态
        # guard —— 归档可能已经把这个 run 硬取消，而本次 resume 的
        # post-commit 投递晚于那次提交。
        session = await self._sessions.get_for_update(
            entity.session_id, user_id=user_id,
        )

        if session is None:
            raise NotFoundError(f"运行不存在：{run_id}")

        if session.status == SessionStatus.ARCHIVED.value:
            raise ConflictError(
                "会话已归档，不能再恢复其中的运行。"
            )

        # 只有停在 waiting_input 的运行可以被恢复。其余状态走 409：请求
        # 本身没毛病（run 存在、也归你），是它与**当前状态**冲突 ——
        # 跑完的没什么可恢复，正在跑的还没停下来、轮不到 resume 插队。
        if entity.status != RunStatus.WAITING_INPUT.value:
            raise ConflictError(
                f"运行没有在等人工输入（当前状态 {entity.status}），"
                "无法恢复。"
            )

        # thread_id 从**会话行**取，这里不再生成：它必须与建运行时取锁用的、
        # 以及 checkpointer 存 state 用的是同一个字符串（见 _schedule 的说明）。
        thread_id = session.thread_id

        if not thread_id:
            # 结构上不该发生：能停在 waiting_input 的运行必然有过一条 thread
            # （create_run 里 _resolve_thread_id 保证非空）。真出现说明库被
            # 人改过 —— 报出来，不要现编一个 thread_id 去 resume。
            raise ConflictError(
                "这个会话没有 thread_id，无法恢复这次运行。"
            )

        await self._require_complete_answers(thread_id, request)

        self._schedule_resume(
            session_id=entity.session_id,
            run_id=run_id,
            thread_id=thread_id,
            request=request,
        )

    async def _require_complete_answers(
            self,
            thread_id: str,
            request: ResumeRequest,
    ) -> None:
        """
        `answers` 必须**不多不少**正好覆盖当前的待处理中断集合。

        ⚠️ 为什么这条校验必须在**请求期**做
        ---------------------------------
        少了会怎样是实测过的：已答的那个生效、没答的**继续停**，图重新回到
        `waiting_input`，而调用方拿到的是 200 —— 一次"成功"的请求什么都没
        推进，且**没有任何地方报错**。那正是最难查的一类问题。

        多了/给错则说明客户端手上的集合是过期的（多半漏读了后到的
        `hitl.requested`）—— 那种情况让它静默通过，等于把一个必然错位的
        回答交给引擎。

        ⚠️ 409 而不是 400
        ---------------
        请求本身没毛病（结构合法、运行归你、状态也对），是它与**当前状态**
        不一致 —— 与"运行不在 waiting_input"同一类。`to_http_error` 把
        ConflictError 映射成 409，API 层不用改。
        """

        if self._react_definition is None:
            # 没接线（单元测试直接构造）—— 那时也没有 dispatcher，resume
            # 不会真的发生，校验也就无从谈起。
            return

        pending = set(
            await pending_interrupt_ids(
                self._react_definition, thread_id=thread_id
            )
        )

        given = {answer.interrupt_id for answer in request.answers}

        if given == pending:
            return

        missing = sorted(pending - given)
        unknown = sorted(given - pending)

        # 错误里把两边的差集都列出来：调用方据此能直接改，不必再去翻
        # 事件流。pending 本身不是秘密 —— 调用方本来就该从 hitl.requested
        # 拿到它，而且这里已经过了归属校验。
        raise ConflictError(
            "answers 与当前待处理的中断不一致："
            f"缺少 {missing or '无'}，多出（或已失效）{unknown or '无'}。"
            f"当前待处理的是 {sorted(pending)}。"
            "一次 resume 必须把**当前所有**待审批一次带上。"
        )

    async def cancel_run(
            self,
            run_id: UUID,
            *,
            user_id: str | None,
    ) -> None:

        entity = await self._runs.get(run_id)

        if entity is None:
            raise NotFoundError(f"运行不存在：{run_id}")

        session = await self._sessions.get(
            entity.session_id, user_id=user_id,
        )

        if session is None:
            raise NotFoundError(f"运行不存在：{run_id}")

        # ⚠️ `waiting_input` 单独给一条文案，因为它是**最容易被误用**的一格
        # ------------------------------------------------------------------
        # 用户看到"这次运行停着"，很自然会想"那取消它"。但停在审批上的运行
        # 没有在跑的任务可取消 —— 它要的是一次回答。笼统地说"当前状态不能
        # 取消"会让人以为这条路坏了。
        if entity.status == RunStatus.WAITING_INPUT.value:
            raise ConflictError(
                "这次运行停在上一次人工审批上，没有正在执行的任务可取消。"
                "请用 POST /runs/{run_id}/resume 回答那次审批"
                "（回答 reject 就等于不要这一步）。"
            )

        if entity.status not in (
                RunStatus.QUEUED.value,
                RunStatus.RUNNING.value,
        ):
            raise ConflictError(
                f"运行已经结束（当前状态 {entity.status}），无法取消。"
            )

        # `_dispatcher is None` 只出现在单元测试直接构造本服务的场合 ——
        # 那时没有任何东西在跑这次运行，也就没有可取消的对象。
        accepted = (
            self._dispatcher is not None
            and await self._dispatcher.cancel(run_id)
        )

        if not accepted:
            # 到这儿有两种成因，都不该报成功：
            #   1. 上面那次读之后它就跑完了（状态还没回写，竞态窗口很小）
            #   2. 跑它的不是这个进程（进程内调度器的固有限制）
            # 调用方重新查一次状态就能分清，所以文案把两个可能都说出来。
            raise ConflictError(
                "取消没有被受理：这次运行要么刚刚已经结束，要么不归本进程"
                "调度。请重新查一次运行状态。"
            )

    # ========================================================
    # 用户消息
    # ========================================================

    async def _record_user_message(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            text: str,
            now: datetime,
            business_agent_id: str | None,
    ) -> None:
        """
        把用户这一轮的输入写进 agent_message 表 —— **接收时**就写。

        为什么在这里而不是派发 handler 里
        ------------------------------
        这是 `POST /sessions/{id}/runs` 的请求事务，写在这里意味着：

            接口一返回，前端就能从 agent_message 表读到自己刚发的那句话，
            不必等 worker 抢到这次运行；
            而且它与运行行**同生共死** —— 回滚一起回滚，不会出现
            "有运行行、没有用户消息"这种半边状态。

        派发是在本事务提交**之后**才登记的（见 _schedule），也就是说这两行
        数据一定比这次运行更早成为事实 —— handler 那边据此回填
        `reply_to_message_id`（它查的就是本方法写下的这一行）。

        ⚠️ seq 的取号在锁内
        ------------------
        本方法排在 `_acquire_thread_lock` 之后，因此"取号 + 插入"这段被
        那个 thread 锁串行化了，同一会话不会有两个请求同时取到同一个号。

        锁是**进程内**的，因此这条与模块 docstring 里那条"部署必须
        workers=1"是同一个前提。更弱的一种情况：装配时没给 dispatcher，
        那时根本不取锁（见 create_run 里的判断），同一会话的并发建运行
        可能取到同一个 seq —— 表上的 uq_agent_message_session_seq 会拦下来
        （不会静默写坏），但报出来的是一次唯一键冲突的 500，而不是 409。
        生产路径上 dispatcher 一定有值，这条只在单元测试里成立。

        ⚠️ status 永远是 completed
        ------------------------
        用户消息不存在"生成到一半"：它是用户完整说出的一句话。运行后来失败
        也好、被取消也好，都不改变"用户确实这么说了一句"这个事实 —— 那是
        运行的状态，不是这条消息的状态（见 MessageStatus 的说明）。
        """

        # 取号与插入必须在同一个事务里（这里天然是 —— 都在请求事务内）。
        seq = await self._messages.next_seq(session_id)

        await self._messages.add_many([
            AgentMessageEntity(
                message_id=uuid4(),
                session_id=session_id,
                run_id=run_id,
                seq=seq,
                role=MessageRole.USER.value,
                kind=MessageKind.CHAT.value,
                # 产出这条消息的**职责**（duty_key），与运行行上那个值
                # 同一个来源（见 create_run 里的 _resolve_duty_key）。
                business_agent_id=business_agent_id,
                # 纯文本也包成内容块数组 —— 契约与库里的 CHECK 都要求数组。
                content=[{"type": "text", "text": text}],
                # reasoning_content 刻意不传（吃实体默认的 None）：思维链是
                # **模型**产出的东西，用户消息没有它。要传也只能是 None。
                status=MessageStatus.COMPLETED.value,
                # 幂等 ID 要客户端给，而 CreateRunRequest 里没有这个字段：
                # 等将来的 POST /sessions/{id}/messages。
                client_message_id=None,
                # 用户消息没有"被回复的那条"。
                reply_to_message_id=None,
                meta={},
                created_at=now,
                # 与 created_at 同一个值：消息不可变，见 message_recorder
                # 的 to_entities 与 agent_message 的 ORM 说明。
                updated_at=now,
            )
        ])

    # ========================================================
    # thread_id 与锁
    # ========================================================

    async def _resolve_thread_id(
            self,
            session: AgentSessionEntity,
    ) -> str:
        """
        本会话生效的 thread_id —— 已有就复用，没有就生成并落库。

        复用是**多轮历史的前提**：thread_id 稳定，checkpointer 才会把
        新一轮的 HumanMessage 追加到上一轮那份 state 上，而不是从零开
        一份。每轮换 thread 正是"同一会话多次 invoke，messages 里只有
        一条 HumanMessage"的成因。

        这里先看实体上的值（快路径，省一次条件更新）；为空才走
        `ensure_thread_id` —— 那条路是并发的（两个请求同时给一个还没
        有 thread_id 的历史会话发运行），所以它必须用条件更新，理由见
        仓储契约的 docstring。
        """

        if session.thread_id:
            return session.thread_id

        return await self._sessions.ensure_thread_id(
            session.session_id,
            new_session_thread_id(
                current_environment().value,
                AGENT_TYPE,
                uuid4(),
            ),
        )

    def _acquire_thread_lock(
            self,
            thread_id: str,
            run_id: UUID,
    ) -> None:
        """
        取该 thread 的运行锁。已被占用则抛 `ConflictError`（-> 409）。

        为什么在这里取（请求期）而不是派发时取
        -------------------------------------
        要在**建运行之前**就告诉调用方"这个会话正忙"。派发是 post-commit
        之后的事，那时响应早就发出去了，只能返回 200 —— 而调用方拿到
        一个永远不会执行的 queued 运行。

        ⚠️ 回滚钩子必须登记在这里
        ------------------------
        锁是在事务**提交之前**拿到的，而事务可能不提交（业务校验失败、
        或 COMMIT 本身失败）。不还的话这个会话此后每次建运行都 409，
        只能重启进程 —— 而"提交失败"那条路业务代码自己完全看不到。
        见 common/service/base.py 的 RollbackHook。
        """

        lock = get_thread_run_lock()

        holder = lock.try_acquire(thread_id, run_id=run_id)

        if holder is None:

            async def _release() -> None:
                # 带 run_id：只放自己那把。迟到的释放落在别人刚拿到的
                # 锁上时是 no-op —— thread_id 按会话稳定，这是会发生的。
                lock.release(thread_id, run_id=run_id)

            self._rollback.append(_release)

            return

        # 409 而不是 400：请求本身没毛病，是它与**当前状态**冲突。
        # to_http_error 把 ConflictError 映射成 409，API 层不需要改。
        raise ConflictError(
            f"该会话已有运行在进行中（run {holder.run_id}）。"
            "同一会话同时只能有一次运行，请等它结束。"
        )

    # ========================================================
    # Dispatch
    # ========================================================

    def _schedule(
            self,
            session_id: UUID,
            run_id: UUID,
            thread_id: str,
            *,
            trace_id: str | None = None,
    ) -> None:
        """
        登记一次调度。**不执行** —— 本方法跑在请求事务里，见模块 docstring。

        `thread_id` 由调用方传入而**不在这里生成**
        ----------------------------------------
        它必须与三处是**同一个字符串**：取锁时用的那个、这里进 envelope
        的那个、以及 checkpointer 存 state 的那个。在这里现算一次就等于
        给"它们真的是同一个"增加一处能漂移的地方 —— 而漂移的表现是
        "锁锁住了一条 thread，图却跑在另一条上"，护栏静默失效。

        ⚠️ 这里没有 agent_id -> 图 的查表，是刻意的
        ------------------------------------------
        从这张 API 进来的只有一种图，前端不选图、也不感知 agent_id
        （决定见 domain/schemas/__init__.py）。因此每个运行都走
        REACT_GRAPH_NAME，agent_id 只是会话上记着的一个标签。
        将来真出现第二种图时，查表加在这里。
        """

        if self._dispatcher is None:
            return

        envelope = DispatchEnvelope(
            graph_name=REACT_GRAPH_NAME,
            # run_id 就是本次运行的 pipeline_run_id：一次运行一个身份，
            # 不另造一个 id 让观测与业务各说各话。
            pipeline_run_id=run_id,
            thread_id=thread_id,
            # handler 从库里读运行行，不从 payload 取内容 ——
            # 这里放 session_id 只是为了日志/trace 里能看出是哪个会话。
            payload={"session_id": str(session_id)},
        )

        dispatcher = self._dispatcher

        async def _hook() -> None:
            handle = await dispatcher.dispatch(envelope)

            if not handle.accepted:
                # ⚠️ 必须在这里放锁
                # ----------------
                # 受理失败 = 永远不会有 handler 来收尾，而 handler 的
                # finally 是**唯一**放锁的地方。不放的话这个会话此后每次
                # 建运行都 409，只能重启进程才能恢复。
                #
                # 三种拒收原因（正在关闭 / 图没注册 / 队列满）在这里不做
                # 区分 —— 对锁而言它们的后果完全一样。
                #
                # release 是同步且永不抛的，因此放在这个 post-commit 钩子里
                # 是安全的：钩子跑在事务提交之后，那时再抛只会变成 500。
                get_thread_run_lock().release(thread_id, run_id=run_id)

                # 受理失败已由 dispatcher 明确返回，不抛 —— 此刻业务事务
                # 已经提交，抛出去只会变成 500，而调用方需要的事实是
                # "运行建好了但没跑起来"。运行行会停在 queued，看得见。
                logger.warning(
                    "Agent run %s dispatch rejected (thread %s): %s",
                    run_id,
                    thread_id,
                    handle.message,
                )
                return

            # run.queued：dispatcher 已受理（进入进程内队列）。拒收时不发
            # —— 那次运行从未进过队列，发了就是伪造。
            await publish_trace_event(
                EventType.RUN_QUEUED,
                trace_id=trace_id,
                run_id=str(run_id),
                thread_id=str(session_id),
                span_id=str(run_id),
                span_kind=SpanKind.RUN,
                span_name="run",
                payload={"queue_name": "agent_run"},
            )

        self._post_commit.append(_hook)

    def _schedule_resume(
            self,
            *,
            session_id: UUID,
            run_id: UUID,
            thread_id: str,
            request: ResumeRequest,
    ) -> None:
        """
        登记一次**恢复**派发。时序与 `_schedule` 完全一样：本方法跑在请求
        事务里，真正进调度器是在 post_commit 之后。

        ⚠️ 这里**不取锁**，与 `_schedule` 不同
        ------------------------------------
        中断时锁没放（见 agent_run_dispatch 的 handler），它仍挂在**同一个
        run_id** 上。恢复是它的合法继承者，不需要再取 —— 再取一次会被自己
        挡住，直接 409。

        ⚠️ 派发被拒收时**不放锁**（与 `_schedule` 相反）
        ---------------------------------------------
        `_schedule` 里那条"受理失败就放锁"的前提是"锁是本次请求取来的"。
        这里锁是上一次运行留下的，放掉它等于让那条 thread 带着一个 pending
        任务重新变得可 invoke —— 正是锁要防的事。

        代价说清楚：受理失败时运行停在 `waiting_input` 且没人推进它。但那是个
        **看得见的**状态（与 create 停在 queued 同理），调用方可以再 resume
        一次。宁可让人重试一次，也不要损坏 checkpoint。
        """

        if self._dispatcher is None:
            return

        envelope = DispatchEnvelope(
            graph_name=REACT_GRAPH_NAME,
            pipeline_run_id=run_id,
            thread_id=thread_id,
            payload={
                # 只为日志/trace，与 _schedule 同一条。
                "session_id": str(session_id),
                # ⚠️ 恢复标记。handler 只看这一个键决定**驱动方式**：
                # 有它就是 `Command(resume=...)`，没有就是一轮新输入。
                #
                # 形状就是 `Command(resume=...)` 要的那个映射，而且**一次
                # 带全** —— 集合的完整性已经在 `_require_complete_answers`
                # 里校验过，handler 不再翻译、也不再检查。
                #
                # 放在 payload 里而不是给 DispatchEnvelope 加字段：调度器
                # 明确"不解释 payload，原样交给 handler"（见 dispatcher.py），
                # 而"怎么跑图"是 handler 的事，不是调度该知道的事。
                "resume": {
                    answer.interrupt_id: dict(answer.data)
                    for answer in request.answers
                },
            },
        )

        dispatcher = self._dispatcher

        async def _hook() -> None:
            handle = await dispatcher.dispatch(envelope)

            if not handle.accepted:
                logger.warning(
                    "Agent run %s resume dispatch rejected (thread %s): %s",
                    run_id,
                    thread_id,
                    handle.message,
                )

        self._post_commit.append(_hook)


# ============================================================
# internals
# ============================================================


async def _resolve_duty_key() -> str:
    """
    这次运行用哪份**职责提示词** —— 「选职责」在建运行这一步落定。

    为什么问 PromptProvider 而不是 import 那个常量
    -------------------------------------------
    取值本身今天确实是个常量（`app/services/prompt/catalog.py` 里的
    `DEFAULT_DUTY`），但**谁来决定它**是提示词域的知识，不是会话域的知识。
    将来"按什么选职责"变了（会话配置？Agent 目录？），要改的是
    `PromptService.default_duty_key` 一个地方 —— 而不是每个建运行的地方。

    为什么在服务里直接取进程级 Provider，而不是走 ServiceFactory 的 pool
    -----------------------------------------------------------------
    与 `UserServiceImpl` 取 `get_casdoor_component()` 同一条已经写明的
    规矩：**进程级、不碰 I/O、不花钱的东西直接取**，不必往 main.py 的
    pool 里加一个键（那是"机制"不是"依赖"）。

    ⚠️ 取不到**不拦住建运行**
    -----------------------
    回落默认职责 + 一条 WARNING。两点理由：

        今天回落值与正常值**相同**（都指向 catalog 里那份通用职责），
        所以它不掩盖任何东西；

        将来选择真的接上外部依据（库 / 配置中心）时，"选择不可用"不该
        让用户连一句话都发不出来 —— 那种失败应当降级到通用职责，而不是
        把建运行整个拒掉。

    因此这里**不抛**：线索在日志的 WARNING 与 `agent_run.duty_key` 那一列
    （回落时它等于默认职责，而"本该是别的职责"从这一列上看不出来 ——
    这是已知的取舍，等职责真的不止一份时再考虑往 run.meta 里记一笔）。
    """

    try:
        return await get_prompt_provider().duty_key()

    except Exception as exc:  # noqa: BLE001 - 降级，理由见上面那段

        logger.warning(
            "解析 duty_key 失败（%s: %s），这次运行回落到默认职责 %r。",
            type(exc).__name__,
            exc,
            DEFAULT_DUTY,
        )

        return DEFAULT_DUTY
