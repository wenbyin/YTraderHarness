"""
运行的读写契约。

对应 session.yaml：
    POST /sessions/{session_id}/runs     建运行
    POST /runs/{run_id}/resume           回答一次人工审批，让图继续
    POST /runs/{run_id}/cancel           取消一次正在跑的运行

⚠️ 契约上**没有** dispatch
-----------------------
派发是**实现细节**：它登记在请求事务的 post_commit 钩子里，提交之后由
组合根执行（见 application/agent_run_service_impl.py）。把它做成一个公开
方法，等于让调用方可以"建而不派"或"派两次" —— 而这两种都不是业务语义。

`resume` / `cancel` 则是真的业务动作，所以它们在契约上、也各有端点。

⚠️ `create_run` 返回的 Run 一定是 `queued`
----------------------------------------
不是"没接图的假象"，是时序使然：create 返回时事务还没提交，图也还没开始
跑。状态的推进（queued -> running -> waiting_input -> completed / failed）
由 application/agent_run_dispatch.py 的 handler 写进库。

`resume_run` **不回 Run**（规格里那个端点也只写 "Resume accepted"，没有
响应体）。这不是偷懒：接受受理的那一刻库里还是 `waiting_input`，而把它说成
`running` 是个还没成为事实的断言 —— 与上面那条同一种诚实。

真要看一次运行最后成了没有，当前只能查库 —— 规格里没有查运行的接口，
事件流（④）也还没做。这是已知的、写在 session.yaml 里的缺口。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import CreateRunRequest, ResumeRequest, Run

__all__ = ["AgentRunService"]


class AgentRunService(BaseService):

    @abstractmethod
    async def create_run(
            self,
            session_id: UUID,
            request: CreateRunRequest,
            *,
            user_id: str | None,
            caller_identity: dict[str, Any] | None = None,
    ) -> Run:
        """
        在某个会话下建一次运行，状态 `queued`，并登记一次派发。

        派发在请求事务提交后执行；受理失败（队列满、图没注册）不会让本
        方法抛错 —— 运行已经建好了，调用方需要的是那个事实，而不是 500。

        user_id 是**发起人**（Casdoor 的 sub），由 api 层从 access token 取。
        它既是归属校验的输入，也会落到 agent_run 上并传给图（见实现）。

        caller_identity 是**同一时刻的身份快照**（groups / roles，形状见
        `app/agent/tools/processor/context.py` 的 build_caller_context）。
        它必须在这里、由调用方一起给：派发是异步的，跑的时候手里已经没有
        token 了，而工具级的 checkauth 拦截器要靠它判"这个人够不够格调这
        个工具"。不给（None）时存空 —— 那时受角色/群组保护的工具会按
        "没有身份"一律拒绝（fail-closed），而不是放行。

        Raises:
            NotFoundError: 会话不存在，**或它不属于 user_id**。
            ValueError: input.text 为空或只有空白（-> 400）。
        """

        raise NotImplementedError

    @abstractmethod
    async def resume_run(
            self,
            run_id: UUID,
            request: ResumeRequest,
            *,
            user_id: str | None,
    ) -> None:
        """
        回答人工审批，让停在 `waiting_input` 的运行继续。

        ⚠️ 一次**所有**，不是一次一条
        ---------------------------
        扇出下多个工具可以**同时**等审批（每个分支各停各的），所以
        `ResumeRequest.answers` 是"当前所有待答"的集合。实现会读 checkpoint
        里真实的待处理集合比对，少了或多了都抛 `ConflictError`。

        为什么必须严格：实测少给一个 id 时，已答的那个生效、没答的继续停，
        运行回到 `waiting_input` —— 而调用方拿到的是 200。一次"成功"的请求
        什么都没推进，且没有任何地方报错。

        ⚠️ 它**不建新的运行行**
        ----------------------
        恢复的是**同一次运行**（同一个 `run_id`、同一条 thread），因为
        checkpoint 里那些 pending 任务就挂在那个 run 的 thread 上。新建一个
        run 去"接着跑"是做不到的：LangGraph 期望的是 `Command(resume=...)`，
        不是一次新的 invoke（见 agent_run_dispatch 的 `_terminal_status`）。

        ⚠️ 它**不取** thread 锁
        ---------------------
        中断时锁**没有放**（这是刻意的，见 agent_run_dispatch 的 handler 和
        infra/graph/thread_lock.py）：图停下来等人期间，那条 thread 上有
        pending 任务，谁都不能在上面再 invoke。锁还挂在**同一个 run_id** 上，
        所以恢复路径不需要重新取，只需要确认它还在。

        Raises:
            NotFoundError: 运行不存在，**或它所属的会话不属于 user_id**。
            ConflictError: 两种情形（-> 409）：
                这次运行**不在** `waiting_input` —— 跑完的没什么可恢复，
                正在跑的还没停下来轮不到插队；
                `answers` 的集合与当前待处理的中断**不一致** —— 少了那个
                会静默空转，多了说明客户端手上的集合过期了。
        """

        raise NotImplementedError

    @abstractmethod
    async def cancel_run(
            self,
            run_id: UUID,
            *,
            user_id: str | None,
    ) -> None:
        """
        取消一次**正在跑**的运行。

        ⚠️ 它取消的是"正在执行的那个任务"，不是"那个 run 行"
        ------------------------------------------------
        真正的动作是 `asyncio` 层面的：任务在下一个 await 点抛出
        `CancelledError`，handler 据此走取消收尾 ——

            rollback 到运行前的 checkpoint（丢掉这轮写下的半截状态）
            状态记成 `cancelled`
            放掉 thread 锁

        `queued` 的运行也能取消：它还没有 task 可取消，于是调度器先登记，
        handler 开工时第一件事就是看这个登记，看到就立刻收尾、图一次都不跑。

        ⚠️ `waiting_input` 的运行**不走这条路**
        -------------------------------------
        停在审批上的运行**没有在跑的任务** —— 图已经停住并把控制权交回来了。
        它要的是一次 `resume_run`（回答审批），而不是取消。这里按 409 处理
        并说明该做什么，比"悄悄替用户点一次拒绝"诚实。

        ⚠️ 接口**不回** Run，也不保证**已经**取消完
        ----------------------------------------
        取消是异步的（任务要在下一个 await 点才真的停）。返回只表示"受理并
        已通知"。要确认，看事件流上的终止事件或查库。

        Raises:
            NotFoundError: 运行不存在，**或它所属的会话不属于 user_id**。
            ConflictError: 运行不在可取消的状态（-> 409）。三种情形：
                已经跑完 / 停在审批等回答 / 调度器不认得它（可能刚跑完）。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_run(
            self,
            run_id: UUID,
            *,
            user_id: str,
    ) -> Run:
        """
        取一次运行。

        ⚠️ 契约里**没有**对应的端点 —— 它是给事件流用的内部读
        ----------------------------------------------------
        session.yaml 只有"建运行"，没有"查运行"。这个方法存在，是因为
        `GET /runs/{run_id}/events` 必须先回答"这个 run 存在吗"：不存在
        要给 404，而不是给客户端一条**永远不会有事件**的空流。

        它不是"偷偷加了个端点"：没有路由指向它，对外 DTO 也没变。

        为什么不让事件流靠 Redis 自己判存在
        ----------------------------------
        键可能因为 TTL 过期而消失（运行还在，流没了），也可能还没建
        （刚建好、第一个事件还没发）。两种情况下"看键在不在"都会得出
        错的结论 —— 真相在库里。

        Raises:
            NotFoundError: 运行不存在，**或它所属的会话不属于 user_id**。
        """

        raise NotImplementedError
