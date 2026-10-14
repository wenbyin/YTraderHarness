"""
"这个 run 还会产出事件吗" —— 探活的**判据实现**。

为什么在这一层
-------------
判据要用到两样东西：

    调度器        这个**进程**手里还有它吗（infra/graph/dispatcher_impl.py）
    agent_run 行  它最后被记成了什么（repository/agent_run.py）

两样都不归事件层管。所以 app/agent/event/run_stream.py 只认一个
`RunLivenessProbe` 协议，具体判据放在这里，由组合根注入。

两个信号为什么要**合起来**看
--------------------------
单独哪一个都会答错：

    只看数据库    执行器崩溃后行会永远停在 running，那条流也就永远挂着
                 （没有人会来关它）。这是本次要修的一半泄漏。
    只看调度器    它是**进程内**的 —— 进程重启后是空的，而数据库里
                 waiting_input / completed 那些信息全在数据库里。

合起来才能把「跑得慢」和「执行器没了」分开。判据见 `probe_run_liveness`。

⚠️ 前提是 workers=1
------------------
"调度器手里没有它" 只在**同一个进程**里等于"没有人在跑它"。多 worker 时，
跑在别的 worker 上的 run 在这里一律是 False，于是"DB 还是 running + 本进程
说 False"会被误判成执行器死了、把一条正常的流关掉。本项目当前 workers=1
（同 infra/graph/thread_lock.py 的前提）；换多 worker 时这个探针要改成查
共享存储（比如执行侧定期写一个带 TTL 的键）。
"""

from __future__ import annotations

from uuid import UUID

from app.agent.event import RunLiveness
from app.common.factories import get_repository_factory
from app.infra.graph import get_graph_dispatcher
from app.services.session.domain.schemas.enum import RunStatus
from app.services.session.repository.agent_run import AgentRunRepository

__all__ = [
    "probe_run_liveness",
]


#: 数据库里"运行已经结束"的那几个状态。
#: 用枚举成员而不是字面量拼 —— 加状态时这里会跟着一起被看到。
_FINISHED_STATUSES: frozenset[str] = frozenset({
    RunStatus.COMPLETED.value,
    RunStatus.FAILED.value,
    RunStatus.CANCELLED.value,
})


async def probe_run_liveness(run_id: UUID) -> RunLiveness:
    """
    问一句"这个 run 现在什么情况"。

    只读，因此 `begin=False`（不开显式事务）—— 与 `_load_run`、recorder 的
    summarize 同一条规矩。

    ⚠️ 它跑在事件流的循环 task 里
    ---------------------------
    也就是说这次查询期间，该 run 的尾随读是停的。调用方（run_stream.py）
    因此**只在空闲时**才问，并且加了超时。这里不做缓存：一次探活就是一次
    主键查询，缓存的失效时机比它省下的那点开销更难写对。

    Raises:
        任何底层异常都**原样抛**。调用方把"探不出来"与"判定它死了"分得很开
        （见 run_stream.py 的 `_probe`）—— 在这里吞掉异常会让那个区分失去
        依据。
    """

    dispatcher = get_graph_dispatcher()

    # ---- 信号一：本进程手里还有它吗 ----
    #
    # 先问这个是因为它**免费**（读两个内存集合，无 I/O）。而且它是唯一能
    # 区分"排队中"与"执行器没了"的信号 —— 数据库那边两者都是 queued。
    if dispatcher is not None and await dispatcher.is_active(run_id):
        return RunLiveness.ALIVE

    # ---- 信号二：数据库怎么记的 ----
    async with get_repository_factory().scope(begin=False) as repos:
        run = await repos.create(AgentRunRepository).get(run_id)

    if run is None:
        return RunLiveness.MISSING

    if run.status in _FINISHED_STATUSES:
        return RunLiveness.FINISHED

    if run.status == RunStatus.WAITING_INPUT.value:
        return RunLiveness.PAUSED

    # ---- 到这里：数据库说它该在跑（queued / running），而本进程没有它 ----
    if dispatcher is None:
        # 探针没装配调度器（比如脚本里单独用事件层）。少一个信号就**不判死**
        # —— 宁可多挂一条流，也不要因为缺一个信号把正常运行的流全关掉。
        return RunLiveness.ALIVE

    return RunLiveness.EXECUTOR_GONE
