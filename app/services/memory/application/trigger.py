"""
"什么时候排哪条记忆任务" —— 四条链的**调度**全部住在这个模块里。

    submit_extract           L0  每一次运行收尾就排（挂在派发 handler 的收尾）
    submit_atom_extract      L1  攒够 N 轮 / 空闲 60s（同样挂在收尾）
    submit_scene_extract     L2  L1 抽出东西之后再延迟一会儿（挂在 L1 handler 末尾）
    submit_persona_extract   L3  L2 完成就排（挂在 L2 handler 末尾）—— **不判定**

⚠️ 前三条：判定在**触发侧**，不在执行侧
--------------------------------
任务一旦排出来，handler 只干事，不再"觉得不该跑就不跑"。把条件带进任务
参数，会让同一条任务的行为取决于它被排出来之后环境变成了什么 —— 那是
最查不出原因的一类失效（逐条理由见各自的 params docstring）。

⚠️ L3 是**例外**：它的判定在执行侧
------------------------------
L3 反过来 —— 这个模块只负责"没有活着的任务就排一条"，把**触发那一刻的
两个事实**（L2 是否举手、是否满 N 轮次）带进参数，由
`memory_persona_extract` 决定要不要真的生成。

理由：L3 的核心判据是"**输入变了没有**"，而那本来就要在执行时刻查一次
（`list_scene_delta`）。放在这边判既重复、又会在排出去到跑起来之间过期。
完整的论证见 `MemoryPersonaExtractParams` 的 docstring —— 那条纪律没有
被推翻，是 L3 的判据性质不同。

代价是每次触发都要**先读一次状态**（只读查询）。换来的是不成立时一条任务
都不排。

⚠️ 失败一律**不往上抛**
------------------
记忆是旁路：任何一条触发失败都不该让一次成功的运行 / 抽取 / 整合变成失败的
任务。所以每个 submit 都自己吞掉异常、留一条 warning —— 那条 warning 是
"这次没排上"的唯一痕迹，而它们都会自愈（下一轮把同一段增量重新带上）。

下面是 L0 那一条的详细说明（它是本仓库接的第一条真 Handler，位置与失败
语义的取舍写得最细；后面三条只在各自的分节里写差异）。

⚠️ 调用位置：run 的终态写完之后
-------------------------
触发点挂在 `agent_run_dispatch.py` 的收尾里，而且必须在 `_finish` **之后**。
理由是那一刻同时成立两件事：

    消息已提交    `_record_produced_messages` 在 `_finish` 之前跑完
    run 已终态    `_finish` 把 status 落库并提交

于是任务**即使被立刻投递**（`submit` 会给 dispatcher 挂一个提交后钩子，
见 task_service_impl），它读到的也是一个一致的快照：消息在、run 不是
running。放在 `_finish` 之前的话，run 那行还是 running，"这次运行到底
结束没有"就得靠任务自己重试去等 —— 白烧一次 attempt。

⚠️ 失败**不往上抛**
---------------
记忆索引是**旁路**：它失败不该让一次成功的运行变成失败的 handler。
所以这里吞掉异常、只留一条 warning。

代价说清楚：吞掉意味着"这次没索引上"不会被自动重试。可接受的原因是
**数据没丢** —— 消息还在 `agent_message` 里，重跑一次 `memory.l0.extract`
（按 run_id）就能补上。真正会丢的是"没人知道要补"，所以日志里那一条
必须是 warning 而不是 debug。

⚠️ 幂等键 = `memory.l0.extract:{run_id}`
---------------------------------
一个 run 可能**多次**走到收尾：

    停在 HITL 等人 -> resume 跑完 -> 再收尾一次（第一次索引了半程）
    任务被重投 / 人工重跑

键取 run_id 而不是 (run_id, 次数)：索引的是"这一轮的全部消息"，第二次
收尾读到的消息是**超集**，重写一遍是对的（仓储 upsert，消息不可变）
—— 而重复提交同一个键只会得到同一条 Task，不会排两条。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import UUID

from app.common.factories import (
    get_repository_factory,
    get_service_factory,
)
from app.common.repository.factory import RepositoryFactory
from app.common.task.models import Task
from app.common.task.types import TaskStatus
from app.services.session.repository.message import MessageRepository
from app.services.task.application import TaskService
from app.services.task.repository import TaskStore

from ..domain.entities.scene import SceneScope
from ..domain.tasks import (
    MEMORY_EXTRACT_HANDLER,
    MEMORY_ATOM_EXTRACT_HANDLER,
    MEMORY_SCENE_EXTRACT_HANDLER,
    MEMORY_PERSONA_EXTRACT_HANDLER,
    MemoryExtractParams,
    MemoryAtomExtractParams,
    MemorySceneExtractParams,
    MemoryPersonaExtractParams,
)
from ..repository.atom import MemoryAtomRepository
from ..repository.persona import MemoryPersonaRepository
from ..repository.scene import MemorySceneRepository
from app.services.session.repository.message import MessageRepository

__all__ = [
    "ATOM_IDLE_DELAY_SECONDS",
    "ATOM_TRIGGER_TURNS",
    "SCENE_AFTER_L1_DELAY_SECONDS",
    "SCENE_MIN_INTERVAL_SECONDS",
    "atom_task_key",
    "persona_task_key",
    "scene_task_key",
    "submit_extract",
    "submit_atom_extract",
    "submit_persona_extract",
    "submit_scene_extract",
    "task_id_of",
]


#: 还能改执行时间的状态。`reschedule` 也只对这两种生效（见它的说明）：
#: 正在跑的任务马上写回结果，改了没意义；终态的更不能改。
_RESCHEDULABLE: tuple[TaskStatus, ...] = (
    TaskStatus.PENDING,
    TaskStatus.RETRYABLE,
)


logger = logging.getLogger(__name__)


def idempotency_key(run_id: UUID) -> str:
    """`memory.l0.extract` 的幂等键。提成函数是为了让断言与实现用同一个。"""

    return f"{MEMORY_EXTRACT_HANDLER}:{run_id}"


async def submit_extract(
        *,
        run_id: UUID,
        session_id: UUID,
) -> UUID | None:
    """
    提交一条 `memory.l0.extract`，返回 task_id；提交失败时返回 None（不抛）。

    ⚠️ 用 `scope(TaskService)` 而不是 `ScopedTaskSubmitter`
    ---------------------------------------------------
    后者（infra/task/submitter.py）刻意不挂 dispatcher —— 它靠 outbox
    relay(10s) 与兜底扫描(15s) 投递，那是给 Handler 派生**子任务**用的
    （子任务不着急）。这里希望尽快投递，所以直接开一个作用域取
    TaskService：它的 `post_commit` 钩子会把 `dispatch()` 挂上，作用域
    一提交就投递。

    ⚠️ 应用没跑过 lifespan 时安静跳过
    ------------------------------
    `get_service_factory()` 在那种情况下抛 RuntimeError —— 冒烟脚本与
    工具脚本都是这种常态（见 agent_builder_node 里同一类的降级处理）。
    那不是故障，所以只留一条 debug 日志：没有任务基建，就没有后台索引，
    而调用方（派发 handler）本来也不该因此失败。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        logger.debug(
            "[Memory] 任务基建不可用，跳过记忆索引（run=%s）：%s",
            run_id, exc,
        )
        return None

    try:
        async with factory.scope(TaskService) as service:

            return await service.submit(
                Task(
                    handler=MEMORY_EXTRACT_HANDLER,
                    params=MemoryExtractParams(
                        run_id=run_id,
                        session_id=session_id,
                    ),
                    # causation_id 指向**运行**，不是父任务 —— 将来问
                    # "这次运行引发了哪些任务"时，答案是这条链的起点
                    # （见 domain/tasks.py 与 task types 里的说明）。
                    causation_id=run_id,
                    idempotency_key=idempotency_key(run_id),
                )
            )

    except Exception as exc:  # noqa: BLE001 —— 旁路，见模块 docstring
        # ⚠️ 这一条**必须**是 warning：吞掉异常之后，这条日志是"这次没
        # 索引上"的唯一痕迹。
        logger.warning(
            "[Memory] 提交记忆索引任务失败（run=%s，消息仍在库里，"
            "可重跑 %s 补上）：%s",
            run_id, MEMORY_EXTRACT_HANDLER, exc,
        )
        return None


# ============================================================
# L1（memory_atom）：按轮次触发
# ============================================================
#
# 与上面那条的分工，一句话：
#
#     L0   每一轮收尾都抽（按 run_id 一轮一轮索引），无脑、幂等、便宜
#     L1   攒够 N 轮才抽，且空闲 60s 也抽一次 —— 因为每次都要调 LLM
#
# 所以 L0 那条是"索引"，L1 这条是"提炼"。前者贵不在模型，后者贵在模型。


#: 轮次阈值：**自上次抽取以来**积累了这么多用户轮次就抽一次。
#:
#: ⚠️ 是"自上次抽取以来"，不是"会话里第 N 轮"
#: -----------------------------------
#: 判据是 `count_user_turns(sid, after_seq=水位) >= 阈值`（见
#: `submit_atom_extract`）。**不要**改回 `总数 % N == 0`：那个写法锚在
#: 绝对轮号上，空闲触发（或兜底 60s 落下）之后，下一次达标可能只隔 1 轮
#: —— 那一次调用的输入只有一两轮，按提示词的「宁缺毋滥」多半什么都抽不
#: 出来，属于纯空转的模型调用。
#:
#: ⚠️ 这是一个**待校准的起点**，不是实测出来的值。调它两个方向各有一个
#: 可观察的症状：
#:
#:     调太小    抽得太勤 —— 每轮都付一次 LLM，而提示词的第一条原则是
#:               「宁缺毋滥」，信息量不足时它本来就不抽，多数轮次白花钱
#:     调太大    记忆更新滞后 —— 用户改了偏好，AI 要过很多轮才「记住」
ATOM_TRIGGER_TURNS = 5

#: 没到阈值时那条兜底任务的延迟（秒）。
#:
#: 语义是**空闲超时**：用户停手这么久就把攒着的抽掉，不必等凑满阈值。
ATOM_IDLE_DELAY_SECONDS = 60.0


def atom_task_key(session_id: UUID) -> str:
    """
    `memory.l1.extract` 的识别键。**粒度是会话** —— 与这条链的语义一致：
    原规格的提取与去重都以会话为单位（水位按会话推进、候选池按会话收窄）。

    ⚠️ 它能**反复用**，是因为任务进终态时会释放键
    --------------------------------------
    幂等键原来是"一辈子只认一次"的（终态任务永久占住它），所以以前只能
    用 per-run 键。现在终态会把它清掉（见 task_impl 的「幂等键的生命周期」），
    这个键的含义变成了"**这个会话当前那条活着的任务**" —— 于是"每来一句
    就把截止时间往后推"（防抖）才成立。

    这也解释了为什么以前库里 45 条任务全是 `generation = 1`：那时候没有
    一条能反复重排的键。
    """

    return f"{MEMORY_ATOM_EXTRACT_HANDLER}:{session_id}"


async def submit_atom_extract(
        *,
        session_id: UUID,
        run_id: UUID,
        repositories: RepositoryFactory,
        trigger_turns: int = ATOM_TRIGGER_TURNS,
        idle_delay_seconds: float = ATOM_IDLE_DELAY_SECONDS,
) -> UUID | None:
    """
    把「这一轮结束了」变成「该给这个会话抽一次 L1 了」。

    与 L0 那条的关系
    --------------
    两者都在收尾路径上、紧挨着调用，但**没有先后依赖**：L1 读的是
    `agent_message`（权威表）与自己的水位（memory_atom_cursor），不读 L0 的
    两张索引表。所以 L0 的任务还没跑完，L1 也能读到完整消息。

    Parameters
    ----------
    repositories
        只读用（`scope(begin=False)`）。轮次计数走 session 模块的
        `MessageRepository.count_user_turns` —— 「一轮」的定义（每条
        用户消息开始一轮、只算 kind=chat）归 session 模块，本模块不去
        直接查 `agent_message`。
    trigger_turns
        轮次阈值。<= 0 表示**永不**因阈值触发，只走空闲超时那条路
        （给想关掉它的部署留的后门，不必改代码）。
    idle_delay_seconds
        空闲超时的秒数。

    两条分支
    ------
        达标（`水位之后的用户轮次 >= trigger_turns`）  execute_at=None，立刻
        不达标                                    now + idle_delay_seconds

    ⚠️ 为什么**每一轮**都提交（而不是只在达标时）
    ----------------------------------------
    没到阈值那条延迟任务的语义是空闲超时（用户停手就抽），而它同时是
    兜底：提交失败、抽取失败都不丢数据 —— 下一轮收尾会把同一段增量重新
    带上，因为抽取**永远按水位读全量**。

    代价说清楚：会话里每一轮都会留下一条任务，其中相当一部分会「空转」
    （跑到时发现增量已经被前一条抽走了，于是正常结束）。这是**刻意**的
    —— 换来的是不需要一条稳定键 + 重排。

    ⚠️ 轮次计数漂了**不会漏抽**
    ------------------------
    计数只决定「什么时候发任务」，不决定「抽什么」—— 抽什么由水位说了
    算。所以计数算错、丢了一次，最坏只是**晚抽一次**。

    ⚠️ 与 L0 一样，失败**不往上抛**：记忆抽取是旁路，它失败不该让一次
    成功的运行变成失败的 handler。吞掉异常之后那条 warning 是"这次没抽
    上"的唯一痕迹。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        logger.debug(
            "[Memory] 任务基建不可用，跳过 L1 抽取（session=%s）：%s",
            session_id, exc,
        )
        return None

    try:
        # 只读：不开显式事务。
        #
        # 两次读，缺一不可：
        #     水位   自上次抽取以来 = 水位**之后**（memory_atom_cursor）
        #     轮次   那之后还有几条**用户消息**（session 模块数的）
        async with repositories.scope(begin=False) as repos:

            cursor = await repos.create(
                MemoryAtomRepository
            ).get_cursor(session_id)

            pending_turns = await repos.create(
                MessageRepository
            ).count_user_turns(session_id, after_seq=cursor.last_seq)

        # ⚠️ `pending_turns` 是**增量**，不是会话累计
        # -------------------------------------
        # 判据必须是"自上次抽取以来攒了几轮"，不是"会话第几轮"。取模那种
        # 写法（`总数 % N == 0`）锚在绝对轮号上，空闲/兜底触发的抽取之后
        # 下一次达标可能只隔 1 轮 —— 那一次只有一两轮输入，按提示词的
        # 「宁缺毋滥」多半什么都抽不出来，是纯空转的一次 LLM 调用。
        #
        # ⚠️ 也别把水位**减**进轮次里
        # ----------------------
        # 水位是 `agent_message.seq`（每条消息都占号），轮次是**用户消息
        # 的条数**，两者单位不同：相减会让 5 轮的阈值静默变成 20 轮，而且
        # 取决于 assistant 有多啰嗦。水位在这里只能当**过滤条件**用
        # （`after_seq=` 进 SQL 的 WHERE），见会话：count_user_turns 的说明。
        #
        # ⚠️ 0 要单独拦
        # ----------
        # 水位之后没有新的用户消息 = 这次收尾没有产生可抽的东西（resume
        # 跑完、或那一轮只有工具活动）。排一条延迟任务必然空转，所以直接
        # 不排。
        #
        # 它同时替掉了原来"0 % N == 0"那个取模的边界坑（对任何 N 都成立，
        # 不拦就会每来一次排一条立刻执行的）。
        if pending_turns <= 0:
            logger.debug(
                "[Memory] 会话 %s 自水位 seq=%d 起没有新的用户轮次，跳过 L1。",
                session_id, cursor.last_seq,
            )
            # ⚠️ 这一条也打到控制台 —— 它是**新的静默路径**
            # --------------------------------------
            # 它由 2026-10-11 的判定改动引入（原来是取模，这种情况会照排
            # 一条延迟任务、跑到之后空转）。而"没排任务"与"排了但抽取空转"
            # 在控制台上原本长得**完全一样**（都什么都不打印）—— 那正是这个
            # 模块反复强调要避免的一类失效。三种判定各一条，一眼能分辨。
            print(
                f"[Memory] L1 不排任务：session={session_id} "
                f"自水位 seq={cursor.last_seq} 起没有新的用户轮次。",
                flush=True,
            )
            return None

        reached = (
            trigger_turns > 0 and pending_turns >= trigger_turns
        )

        execute_at = (
            None
            if reached
            else datetime.now(timezone.utc)
            + timedelta(seconds=idle_delay_seconds)
        )

        logger.debug(
            "[Memory] L1 触发判定：session=%s 待抽轮次=%d/%s（水位 seq=%d）-> %s",
            session_id,
            pending_turns,
            trigger_turns or "关闭",
            cursor.last_seq,
            "立即" if reached else f"{idle_delay_seconds:.0f}s 后",
        )

        # ---- 这个会话已经有一条活着的 L1 任务？往后推它（真·防抖）------
        #
        # **会话就是识别键**（`memory.l1.extract:{session_id}`）：原规格的
        # 提取与去重都以会话为单位，键自然也该是它。
        #
        # 有活着的就不新提交，而是改它的执行时间：用户每说一句就把截止时间
        # 往后挪 60 秒，最后那次赢 —— 一个会话始终只有**一条** L1 任务。
        # 达标时 `reschedule(execute_at=None)` 会**立刻投递**它（见
        # TaskService.reschedule 的说明），所以"攒够 N 轮立刻抽"走同一段。
        key = atom_task_key(session_id)

        existing = await _task_by_key(repositories, key)

        if existing is not None and existing.status is TaskStatus.RUNNING:
            # 正在跑：它读的就是同一段增量。这一轮**不新排、也不改它的
            # 时间**（`reschedule` 对 running 会拒）。顺带把"同一会话两条
            # 并发抽取"这件事挡掉了 —— 两条都会写 memory_atom。
            logger.info(
                "[Memory] 会话 %s 的 L1 任务 %s 正在跑，本次不新排。",
                session_id, existing.task_id,
            )
            return existing.task_id

        if existing is not None and existing.status in _RESCHEDULABLE:

            # 查到了就**改它的执行时间**，不新建：
            #
            #   达标（execute_at=None）  -> **强制立刻执行它**
            #                              （reschedule 到"已到点"时服务层会
            #                              当场投递一次，见 TaskService.
            #                              reschedule 的说明）
            #   没达标                   -> 往后推 60 秒（真·防抖）
            async with factory.scope(TaskService) as service:
                await service.reschedule(
                    existing.task_id, execute_at=execute_at,
                )

            logger.debug(
                "[Memory] 复用会话 %s 的 L1 任务 %s -> %s",
                session_id,
                existing.task_id,
                "立刻执行" if execute_at is None else "往后推",
            )

            _log_trigger(
                session_id, pending_turns, trigger_turns,
                idle_delay_seconds, execute_at, existing.task_id,
            )
            return existing.task_id

        # ⚠️ 与 L0 一样用 `scope(TaskService)` 而不是 ScopedTaskSubmitter：
        # 后者刻意不挂 dispatcher（靠 outbox relay 与兜底扫描投递，那是对
        # 付**子任务**的），而这条是收尾路径上希望尽快投递的入口。
        #
        # 走到这里说明这个会话**没有活着的任务**（没有、或上一条已经终结
        # 并释放了键，见 task_impl 的「幂等键的生命周期」）—— 新建一条。
        async with factory.scope(TaskService) as service:

            task_id = await service.submit(
                Task(
                    handler=MEMORY_ATOM_EXTRACT_HANDLER,
                    params=MemoryAtomExtractParams(session_id=session_id),
                    execute_at=execute_at,
                    # causation_id 指向**运行**（不是父任务）—— 与 L0 同一条
                    # 取舍，理由见 domain/tasks.py 与 memory.l0.extract 的说明。
                    causation_id=run_id,
                    idempotency_key=key,
                    # ⚠️ 声明这个键**可以反复用**：任务一进终态就把它释放，
                    # 于是下一个窗口能重新提交同一条键。
                    #
                    # 默认是不释放的（老语义，L0 靠它），所以这里必须显式开 ——
                    # 否则第二次触发会拿回那条已终结的任务、reschedule 直接抛。
                    # 见 task_impl 的「幂等键的生命周期」。
                    reusable_key=True,
                )
            )

        _log_trigger(
            session_id, pending_turns, trigger_turns,
            idle_delay_seconds, execute_at, task_id,
        )

        return task_id


    except Exception as exc:  # noqa: BLE001 —— 旁路，见模块 docstring
        # ⚠️ 这一条**必须**是 warning：吞掉异常之后，它是"这次没抽上"的
        # 唯一痕迹。与 L0 不同的一点：这里**不需要**提示"可重跑什么" ——
        # 抽取按水位读增量，用户下一轮说话就会把这段重新带上。
        logger.warning(
            "[Memory] 提交 L1 抽取任务失败（session=%s，增量仍在库里，"
            "下一轮会自动带上）：%s",
            session_id, exc,
        )
        return None


# ============================================================
# L2（memory_scene）：L1 完成之后
# ============================================================
#
# 三层的触发时机，一句话对照：
#
#     L0   每一轮收尾都索引          无脑、幂等、便宜
#     L1   攒够 N 轮 / 空闲 60s 才抽   每次都要调 LLM
#     L2   L1 抽出东西之后再等一会儿   每次都要调 LLM，而且要把整个 scope
#                                    的场景全文喂进去 —— 最贵的一层


#: L1 完成之后至少等这么久才排 L2（秒）。对应原规格的
#: `T = max(now + 90s, lastL2 + 900s)` 里那个 `now + 90s`。
#:
#: 为什么要等：L1 刚抽出几条记忆，用户往往**还在说**。立刻整合会让 90 秒
#: 后的那批记忆再触发一次整合 —— 两次模型调用做同一件事，而第一次的结论
#: 马上就被第二次覆盖。等一小段，让"这一轮"讲完。
SCENE_AFTER_L1_DELAY_SECONDS = 90.0

#: 同一个会话两次 L2 之间至少隔这么久（秒）。对应原规格里那个
#: `lastL2 + 900s`。
#:
#: 它是**成本闸门**：L2 一次要读该 scope 的全部场景（十几个文件全文）再
#: 写回，比 L1 贵一个量级。没有它的话，一个聊得很快的会话会连续排 L2 ——
#: 每次的输入几乎一样，输出几乎一样。
SCENE_MIN_INTERVAL_SECONDS = 900.0


def scene_task_key(session_id: UUID) -> str:
    """
    `memory.l2.extract` 的识别键。**粒度是会话**（与 L1 那条同一粒度）。

    为什么是会话而不是 scope：水位（"哪些 L1 已经整合过"）是会话级的
    （见 domain/tasks.py 里 `MemorySceneExtractParams` 的说明）—— 一个会话
    一条活着的整合任务，与"这一批增量由谁负责带走"是同一件事。
    产物落哪个 scope 是执行阶段的事，不参与识别。
    """

    return f"{MEMORY_SCENE_EXTRACT_HANDLER}:{session_id}"


async def submit_scene_extract(
        *,
        session_id: UUID,
        causation_id: UUID | None = None,
        after_l1_delay_seconds: float = SCENE_AFTER_L1_DELAY_SECONDS,
        min_interval_seconds: float = SCENE_MIN_INTERVAL_SECONDS,
) -> UUID | None:
    """
    把「L1 抽完了」变成「过一会儿该把这个会话的记忆整合进场景了」。

    ⚠️ 调用位置：L1 handler 的**末尾**（那里知道"确实抽完了"）
    ------------------------------------------------
    与 L1 那条挂在运行收尾不同，这条不能挂在收尾：收尾时 L1 的任务**还没
    跑**（它排在 60s 后或立刻投递），而且相当一部分会空转。挂在 L1 的
    handler 末尾才有"这一轮真的抽出了东西"这个事实可依。

    代价说清楚：**它跑在 L1 那次事务提交之前**。若 L1 随后回滚，L2 已经
    排在库里了 —— 届时它读到空增量（水位没动、那批记忆也没落库），只把
    `last_l2_at` 戳一下、正常结束。空转一次，不是数据丢失。要消掉它就得
    等提交之后再排，那需要一条 post-commit 钩子，而任务基建那层
    （`scope(TaskService)`）恰好就是那条钩子 —— 所以这里的取舍是"不等"。

    ⚠️ 时间：`T = max(now + 90s, lastL2 + 900s)`
    ---------------------------------------
    两个数各挡一件事，见上面两个常量的说明。`lastL2` 从会话的 L2 水位上
    读（`memory_scene_cursor.last_l2_at`）—— **含空转那一次**，所以一个反复
    空转的会话不会每 90 秒白排一次。

    ⚠️ 只允许提前，不允许推迟
    --------------------
    已经有一条活着的任务时，**只有新时间更早才改它**（`reschedule`），否则
    什么都不做。`reschedule` 本身不限制方向（往前推也好、往后推也好都
    算一次重排，见 TaskStore 的说明），所以"只提前"这条得在这里夹住。

    为什么要夹：T 是关于 `now` 单调不减的，所以正常情况下后续几次触发算出
    的 T 只会更晚 —— 让它生效就等于**每次 L1 完成都把整合往后推**，而
    L1 完成得很频繁（攒够 5 轮或空闲 60s），结果是这个会话永远等不到整合。
    再叠加"跑完一次 L2 之后 `lastL2` 跳到当下、T 直接变成 lastL2+900"，
    那条已经排好、本来马上就到的任务会被推出去整整 15 分钟。

    反过来（只提前）的行为是可解释的：**这条任务只会比排它的时候更早跑**。

    ⚠️ 与 L0/L1 一样，失败**不往上抛**
    ------------------------------
    记忆整合是旁路，它失败不该让一次成功的 L1 抽取变成失败的任务。吞掉
    异常之后那条 warning 是"这次没排上"的唯一痕迹 —— 而它**会自愈**：
    下一次 L1 完成会把同一个会话重新带上（水位没动，增量还在）。

    ⚠️ 为什么自己取仓储工厂，而不是让调用方传
    --------------------------------
    调用方是 L1 的 handler，它手里只有注入的仓储实例，没有工厂（见
    `submit_atom_extract` 那边为什么带 `repositories`：它的调用方
    ——派发 handler —— 恰好握着一个）。多传一个参数只为了让签名对称，
    不如让这里自己取（`get_repository_factory()`）。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        logger.debug(
            "[Memory] 任务基建不可用，跳过 L2 整合（session=%s）：%s",
            session_id, exc,
        )
        return None

    try:
        repositories = get_repository_factory()

        # 只读：不开显式事务。
        async with repositories.scope(begin=False) as repos:

            cursor = await repos.create(
                MemorySceneRepository
            ).get_l2_cursor(session_id)

        execute_at = _scene_execute_at(
            last_l2_at=cursor.last_l2_at,
            after_l1_delay_seconds=after_l1_delay_seconds,
            min_interval_seconds=min_interval_seconds,
        )

        key = scene_task_key(session_id)

        existing = await _task_by_key(repositories, key)

        if existing is not None and existing.status is TaskStatus.RUNNING:
            # 正在跑：它读的就是同一段增量，而且**跑完之后自己会再排下一次**
            # （见 L1 那边的同一段）。这一轮不新排、也不改它的时间
            # （`reschedule` 对 running 会拒）。
            logger.info(
                "[Memory] 会话 %s 的 L2 任务 %s 正在跑，本次不新排。",
                session_id, existing.task_id,
            )
            # ⚠️ 控制台也要一条 —— 理由与 L1 那条一样（本应用没配 logging，
            # logger.info 在控制台上一个字都不会出现，见 _log_trigger 的说明）。
            # 这一支尤其要看得见：它是"排了但被跳过"，与"根本没排上"长得一样。
            print(
                f"[Memory] L2 不新排：session={session_id} "
                f"已有任务正在跑, task={existing.task_id}",
                flush=True,
            )
            return existing.task_id

        if existing is not None and existing.status in _RESCHEDULABLE:

            # ⚠️ 只允许提前
            # ----------
            # 已经排到点上的（execute_at 为 None）更早，不动它。
            if (
                    existing.execute_at is not None
                    and execute_at >= existing.execute_at
            ):
                logger.debug(
                    "[Memory] 会话 %s 的 L2 任务 %s 已排在 %s，本次算出的 %s "
                    "更晚 —— 不推迟它。",
                    session_id, existing.task_id,
                    existing.execute_at, execute_at,
                )
                print(
                    f"[Memory] L2 不推迟：session={session_id} "
                    f"已排在 {existing.execute_at:%H:%M:%S}，本次算出的 "
                    f"{execute_at:%H:%M:%S} 更晚（只允许提前）。",
                    flush=True,
                )
                return existing.task_id

            async with factory.scope(TaskService) as service:
                await service.reschedule(
                    existing.task_id, execute_at=execute_at,
                )

            logger.debug(
                "[Memory] 会话 %s 的 L2 任务 %s 提前到 %s。",
                session_id, existing.task_id, execute_at,
            )
            print(
                f"[Memory] L2 提前：session={session_id} "
                f"从 {existing.execute_at:%H:%M:%S} 提前到 {execute_at:%H:%M:%S}, "
                f"task={existing.task_id}",
                flush=True,
            )
            return existing.task_id

        # 走到这里说明这个会话**没有活着的 L2 任务**（没有、或上一条已经
        # 终结并释放了键，见 task_impl 的「幂等键的生命周期」）—— 新建一条。
        async with factory.scope(TaskService) as service:

            task_id = await service.submit(
                Task(
                    handler=MEMORY_SCENE_EXTRACT_HANDLER,
                    params=MemorySceneExtractParams(session_id=session_id),
                    execute_at=execute_at,
                    # causation_id 指向**触发它的那条 L1 任务**（调用方给），
                    # 于是"这次 L1 引发了哪些整合"是可查的。
                    causation_id=causation_id,
                    idempotency_key=key,
                    # 与 L1 同一条：这个键要能反复用（任务一进终态就释放），
                    # 否则第二次触发会拿回那条已终结的任务、reschedule 直接抛。
                    reusable_key=True,
                )
            )

        logger.debug(
            "[Memory] 会话 %s 排入 L2 整合：%s 执行，task=%s",
            session_id, execute_at, task_id,
        )

        print(
            f"[Memory] L2 入队：session={session_id} "
            f"排在 {execute_at:%H:%M:%S}（= max(now+{after_l1_delay_seconds:.0f}s, "
            f"lastL2+{min_interval_seconds:.0f}s)）, task={task_id}",
            flush=True,
        )

        return task_id

    except Exception as exc:  # noqa: BLE001 —— 旁路，见函数说明
        # ⚠️ 这一条**必须**是 warning：吞掉异常之后，它是"这次没排上"的
        # 唯一痕迹。与 L1 那条不同的是**不需要**提示怎么补 —— 增量还在
        # （水位没动），下一次 L1 完成会重新排。
        logger.warning(
            "[Memory] 提交 L2 整合任务失败（session=%s，增量仍在库里，"
            "下一次 L1 完成会重新排）：%s",
            session_id, exc,
        )
        return None


# ============================================================
# L3（memory_persona）：L2 完成之后
# ============================================================
#
# 四层的触发时机，一句话对照：
#
#     L0   每一轮收尾都索引             无脑、幂等、便宜
#     L1   攒够 N 轮 / 空闲 60s 才抽      每次都要调 LLM
#     L2   L1 抽出东西之后再等一会儿      每次都要调 LLM，且要读整个 scope
#     L3   L2 完成之后**一律排**           每次都要调 LLM，且要读整篇画像 +
#          （判定在执行侧）                全部变化场景 —— 最贵的一层
#
# L3 与上面三条**最大的不同**：它的门在**执行侧**，不在这里。L2 完成得很
# 频繁（每 15 分钟一个会话），而画像变化得慢 —— 所以任务排出来之后由
# `memory_persona_extract` 自己判"输入变了没有"：没有变化场景、也没收到
# 举手、也没满轮次，就一条 SELECT 走人，不调模型。
#
# 为什么这一层可以这样而 L1/L2 不行，见 `MemoryPersonaExtractParams` 的
# docstring（一句话：L3 的判据是"这个 scope 有没有画像"，那本来就要在执行
# 时刻查一次主键，放在这里判是重复且会过期）。
#
# ⚠️ "满 N 轮次"那条判据**不在这里**（2026-10-13）
# ----------------------------------------------
# 它原来在触发侧算好、当一个 `force` 参数带进任务（`_persona_force` + 一次
# 跨会话 COUNT）。现在三条判据全在执行侧现判，那次计数也搬到了
# `MemoryPersonaServiceImpl._enough_turns`（常量 `PERSONA_TRIGGER_TURNS`）。
#
# 搬家的理由：触发侧挂着**每一次** L2 完成，在那里算等于"提交那一刻的事实"，
# 而任务可能等一会儿才跑；执行侧算读的是此刻的水位，更新鲜，也让这条路径
# 真正做到**只写不读**（见 submit_persona_extract 的说明）。
#
# ⚠️ 而那个判据原来**从未生效过**：基准 `last_l3_at` 在每次空转时也被推，
# 每跳过一次窗口就归零。执行侧现在不推它了（见 memory_persona_service_impl
# 第 3 步那段）—— 两件事是一起改的，少改一处这条判据仍然是死的。


def persona_task_key(scope: SceneScope) -> str:
    """
    `memory.l3.extract` 的识别键。**粒度是 scope** —— 与上面三条都不同。

    ⚠️ 为什么是 scope 而不是会话
    ------------------------
    产物（一篇画像）与水位（`memory_persona_cursor`）**都是 scope 级的**，
    而 L2 的完成是会话级的 —— 一个 scope 下 N 个会话的 L2 完成最终指向
    同一条 L3。键取 scope，这些完成才会**合并成一条**；取会话的话，同一个
    scope 会被并发写 N 次，全靠乐观锁互相打回重来。

    ⚠️ 它能**反复用**，是因为任务进终态时会释放键
    ------------------------------------
    与 L1/L2 那两条同一条理由（见 task_impl 的「幂等键的生命周期」）：
    "这个键查得到" == "这个 scope 此刻有一条活着的 L3"，而不是"一辈子只能
    有一条"。
    """

    return f"{MEMORY_PERSONA_EXTRACT_HANDLER}:{scope.key}"


async def submit_persona_extract(
        *,
        scope: SceneScope,
        persona_update_request: bool = False,
        causation_id: UUID | None = None,
) -> UUID | None:
    """
    把「L2 整合完了」变成「这个 scope 该跑一次画像生成」。

    ⚠️ 调用位置：L2 handler 的**末尾**（那里才知道 scope 与那个信号）
    ------------------------------------------------------
    与 L2 那条挂在 L1 handler 末尾是同一手法、同一理由：只有那一刻同时
    成立三件事 ——

        这个 scope 是谁       L2 刚解析出来的（`_duty_key` + 会话的归属）
        这一轮真的整合完了    上面的写入与水位都做完了
        模型给没给信号        `result.persona_update_request` 只在这次回复里

    ⚠️ 它**不判定该不该生成** —— 那是执行侧的事（与 L1/L2 相反）
    ------------------------------------------------------
    这里只做两件事：

        没有活着的任务  -> 排一条，把**触发那一刻的两个事实**带进参数
        有活着的        -> 复用 / 提前，与 L1/L2 同一套

    "要不要真的生成"由 `memory_persona_extract` 判：没有变化场景、也没收到
    举手、也没被要求强制，就只戳一下水位、不调模型。**为什么 L3 可以这样而
    L1/L2 不行**，理由写在 `MemoryPersonaExtractParams` 的 docstring 里
    （L3 的核心判据是"输入变了没有"，那本来就要在执行时刻重算一次）。

    ⚠️ 这条路径**只写不读**（2026-10-13）
    --------------------------------
    原来要读一次 L3 水位、再跑一次跨会话的轮次 COUNT（算 `force`）。
    `force` 去掉之后两件事一起消失 —— 现在这里唯一的交互是按幂等键
    取那条任务（`_task_by_key`），外加最终的那次 submit。

    ⚠️ 带进去的是**结论**，不是判据
    ---------------------------
        persona_update_request   L2 的模型在整合现场举没举手（一个布尔）

    ⚠️ 判定**不需要 session_id**
    ------------------------
    "哪个会话刚跑完 L2"不影响任何东西 —— 触发它的那个会话只是**时机**
    （谁先完成谁来看一眼），不是判据的一部分。判据是"这个 scope 有没有
    画像"，那是 scope 级的。这也是为什么参数里没有它。

    ⚠️ 立即可执行（`execute_at=None`），**不延迟**
    ---------------------------------------
    与 L2 那条不同（那里有 90s 的"等用户说完"与 900s 的成本闸门）。这里
    不需要：任务跑起来是廉价的（没有变化场景就一条 SELECT 走人），而真正
    该生成的那一条越早生效越好。

    同一 scope 下多个会话的 L2 完成会**撞同一个 key**，那时的行为与 L1/L2
    一致：有活着的就复用。

    ⚠️ 与上面三条一样，失败**不往上抛**
    ----------------------------
    画像是旁路，它失败不该让一次成功的 L2 整合变成失败的任务。吞掉异常
    之后那条 warning 是"这次没排上"的唯一痕迹 —— 而它**会自愈**：下一次
    L2 完成会重新排一遍（水位没动，场景还在）。
    """

    try:
        factory = get_service_factory()

    except RuntimeError as exc:
        logger.debug(
            "[Memory] 任务基建不可用，跳过 L3 触发（scope=%s）：%s",
            scope.key, exc,
        )
        return None

    try:
        repositories = get_repository_factory()

        # ⚠️ 这里**一次读都没有**了（2026-10-13）
        # -------------------------------------
        # 原来要读一次 L3 水位、再跑一次跨会话的轮次 COUNT（算 `force`）。
        # `force` 去掉之后那两个都没了 —— 触发侧现在**只写不读**，唯一剩下
        # 的交互是后面按幂等键取那一条任务。
        key = persona_task_key(scope)

        existing = await _task_by_key(repositories, key)

        if existing is not None and existing.status is TaskStatus.RUNNING:
            # 正在跑：它读的就是同一批场景（水位没动之前它们都在），而且
            # 跑完会自己把水位推上去。这一轮不新排、也不改它的时间
            # （`reschedule` 对 running 会拒）。
            #
            # ⚠️ 这一轮算出来的举手 / 强制因此**丢了** —— 它们进不了正在
            # 跑的那条任务的参数（TaskService 没有"改 params"）。代价可接受：
            # L3 是立刻执行的，这个窗口只到 worker 领走为止。
            logger.info(
                "[Memory] scope %s 的 L3 任务 %s 正在跑，本次不新排。",
                scope.key, existing.task_id,
            )
            print(
                f"[Memory] L3 不新排：scope={scope.key} "
                f"已有任务正在跑, task={existing.task_id}",
                flush=True,
            )
            return existing.task_id

        if existing is not None and existing.status in _RESCHEDULABLE:

            # ⚠️ 已经排在"立刻"的更早，不动它
            # ------------------------------
            # 目标时间恒为 None（立刻），所以只有"它现在不是立刻"才需要改。
            # 与 L2 那边"只允许提前"是同一条纪律：这里的目标就是最早的那个
            # 时刻，任何"推迟"都会让一条本该马上跑的任务被推出去。
            if existing.execute_at is None:
                logger.debug(
                    "[Memory] scope %s 的 L3 任务 %s 已排在立刻，不动它。",
                    scope.key, existing.task_id,
                )
                return existing.task_id

            async with factory.scope(TaskService) as service:
                await service.reschedule(existing.task_id, execute_at=None)

            logger.debug(
                "[Memory] scope %s 的 L3 任务 %s 提前到立刻。",
                scope.key, existing.task_id,
            )
            print(
                f"[Memory] L3 提前：scope={scope.key} 从 "
                f"{existing.execute_at:%H:%M:%S} 提前到立刻, "
                f"task={existing.task_id}",
                flush=True,
            )
            return existing.task_id

        # 走到这里说明这个 scope **没有活着的 L3 任务**（没有、或上一条已经
        # 终结并释放了键，见 task_impl 的「幂等键的生命周期」）—— 新建一条。
        async with factory.scope(TaskService) as service:

            task_id = await service.submit(
                Task(
                    handler=MEMORY_PERSONA_EXTRACT_HANDLER,
                    params=MemoryPersonaExtractParams(
                        agent_id=scope.agent_id,
                        user_id=scope.user_id,
                        org_id=scope.org_id,
                        # ⚠️ L2 的"举手"理由**原样带上**（只去掉了两端空白）。
                        # 丢掉它是最容易犯的一个错 —— L3 照常跑，只是模型
                        # 不知道 L2 指的是哪件事。见 params 与提示词模块。
                        persona_update_request=persona_update_request,
                    ),
                    # 立刻执行，不延迟 —— 见上面那段说明。
                    execute_at=None,
                    # causation_id 指向**触发它的那条 L2 任务**（调用方给），
                    # 于是"这次 L2 引发了哪些画像生成"是可查的。
                    causation_id=causation_id,
                    idempotency_key=key,
                    # 与 L1/L2 同一条：这个键要能反复用（任务一进终态就释放），
                    # 否则第二次触发会拿回那条已终结的任务、reschedule 直接抛。
                    reusable_key=True,
                )
            )

        logger.debug(
            "[Memory] scope %s 排入 L3 生成（举手=%s）：task=%s",
            scope.key, persona_update_request, task_id,
        )

        print(
            f"[Memory] L3 入队：scope={scope.key} 立刻执行"
            f"（{_trigger_label(persona_update_request)}）"
            f", task={task_id}",
            flush=True,
        )

        return task_id

    except Exception as exc:  # noqa: BLE001 —— 旁路，见函数说明
        # ⚠️ 这一条**必须**是 warning：吞掉异常之后，它是"这次没排上"的
        # 唯一痕迹。与 L1/L2 不同的是**不需要**提示怎么补 —— 下一次 L2 完成
        # 会把同一个 scope 重新排一遍（水位没动，场景还在）。
        logger.warning(
            "[Memory] 提交 L3 生成任务失败（scope=%s，场景仍在库里，"
            "下一次 L2 完成会重新排）：%s",
            scope.key, exc,
        )
        return None


def _trigger_label(
        persona_update_request: bool,
) -> str:
    """入队那一行里的触发原因 —— **一条都没有时要说清楚**。"""

    labels: list[str] = []

    if persona_update_request:
        labels.append("L2 举手")

    # ⚠️ 空的时候不能只留一对空括号
    # --------------------------
    # 没有举手时任务**仍然可能**真的生成 —— 判据是"这个 scope 还没有画像"
    # （首次创建），而那是执行侧查出来的事实，触发侧看不到。所以这里的措辞
    # 要把那个可能性说出来，而不是让人觉得"没信号 = 一定会空转"。
    return "、".join(labels) or "无举手（执行侧按首次创建判）"


# ============================================================
# internals
# ============================================================


def task_id_of(ctx: Any) -> UUID | None:
    """
    正在跑的这条任务的 id —— 给下游任务的 `causation_id` 用。

    ⚠️ 为什么住这里，而不是各自散在 handler 里
    ---------------------------------
    三个 handler（L1 / L2 / L3）都要它，而它有一个**不显然**的写法。抄三份
    的话，某一份写成 `ctx.task.task_id` 就会在冒烟里炸（见下），而那时离
    "我复制了一份"这个原因很远。

    ⚠️ 用 `getattr` 链而不是 `ctx.task.task_id`
    -------------------------------------
    `ctx` 是 `TaskContext`（见 app/common/task/models.py），由 Worker 注入。
    但**冒烟脚本会传 None**（它们直接调 handler，不经过 Worker）与
    `SimpleNamespace`（只带它关心的字段）。为一条**可选的溯源信息**让冒烟
    脚本炸掉不划算 —— 拿不到就是 None，`causation_id` 本来就允许为空
    （见 Task 的字段说明）。
    """

    return getattr(getattr(ctx, "task", None), "task_id", None)


def _scene_execute_at(
        *,
        last_l2_at: datetime | None,
        after_l1_delay_seconds: float,
        min_interval_seconds: float,
) -> datetime:
    """
    下一次整合的**最早**时间：`max(now + 90s, lastL2 + 900s)`。

    两个来源各挡一件事，见 `SCENE_AFTER_L1_DELAY_SECONDS` /
    `SCENE_MIN_INTERVAL_SECONDS` 的说明。`last_l2_at` 为 None（这个会话还没
    整合过）时只有第一个数起作用 —— 首次整合本来就该快一点。

    抽成纯函数是为了**可测**：那两个常数是"待校准的起点"，而校准的方式
    就是断言这里算出来的时间（不必起任务基建）。
    """

    earliest = (
        datetime.now(timezone.utc)
        + timedelta(seconds=after_l1_delay_seconds)
    )

    if last_l2_at is not None:
        earliest = max(
            earliest,
            last_l2_at + timedelta(seconds=min_interval_seconds),
        )

    return earliest


async def _task_by_key(
        repositories: RepositoryFactory,
        key: str,
) -> Any | None:
    """
    按幂等键取那条任务；没有就 None。

    ⚠️ 为什么不按状态扫一遍
    -------------------
    会话就是识别键，键上直接就能查到那一条 —— 一次唯一索引级的查询。扫
    "该 handler 的全部 pending 任务再在 Python 里筛"要多一次查询、还要处理
    分页：**扫不到与没有待办长得一样**，而扫不到的后果是多排一条本该被
    推迟的任务。

    ⚠️ 为什么这个键能反复用（以前不能）
    -----------------------------
    任务进终态时会把键释放（见 task_impl 的「幂等键的生命周期」）。所以
    "这个键查得到" == "这个会话此刻有一条活着的任务"，而不是"这个会话这一
    辈子只能有一条任务"。
    """

    async with repositories.scope(begin=False) as repos:

        return await repos.create(
            TaskStore
        ).get_by_idempotency_key(key)


def _log_trigger(
        session_id: UUID,
        pending_turns: int,
        trigger_turns: int,
        idle_delay_seconds: float,
        execute_at: datetime | None,
        task_id: UUID,
) -> None:
    """
    这一轮的触发判定打到控制台 —— **两种判定都打**。

    ⚠️ 为什么"没到阈值"那条尤其要看得见
    ------------------------------
    它是"先排一条 60s 后的"分支，平时最容易被当成"什么都没发生"：任务不在
    当下执行，也没有别的痕迹。而它恰恰是排查"为什么记忆没更新"时的第一站
    （要么没排上，要么排上了但抽取空转）。

    ⚠️ 为什么"立即执行"那条也打（2026-10-11 补）
    --------------------------------------
    原来是 `execute_at is None 就 return`，于是控制台上只看得见延迟入队 ——
    排查时看到一串"延迟入队"却看不出"哪一次真的到了阈值"，而这两件事的
    区别正是"触发条件对不对"的答案。现在两种判定各一条，措辞不同：

        [Memory] L1 延迟入队   ... -> 60s 后执行
        [Memory] L1 立即入队   ... -> 达标，立刻执行

    频率没有变（每次收尾本来就只走其中一条），所以不会变吵。

    ⚠️ 为什么是 print 而不是 logger.info
    --------------------------------
    本应用**没有配置 logging**（main.py 里没有 basicConfig，也没有日志配置
    模块），所以 root logger 停在默认的 WARNING —— `logger.info` /
    `logger.debug` 在控制台上**一个字都不会出现**。要"控制台上看得见"就
    只有 print。（`submit_atom_extract` 里那句 `logger.debug` 是给将来配上
    日志之后用的诊断信息，两件事不冲突。）
    """

    # 阈值关掉时（trigger_turns <= 0）走的永远是延迟那条，说"未到 N 轮"
    # 读起来像 bug，所以分开措辞。
    reason = (
        f"待抽 {pending_turns} 轮，未到 {trigger_turns}"
        if trigger_turns > 0
        else "阈值已关闭"
    )

    if execute_at is None:
        print(
            f"[Memory] L1 立即入队：session={session_id} "
            f"待抽 {pending_turns} 轮（已达阈值 {trigger_turns}）"
            f" -> 立刻执行, task={task_id}",
            flush=True,
        )
        return

    print(
        f"[Memory] L1 延迟入队：session={session_id} "
        f"{reason}"
        f" -> {idle_delay_seconds:.0f}s 后执行, task={task_id}",
        flush=True,
    )
