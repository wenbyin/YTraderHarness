"""
MemoryPersonaServiceImpl —— L3 画像生成（一次模型调用）。

一次生成的形状
------------
    解析 scope -> 读水位 -> 按水位取变化场景
      -> 读该 scope 的现有画像（可能没有）+ 场景总数
      -> [模型：一次回复给出 no_change / 整篇 content] -> 解析
      -> 写画像（新建走 insert、改写走乐观锁 update）
      -> 推水位

⚠️ "该不该跑"由**这一侧**判（与 L1/L2 相反）
------------------------------------
触发侧（`trigger.submit_persona_extract`）挂在**每一次 L2 完成**上、
**不做任何判断**：没有活着的任务就排一条，有就复用/提前。它只把那一刻的
**一个事实**带进参数 —— `persona_update_request`（L2 的模型举没举手）。

真正决定要不要调模型的是这里。**task 跑 ≠ 提取跑**：排出来不代表执行过，
三条判据全在这一侧现判：

    首次创建    这个 scope 有场景、但**还没有画像**
    L2 举手      `persona_update_request` 为 True
    满 N 轮次    自上次**真正生成**以来，这个 scope 攒够
                 `PERSONA_TRIGGER_TURNS` 个 L0 用户轮次（**跨会话**）

三条**任一条成立**就真提取；都不成立就什么都不做 —— 不调模型、不写画像、
**也不推 cursor**。**变化场景不是触发条件**，它只是输入（见第 4 步）。

**为什么 L3 可以这样而 L1/L2 不行**，见 `MemoryPersonaExtractParams` 的
docstring。一句话：L3 的判据都是"查一次就知道"的事实（画像行在不在、
水位到哪了），放在这里判不但不重复，而且比在触发侧判**更新鲜**。

⚠️ 模型**一轮定输赢**，没有工具、没有多轮
--------------------------------
与 L2 逐字同一条理由（见 `memory_scene_service_impl.py` 的模块 docstring
开头那段）：后台任务里没有工具回路。原项目让模型用 read / write / edit
操作 `persona.md`，这里改成"它在一段 JSON 里给出整篇内容，工程侧写库"。

代价说清楚：模型**看不到自己的执行结果**，也没有第二轮去补救。所以下面
每一处"看不懂就不做"的取舍都倾向**保留数据**，而不是猜它的意图。

⚠️ `no_change` 也**推水位**
---------------------
这一处值得单独说：模型看了这批场景、判断画像不用改，那么这批场景就是
**已经抽象过**了 —— 不推水位的话，下一轮（比如 50 轮次之后）会把同一批
场景原样再喂一遍，而结论必然还是不改。

判断本身也是"消费"，与 L2 里"模型给了空 operations 也照样推水位"是同一
件事。

⚠️ 内容没变（md5 相同）**同样推水位、但不推 version**
----------------------------------------
推水位：这批场景看过了。不推 version：否则"版本号"会变成"跑过几轮"的
同义词，乐观锁的基线失去意义（与 L2 的 `_apply` 同一条）。

⚠️ 水位与画像行**同一个事务**
----------------------
调用方的作用域（`scope(begin=...)`）退出时一起提交。见契约里那段 ——
这是这条链上唯一不能让步的地方。

⚠️ 模型调用走 ModelRuntime，不是自己拼客户端
-----------------------------------
超时、重试、降级、用量观测都在那一层，而且那里重试的是 **provider
attempt**，不是重跑本方法。本方法因此不为"模型抽风"写任何重试。

⚠️ 解析失败要抛，不要吞
------------------
"这一轮什么都没改"与"模型输出坏了"在数据上长得一样，但一个是常态、
一个是 bug。吞掉等于让后者永远查不出来（见 domain/schemas/persona.py）。
"""

from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context.processor.messages import (
    message_text as llm_message_text,
)
from app.agent.model import ModelRequest, ModelRole, get_model_registry
from app.agent.prompt import get_prompt_provider
from app.agent.prompt.prompts.memory_persona_extract import (
    render_user_message,
)
from app.common.exceptions import ConflictError
from app.common.service.base import ServiceContext

from ..domain.entities.persona import (
    PersonaCursorState,
    PersonaRow,
)
from ..domain.entities.scene import (
    DEFAULT_AGENT_ID,
    PERSONA_FILENAME,
    SceneScope,
    SceneRow,
    persona_id,
)
from ..domain.schemas.persona import PersonaResult, parse_persona
from app.services.session.repository.message import MessageRepository
from ..domain.tasks import MemoryPersonaExtractParams
from ..repository.persona import MemoryPersonaRepository
from ..repository.scene import MemorySceneRepository
from .memory_persona_service import MemoryPersonaService
from .profile_cache import profile_cache_invalidation_hook

__all__ = [
    "AGENT_TYPE",
    "DEFAULT_PERSONA_BATCH",
    "PERSONA_MAX_CHARS",
    "TEAM_MAX_CHARS",
    "MemoryPersonaServiceImpl",
]

logger = logging.getLogger(__name__)


#: 模型绑定用的 agent_type（见 model.yaml 的 bindings）。
#:
#: ⚠️ 与 model_role 不是一个东西：model_role 是**动作**
#: （`memory.persona.extract`，见 ModelRole），agent_type 是**产物/服务**。
#: 与 L1/L2 那三条同一条口径（它们取了表名）。
AGENT_TYPE = "memory_persona_extract"


#: 一次生成最多消费几个变化场景。
#:
#: ⚠️ 正常情况下**永远够用**：一个 scope 的场景数由 L2 的
#: `DEFAULT_MAX_SCENES`（12）把着，模型应当自己 MERGE 下来。但那条上限只是
#: **告警**（L2 的实现明确不截断），所以场景数真的可能涨上去 —— 这个数就是
#: 那种情况下的兜底。
#:
#: ⚠️ 超了**不是丢数据**，是分几轮追上：增量按 `updated_at` **升序**取，
#: 水位只推到"真的消费完的那一条"，剩下的下一轮接着来。反过来（取最新的
#: 一批）会把中间那段永久跳过 —— 那是静默丢数据。
#:
#: 50 的理由：每篇场景正文上限约 1500 字符（L2 提示词的约束），50 篇 ≈
#: 75KB 输入。再多就该怀疑 L2 那条链出问题了，而不是继续往上下文里塞。
DEFAULT_PERSONA_BATCH = 50


#: 画像正文的字数上限（**给模型看的约束在提示词里**，这里是观测用的）。
#:
#: ⚠️ 超了**只告警，不截断** —— 与 L2 的 `DEFAULT_MAX_SCENES` 同一条取舍：
#: 砍掉尾巴等于毁掉一份完整文档（画像是一整篇，断在半句上比啰嗦糟得多），
#: 而超长本身只是"这一篇有点啰嗦"，下一次生成会压回来。
#:
#: 分两个值是因为提示词给个人与团队定了不同的上限（2000 / 1200 字符）。
PERSONA_MAX_CHARS = 2000
TEAM_MAX_CHARS = 1200


#: 「满 N 轮次」那条触发的阈值：自上次**真正生成**以来，**这个 scope**
#: 攒了多少 L0 用户轮次就把画像重算一遍。
#:
#: ⚠️ 数与判都在**执行侧**（2026-10-13）
#: ----------------------------------
#: 它曾经在触发侧算（`trigger._persona_force`，把结论当 `force` 参数带进来）。
#: 现在触发侧只排队、不做任何判断，三个条件全在执行侧现判 —— 于是这次
#: 计数也搬到了这里。好处是它读的是**此刻**的水位，不会在排队期间过期。
#:
#: ⚠️ 数的是**整个 scope**（跨会话），不是触发它的那一个会话
#: --------------------------------------------------
#: 画像本身就是跨会话聚合的（一个 scope 一篇，被该 scope 下所有会话共同
#: 书写），所以"距上次生成攒了多少轮"必须也按 scope 数。只数触发它的那
#: 一个会话的话，一个习惯开很多短会话的用户**永远到不了阈值**，而**不报错**。
#:
#: 口径落在 `MessageRepository.count_user_turns_in_scope`（agent_message
#: join agent_session，duty_key + user/org 三个维度都匹配上）—— 与 L1 那边
#: "一轮"的定义是同一处，本模块不另数一遍。
#:
#: ⚠️ 这是一个**待校准的起点**，不是实测出来的值。调它两个方向各有一个
#: 可观察的症状：调太小则画像每改一点就重写整篇；调太大则用户在几十轮里
#: 明显变了、Agent 还用着旧的画像。
PERSONA_TRIGGER_TURNS = 50


class MemoryPersonaServiceImpl(MemoryPersonaService):

    def __init__(
            self,
            *,
            personas: MemoryPersonaRepository,
            scenes: MemorySceneRepository,
            messages: MessageRepository,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._personas = personas
        self._scenes = scenes
        #: 「满 N 轮次」那条触发要跨会话数用户轮次 —— 口径归 session 模块，
        #: 所以注入它而不是在这里拼一条 SQL（见 PERSONA_TRIGGER_TURNS）。
        self._messages = messages

    # ========================================================
    # Handler 入口
    # ========================================================

    async def memory_persona_extract(
            self,
            params: MemoryPersonaExtractParams,
            ctx: Any,
    ) -> None:

        # ---- 1. 解析 scope ----------------------------------------
        #
        # ⚠️ 走 `SceneScope.resolve`，与 L2 **同一个落点**
        # ------------------------------------------
        # "这份画像属于谁"在两层的口径必须逐字相同（L3 的输入就是 L2 的
        # 产物）。自己在这里拼一个 = 第二个口径，而它一旦漂了，症状是
        # "L2 写进 scope A 的场景、L3 从 scope B 读"—— 读出来是空的，
        # 不报错，画像永远不更新。
        #
        # 参数里那三个值来自 L2 完成时的判定（见 trigger 的
        # `submit_persona_extract`），已经过一次同样的 resolve —— 这里是
        # 第二次，代价是一次纯计算，换来的是"任务参数被人工改过"也能被挡住。
        scope = SceneScope.resolve(
            agent_id=params.agent_id,
            user_id=params.user_id,
            org_id=params.org_id,
        )

        now = datetime.now(timezone.utc)

        # ---- 2. 画像在不在 —— 这是**跳过判据**，所以排在最前 ----------
        #
        # ⚠️ 2026-10-13 从这里改了触发条件
        # ---------------------------------
        # 原来是「自水位以来有变化场景」也算一条触发路径，于是 L2 每次改写了
        # 场景、画像就跟着重写一遍 —— 而画像是**慢变量**，"随便改了一条场景"
        # 不等于"这个人变了"。
        #
        # 三条，满足任一条就真提取 —— **全在执行侧现判**：
        #
        #     首次创建    这个 scope 有场景、但**还没有画像**
        #     L2 举手      `params.persona_update_request` 为 True
        #     满 N 轮次    自上次**真正生成**以来，这个 scope 攒够
        #                  `PERSONA_TRIGGER_TURNS` 个 L0 用户轮次（跨会话）
        #
        # **变化场景不再是触发条件**，它退回成纯粹的**输入**。所以判定必须
        # 先取画像 —— 它原来是第 4 步，现在提到第 2 步。
        #
        # ⚠️ task 跑 ≠ 提取跑
        # -----------------
        # 触发侧挂在每一次 L2 完成上、**无条件**排队，所以"排了一条 L3 任务"
        # 不代表这一层提取执行过。三条都不成立时这里**什么都不做**：不调模型、
        # 不写画像、**也不推水位**（下一步那段说明）。
        persona = await self._personas.get_persona(scope)

        # 场景水位（增量起点）与轮次基准都在这，取一次。
        cursor = await self._personas.get_l3_cursor(scope)

        # ---- 3. 三条都不成立：空转，什么都不做 --------------------------
        #
        # ⚠️ 空转**是常态**，不是失败：L3 挂在每一次 L2 完成上（每个会话至少
        # 15 分钟一次），而画像只在上面那三条之一成立时才该重算。
        #
        # ⚠️ 这里**不推 cursor**（2026-10-13 改）
        # -----------------------------------
        # 原来跳过时会把 `last_l3_at` 推到当下。那件事有两个问题：
        #
        #   语义上   `memory_persona_cursor` 整行回答的是"这一层的提取**消费**
        #            到哪了"（另两列是 last_scene_updated_time / ids）。空转
        #            没有消费任何东西 —— 场景水位没动、没调模型、没写画像。
        #            往一张进度表里记一笔没发生的事，是错的定义。
        #
        #   功能上   第三条第判据的基准**就是** `last_l3_at`。空转推它，等于
        #            每跳过一次就把"攒够 50 轮"的窗口归零 —— 窗口于是永远只有
        #            "两次 L3 之间"那么长（≈ L2 的最小间隔 15 分钟），第三条
        #            永远到不了。实测历史 6 条任务的 force 一次都没真过。
        #
        # 所以判据不成立时**一条 cursor 都不推**：`last_l3_at` 回到它字段名
        # 的字面意思 —— 上次 L3 **提取**跑完。
        if (persona is not None
                and not params.persona_update_request
                and not await self._enough_turns(scope, cursor=cursor)):

            # ⚠️ 这条 print 是"为什么画像没更新"的第一站
            # --------------------------------------
            # 触发侧那条同义的 print 已经随判定一起删掉，于是"排了但跳过"
            # 与"压根没排"在控制台上会重新变得一样 —— 所以这里必须把**三
            # 条各自**说清楚（与 `_log_result` 那四种措辞同一个理由）。
            print(
                f"[Memory] L3 跳过：scope={scope.key} 已有画像"
                f"（v{persona.version}）、L2 没举手、且自上次生成未满 "
                f"{PERSONA_TRIGGER_TURNS} 轮 —— 什么都不做。",
                flush=True,
            )

            return

        # ---- 4. 按水位取变化场景（**只是输入**，不再是判据）--------------

        scenes = await self._scenes.list_scene_delta(
            scope,
            updated_after=cursor.last_scene_updated_time,
            consumed_ids=cursor.last_scene_ids,
            limit=DEFAULT_PERSONA_BATCH,
        )

        if not scenes:
            # 判据成立了但没有可抽象的场景：调一次模型也没有输入。跳过，并
            # 把理由说清楚 —— 它与"三条都不成立"那一条**是两回事**（一条是
            # 不该重算，这一条是没东西可算）。
            #
            # ⚠️ 这里也**不推 cursor**（同第 3 步）：没有发生提取，就没有
            # 需要记录"消费到哪了"的东西。场景水位本来也没动。
            print(
                f"[Memory] L3 跳过：scope={scope.key} 判据成立但自 "
                f"{cursor.last_scene_updated_time or '（从未生成）'} 起没有"
                "变化的场景，没有可抽象的输入 —— 什么都不做。",
                flush=True,
            )

            return

        # 这一批被截断了（还有更多）—— 说清楚，别让它看起来像"全抽象完了"。
        # 见 DEFAULT_PERSONA_BATCH 的说明：剩下的下一轮接着来。
        if len(scenes) >= DEFAULT_PERSONA_BATCH:
            print(
                f"[Memory] L3 变化场景达到单批上限 {DEFAULT_PERSONA_BATCH} 个"
                f"（scope={scope.key}），本轮只抽象最早的这一批，"
                "余下的下一轮继续。",
                flush=True,
            )

        # ---- 5. 场景总数 --------------------------------------------
        #
        # `persona` 为 None 是**正常**的两种情形之一：
        #
        #     从来没有生成过          -> 提示词进入"首次生成"（不另开分支，
        #                               见 prompts 里 render_user_message）
        #     另一个任务刚刚建了行     -> 不冲突，我们读到那一行就是了
        #
        # ⚠️ 刻意**不**在这里抛"没有画像行"：首次生成正是这条路径本身。
        scene_count = await self._scenes.count_scenes(scope)

        # ---- 5. 一次模型调用 --------------------------------------
        result = await self._generate(
            scope=scope,
            scenes=scenes,
            persona=persona,
            scene_count=scene_count,
            # ⚠️ L2 的"举手"理由原样透传进提示词（原项目的 `triggerInfo`）。
            # 它是**上下文**，不参与这里任何分支 —— 见 params 的说明。
            persona_update_request=params.persona_update_request,
            now=now,
        )

        # ---- 6. 落库 ----------------------------------------------
        written = await self._apply(
            scope,
            persona=persona,
            content=None if result.no_change else result.content,
        )

        # ---- 7. 推水位 --------------------------------------------
        #
        # ⚠️ 推到最后一条**被消费的**场景（升序取出来，所以是最后一个）。
        # 边界时间戳上已消费的 id 一起记下：同一事务写入的场景共享一个
        # `now()`，只按 `>` 取会让恰好落在同一微秒的那批被静默跳过。
        #
        # ⚠️ **第三步与第六步都推**（no_change 也推）：判断本身也是消费。
        # 漏掉它会造出一个死循环 —— 同一批场景每 50 轮被重新喂一次，
        # 而模型每次都答"不用改"。
        #
        # ⚠️ 必须在上面全部做完之后、且与它们同一个事务里（调用方的作用域）。
        # 提前推 = 那批场景永久进不了画像；分事务 = 画像没写而水位动了，
        # 症状同样是永久跳过。
        if scenes:

            await self._advance(
                scope,
                cursor=cursor,
                last_l3_at=now,
                last_scene_updated_time=scenes[-1].updated_at,
                last_scene_ids=[
                    scene.id
                    for scene in scenes
                    if scene.updated_at == scenes[-1].updated_at
                ],
            )

        else:

            # ⚠️ 强制触发（举手 / 满轮次）但**一条新场景都没有**
            # --------------------------------------------------
            # 只戳 `last_l3_at`，场景水位**不动** —— 没有可消费的东西，推它
            # 等于宣称"这些都看过了"，而它们本来就没进过这一轮。
            #
            # 这条分支只有"被强制"时才走得到：没有强制信号时空转在第三步就
            # 已经返回了（那里也戳 `last_l3_at`，两次含义相同）。
            await self._advance(scope, cursor=cursor, last_l3_at=now)

        self._log_result(
            scope=scope,
            scene_count=scene_count,
            changed=len(scenes),
            result=result,
            written=written,
            persona=persona,
        )

        # 画像可能刚被改写。**提交成功之后**失效 L3 缓存（L2 的键不
        # 受影响）；提前删会让读者在事务提交前把旧值重新写回缓存。
        self.post_commit.append(
            profile_cache_invalidation_hook(scope, "l3")
        )

    # ========================================================
    # 触发判据之一：满 N 轮次
    # ========================================================

    async def _enough_turns(
            self,
            scope: SceneScope,
            *,
            cursor: PersonaCursorState,
    ) -> bool:
        """
        自上次**真正生成**以来，这个 scope 攒够 `PERSONA_TRIGGER_TURNS` 个
        L0 用户轮次了吗。**跨会话**数（见那个常量的说明）。

        ⚠️ 基准是 `cursor.last_l3_at`，而它**只在真正提取之后才推进**
        ------------------------------------------------------
        判据不成立的那两条路径（第 3、4 步）都不推 cursor，所以这个基准
        就是"上次真正生成"的时刻 —— 这正是第三条触发要的语义。

        它曾经被跳过路径一起推（于是每空转一次窗口就归零、这条永远到不了），
        那是这次改动要修的东西，见第 3 步那段说明。

        ⚠️ 没有基准时不数
        -------------
        `last_l3_at` 为空 = 这个 scope **从没生成过**。那种情况下"距上次
        生成攒了多少轮"没有起点，而首次生成由第 2 步（`persona is None`）
        负责 —— 调用方也只在 `persona is not None` 时才问这里，所以这条
        分支实际到不了。留着是防御：水位行与画像行理论上可能不同步。

        取不到计数是 **0**（不是抛）：会话模块那边读不到就说读不到，而
        "这轮不生成"比"整轮失败"轻得多。
        """

        if cursor.last_l3_at is None:
            return False

        pending_turns = await self._messages.count_user_turns_in_scope(
            agent_id=scope.agent_id,
            user_id=scope.user_id,
            org_id=scope.org_id,
            since=cursor.last_l3_at,
            # ⚠️ 兜底职责从这儿传进去，不让 session 模块 import memory 模块
            # —— 口径的**来源**仍然只有一处（SceneScope.resolve 那条链），
            # 这里只是把它送过去。
            default_agent_id=DEFAULT_AGENT_ID,
        )

        return pending_turns >= PERSONA_TRIGGER_TURNS

    # ========================================================
    # 模型调用
    # ========================================================

    async def _generate(
            self,
            *,
            scope: SceneScope,
            scenes: list[SceneRow],
            persona: PersonaRow | None,
            scene_count: int,
            persona_update_request: bool,
            now: datetime,
    ) -> PersonaResult:

        runtime = get_model_registry().get_runtime(
            agent_type=AGENT_TYPE,
            model_role=ModelRole.MEMORY_PERSONA_EXTRACT,
        )

        response = await runtime.ainvoke(
            self._build_request(
                scope=scope,
                scenes=scenes,
                persona=persona,
                scene_count=scene_count,
                persona_update_request=persona_update_request,
                now=now,
            )
        )

        return parse_persona(llm_message_text(response.message))

    def _build_request(
            self,
            *,
            scope: SceneScope,
            scenes: list[SceneRow],
            persona: PersonaRow | None,
            scene_count: int,
            persona_update_request: bool,
            now: datetime,
    ) -> ModelRequest:
        """
        提示词 + 这次的变化场景与现有画像 -> 一次模型请求。

        单独成方法是为了**可测**（冒烟里断言"喂进去的长什么样"不必真的调
        模型）—— 与 L1/L2 的 `_build_*_request` 同一条理由。
        """

        # ⚠️ `is_team` 是这个 role 唯一的变量，也是**两套系统提示词的分派点**
        # --------------------------------------------------------
        # scope 带 org 的走团队 Operating Doctrine（≤1200 字），否则走个人
        # User Narrative Profile（≤2000 字）。见 prompts 里 `_render`。
        system_prompt = get_prompt_provider().node_prompt(
            ModelRole.MEMORY_PERSONA_EXTRACT,
            is_team=scope.is_team,
        )

        # 用户消息**怎么排版**归提示词模块（统计块、变化场景、现有画像、
        # 迭代指南、时间）—— 见 prompts/__init__.py 的约定。
        # 这里只负责**取**：哪个 scope 的哪些场景、现有画像是什么、几点。
        human = render_user_message(
            changed_scenes=scenes,
            scene_count=scene_count,
            existing_persona=persona.content if persona is not None else None,
            persona_update_request=persona_update_request,
            now=now,
        )

        return ModelRequest(
            messages=(
                SystemMessage(content=system_prompt),
                HumanMessage(content=human),
            ),
            # 与 L1/L2 那几次一样**不关思考**：这一次要做的判断（哪些证据
            # 值得进画像、旧结论该不该被推翻、该不该干脆不改）比重写一篇
            # 文档更依赖推理，关掉的代价更大。
            metadata={
                "purpose": "memory_persona_extract",
                "agent_type": AGENT_TYPE,
                "model_role": ModelRole.MEMORY_PERSONA_EXTRACT,
                "scope": scope.key,
                "is_team": scope.is_team,
                "changed_scene_count": len(scenes),
                "scene_count": scene_count,
                "has_persona": persona is not None,
                # 只记"有没有"，不记理由原文 —— 它已经在提示词里了，
                # 再抄一份到 metadata 等于让同一段文本有两个来源（排查时
                # 不知道哪一份是模型当时看到的）。
                "persona_update_requested": persona_update_request,
            },
        )

    # ========================================================
    # 落库
    # ========================================================

    async def _apply(
            self,
            scope: SceneScope,
            *,
            persona: PersonaRow | None,
            content: str | None,
    ) -> str:
        """
        模型给的整篇内容 -> 真的改库。返回这次做了什么（落日志 / 冒烟断言）：

            "no_change"  模型说不用改，一行库都没动
            "unchanged"  内容与库里逐字相同，跳过写入、也**不推 version**
            "created"    新建了一行
            "updated"    原地改写（version +1）

        ⚠️ 内容为空**不是**"不改"
        ---------------------
        `content` 为 None 只有一条来路：`no_change` 为真（解析层把"没给
        content"与"给了空 content"分开了，见 domain/schemas/persona.py）。
        到这里再判一次空串是为了防"模型给了 `   `"那种 —— 它会撞
        `content <> ''` 的 CHECK，而报错离原因很远。
        """

        if content is None:
            return "no_change"

        digest = hashlib.md5(content.encode("utf-8")).hexdigest()

        if persona is None:
            inserted = await self._personas.insert_persona(
                PersonaRow(
                    id=persona_id(scope),
                    filename=PERSONA_FILENAME,
                    content=content,
                    content_md5=digest,
                    agent_id=scope.agent_id,
                    user_id=scope.user_id,
                    org_id=scope.org_id,
                )
            )

            if inserted == 0:
                # id 是 scope 派生的，所以冲突意味着"同一个 scope 的画像刚刚
                # 被另一个任务建了行"（两个会话的 L2 完成各排了一条，而它们
                # 都读到"还没有画像"）。与 L2 的 `insert_scenes` 同一条处理：
                # 不覆盖别人的结果，抛出去重来。整个事务会回滚。
                raise ConflictError(
                    f"scope={scope.key} 的画像在这次生成期间被别的任务创建，"
                    "本次放弃，重排后重来。"
                )

            return "created"

        if persona.content_md5 == digest:
            # 内容逐字相同：不写、**也不推 version**。推的话"版本号"会变成
            # "跑过几轮"的同义词，乐观锁的基线就没意义了。
            return "unchanged"

        affected = await self._personas.update_persona(
            persona.id,
            content=content,
            content_md5=digest,
            expected_version=persona.version,
        )

        if affected == 0:
            # 读出来到现在，另一个 L3 任务（同一个 scope 的另一个触发）改过
            # 这一行。抛出去让任务重入 —— 它会读到新内容、重新调模型。
            # 不重试就是拿一份过期画像的结论覆盖掉别人的。
            raise ConflictError(
                f"scope={scope.key} 的画像在本次生成期间被别的任务改过，"
                "本次放弃，重排后重来。"
            )

        return "updated"

    # ========================================================
    # 日志
    # ========================================================

    def _log_result(
            self,
            *,
            scope: SceneScope,
            scene_count: int,
            changed: int,
            result: PersonaResult,
            written: str,
            persona: PersonaRow | None,
    ) -> None:
        """
        这一轮到底做了什么 —— **打到控制台**。

        ⚠️ 为什么是 print 而不是 logger.info
        --------------------------------
        与 L1/L2 同一条：本应用**没有配置 logging**（main.py 里没有
        basicConfig），所以 root logger 停在默认的 WARNING —— `logger.info`
        / `logger.debug` 在控制台上**一个字都不会出现**。而"这一轮到底改了
        什么"是最常见的问题：

            no_change  模型判"现有画像已经够了"（不是没跑）
            unchanged  它重写了一遍但逐字没变（也不是没跑）
            created    建了第一篇
            updated    真的改了

        这四种在控制台上原本长得**完全一样**（都什么都不打印），而它们的
        原因完全不同。所以四种各一条，措辞可区分。
        """

        # 字数上限只**告警**，不截断（见 PERSONA_MAX_CHARS 的说明）。
        limit = TEAM_MAX_CHARS if scope.is_team else PERSONA_MAX_CHARS

        if result.content is not None and len(result.content) > limit:
            logger.warning(
                "[Memory] scope=%s 的画像 %d 字符，超过提示词里的上限 %d"
                "（不截断，下一次生成会压回来）。",
                scope.key, len(result.content), limit,
            )

        verbose = {
            "no_change": f"模型判断不用改（{changed} 个场景已看过）",
            "unchanged": f"重写后逐字未变（{changed} 个场景已看过）",
            "created": f"新建画像（{len(result.content or '')} 字符）",
            "updated": f"改写画像（{len(result.content or '')} 字符，"
                       f"version {persona.version} -> {persona.version + 1}）"
                       if persona is not None else "改写画像",
        }[written]

        logger.info(
            "[Memory] scope=%s 的 L3 完成：%d 个变化场景（共 %d 个）-> %s。",
            scope.key, changed, scene_count, verbose,
        )

        print(
            f"[Memory] L3 完成：scope={scope.key} "
            f"{changed} 个变化场景（共 {scene_count} 个）-> "
            f"{written}（{verbose}）",
            flush=True,
        )

    # ========================================================
    # internals
    # ========================================================

    async def _advance(
            self,
            scope: SceneScope,
            *,
            cursor: PersonaCursorState,
            last_l3_at: datetime,
            last_scene_updated_time: datetime | None = None,
            last_scene_ids: list[str] | None = None,
    ) -> None:
        """
        推水位。**不传 `last_scene_updated_time` 就是"只戳 last_l3_at"**
        （场景水位保持原值）—— 空转那条路径走的就是它。
        """

        await self._personas.advance_l3_cursor(
            scope,
            last_scene_updated_time=(
                cursor.last_scene_updated_time
                if last_scene_updated_time is None
                else last_scene_updated_time
            ),
            last_scene_ids=(
                cursor.last_scene_ids
                if last_scene_ids is None
                else last_scene_ids
            ),
            last_l3_at=last_l3_at,
        )
