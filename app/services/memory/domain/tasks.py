"""
Memory 后台任务的参数模型。

`Task.params` 必须是 pydantic 模型（裸 dict 会被 submit 拒绝，见
app/common/task/models.py 的 `_reject_bare_dict`），而且**只放业务 ID 与
执行所需的最小参数** —— 大状态按 ID 在执行阶段重新加载。这条纪律在这里的
落点很具体：**不要把消息正文塞进来**。

    ✗ 带 messages    Task 会被反复读出（Dispatcher 每轮轮询读、Worker claim
                     读、查询接口读、重试读），塞一份正文等于把最热的那些
                     查询全部变重；而且消息已经是 `agent_message` 里的
                     权威数据，任务里再存一份就是从第二个地方回答"这一轮
                     说了什么"
    ✓ 带 run_id      执行阶段用 session 模块的 `MessageRepository.list_by_run`
                     按它取回来

⚠️ handler 名字是**持久化数据**
---------------------------
`handler="memory.l0.extract"` 这个字符串会按字面存进 tasks 表。改名 = 让库里
已有的行解析失败（判 FAILED）。要改就得新旧名字都注册一段时间 ——
见 app/infra/task/registry.py。
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

__all__ = [
    "MEMORY_EXTRACT_HANDLER",
    "MEMORY_ATOM_EXTRACT_HANDLER",
    "MEMORY_SCENE_EXTRACT_HANDLER",
    "MEMORY_PERSONA_EXTRACT_HANDLER",
    "MemoryExtractParams",
    "MemoryAtomExtractParams",
    "MemorySceneExtractParams",
    "MemoryPersonaExtractParams",
]

#: handler 的逻辑名。稳定契约，不要跟着模块改名走。
#:
#: ⚠️ 命名空间是 `memory.<层>.<动作>`，与**表名**是两回事：表/类叫
#: memory_atom（L1 原子记忆），handler 仍然叫 memory.l1.* —— 与既有的
#: memory.l0.extract 对齐。改 handler 名 = 让库里已有的任务行解析失败，
#: 所以这条命名不跟着存储层的改名走。
#:
#: ⚠️ L1 只有**一个** handler
#: -----------------------
#: 抽取与去重（冲突检测）在同一个 handler 里**先后**执行 —— 不拆成两条
#: 任务。理由见 memory_atom_service_impl.memory_atom_extract 的说明：两步之间是"刚写下的
#: 那批记录"这个只在内存里存在的上下文，拆成两条任务就得把它塞进
#: params 或让第二步去反推。
#:
#: ⚠️ L2 与 L1 是**两条**任务，不是一条链里的两步
#: ----------------------------------------
#: 表/类叫 memory_scene（场景文件），handler 叫 memory.l2.extract。
#: 拆分而不是并进 L1 的理由有两条，都硬：
#:
#:     时机不同   L1 攒够轮次就抽；L2 抽完还要再等一会儿才整合（见
#:                application/trigger.py 的 `submit_scene_extract`），
#:                并在一起就没法各排各的时间
#:     上下文在库里  L1 的抽取与去重之间隔着一个"刚写下的那批"（只在内存
#:                里）；L1 与 L2 之间隔着的是**水位**（memory_scene_cursor），
#:                那在库里。所以拆开不丢信息，还换来一件事：L2 重试不必
#:                把 L1 已经付过的模型调用再付一遍
#:
#: ⚠️ 但 model_role 叫 `memory.scene.consolidate`（动作），不叫
#: `memory.l2.extract`（handler）—— 见 app/agent/model/types.py 的说明。
MEMORY_EXTRACT_HANDLER = "memory.l0.extract"
MEMORY_ATOM_EXTRACT_HANDLER = "memory.l1.extract"
MEMORY_SCENE_EXTRACT_HANDLER = "memory.l2.extract"

#: L3 画像生成的 handler。
#:
#: ⚠️ 与 L2 一样是**独立的**一条任务，而不是 L2 的后半段
#: ------------------------------------------------
#: 三条理由，都不是"顺手拆"：
#:
#:     粒度不同   任务的**识别键**是 scope（`memory.l3.extract:{scope}`），
#:                而 L2 是会话。一个 scope 下 N 个会话的 L2 完成最终指向
#:                同一条 L3 —— 并进 L2 就没有"合并同一 scope 的多次完成"
#:                这件事了
#:     产物不同   写的是 `memory_persona`，与 L2 的 `memory_scene` 是两张
#:                表、两套乐观锁
#:     多数时候空转 每次 L2 完成都排一条，但**多数会被执行侧跳过**（没有
#:                变化场景、也没举手、也没满轮次）—— 跳过很便宜：一条
#:                SELECT 走人，不调模型。判定为什么在这里而不在触发侧，
#:                见 `MemoryPersonaExtractParams` 的 docstring
#:
#: ⚠️ model_role 叫 `memory.persona.extract`（动作），不叫这个 handler 名
#: —— 与 L1/L2 同一条口径，见 app/agent/model/types.py 的说明。
MEMORY_PERSONA_EXTRACT_HANDLER = "memory.l3.extract"


class MemoryExtractParams(BaseModel):
    """
    给一次运行建立 L0 索引。

    ⚠️ 为什么 session_id 也带着（run 上就能查到）
    ---------------------------------------
    索引行要冗余 `session_id` / `user_id` / `org_id` 三列（见
    models/sqlalchemy/_scope.py），而 `user_id` / `org_id` 在**会话**上、
    不在运行上。带 session_id 让执行阶段少一次"先查运行、再查会话"的
    往返 —— 而那次往返的中间结果（run.session_id）在提交那一刻本来就有。

    代价是两处可能不一致（参数里的 session_id 与 run.session_id 不是
    同一个值）。今天不可能发生：两个值都由同一个调用点填。真担心的话
    执行阶段以**运行**为准校验一次，但那就把刚省掉的往返又加回来了。

    ⚠️ 没有 idempotency_key
    ---------------------
    幂等键在**提交侧**给（`memory.l0.extract:{run_id}`），不在参数里 ——
    它是 Task 的字段，不是业务参数。为什么必须有它见提交点的说明。
    """

    model_config = ConfigDict(extra="forbid")

    run_id: UUID = Field(
        description="要索引的那次运行。执行阶段按它取消息、并据此回填 row.run_id。",
    )

    session_id: UUID = Field(
        description=(
            "该运行所属会话。用来取 user_id / org_id 两列（它们在会话上），"
            "并且是索引行的过滤键之一。"
        ),
    )


class MemoryAtomExtractParams(BaseModel):
    """
    给一个会话做一次 L1（memory_atom）抽取。

    ⚠️ 只有 session_id —— **没有 run_id**，与 L0 那条不同
    ------------------------------------------------
    L0 是"给**这一次运行**建索引"（按 run 取消息），所以它带 run_id；
    L1 是"把这个会话里**还没抽过的那一段**抽掉"，而水位
    （`memory_atom_cursor.last_seq`）是**会话级**的。执行阶段要的是会话，
    不是运行。

    ⚠️ 为什么阈值判定不带进来
    ----------------------
    "这次该不该抽"是**触发侧**的判断，不是执行侧的条件。任务一旦被排
    出来，执行侧只做一件事：按水位读增量、抽、落库、推水位。把阈值再
    带进来会让执行侧有条件"觉得不该抽就不抽"，于是同一条任务的行为取决
    于它被排出来之后环境变成了什么 —— 那是最查不出原因的一类失效。

    水位读出来是空的（那一批已经被别人抽走了）时它**正常结束**，与 L0
    那条"没有消息就 return"是同一条纪律。

    ⚠️ 没有 idempotency_key / 没有消息
    ------------------------------
    与 L0 同一条纪律：任务里只放业务 ID。幂等键在提交侧给，消息在执行
    阶段按水位重新读。
    """

    model_config = ConfigDict(extra="forbid")

    session_id: UUID = Field(
        description=(
            "要抽取的会话。水位（memory_atom_cursor）、增量消息、以及写入行的"
            " scope 列都按它取。"
        ),
    )


class MemorySceneExtractParams(BaseModel):
    """
    把一个会话里**还没整合过的** L1 记忆融进场景文件。

    ⚠️ 与 L1 那条同形（只有 session_id），但粒度是**两件事拼起来的**
    ------------------------------------------------------------
    水位按会话（`memory_scene_cursor.session_id`），产物却是按 scope 的
    （一个 scope 十几行画像，被该 scope 下所有会话共享）。两者不是一回事：
    任务可以被同一个 scope 下的两个会话各排一条，而两条都会写同几行画像
    —— 那是刻意的（画像就是跨会话聚合的产物），并发由行上的 `version`
    乐观锁兜住，冲突抛 `ConflictError` 交给任务重入。

    所以**不要**把这个参数改成 (scope, ...)：那会让"哪些 L1 已经整合过"
    无从记起（水位是会话级的），而漏掉的增量再也不会回来。

    ⚠️ 为什么阈值/调度参数不带进来
    --------------------------
    与 L1 同一条纪律：`lastL2 + 900s` 那种判断是**触发侧**的事，不是执行
    侧的条件。任务一旦排出来，执行侧只做一件事：按水位读增量、整合、落库、
    推水位。带进来会让同一条任务的行为取决于它被排出来之后环境变成了什么。

    水位读出来是空的（那批已经被别人整合走了）时它**正常结束**（并且把
    `last_l2_at` 戳一下，见 handler 的说明），与 L1/L0 同一条纪律。

    ⚠️ 没有 idempotency_key / 没有消息
    ------------------------------
    与 L1 同一条：任务里只放业务 ID。幂等键在提交侧给
    （`memory.l2.extract:{session_id}`），消息与增量都在执行阶段重新读。
    """

    model_config = ConfigDict(extra="forbid")

    session_id: UUID = Field(
        description=(
            "要整合的会话。水位（memory_scene_cursor）、L1 增量、以及写入行的"
            " scope（user_id / org_id 两列）都按它取。"
        ),
    )


class MemoryPersonaExtractParams(BaseModel):
    """
    把一个 scope 下**变化过的场景**抽象成一篇画像。

    ⚠️ 与 L1/L2 那两条**形状不同**：这里是 scope，不是 session
    ------------------------------------------------
    L1/L2 的参数都是 `session_id`（水位与增量都是会话级的）；L3 的产物与
    水位**都是 scope 级**，而 `memory_scene` 行上刻意没有 session_id ——
    所以这条链上根本凑不出会话这个维度（见 models/sqlalchemy/
    memory_persona_cursor.py 的模块 docstring）。

    ⚠️ 判定在**执行侧**，而这与 L1/L2 是**相反**的
    -----------------------------------------
    任务一旦排出来，执行侧自己决定要不要真的跑：

        **已经有画像**，且 L2 没有举手
            -> 跳过生成（只戳一下 `last_l3_at`，见 service impl 第 3 步）

    ⚠️ 触发条件是「**首次创建**」+「L2 举手」—— 2026-10-13 改的
    ------------------------------------------------------
    原来是「自水位以来有变化场景」也算一条触发路径。那条被去掉了：L2
    每次整合只要改写了场景，画像就跟着重写一遍，而画像本该是**慢变量**
    ——「随便改了一条场景」不等于「这个人变了」。

    现在只剩两条：

        该 scope 有场景、但**还没有画像**     首次创建（`persona` 为 None）
        L2 那次的模型举了手                    `persona_update_request`

    **变化场景不再是触发条件**，它只是**输入**（有待抽象的场景才谈得上
    生成）。所以"有画像 + 场景变了 + 没举手"= 跳过。

    ⚠️ 为什么 L3 可以这样，而 L1/L2 不行
    --------------------------------
    L1/L2 的判据是"该不该**现在**做"（攒够轮次 / 空闲 60s），那个答案只有
    触发那一刻知道，带进任务就变成"执行侧自己重判"，于是同一条任务的行为
    取决于它被排出来之后环境变成了什么。

    而 L3 的判据是"**画像还在不在**" —— 那是一个执行时刻查一次主键就知道
    的事实。在触发侧再算一遍是重复的，而且那个结论到执行时可能已经过期
    （同 scope 下另一个任务先建了行）。所以这里宁可让执行侧判，只在触发侧
    留下**触发那一刻的那个事实**：

        persona_update_request   L2 的模型举没举手（一个布尔）

    ⚠️ 为什么没有 `session_id`
    ----------------------
    它在这条链上没有消费者：水位按 scope，场景按 scope，画像按 scope。
    带上它只会多一个"参数里的 session 与别处不一致"的可能性，而这个不一致
    没有任何地方会读到。触发侧的轮次计数在**提交那一刻**就用掉了（那不是
    执行侧的事）。

    ⚠️ 没有 idempotency_key
    ---------------------
    与 L1/L2 一样：幂等键在提交侧给（`memory.l3.extract:{scope}`），它是
    Task 的字段，不是业务参数。见 trigger.py 的 `persona_task_key`。
    """

    model_config = ConfigDict(extra="forbid")

    agent_id: str = Field(
        min_length=1,
        description=(
            "画像归属的 agent —— 业务键的一半。取值口径与 memory_scene.agent_id "
            "逐字一致（L1 记忆上的 duty_key，回退会话的 agent_id，再回退 "
            "DEFAULT_AGENT_ID）。"
        ),
    )

    user_id: str | None = Field(
        default=None,
        description="个人画像的归属用户。与 `org_id` 恰有一个非空。",
    )

    org_id: str | None = Field(
        default=None,
        description="团队画像的归属组织。与 `user_id` 恰有一个非空。",
    )

    persona_update_request: bool = Field(
        default=False,
        description=(
            "L2 那次整合的模型**主动举手** —— 一个布尔。\n\n"
            "True = 模型认为发生了重大价值观转变 / 跨场景突破性洞察，"
            "画像该重写了。执行侧据此**不跳过**生成。\n\n"
            "⚠️ 2026-10-13 从「理由原文」压成布尔\n"
            "--------------------------------\n"
            "消费方只问「要不要」，从不读那段文字；而「填一段描述」这种"
            "要求模型给得很不稳定（实测一次都没给过）。代价是 L3 的提示词"
            "里少了一格「L2 指出的是哪件事」。\n\n"
            "⚠️ 它只免掉「已有画像就跳过」那一条 —— 首次生成与 `no_change`"
            "不受它影响。"
        ),
    )

    @model_validator(mode="after")
    def _scope_ok(self) -> "MemoryPersonaExtractParams":
        """
        两个归属**恰有一个** —— 与 `SceneScope` 同一条不变量。

        ⚠️ 这里再挡一次不是重复：任务参数是**从库里读回来**的
        （`tasks.params` 是一个 JSONB 列），也就是说它可能来自一个更早的
        版本、或一次人工订正。等执行侧构造 `SceneScope` 时才炸，症状是
        "这条任务反复 FAILED"而原因在参数里 —— 在**入参校验**这一层挡掉，
        错误信息才贴着出问题的那个值。
        """

        if (self.user_id is None) == (self.org_id is None):
            raise ValueError(
                "L3 的 scope 必须恰好有 user_id 或 org_id 之一："
                "有 org 的是团队画像，没有的是个人画像。"
            )

        return self

    @property
    def is_team(self) -> bool:
        """团队画像（org + agent）还是个人画像（user + agent）。"""

        return self.org_id is not None
