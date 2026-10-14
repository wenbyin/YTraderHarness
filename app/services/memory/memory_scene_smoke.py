"""
L2（memory_scene）的冒烟测试：增量水位 -> 模型 -> 场景行 + 乐观锁 + 越权。

    .venv/Scripts/python.exe -m app.services.memory.memory_scene_smoke

直接用真数据库跑（本地 Postgres），但**模型是假的** —— 本脚本要验的是
这条链的读写与取舍（水位、scope、乐观锁、文件名归一化、[DELETED]），
不是某一次 Prompt 调出来的结果好不好。

⚠️ 一个注入点：`memory_scene_service_impl` 的模块级 getter
--------------------------------------------------
    get_model_registry    换成一台**按剧本应答**的假 runtime（每次调用
                          弹出一个预先写好的 JSON），于是"这一轮该改哪些
                          场景"完全由脚本决定

提示词那一侧**不注入** —— 走真 provider，顺带把"这份提示词确实注册上了"
一起验了（见 `_install` 的说明）。

⚠️ 用过即清
--------
脚本自己造会话与 L1 记忆，跑完按 session_id 删记忆与水位、按 scope 删画像
行。**失败路径也清**（finally），否则库里会留下垃圾行，而它们会在下次跑
别的冒烟脚本时以"多出来几条"的形式出现。

⚠️ 它绕过了 HTTP 层
---------------
与仓库其他冒烟脚本一样，这里直接调服务 —— 因此**不能**用它证明"接口在
HTTP 上通"。L2 当前没有 HTTP 入口（见 memory_scene_service.py）。
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import sys
import uuid
from collections import deque
from types import SimpleNamespace

from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import app.services.memory.application.memory_scene_service_impl as impl
from app.agent.model import ModelRole
from app.common.exceptions import ConflictError
from app.services.memory.domain.entities.scene import (
    SceneCursorState,
    SceneScope,
    scene_id,
)
from app.services.memory.domain.schemas.scene import (
    DELETED_MARKER,
    ConsolidationParseError,
)
from app.services.memory.domain.tasks import MemorySceneExtractParams
from app.services.memory.models.sqlalchemy.memory_atom import MemoryAtom as MemoryAtomModel
from app.services.memory.models.sqlalchemy.memory_scene_cursor import MemorySceneCursor
from app.services.memory.models.sqlalchemy.memory_scene import MemoryScene
from app.services.memory.repository.atom_impl import MemoryAtomRepositoryImpl
from app.services.memory.repository.scene_impl import MemorySceneRepositoryImpl
from app.services.session.models.sqlalchemy.agent_session import (
    AgentSession as AgentSessionModel,
)
from app.services.session.repository.agent_session_impl import (
    AgentSessionRepositoryImpl,
)
from app.services.task.models.sqlalchemy.task import Task as TaskModelRow

DSN = "postgresql+asyncpg://postgres:123456@localhost:5432/ytrader_agent"

#: 冒烟用的身份。取正交的值，方便一眼认出库里的垃圾行是谁留下的。
AGENT_ID = "smoke-l2-agent"
USER_A = "smoke-l2-user-a"
USER_B = "smoke-l2-user-b"
ORG_T = "smoke-l2-org"

_FAILURES: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    """断言并记账。失败**不中断**：一次跑完能看见所有问题。"""

    mark = "OK  " if condition else "FAIL"
    line = f"  [{mark}] {label}"

    if detail and not condition:
        line += f"  <- {detail}"

    print(line, flush=True)

    if not condition:
        _FAILURES.append(label)


# ============================================================
# 假模型
# ============================================================


class _ScriptedRuntime:
    """
    按剧本应答的 runtime：每次 `ainvoke` 弹出一个预设的 JSON 字符串。

    顺带记下每次收到的请求（`requests`），于是"喂进去的长什么样"可以被
    断言 —— 那是本脚本唯一能验提示词输入的地方（模型是假的）。
    """

    def __init__(self, replies: list[str]) -> None:
        self.replies = deque(replies)
        self.requests: list = []

    async def ainvoke(self, request):
        self.requests.append(request)

        if not self.replies:
            raise AssertionError(
                "假 runtime 的剧本用完了 —— 这次整合比预期多调了一次模型。"
            )

        return SimpleNamespace(
            message=SimpleNamespace(content=self.replies.popleft())
        )


class _FakeModelRegistry:
    def __init__(self, runtime: _ScriptedRuntime) -> None:
        self._runtime = runtime
        self.calls: list[dict] = []

    def get_runtime(self, *, agent_type: str, model_role: str, context=None):
        self.calls.append({"agent_type": agent_type, "model_role": model_role})
        return self._runtime


def _install(runtime: _ScriptedRuntime) -> _FakeModelRegistry:
    """
    换掉模型注册表，返回假 registry（调用方要断言它被怎么调）。

    ⚠️ **只换模型，不换提示词**
    --------------------
    提示词走真的 `get_prompt_provider()` —— 于是本脚本顺带验了
    "`memory.scene.consolidate` 确实被目录装配收进来了"（少了那份提示词
    就抛 PromptNotFoundError，而不是悄悄地发一份空的系统提示词给模型）。

    它曾经是假的：那时 `prompts/` 下有两个 0 字节的占位文件让整个目录
    装配抛 PromptRegistryError。那两个文件已删，所以这一层假的没必要留。
    """

    registry = _FakeModelRegistry(runtime)

    impl.get_model_registry = lambda: registry

    return registry


# ============================================================
# 造数据 / 清理
# ============================================================


async def _seed_session(
        session,
        *,
        user_id: str | None,
        org_id: str | None,
) -> uuid.UUID:
    """造一个会话。**不造运行与消息** —— L2 只读 memory_atom 与水位。"""

    now = dt.datetime.now(dt.timezone.utc)
    session_id = uuid.uuid4()

    session.add(AgentSessionModel(
        session_id=session_id,
        agent_id=AGENT_ID,
        status="active",
        meta={},
        thread_id=f"smoke-l2-{session_id}",
        user_id=user_id,
        org_id=org_id,
        created_at=now,
        updated_at=now,
    ))

    await session.flush()

    return session_id


#: 造数据用的**单调时钟**：每个 atom 比前一个晚一秒。
#:
#: ⚠️ 不能用 `datetime.now()` 现取
#: ---------------------------
#: `memory_atom.updated_time` 就是 L2 的增量水位，而一次 L1 抽取写的若干条
#: 记忆共享一个 `now()`（同一事务）。现取时间的话，"第二轮造的记忆"有可能
#: 落在**第一轮的水位之前**（两次 `now()` 只差几毫秒，而第一条 atom 到
#: 最后一条之间差了好几秒）—— 于是 L2 合法地读不到它，而脚本会以为是
#: 产品的 bug。用单调时钟把"后来的记忆一定更晚"这件事钉死。
_SEED_CLOCK = dt.datetime.now(dt.timezone.utc)


async def _seed_atoms(
        session,
        *,
        session_id: uuid.UUID,
        contents: list[str],
        user_id: str | None,
        org_id: str | None,
        duty: str | None = AGENT_ID,
        same_instant: bool = False,
) -> list[uuid.UUID]:
    """
    造 L1 记忆。**显式给 updated_time** —— 它就是 L2 的水位，不定死的话
    "增量取到哪一条"没法断言。

    `same_instant=True` 时这一批**共享同一个时间戳**（模拟 L1 一次事务写
    若干条的真实形态），用来验水位边界那批 id 的排除逻辑。
    """

    global _SEED_CLOCK

    record_ids: list[uuid.UUID] = []

    batch_stamp: dt.datetime | None = None

    for content in contents:

        record_id = uuid.uuid4()
        record_ids.append(record_id)

        if same_instant:
            if batch_stamp is None:
                _SEED_CLOCK += dt.timedelta(seconds=1)
                batch_stamp = _SEED_CLOCK
            stamp = batch_stamp
        else:
            _SEED_CLOCK += dt.timedelta(seconds=1)
            stamp = _SEED_CLOCK

        session.add(MemoryAtomModel(
            record_id=record_id,
            content=content,
            type="persona",
            priority=80,
            scene_name="冒烟情境",
            session_key=f"smoke-l2-{session_id}",
            version=0,
            timestamp_str=stamp.isoformat(),
            timestamp_start="",
            timestamp_end="",
            created_time=stamp,
            updated_time=stamp,
            source_message_ids=[],
            meta={},
            session_id=session_id,
            business_agent_id=duty,
            user_id=user_id,
            org_id=org_id,
        ))

    await session.flush()

    return record_ids


async def _cleanup(session, *, session_ids: list[uuid.UUID]) -> None:
    """按会话清干净（画像行按 scope 清，它们与 session_id 没有直接关系）。"""

    await session.execute(
        delete(MemoryAtomModel).where(
            MemoryAtomModel.session_id.in_(session_ids)
        )
    )
    await session.execute(
        delete(MemorySceneCursor).where(
            MemorySceneCursor.session_id.in_(session_ids)
        )
    )
    await session.execute(
        delete(MemoryScene).where(MemoryScene.agent_id == AGENT_ID)
    )
    await session.execute(
        delete(AgentSessionModel).where(
            AgentSessionModel.session_id.in_(session_ids)
        )
    )
    await session.flush()


# ============================================================
# 读回来的小工具
# ============================================================


async def _scenes(session, scope: SceneScope) -> dict[str, MemoryScene]:
    """该 scope 的场景行，按 filename 索引。"""

    rows = await session.scalars(
        select(MemoryScene).where(
            MemoryScene.agent_id == scope.agent_id,
            MemoryScene.user_id == scope.user_id,
            MemoryScene.org_id == scope.org_id,
        )
    )

    return {row.filename: row for row in rows}


async def _cursor(session, session_id: uuid.UUID) -> SceneCursorState:
    """
    读水位 —— 返回**值对象**，不是 ORM 行。

    ⚠️ 这里踩过一次坑，别改回 `session.scalar(select(MemorySceneCursor)...)`
    ------------------------------------------------------------
    ORM 实例在 identity map 里是**同一个对象**：`before = await _cursor(...)`
    与 `after = await _cursor(...)` 拿到的是同一个 Python 对象，而仓储那侧
    用了 `populate_existing`（见 scene_impl 的 `_fresh`，那里是必须的），
    于是第二次读会**原地刷新**那个对象 —— 结果 `after > before` 恒为假，
    两个值打印出来一模一样，看起来像"水位没动"。

    那是一个**断言写错**的错误，而不是产品的问题：库里其实变了（换成裸
    SQL 读就看得见）。值对象（pydantic）在每次读时新建，不会被后续的读
    改写，所以断言比较的是两个时刻的快照。

    ⚠️ "没有行"返回的是默认值（水位 None），不是 None
    ------------------------------------------
    与契约一致（见 MemorySceneRepository.get_l2_cursor）。所以判"这个
    会话还没整合过"要看字段是不是 None，别判对象是不是 None。
    """

    return await MemorySceneRepositoryImpl(
        session=session
    ).get_l2_cursor(session_id)


def _write(path: str, content: str) -> dict:
    return {"path": path, "content": content}


def _reply(*operations: dict, persona: str | None = None) -> str:
    return json.dumps(
        {"operations": list(operations), "persona_update_request": persona},
        ensure_ascii=False,
    )


def _service(session) -> impl.MemorySceneServiceImpl:
    return impl.MemorySceneServiceImpl(
        profiles=MemorySceneRepositoryImpl(session=session),
        atoms=MemoryAtomRepositoryImpl(session=session),
        sessions=AgentSessionRepositoryImpl(session=session),
    )


async def _run(session, session_id: uuid.UUID) -> None:
    await _service(session).memory_scene_extract(
        MemorySceneExtractParams(session_id=session_id), None,
    )


# ============================================================
# 场景 A：个人画像全流程
# ============================================================


async def smoke_personal(session) -> uuid.UUID:

    print("\n--- A 个人画像：新建 / 改写 / 未变 / 删除 / 水位 ---")

    first = "用户偏好 Rust 的异步运行时。"
    second = "用户住在合肥。"

    session_id = await _seed_session(session, user_id=USER_A, org_id=None)
    await _seed_atoms(
        session, session_id=session_id, contents=[first, second],
        user_id=USER_A, org_id=None,
    )

    scope = SceneScope.resolve(
        agent_id=AGENT_ID, user_id=USER_A, org_id=None,
    )

    # ---- 第一次：两个新文件 --------------------------------------------------
    runtime = _ScriptedRuntime([
        _reply(
            _write("技术研究-Rust学习.md", "# Rust\n\n## 用户核心特征\n偏好异步。"),
            _write("日常生活.md", "# 生活\n\n## 用户核心特征\n住在合肥。"),
        ),
    ])
    registry = _install(runtime)

    await _run(session, session_id)

    scenes = await _scenes(session, scope)

    check("A1 新建两个场景", set(scenes) == {"技术研究-Rust学习.md", "日常生活.md"},
          f"实际 {sorted(scenes)}")
    check("A2 version 从 0 起", all(row.version == 0 for row in scenes.values()),
          f"实际 {[row.version for row in scenes.values()]}")
    check(
        "A3 行归属是个人画像（user 有值 / org 为空）",
        all(row.user_id == USER_A and row.org_id is None for row in scenes.values()),
    )
    check(
        "A4 主键是派生 id",
        scenes["日常生活.md"].id
        == scene_id(scope, filename="日常生活.md"),
    )
    check(
        "A5 content_md5 与内容对得上",
        scenes["日常生活.md"].content_md5
        == hashlib.md5(scenes["日常生活.md"].content.encode()).hexdigest(),
    )
    check(
        "A6 模型绑的是 L2 那个 agent_type",
        registry.calls[0]["agent_type"] == "memory_scene_consolidate",
    )
    check(
        "A7 提示词要的是 L2 的 role",
        runtime.requests[0].metadata["model_role"] == ModelRole.MEMORY_SCENE_CONSOLIDATE,
    )

    # 喂进去的内容：新增记忆与"当前时间"都得在 HumanMessage 里
    human = runtime.requests[0].messages[-1].content
    check("A8 新增记忆进了提示词", first in human and second in human)
    check("A9 清单标了总数与上限", "共 0 个，上限 12 个" in human, human[:120])
    check("A10 当前时间进了提示词", "### 3️⃣ 当前时间" in human, human[:200])

    cursor = await _cursor(session, session_id)
    check(
        "A11 水位推到这一批的最后一条",
        cursor is not None
        and cursor.last_atom_updated_time is not None
        and cursor.last_l2_at is not None,
    )
    check(
        "A12 边界 id 只含最后那个时间戳上的记忆",
        cursor is not None and len(cursor.last_atom_ids) == 1,
        f"实际 {cursor.last_atom_ids if cursor else None}",
    )

    # ---- 第二次：改一个、不动另一个 -------------------------------------------
    await _seed_atoms(
        session, session_id=session_id, contents=["用户改用 tokio 了。"],
        user_id=USER_A, org_id=None,
    )

    runtime = _ScriptedRuntime([
        _reply(_write("技术研究-Rust学习.md", "# Rust\n\n改用 tokio。")),
    ])
    _install(runtime)

    await _run(session, session_id)

    scenes = await _scenes(session, scope)
    check("A13 改写只动被写的那个", scenes["技术研究-Rust学习.md"].version == 1,
          f"version={scenes['技术研究-Rust学习.md'].version}")
    check("A14 没被写的场景 version 不动", scenes["日常生活.md"].version == 0)
    check("A15 内容确实换了", "tokio" in scenes["技术研究-Rust学习.md"].content)
    check(
        "A16 这一轮只喂了**新增**的那一条（水位生效）",
        "用户改用 tokio 了。" in runtime.requests[0].messages[-1].content
        and first not in runtime.requests[0].messages[-1].content,
    )
    check(
        "A17 清单里带上了现有场景的全文",
        "# Rust" in runtime.requests[0].messages[-1].content,
    )

    # ---- 第三次：原样吐回来 -> 不写、不推 version ------------------------------
    await _seed_atoms(
        session, session_id=session_id, contents=["用户还是喜欢 Rust。"],
        user_id=USER_A, org_id=None,
    )

    same = scenes["技术研究-Rust学习.md"].content
    runtime = _ScriptedRuntime([_reply(_write("技术研究-Rust学习.md", same))])
    _install(runtime)

    await _run(session, session_id)

    scenes = await _scenes(session, scope)
    check(
        "A18 内容逐字相同 -> version 不动",
        scenes["技术研究-Rust学习.md"].version == 1,
        f"version={scenes['技术研究-Rust学习.md'].version}",
    )

    # ---- 第四次：[DELETED] 真删 + 归一化 + 写赢删除 ---------------------------
    await _seed_atoms(
        session, session_id=session_id, contents=["用户开始写周报了。"],
        user_id=USER_A, org_id=None,
    )

    runtime = _ScriptedRuntime([
        _reply(
            # 带空格与括号的名字 -> 归一化
            _write("Coffee (Yirgacheffe).md", "# 咖啡"),
            # 删除标记
            _write("日常生活.md", DELETED_MARKER),
            # 既写又删同一个 -> **写赢**
            _write("周报.md", "# 周报\n\n每周五写。"),
            _write("周报.md", DELETED_MARKER),
        ),
    ])
    _install(runtime)

    await _run(session, session_id)

    scenes = await _scenes(session, scope)
    check("A19 [DELETED] 真删掉了", "日常生活.md" not in scenes, f"实际 {sorted(scenes)}")
    check(
        "A20 文件名被归一化",
        "Coffee-Yirgacheffe.md" in scenes and "Coffee (Yirgacheffe).md" not in scenes,
        f"实际 {sorted(scenes)}",
    )
    check(
        "A21 既写又删时写赢",
        "周报.md" in scenes and scenes["周报.md"].content != DELETED_MARKER,
        f"实际 {sorted(scenes)}",
    )
    check(
        "A22 写赢之后不是 [DELETED] 的内容",
        scenes["周报.md"].content.startswith("# 周报"),
    )

    # ---- 第五次：没有新记忆 -> 不调模型 ---------------------------------------
    before = await _cursor(session, session_id)
    runtime = _ScriptedRuntime([])
    _install(runtime)

    await _run(session, session_id)

    check("A23 没有增量就不调模型", runtime.requests == [])
    after = await _cursor(session, session_id)
    check(
        "A24 空转仍推 last_l2_at",
        after is not None
        and before.last_l2_at is not None
        and after.last_l2_at > before.last_l2_at,
        f"before={before.last_l2_at} after={after.last_l2_at if after else None}",
    )
    check(
        "A25 空转不动增量水位",
        after.last_atom_updated_time == before.last_atom_updated_time
        and after.last_atom_ids == before.last_atom_ids,
    )

    return session_id


# ============================================================
# 场景 B：团队画像
# ============================================================


async def smoke_team(session) -> uuid.UUID:

    print("\n--- B 团队画像：org 有值就按 org 聚合 ---")

    session_id = await _seed_session(session, user_id=USER_B, org_id=ORG_T)
    await _seed_atoms(
        session, session_id=session_id, contents=["团队约定周五发布。"],
        user_id=USER_B, org_id=ORG_T,
    )

    runtime = _ScriptedRuntime([_reply(_write("发布节奏.md", "# 发布\n\n周五。"))])
    _install(runtime)

    await _run(session, session_id)

    team = SceneScope.resolve(agent_id=AGENT_ID, user_id=USER_B, org_id=ORG_T)
    personal = SceneScope.resolve(agent_id=AGENT_ID, user_id=USER_B, org_id=None)

    check("B1 落到团队 scope", len(await _scenes(session, team)) == 1)
    check("B2 没有落到个人 scope", await _scenes(session, personal) == {})

    row = (await _scenes(session, team))["发布节奏.md"]
    check(
        "B3 团队行的 org 有值、user 为空",
        row.org_id == ORG_T and row.user_id is None,
        f"org={row.org_id!r} user={row.user_id!r}",
    )

    # 同一个 user 但**没有 org** 的会话，落的是另一个 scope（B2 已经验了空）
    return session_id


# ============================================================
# 场景 C：乐观锁冲突
# ============================================================


async def smoke_conflict(session) -> uuid.UUID:

    print("\n--- C 乐观锁：整合期间被别的任务改过 ---")

    session_id = await _seed_session(session, user_id=USER_A, org_id=ORG_T)
    await _seed_atoms(
        session, session_id=session_id, contents=["用户喜欢手冲咖啡。"],
        user_id=USER_A, org_id=ORG_T,
    )

    scope = SceneScope.resolve(agent_id=AGENT_ID, user_id=USER_A, org_id=ORG_T)

    # 先正常建一行
    _install(_ScriptedRuntime([_reply(_write("咖啡.md", "# 咖啡 v1"))]))
    await _run(session, session_id)

    row = (await _scenes(session, scope))["咖啡.md"]
    cursor_after_first = await _cursor(session, session_id)

    # ⚠️ 模拟"另一个 L2 任务"在这一轮**读完之后、写回之前**改了它
    # ------------------------------------------------------
    # 时机就卡在模型调用里：服务读完场景 -> 调模型 -> 写回。假的 runtime
    # 在 `ainvoke` 里把 version 顶掉一步，等价于另一个会话的整合先提交了。
    class _BumpingRuntime(_ScriptedRuntime):
        async def ainvoke(self, request):
            await session.execute(
                update(MemoryScene)
                .where(MemoryScene.id == row.id)
                .values(version=MemoryScene.version + 1)
            )
            return await super().ainvoke(request)

    await _seed_atoms(
        session, session_id=session_id, contents=["用户改喝浅烘了。"],
        user_id=USER_A, org_id=ORG_T,
    )

    _install(_BumpingRuntime([_reply(_write("咖啡.md", "# 咖啡 v2"))]))

    raised = None

    try:
        await _run(session, session_id)
    except ConflictError as exc:
        raised = exc

    check("C1 冲突抛 ConflictError", raised is not None, "没抛")
    check(
        "C2 冲突时**不覆盖**别人的内容",
        (await _scenes(session, scope))["咖啡.md"].content == "# 咖啡 v1",
    )

    # ⚠️ 冲突之后水位必须**一点都没动** —— 那批记忆下一轮还得再消化一次。
    # 只断言"两张表都没变"是不够的（它们本来就不可能变），这里比的是第一轮
    # 之后那个快照：水位、边界 id 集、last_l2_at 三样全都不许前移。
    cursor_after_conflict = await _cursor(session, session_id)

    check(
        "C3 冲突时增量水位不推进（那批记忆下一轮重来）",
        cursor_after_conflict.last_atom_updated_time
        == cursor_after_first.last_atom_updated_time
        and cursor_after_conflict.last_atom_ids == cursor_after_first.last_atom_ids,
        f"before={cursor_after_first.last_atom_updated_time} "
        f"after={cursor_after_conflict.last_atom_updated_time}",
    )
    check(
        "C4 冲突时 last_l2_at 也不动",
        cursor_after_conflict.last_l2_at == cursor_after_first.last_l2_at,
        f"before={cursor_after_first.last_l2_at} "
        f"after={cursor_after_conflict.last_l2_at}",
    )

    return session_id


# ============================================================
# 场景 D：模型输出坏了 -> 抛，且水位不动
# ============================================================


async def smoke_bad_output(session) -> uuid.UUID:

    print("\n--- D 模型输出坏了：抛（可重试）且水位不动 ---")

    session_id = await _seed_session(session, user_id=USER_A, org_id="smoke-l2-bad")
    await _seed_atoms(
        session, session_id=session_id, contents=["用户养了只猫。"],
        user_id=USER_A, org_id="smoke-l2-bad",
    )

    cursor_before = await _cursor(session, session_id)

    _install(_ScriptedRuntime(["这不是 JSON"]))

    raised = None

    try:
        await _run(session, session_id)
    except ConsolidationParseError as exc:
        raised = exc

    check("D1 坏输出抛 ConsolidationParseError", raised is not None, "没抛")

    cursor_after = await _cursor(session, session_id)
    check(
        "D2 失败时水位**完全没动**（下一轮重来）",
        # ⚠️ 判字段而不是判对象：没有行时读回来的是**默认值对象**（水位
        # None），不是 None —— 见 `_cursor` 的说明。
        cursor_after.last_atom_updated_time is None
        and cursor_before.last_atom_updated_time is None
        and cursor_after.last_l2_at is None,
        f"before={cursor_before.last_atom_updated_time} "
        f"after={cursor_after.last_atom_updated_time} "
        f"last_l2_at={cursor_after.last_l2_at}",
    )

    # 缺 operations 也是坏输出（不是"没有变更"）
    _install(_ScriptedRuntime(['{"persona_update_request": "x"}']))

    raised = None

    try:
        await _run(session, session_id)
    except ConsolidationParseError as exc:
        raised = exc

    check("D3 缺 operations 也抛", raised is not None, "没抛")

    return session_id


# ============================================================
# 场景 E：分批不丢
# ============================================================


async def smoke_batch(session) -> uuid.UUID:

    print("\n--- E 分批：单批上限截断时不丢数据 ---")

    session_id = await _seed_session(session, user_id=USER_A, org_id="smoke-l2-batch")
    await _seed_atoms(
        session, session_id=session_id, contents=["记忆一", "记忆二", "记忆三"],
        user_id=USER_A, org_id="smoke-l2-batch",
    )

    # 把单批上限压到 1，逼出"还有更多、下一轮接着来"那条路
    original = impl.DEFAULT_CONSOLIDATE_BATCH
    impl.DEFAULT_CONSOLIDATE_BATCH = 1

    try:
        runtime = _ScriptedRuntime([_reply(_write("一批.md", "# 一"))])
        _install(runtime)
        await _run(session, session_id)

        human = runtime.requests[0].messages[-1].content
        check("E1 第一批只喂了一条", "记忆一" in human and "记忆二" not in human,
              human[-200:])

        cursor = await _cursor(session, session_id)
        check(
            "E2 水位只推到真的消费掉的那一条（不跳到最新）",
            cursor is not None and len(cursor.last_atom_ids) == 1,
            f"实际 {cursor.last_atom_ids if cursor else None}",
        )

        # 下一轮把剩下的带走 —— 一条都不该丢
        runtime = _ScriptedRuntime([_reply(_write("二批.md", "# 二"))])
        _install(runtime)
        await _run(session, session_id)

        human = runtime.requests[0].messages[-1].content
        check("E3 下一轮拿到的是接着的那条", "记忆二" in human, human[-200:])

    finally:
        impl.DEFAULT_CONSOLIDATE_BATCH = original

    return session_id


# ============================================================
# 场景 F：没有归属的会话
# ============================================================


async def smoke_no_owner(session) -> uuid.UUID:

    print("\n--- F 无主会话：不抛、不写画像、水位不动 ---")

    session_id = await _seed_session(session, user_id=None, org_id=None)
    await _seed_atoms(
        session, session_id=session_id, contents=["没有归属的记忆。"],
        user_id=None, org_id=None,
    )

    runtime = _ScriptedRuntime([])
    _install(runtime)

    raised = None

    try:
        await _run(session, session_id)
    except Exception as exc:  # noqa: BLE001 —— 冒烟要的是"什么都不抛"
        raised = exc

    check("F1 不抛（重试改变不了它）", raised is None, repr(raised))
    check("F2 没有调模型", runtime.requests == [])

    cursor = await _cursor(session, session_id)
    check(
        "F3 水位不动（那批记忆留着，别静默跳过）",
        cursor.last_atom_updated_time is None,
        f"实际 {cursor.last_atom_updated_time}",
    )
    check(
        "F4 但 last_l2_at 戳上了（别每 90 秒重来）",
        cursor.last_l2_at is not None,
    )

    return session_id


# ============================================================
# 场景 I：提示词的拼装（预警 / 清单 / 三段顺序）
# ============================================================


async def smoke_prompt_assembly(session) -> uuid.UUID:
    """
    验 `_build_request` 拼出来的那一份：**预警段只在超限时出现**、清单里
    有真实文件名、三段顺序固定。

    ⚠️ 为什么要单独验这个
    -----------------
    提示词的正文（系统提示词）是常量，改了看得见；而**用户输入那一半是
    拼出来的**，少一段、顺序变了都不会报错 —— 症状只是"模型忽然不合并了"。
    把上限压到 1 就能用两个场景走完"无预警 -> 红色预警"这条边。
    """

    print("\n--- I 提示词拼装：预警 / 清单 / 顺序 ---")

    session_id = await _seed_session(session, user_id=USER_A, org_id="smoke-l2-warn")
    await _seed_atoms(
        session, session_id=session_id, contents=["用户在用 Rust。"],
        user_id=USER_A, org_id="smoke-l2-warn",
    )

    original = impl.DEFAULT_MAX_SCENES
    impl.DEFAULT_MAX_SCENES = 1

    try:
        # ---- 第一次：0 个场景 -> 不该有预警 ----------------------------------
        runtime = _ScriptedRuntime([_reply(_write("技术.md", "# 技术"))])
        _install(runtime)
        await _run(session, session_id)

        human = runtime.requests[0].messages[-1].content

        check("I1 没有场景时不出现数量预警", "场景数量警告" not in human,
              human[:120])
        check(
            "I2 三段都在，且顺序是 新增记忆 -> 现有场景 -> 当前时间",
            human.index("### 1️⃣ 新增记忆")
            < human.index("### 2️⃣ 现有场景文件")
            < human.index("### 3️⃣ 当前时间"),
            human[:160],
        )
        check("I3 空清单也写明”第一次整合“", "第一次整合" in human)

        # ---- 第二次：1 个场景 = 上限 -> 红色预警 -----------------------------
        await _seed_atoms(
            session, session_id=session_id, contents=["用户改用 tokio 了。"],
            user_id=USER_A, org_id="smoke-l2-warn",
        )

        runtime = _ScriptedRuntime([_reply(_write("技术.md", "# 技术 v2"))])
        _install(runtime)
        await _run(session, session_id)

        human = runtime.requests[0].messages[-1].content

        check("I4 达到上限时出现预警段", "场景数量警告" in human, human[:160])
        check("I5 红色预警要求先 MERGE", "必须先 MERGE" in human, human[:400])
        check("I6 红色预警禁止新建", "禁止新建" in human)
        check(
            "I7 清单列出了真实文件名（模型只能沿用这些名字）",
            "- `技术.md`" in human,
            human[:400],
        )
        check(
            "I8 清单出现在全文块之前",
            human.index("- `技术.md`") < human.index("-----BEGIN CONTENT-----"),
            human[:400],
        )
        check(
            "I9 头上标了总数与上限",
            "共 1 个，上限 1 个" in human,
            human[:200],
        )

    finally:
        impl.DEFAULT_MAX_SCENES = original

    return session_id


# ============================================================
# 场景 H：触发 A（真的把任务排出来）
# ============================================================


async def smoke_trigger(maker) -> None:
    """
    验 `trigger.submit_scene_extract`：任务真的排出来了，时间对、键对，
    而且**只允许提前、不允许推迟**。

    ⚠️ 清理只针对自己造的那条会话
    -----------------------
    见 finally 里那段：**别按 handler 删** —— 那会波及用户库里的历史任务，
    甚至撞上"正在跑的那条"，伪装成 cancel / 租约过期的错误。

    ⚠️ 它必须跑在**独立的事务**里
    -------------------------
    `submit_scene_extract` 自己开作用域（`get_service_factory()` +
    `scope(TaskService)`），所以它读得到的东西必须**已经提交**。上面那批
    场景都挤在一个未提交的事务里（那样才验得了"同一个 session 连着调几次
    handler"），所以这一场单独放在外层会话之后。

    ⚠️ 装配是真的，只有 dispatcher 故意不给
    ---------------------------------
    `build_factories` + `_register_handlers` 用的就是应用启动时那一套
    （顺带把"注册表里确实有 memory.l2.extract"这件事也验了）。不给
    `task_dispatcher` 是有意的 —— 它是可选的（投递由 outbox relay 兜底，
    见 task_service_impl 的说明），而"延迟任务排进库"这件事不需要有人
    真的去跑它。本场景只读 `tasks` 表。
    """

    print("\n--- H 触发 A：排任务 / 只允许提前 ---")

    from app.bootstrap.task import _register_handlers
    from app.common.factories import build_factories, reset_factories
    from app.infra.task import TaskHandlerRegistryImpl
    from app.services.memory.application.trigger import (
        SCENE_AFTER_L1_DELAY_SECONDS,
        scene_task_key,
        submit_scene_extract,
    )
    registry = TaskHandlerRegistryImpl()
    _register_handlers(registry)

    check(
        "H1 注册表里有 memory.l2.extract",
        "memory.l2.extract" in registry.known(),
        f"实际 {registry.known()}",
    )

    build_factories(maker, pool={"handler_registry": registry})

    session_id = None

    try:
        async with maker() as session:
            session_id = await _seed_session(session, user_id=USER_A, org_id=None)
            await session.commit()

        key = scene_task_key(session_id)
        now = dt.datetime.now(dt.timezone.utc)

        first = await submit_scene_extract(session_id=session_id)
        check("H2 排出了一条任务", first is not None, "返回了 None")

        task = await _task(maker, key)
        check("H3 handler 是 memory.l2.extract", task is not None and task.handler == "memory.l2.extract",
              f"实际 {task.handler if task else None}")
        check("H4 幂等键按会话", task is not None and task.idempotency_key == key)
        check(
            "H5 键声明可反复用（终态会释放，否则第二次触发会拿回已终结的那条）",
            task is not None and task.reusable_key is True,
        )

        offset = (task.execute_at - now).total_seconds() if task and task.execute_at else None
        check(
            "H6 首次整合排在 L1 之后那 90s",
            offset is not None and 80 <= offset <= 120,
            f"实际 {offset}",
        )

        # ---- 只允许提前：把它挪到很晚，再触发一次应当**拉回来** ----------
        async with maker() as session:
            await session.execute(
                update(TaskModelRow).where(TaskModelRow.idempotency_key == key)
                .values(execute_at=now + dt.timedelta(hours=1))
            )
            await session.commit()

        await submit_scene_extract(session_id=session_id)

        pulled = await _task(maker, key)
        pulled_offset = (pulled.execute_at - now).total_seconds()
        check(
            "H7 算出的时间更早就**提前**它",
            pulled_offset is not None and pulled_offset < 3600,
            f"实际 {pulled_offset}",
        )

        # ---- 不允许推迟：连着再触发一次，算出的时间只晚一点点 ------------
        before_second = pulled.execute_at
        await submit_scene_extract(session_id=session_id)
        after_second = (await _task(maker, key)).execute_at
        check(
            "H8 算出的时间更晚时**不动它**（不允许推迟）",
            after_second == before_second,
            f"before={before_second} after={after_second}",
        )

        # ---- lastL2 + 900s 那条底线 --------------------------------------
        async with maker() as session:
            await MemorySceneRepositoryImpl(
                session=session
            ).advance_l2_cursor(
                session_id,
                last_atom_updated_time=None,
                last_atom_ids=[],
                last_l2_at=now,
                user_id=USER_A,
            )
            await session.execute(
                update(TaskModelRow).where(TaskModelRow.idempotency_key == key)
                .values(execute_at=now + dt.timedelta(seconds=10))
            )
            await session.commit()

        # 现在 T = max(now+90, now+900) = now+900，比排着的那条（now+10）
        # **晚** —— 不许推它。
        await submit_scene_extract(session_id=session_id)
        guarded = (await _task(maker, key)).execute_at
        check(
            "H9 刚整合过时，900s 底线不会把已有的更早定时推迟",
            guarded == now + dt.timedelta(seconds=10),
            f"实际 {(guarded - now).total_seconds()}s",
        )

        check(
            "H10 默认延迟常量就是文档里那两个数",
            SCENE_AFTER_L1_DELAY_SECONDS == 90.0,
            f"实际 {SCENE_AFTER_L1_DELAY_SECONDS}",
        )

    finally:
        reset_factories()

        async with maker() as session:
            # ⚠️ 只删**这一场自己造的那条**任务
            # ---------------------------------
            # 曾经写的是 `where(handler == "memory.l2.extract")` —— 那是把用户
            # 库里**全部** L2 任务都删掉（2026-10-11 实际发生过），而且后果比
            # "少几行历史"严重得多：如果被删的那条**正在跑**，worker 跑完之后
            # `mark_success`（守卫是 `status = RUNNING`）会更新到 0 行，于是报
            # "执行成功但状态写回无效（多半是期间被取消 / 租约被收回）" ——
            # 一条把人引向 cancel/lease 的假线索。
            #
            # 按 `params_json->>'session_id'` 删而不是按幂等键：任务**进终态时
            # 会释放幂等键**（置 NULL），按键删会漏掉已完成的那条、留下垃圾。
            if session_id is not None:
                await session.execute(
                    delete(TaskModelRow).where(
                        TaskModelRow.params_json["session_id"].astext
                        == str(session_id)
                    )
                )
            if session_id is not None:
                await _cleanup(session, session_ids=[session_id])
            await session.commit()


async def _task(maker, key: str) -> TaskModelRow | None:
    """按幂等键取那条任务（新开一个会话，绕开 identity map）。"""

    async with maker() as session:
        return await session.scalar(
            select(TaskModelRow)
            .where(TaskModelRow.idempotency_key == key)
        )


# ============================================================
# 纯函数
# ============================================================


def smoke_pure() -> None:

    print("\n--- G 纯函数：文件名归一化 / 计划时间 ---")

    from app.services.memory.application.trigger import (
        SCENE_AFTER_L1_DELAY_SECONDS,
        SCENE_MIN_INTERVAL_SECONDS,
        _scene_execute_at,
    )

    n = impl.normalize_filename

    check("G1 合规名字不动", n("技术研究-Rust学习.md") == "技术研究-Rust学习.md")
    check("G2 空格换短横线",
          n("Daily Rhythm in Shanghai.md") == "Daily-Rhythm-in-Shanghai.md")
    check("G3 括号被删", n("Coffee (Yirgacheffe).md") == "Coffee-Yirgacheffe.md")
    check("G4 目录被截掉", n("a/b/c/深夜食堂.md") == "深夜食堂.md")
    check("G5 没有后缀就补上", n("no-ext") == "no-ext.md")
    check("G6 全非法 -> 空串（调用方跳过）", n("!!!") == "")
    check("G7 全角空格也算空白", n("中文　全角.md") == "中文-全角.md")

    now = dt.datetime.now(dt.timezone.utc)

    first = _scene_execute_at(
        last_l2_at=None,
        after_l1_delay_seconds=SCENE_AFTER_L1_DELAY_SECONDS,
        min_interval_seconds=SCENE_MIN_INTERVAL_SECONDS,
    )
    check("G8 首次整合只等 L1 之后那 90s",
          round((first - now).total_seconds()) == int(SCENE_AFTER_L1_DELAY_SECONDS),
          f"实际 {round((first - now).total_seconds())}")

    recent = _scene_execute_at(
        last_l2_at=now,
        after_l1_delay_seconds=SCENE_AFTER_L1_DELAY_SECONDS,
        min_interval_seconds=SCENE_MIN_INTERVAL_SECONDS,
    )
    check("G9 刚跑完就受 900s 底线约束",
          round((recent - now).total_seconds()) == int(SCENE_MIN_INTERVAL_SECONDS),
          f"实际 {round((recent - now).total_seconds())}")

    old = _scene_execute_at(
        last_l2_at=now - dt.timedelta(hours=1),
        after_l1_delay_seconds=SCENE_AFTER_L1_DELAY_SECONDS,
        min_interval_seconds=SCENE_MIN_INTERVAL_SECONDS,
    )
    check("G10 早就跑过就还是 90s",
          round((old - now).total_seconds()) == int(SCENE_AFTER_L1_DELAY_SECONDS))

    # ---- 场景数量的分级（工程侧算好的那一份，模型不再自己数）----------
    #
    # ⚠️ 从**提示词模块**导入：分级的话术与阈值必须与 TEXT 里那三级一致，
    # 所以它与模板住在一起（见那个模块的说明）。
    from app.agent.prompt.prompts.memory_scene_consolidate import (
        scene_count_warning,
    )

    MAX = 12

    check(
        "G11 远未到上限时不预警",
        scene_count_warning(0, max_scenes=MAX) == ""
        and scene_count_warning(9, max_scenes=MAX) == "",
    )
    check(
        "G11b 一个场景都没有时不预警（上限再小也不）",
        scene_count_warning(0, max_scenes=1) == ""
        and scene_count_warning(1, max_scenes=1) != "",
        f"0 个 -> {scene_count_warning(0, max_scenes=1)!r}",
    )
    check(
        "G12 接近上限 -> 黄色（优先合并）",
        "优先" in scene_count_warning(10, max_scenes=MAX),
    )
    check(
        "G13 只剩一个名额 -> 橙色（只能重写）",
        "只能重写" in scene_count_warning(11, max_scenes=MAX),
    )
    check(
        "G14 达到上限 -> 红色（必须先 MERGE + 禁止新建）",
        "必须先 MERGE" in scene_count_warning(12, max_scenes=MAX)
        and "禁止新建" in scene_count_warning(15, max_scenes=MAX),
    )

    # ---- META 块的解析（也住在提示词模块里，与模板同级）----------------
    from app.agent.prompt.prompts.memory_scene_consolidate import parse_meta

    meta = parse_meta(
        "-----META-START-----\n"
        "created: 2026-10-01T10:00:00Z\n"
        'updated: "2026-10-11T09:12:00Z"\n'
        "summary: [30-40 words concise summary for indexing]\n"
        "heat: 7\n"
        "-----META-END-----\n\n## 正文\nheat: 99（这里的不算）\n"
    )
    check(
        "G15 META 块解析：字段、去引号、剥模板方括号",
        meta.has_meta_block
        and meta.updated == "2026-10-11T09:12:00Z"
        and meta.summary == "30-40 words concise summary for indexing"
        and meta.heat == 7,
        f"实际 {meta}",
    )
    check(
        "G16 没有 META 块时不抛、标记为缺块",
        parse_meta("裸正文").has_meta_block is False,
    )
    check(
        "G17 heat 解析不出整数时当没给（不是 0）",
        parse_meta(
            "-----META-START-----\nheat: 很多\n-----META-END-----"
        ).heat is None,
    )


# ============================================================
# main
# ============================================================


async def main() -> int:

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    engine = create_async_engine(DSN)
    maker = async_sessionmaker(engine, expire_on_commit=False)

    smoke_pure()

    session_ids: list[uuid.UUID] = []

    async with maker() as session:

        try:
            session_ids.append(await smoke_personal(session))
            session_ids.append(await smoke_team(session))
            session_ids.append(await smoke_conflict(session))
            session_ids.append(await smoke_bad_output(session))
            session_ids.append(await smoke_batch(session))
            session_ids.append(await smoke_no_owner(session))
            session_ids.append(await smoke_prompt_assembly(session))

            await session.commit()

        finally:
            # ⚠️ 清理要**独立于上面是否成功**：失败时库里同样留下了垃圾行，
            # 而它们会在下次跑别的冒烟脚本时以"多出来几条"的形式出现。
            try:
                if session_ids:
                    await _cleanup(session, session_ids=session_ids)
                    await session.commit()
            except Exception as exc:  # noqa: BLE001 —— 清理失败不能再盖掉原错误
                print(f"  [WARN] 清理失败：{exc}", flush=True)
                await session.rollback()

    # ⚠️ 触发 A 单独放在这里：`submit_scene_extract` 自己开作用域，读得到的东西
    # 必须是**已提交**的，而上面那批场景挤在同一个未提交的事务里（那正是它
    # 要验的东西）。见 `smoke_trigger` 的说明。
    await smoke_trigger(maker)

    await engine.dispose()

    print("\n" + "=" * 60)

    if _FAILURES:
        print(f"失败 {len(_FAILURES)} 项：")
        for label in _FAILURES:
            print(f"  - {label}")
        return 1

    print("全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
