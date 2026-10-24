"""
MemorySceneServiceImpl —— L2 场景整合（一次模型调用）。

一次整合的形状
------------
    读水位 -> 按水位取 L1 增量 -> 解析 scope（写哪一行）
      -> 读该 scope 的现有场景（清单 + 全文）
      -> [模型：整篇重写 + MERGE + [DELETED]] -> 归一化文件名
      -> 先写后删（写入带乐观锁）-> 推水位
      -> 问一次"这个 scope 该不该重生成 L3 画像"

⚠️ 模型**一轮定输赢**，没有工具、没有多轮
--------------------------------
原项目让模型用 read / write / edit 文件工具去操作 `scene_blocks/`，跑完
还有一次 Phase 5 清理。这里没有工具回路（后台任务里没有那一套基建），
所以三件事各由一处负责，别把它们混起来：

    读        调用前**全部内联**（该 scope 的场景全文进 HumanMessage）
    写        模型在一次回复里给出写入清单，工程侧翻成行（见 `_apply`）
    删        `content` 写成 `[DELETED]` 即删除标记，工程侧**最后**真删
              —— 模型永远拿不到删除能力（原项目的设计意图，照搬）

代价说清楚：模型**看不到自己的执行结果**，也没有第二轮去补救。所以
`_apply` 里每一处"看不懂就不做"的取舍都倾向**保留数据**（宁可多留一个
场景文件、宁可跳过一条看不懂的操作），而不是猜它的意图。

⚠️ 顺序：先写后删
------------
与数据层级一致（先立新、再清旧），也与原项目 Phase 5 的位置一致
（"remove soft-deletes" 在结尾）。两者按构造不相交（`_apply` 里 write
赢了就把同名的 delete 摘掉），所以顺序不影响结果。

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
一个是 bug。吞掉等于让后者永远查不出来（见 domain/schemas/scene.py）。

⚠️ 与 L1 的关系
-----------
本服务**不**写 memory_atom：增量的读取走 `MemoryAtomRepository.
list_session_delta`（表是它的，口径也只有一份），画像的写入走
`MemorySceneRepository`。两条链因此可以各自改而不牵扯对方的列。
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage

from app.agent.context.processor.messages import (
    message_text as llm_message_text,
)
from app.agent.model import ModelRequest, ModelRole, get_model_registry
from app.agent.prompt import get_prompt_provider
from app.agent.prompt.prompts.memory_scene_consolidate import (
    parse_meta,
    render_user_message,
)
from app.common.exceptions import ConflictError, NotFoundError
from app.common.service.base import ServiceContext
from app.services.session.domain.entities import AgentSession
from app.services.session.repository.agent_session import (
    AgentSessionRepository,
)

from ..domain.entities.atom import MemoryAtom
from ..domain.entities.scene import (
    SceneCursorState,
    SceneScope,
    SceneRow,
    scene_id,
)
from ..domain.schemas.scene import (
    ConsolidationOperation,
    ConsolidationResult,
    is_deletion,
    parse_consolidation,
)
from ..domain.tasks import MemorySceneExtractParams
from ..repository.atom import MemoryAtomRepository
from ..repository.scene import MemorySceneRepository
from .memory_scene_service import MemorySceneService
from .profile_cache import profile_cache_invalidation_hook
from .trigger import submit_persona_extract, task_id_of

__all__ = [
    "AGENT_TYPE",
    "DEFAULT_CONSOLIDATE_BATCH",
    "DEFAULT_MAX_SCENES",
    "MemorySceneServiceImpl",
    "normalize_filename",
]

logger = logging.getLogger(__name__)


#: 模型绑定用的 agent_type（见 model.yaml 的 bindings）。
#:
#: ⚠️ 与 model_role 不是一个东西：model_role 是**动作**
#: （`memory.scene.consolidate`，见 ModelRole），agent_type 是**产物/服务**。
#: L1 那两条用了 `memory_atom_*`（表名）的形状，这里跟着它走。
AGENT_TYPE = "memory_scene_consolidate"


#: 一个 scope 最多几个场景文件。**待校准的起点**，不是实测出来的值。
#:
#: 提示词按它做三级预警（红：必须先 MERGE 才能处理新记忆 / 橙：只能重写
#: 不能新建 / 黄：优先重写或合并）。调它两个方向各有一个可观察的症状：
#:
#:     调太小    过早合并 —— 不同主题的场景被压成一个，叙事里出现
#:               "技术与生活"这种什么都沾一点的文档，信息密度反而降
#:     调太大    清单变长 —— 每次整合都要把十几个场景全文喂进去，
#:               费用随上限线性涨，而且模型要在更长的清单里挑 UPDATE 目标
DEFAULT_MAX_SCENES = 12


#: 一次整合最多消费几条 L1 记忆。
#:
#: ⚠️ 超了**不是丢数据**，是分几轮追上
#: -------------------------------
#: 增量按 `updated_time` **升序**取（见 MemoryAtomRepository.
#: list_session_delta），水位只推到"真的消费完的那一条"，所以剩下的下一轮
#: 接着取。反过来（取最新的一批）就会把中间那段永久跳过 —— 那是静默丢数据，
#: 所以顺序在这个数存在的前提下才成立。
#:
#: 200 的理由：一段长会话攒下的记忆大多是几十条（L1 有"宁缺毋滥"的抽取
#: 原则），200 覆盖绝大多数情况；真超了说明这个会话很久没整合（或者
#: L1 的阈值该调了），分两轮比一次喂进去几千条更省也更稳。
DEFAULT_CONSOLIDATE_BATCH = 200


#: 文件名里允许出现的字符之外的一切。
#:
#: 提示词里那份规范（见 prompts/memory_scene_consolidate.py）是**给模型看的**，
#: 用自然语言写的；这一份是**给工程用的**，用字符类写的。两者必须一致 ——
#: 不一致的症状是"提示词说合规、工程侧还是改了名"，而日志里只会看到
#: 一堆重命名噪音。
#:
#: 允许：数字 / 拉丁字母 / CJK（含扩展 A、兼容表意、假名、谚文）/ `-` `_` `.`
_ALLOWED_FILENAME = re.compile(
    "[^0-9A-Za-z"
    "㐀-䶿"   # CJK 扩展 A
    "一-鿿"   # CJK 基本区（中日韩统一表意）
    "豈-﫿"   # CJK 兼容表意
    "぀-ヿ"   # 平假名 / 片假名
    "가-힯"   # 谚文
    "._-]"
)

#: 已有的 `.md` 后缀（大小写都认）。
_MD_SUFFIX = re.compile(r"(?i)\.md$")


def normalize_filename(raw: str) -> str:
    """
    模型给的文件名 -> 能落库的那个名字（原项目 `classifyPath` 的对应物）。

    步骤（顺序有意义）：

        1  去掉目录部分      `../../etc/passwd` -> `passwd`
        2  空白（含全角）-> `-`      提示词要求用短横线连接多词
        3  删掉白名单外的字符        括号、引号、`?`、`!`、`:` …
        4  摘掉已有的 `.md` 后缀    为了下一步的 strip 不会吃掉点号
        5  去掉两端的 `-` `.` `_`    否则会造出 `-foo.md` 这种名字
        6  补回小写的 `.md`

    归一化后为空（整个名字都是非法字符）时返回**空串**，由调用方跳过该
    操作并记一条 warning。

    ⚠️ 与原项目的差别：**不**生成兜底名字
    -------------------------------
    原项目归一化失败时会按序号造一个名字，因为它跑完之后还能去读目录、
    把新名字带回给模型。我们是一轮定输赢（模型看不到执行结果），自己
    造的名字模型永远不知道 —— 下一轮清单里会出现一个它从没写过的文件，
    而它可能又"重新创建"一遍同一个场景。跳过 + 记日志是可解释的那一种。

    ⚠️ 第 1 步是**安全**措施，不只是整洁
    -------------------------------
    `path` 来自模型输出，而模型可以吐一个带 `..` 的路径。这里虽然只操作
    数据库（没有文件系统可以穿越），但让一个不可信的路径原样成为一行行的
    `filename`，等于把"这条删除标记删的是谁"交给模型决定 —— 而删除是按
    filename 匹配的。截成最后一段之后，`_apply` 里那层 scope 校验才真正
    兜得住。
    """

    name = (raw or "").strip().replace("\\", "/").rsplit("/", 1)[-1]

    name = re.sub(r"\s+", "-", name)
    name = _ALLOWED_FILENAME.sub("", name)
    name = _MD_SUFFIX.sub("", name)
    name = name.strip("-._")

    if not name:
        return ""

    return f"{name}.md"


class MemorySceneServiceImpl(MemorySceneService):

    def __init__(
            self,
            *,
            profiles: MemorySceneRepository,
            atoms: MemoryAtomRepository,
            sessions: AgentSessionRepository,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._profiles = profiles
        self._atoms = atoms
        self._sessions = sessions

    # ========================================================
    # Handler 入口
    # ========================================================

    async def memory_scene_extract(
            self,
            params: MemorySceneExtractParams,
            ctx: Any,
    ) -> None:

        # ---- 1. 会话与水位 ----------------------------------------
        #
        # 会话先取：user_id / org_id 两列要从它来（它们决定**写哪一行**），
        # 而 agent_id 那一半要从增量记忆上取。取不到会话就**抛** ——
        # 画像不带归属的话，它对谁都可见（NULL 的存量语义），那是一条越权
        # 路径。与 L1 的抽取同一取舍。
        session = await self._sessions.get_any(params.session_id)

        if session is None:
            raise NotFoundError(
                f"会话 {params.session_id} 不存在，无法确定画像的归属。"
            )

        cursor = await self._profiles.get_l2_cursor(params.session_id)

        atoms = await self._atoms.list_session_delta(
            params.session_id,
            updated_after=cursor.last_atom_updated_time,
            consumed_ids=_uuid_list(cursor.last_atom_ids),
            limit=DEFAULT_CONSOLIDATE_BATCH,
        )

        now = datetime.now(timezone.utc)

        # ---- 2. 没有增量：只戳一下"跑过了" --------------------------
        #
        # ⚠️ 空转**是常态**，不是失败：这一批可能已经被同一个 scope 下
        # 另一个会话的任务整合走了，也可能这一轮 L1 什么都没抽到。
        #
        # 而"戳一下 last_l2_at"是必须的：触发 A 算的是
        # `T = max(now + 90s, lastL2 + 900s)`（见 trigger.submit_scene_extract），
        # 不戳的话这个会话每 90 秒就会白跑一次 —— 一次模型都不调，但任务
        # 与查询照付。
        if not atoms:
            await self._advance(
                params.session_id,
                cursor=cursor,
                session=session,
                last_l2_at=now,
            )
            logger.debug(
                "[Memory] 会话 %s 自 %s 起没有新的 L1 记忆，跳过 L2 整合。",
                params.session_id, cursor.last_atom_updated_time,
            )
            return

        # 这一批被截断了（还有更多）—— 说清楚，别让它看起来像"全整合完了"。
        # 见 DEFAULT_CONSOLIDATE_BATCH 的说明：剩下的下一轮接着来。
        if len(atoms) >= DEFAULT_CONSOLIDATE_BATCH:
            print(
                f"[Memory] L2 增量达到单批上限 {DEFAULT_CONSOLIDATE_BATCH} 条"
                f"（session={params.session_id}），本轮只整合最早的这一批，"
                "余下的下一轮继续。",
                flush=True,
            )

        # ---- 3. 解析 scope（写哪一行）------------------------------
        #
        # ⚠️ 拿不到归属会话就**正常结束**，不是抛
        # ------------------------------------
        # 一个既没有 user_id 也没有 org_id 的会话（存量数据里 user_id 是
        # NULL 表示"对谁都可见"）算不出业务键：画像没有归属就没有意义。
        # 抛出去会让任务重试三次、每次得到同一个结论 —— 而重试不会让它
        # 长出归属来。所以打印一行、戳一下 last_l2_at、正常结束。
        #
        # ⚠️ 水位**不动**：那批记忆留着，等归属补上（或者人工处理）之后
        # 还能整合，而不是被静默跳过。
        if not (session.user_id or (session.org_id or "").strip()):
            await self._advance(
                params.session_id,
                cursor=cursor,
                session=session,
                last_l2_at=now,
            )
            print(
                f"[Memory] L2 跳过：会话 {params.session_id} 既没有 user_id "
                f"也没有 org_id，{len(atoms)} 条 L1 记忆没有归属（水位未推进）。",
                flush=True,
            )
            return

        scope = SceneScope.resolve(
            # ⚠️ agent_id 取**记忆上的** duty_key（最近的那个），不是会话的
            # agent_id —— 一条记忆是"为哪份职责而记的"，画像是按那份职责
            # 聚合的。规则与 L1 的 `_duty_key` 逐字一致（那边取消息上的，
            # 这边取记忆上的），改一处要一起看。回退顺序见 SceneScope.resolve。
            agent_id=_duty_key(atoms) or session.agent_id,
            user_id=session.user_id,
            org_id=session.org_id,
        )

        # ---- 4. 现有场景（清单 + 全文）-----------------------------
        scenes = await self._profiles.list_scenes(scope)

        # ---- 5. 一次模型调用 --------------------------------------
        result = await self._consolidate(
            atoms=atoms, scenes=scenes, now=now,
        )

        # ---- 6. 落库（先写后删）-----------------------------------
        report = await self._apply(
            scope, scenes=scenes, operations=result.operations,
        )

        # ---- 7. 推水位 --------------------------------------------
        #
        # ⚠️ 推到最后一条**被消费的**记忆（不是这一批里 updated_time 最大的
        # 那个 —— 升序取出来它们本来就是同一个）。边界时间戳上已消费的 id
        # 一起记下：同一事务写入的 atom 共享一个 now()，只按 `>` 取会让
        # 恰好落在同一微秒的那批被静默跳过。
        #
        # ⚠️ 必须在上面全部做完之后、且与它们同一个事务里（调用方的作用域）。
        # 提前推 = 那批记忆永久进不了画像；分事务 = 画像没写而水位动了，
        # 症状同样是永久跳过。
        await self._advance(
            params.session_id,
            cursor=cursor,
            session=session,
            last_l2_at=now,
            last_atom_updated_time=atoms[-1].updated_time,
            last_atom_ids=[
                atom.record_id
                for atom in atoms
                if atom.updated_time == atoms[-1].updated_time
            ],
        )

        logger.info(
            "[Memory] 会话 %s 的 L2 完成：%d 条 L1 -> 新 %d / 改 %d / "
            "未变 %d / 删 %d / 跳过 %d（scope=%s，共 %d 个场景）。",
            params.session_id,
            len(atoms),
            report.created,
            report.updated,
            report.unchanged,
            report.deleted,
            report.skipped,
            scope.key,
            report.total_scenes,
        )

        # ⚠️ 同一件事也打到控制台
        # ---------------------
        # 本应用**没有配置 logging**（main.py 里没有 basicConfig），所以
        # 上面那句 `logger.info` 控制台上一个字都不会出现。而"这一轮到底
        # 改了什么"是最常见的问题：
        #
        #     全是 0  ->  跑了，模型判"现有场景已经够了"（不是没跑）
        #     有计数   ->  真的改了；`跳过` 非 0 则是它给了没法用的东西
        print(
            f"[Memory] L2 完成：session={params.session_id} "
            f"{len(atoms)} 条 L1 -> 新 {report.created} / 改 {report.updated}"
            f" / 未变 {report.unchanged} / 删 {report.deleted}"
            f" / 跳过 {report.skipped}，scope={scope.key}，"
            f"场景 {report.total_scenes} 个",
            flush=True,
        )

        # L2 的写入全部完成（仍在同一事务里）。**提交成功之后**失效
        # L2 缓存（只失效 L2 —— L3 是独立的键，由它自己的任务负责）——
        # 提前删会让读者在提交前把旧值写回去。
        self.post_commit.append(
            profile_cache_invalidation_hook(scope, "l2")
        )

        # ---- 8. 问一次"这个 scope 该不该重生成画像" ------------------
        #
        # ⚠️ 位置在这里（L2 handler 的末尾）是**刻意的**，不是"顺手"：
        # 只有这一刻同时成立三件事 ——
        #
        #     这个 scope 是谁       上面第 3 步刚解析出来的
        #     这一轮真的整合完了     上面的写入与水位都做完了
        #     模型给没给信号         `persona_update_request` 只存在于这次回复
        #
        # 挂在运行收尾那条路不行：那时 L2 还没跑（它排在 90s 后），"整合完了
        # 没有"无从判断；挂在 L1 末尾更不行 —— L2 的产物那时还不存在。
        #
        # ⚠️ 调度在 `submit_persona_extract` 里，**判定不在这里也不在那里**
        # ----------------------------------------------------------
        # 它只负责排一条任务（没有活着的就新建、有就复用），两个触发事实
        # 随参数走。该不该真生成由 `memory_persona_extract` 判 —— 见
        # `MemoryPersonaExtractParams` 的 docstring。
        # 失败不抛（自己吞掉并留 warning）：画像生成是旁路，它失败不该让
        # 一次成功的 L2 整合变成失败的任务。而它会自愈 —— 下一次 L2 完成
        # 会重新排一遍（水位没动，场景还在）。
        #
        # ⚠️ 已知代价：这一刻上面的写入**还没提交**（提交发生在 handler
        # 返回之后，由调用方的作用域做）。若随后回滚，L3 已经排在库里了，
        # 届时它读到空增量、空转一次。与 L1 末尾排 L2 是同一条取舍。
        #
        # ⚠️ `persona_update_request` 是模型主动给的信号（提示词里的
        # "重大价值观转变、跨场景突破性洞察"），一个 **true / false**。
        # 它在执行侧的身份是一条**免跳过信号** —— 已经生成过画像的 scope
        # 也会因为举手而重写一次。
        if result.persona_update_request:
            print(
                f"[Memory] L2 请求更新 L3 画像（scope={scope.key}）",
                flush=True,
            )

        await submit_persona_extract(
            scope=scope,
            persona_update_request=result.persona_update_request,
            causation_id=task_id_of(ctx),
        )

    # ========================================================
    # 模型调用
    # ========================================================

    async def _consolidate(
            self,
            *,
            atoms: list[MemoryAtom],
            scenes: list[SceneRow],
            now: datetime,
    ) -> ConsolidationResult:

        runtime = get_model_registry().get_runtime(
            agent_type=AGENT_TYPE,
            model_role=ModelRole.MEMORY_SCENE_CONSOLIDATE,
        )

        response = await runtime.ainvoke(
            self._build_request(atoms=atoms, scenes=scenes, now=now)
        )

        return parse_consolidation(llm_message_text(response.message))

    def _build_request(
            self,
            *,
            atoms: list[MemoryAtom],
            scenes: list[SceneRow],
            now: datetime,
    ) -> ModelRequest:
        """
        提示词 + 这次的场景清单与新增记忆 -> 一次模型请求。

        单独成方法是为了**可测**（冒烟里断言"喂进去的长什么样"不必真的调
        模型）—— 与 L1 的 `_build_extract_request` 同一条理由。
        """

        system_prompt = get_prompt_provider().node_prompt(
            ModelRole.MEMORY_SCENE_CONSOLIDATE,
            max_scenes=DEFAULT_MAX_SCENES,
        )

        # ⚠️ 摘要缺失会明显拉低清单的可用性
        # -----------------------------
        # 模型靠 summary 认"这条新记忆该并进哪个文件"，而它是场景之间
        # 唯一可比的短信号（全文太长、heat 不表达主题）。缺了不是错（存量
        # 场景可能是别的路径写进去的），但必须看得见 —— 否则症状是
        # "模型老是新建场景、不合并"，而没人知道原因在 META 块上。
        missing = [
            scene.filename for scene in scenes
            if not parse_meta(scene.content).summary
        ]

        if missing:
            logger.warning(
                "[Memory] %d 个场景没有 summary，模型会更难判断该合并进"
                "哪一个：%s", len(missing), ", ".join(missing[:5]),
            )

        # 用户消息**怎么排版**归提示词模块（含那个"共 N 个"计数与三段
        # 编号、以及预警段的位置）—— 见 prompts/__init__.py 的约定。
        # 这里只负责**取**：哪个 scope 的哪些场景、哪些新记忆、几点。
        human = render_user_message(
            atoms=atoms,
            scenes=scenes,
            max_scenes=DEFAULT_MAX_SCENES,
            now=now,
        )

        return ModelRequest(
            messages=(
                SystemMessage(content=system_prompt),
                HumanMessage(content=human),
            ),
            # 与 L1 那两次一样**不关思考**：这一次要做的判断（并进哪个
            # 场景、要不要先合并、哪些该删）比重写一整篇文档更依赖推理，
            # 关掉的代价更大。
            metadata={
                "purpose": "memory_scene_consolidate",
                "agent_type": AGENT_TYPE,
                "model_role": ModelRole.MEMORY_SCENE_CONSOLIDATE,
                "atom_count": len(atoms),
                "scene_count": len(scenes),
            },
        )

    # ========================================================
    # 落库
    # ========================================================

    async def _apply(
            self,
            scope: SceneScope,
            *,
            scenes: list[SceneRow],
            operations: list[ConsolidationOperation],
    ) -> "_Report":
        """
        写入清单 -> 真的改库。

        ⚠️ 每一处"看不懂就不做"都倾向**保留数据**
        -------------------------------------
        模型看不到执行结果（见模块 docstring），所以这里是最后一道判断，
        而它的失败模式不对称：

            多留一个场景文件     只是下次清单长一点，模型还有机会合并
            误删一个场景文件     LLM 的结论，**不可逆**（重跑得到的是
                                 不同的文档）

        所以下面三条规则都偏保守：归一化不出来就跳过、`content` 为空就跳过、
        同一个文件既写又删时**写赢**（把删除摘掉）。
        """

        known = {scene.filename: scene for scene in scenes}

        report = _Report()

        writes: dict[str, str] = {}
        deletes: set[str] = set()
        renamed: list[str] = []

        for operation in operations:

            filename = normalize_filename(operation.path)

            if not filename:
                report.skipped += 1
                logger.warning(
                    "[Memory] 场景文件名归一化之后为空，跳过这条操作：%r",
                    operation.path,
                )
                continue

            if filename != operation.path.strip():
                renamed.append(f"{operation.path} -> {filename}")

            if is_deletion(operation.content):
                deletes.add(filename)
                continue

            content = operation.content.strip()

            if not content:
                # 空 content 会在写库那一刻撞 `content <> ''` 的 CHECK，
                # 报错离原因很远（提示词里明写了禁止写空字符串，模型仍会）。
                report.skipped += 1
                logger.warning(
                    "[Memory] 场景 %s 的内容为空，跳过（内容为空的正确"
                    "写法是 [DELETED]，那表示删除）。", filename,
                )
                continue

            # dict 赋值 = 同一个文件名写两次时**留最后一条**（模型修正过
            # 的那一版），与 repository/_upsert.dedupe_by 同一条规则。
            writes[filename] = content

        if renamed:
            # 提示词里那份命名规范是给模型看的，这条日志是它没遵守时的
            # 噪音来源（原项目里据说是"每次都会触发"的那条）。
            logger.warning(
                "[Memory] %d 个文件名被归一化：%s",
                len(renamed), "; ".join(renamed[:5]),
            )

        # ⚠️ 既写又删时**写赢**
        # ------------------
        # 那是模型的一次不一致（MERGE 里"写新文件 + 删旧文件"写串了）。
        # 删赢 = 连它刚写出来的内容一起丢掉；写赢 = 那个文件留着新内容，
        # 只是数量没减下去（下次还能合并）。见方法 docstring 里那段取舍。
        conflicted = deletes & set(writes)

        if conflicted:
            logger.warning(
                "[Memory] %d 个文件既被写入又被标记删除，保留写入、摘掉"
                "删除：%s", len(conflicted), ", ".join(sorted(conflicted)),
            )
            deletes -= conflicted

        # ---- 1. 先写（新建 / 原地改写）-----------------------------
        to_insert: list[SceneRow] = []

        for filename, content in writes.items():

            digest = hashlib.md5(content.encode("utf-8")).hexdigest()

            existing = known.get(filename)

            if existing is None:
                to_insert.append(
                    SceneRow(
                        id=scene_id(scope, filename=filename),
                        filename=filename,
                        content=content,
                        content_md5=digest,
                        agent_id=scope.agent_id,
                        user_id=scope.user_id,
                        org_id=scope.org_id,
                    )
                )
                continue

            if existing.content_md5 == digest:
                # 内容逐字相同：不写、**也不推 version**。推的话"版本号"
                # 会变成"跑过几轮"的同义词，乐观锁的基线就没意义了。
                report.unchanged += 1
                continue

            affected = await self._profiles.update_scene(
                existing.id,
                content=content,
                content_md5=digest,
                expected_version=existing.version,
            )

            if affected == 0:
                # 读出来到现在，另一个 L2 任务（同一个 scope 的另一个会话）
                # 改过这一行。抛出去让任务重入 —— 它会读到新内容、重新
                # 调模型。不重试就是拿一份过期内容覆盖掉别人的结论。
                raise ConflictError(
                    f"场景 {filename} 在本次整合期间被别的任务改过"
                    f"（scope={scope.key}），本次放弃，重排后重来。"
                )

            report.updated += 1

        if to_insert:
            inserted = await self._profiles.insert_scenes(to_insert)

            if inserted < len(to_insert):
                # id 是内容派生的，所以冲突意味着"同一个 scope + 同一个
                # 文件名"刚刚被另一个任务建了行。与上面那条同一个处理：
                # 不覆盖别人的结果，抛出去重来。整个事务会回滚，所以
                # 部分插入的几行也不会留下。
                raise ConflictError(
                    f"有 {len(to_insert) - inserted} 个新场景在这次整合期间"
                    f"被别的任务创建（scope={scope.key}），本次放弃，"
                    "重排后重来。"
                )

            report.created += len(to_insert)

        # ---- 2. 后删（[DELETED] 的清理，原项目的 Phase 5）-----------
        if deletes:
            removed = await self._profiles.delete_scenes(
                scope, filenames=sorted(deletes),
            )
            report.deleted = removed

            # 删得比要求少 = 那几个文件本来就不在（模型删了一个它记错的名字）。
            # 不是错误：删除是幂等的意图，而"要求删 3 个、删掉 2 个"在日志里
            # 已经通过计数看得见。
            if removed < len(deletes):
                logger.warning(
                    "[Memory] 要求删 %d 个场景，实际删掉 %d 个（其余不在"
                    "该 scope 下）：%s",
                    len(deletes), removed, ", ".join(sorted(deletes)),
                )

        # ---- 3. 数量上限：只告警，不截断 ---------------------------
        #
        # ⚠️ 超限**不丢东西**
        # --------------
        # 提示词要求模型自己 MERGE 下来（阶段 0 的三级预警），但那是它的
        # 承诺，不是保证。工程侧的选择只有两个：丢掉超出的场景（丢弃模型
        # 的输出，不可逆），或者留着并大声说出来（下次整合时清单顶上会
        # 标着"共 N 个，上限 M"，模型有机会处理）。选后者。
        report.total_scenes = len(known) + report.created - report.deleted

        if report.total_scenes >= DEFAULT_MAX_SCENES:
            logger.warning(
                "[Memory] scope=%s 的场景数达到 %d（上限 %d），下一次整合"
                "的提示词会进入红色预警，模型应当先合并。",
                scope.key, report.total_scenes, DEFAULT_MAX_SCENES,
            )

        return report

    # ========================================================
    # internals
    # ========================================================

    async def _advance(
            self,
            session_id: UUID,
            *,
            cursor: SceneCursorState,
            session: AgentSession,
            last_l2_at: datetime,
            last_atom_updated_time: datetime | None = None,
            last_atom_ids: list[UUID] | None = None,
    ) -> None:
        """
        推水位。**不传 `last_atom_updated_time` 就是"只戳 last_l2_at"**
        （水位保持原值）—— 空转与"没有归属"两条路径走的就是它。
        """

        await self._profiles.advance_l2_cursor(
            session_id,
            last_atom_updated_time=(
                cursor.last_atom_updated_time
                if last_atom_updated_time is None
                else last_atom_updated_time
            ),
            last_atom_ids=(
                cursor.last_atom_ids
                if last_atom_ids is None
                else [str(record_id) for record_id in last_atom_ids]
            ),
            last_l2_at=last_l2_at,
            user_id=session.user_id,
        )


# ============================================================
# 纯函数与中间结构
# ============================================================


class _Report:
    """一次整合的账 —— 落日志用，也方便冒烟断言。"""

    __slots__ = (
        "created", "deleted", "skipped", "total_scenes", "unchanged", "updated",
    )

    def __init__(self) -> None:
        self.created = 0
        self.updated = 0
        self.unchanged = 0
        self.deleted = 0
        self.skipped = 0
        self.total_scenes = 0


def _duty_key(atoms: list[MemoryAtom]) -> str | None:
    """
    这批记忆里**最近的那个** duty_key —— 画像归属的 agent。

    ⚠️ 为什么是"最近的"而不是"出现次数最多的"
    ----------------------------------
    画像属于**当下在履职的那个角色**：一批增量可能跨了好几次运行
    （duty 换过），而这批记忆讲完时在履职的那份职责就是它的归属。
    取众数会让"旧职责的记忆更多"这种偶然情况决定归属 —— 那是查不出来的
    一类错误。

    ⚠️ 与 L1 的 `_duty_key` 是**同一条规则、不同的输入**
    --------------------------------------------
    那边取 `agent_message.business_agent_id`（消息上），这边取
    `memory_atom.business_agent_id`（记忆上，由那次抽取写下的）。
    改一处要一起看 —— 两边分叉的症状是"记忆归属 A、画像归属 B"，
    画像因此在错误的 scope 下聚合。

    ⚠️ 返回 None 是**允许**的状态
    ------------------------
    存量行的 business_agent_id 是 NULL。此时返回 None 让调用方去兜底
    （会话的 agent_id → `DEFAULT_AGENT_ID`），而不是在这里编一个值。
    """

    for atom in reversed(atoms):

        if atom.business_agent_id:
            return atom.business_agent_id

    return None


def _uuid_list(values: list[str]) -> list[UUID]:
    """
    水位里存的 id 字符串 -> UUID，解析不出的丢掉。

    ⚠️ 为什么不能直接 `UUID(v) for v in values`
    ------------------------------
    那一列是 JSONB，内容是**我们自己**写进去的（所以正常情况下一定解析
    得出）。但"正常情况下"不是契约：它可能被人工订正过、也可能是更早
    版本写下的格式。一条解析不了就让整个任务炸，症状是"这个会话的记忆
    再也整合不了"，而原因（水位里有一个坏 id）离得很远。丢掉它最多让
    同一微秒的那批多读一遍（幂等）。
    """

    parsed: list[UUID] = []

    for value in values:

        try:
            parsed.append(UUID(str(value)))

        except (ValueError, AttributeError, TypeError):
            logger.warning(
                "[Memory] 水位里的 id %r 不是合法 UUID，忽略它。", value,
            )

    return parsed
