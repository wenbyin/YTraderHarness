"""
MemorySceneRepository —— L2/L3 画像的持久化契约（memory_scene + 水位）。

与 `MemoryAtomRepository`（L1）分开的理由：**读写形状完全不同**。

    L1   按会话增量抽取 + 原地改写 + 有会话级水位 + 两路召回 + 破坏性删除
    L2   按 scope 读全部 + 按 scope 写（主键 upsert，带乐观锁）+ 会话级水位

合成一个契约只会让两边互相牵制 —— 与 session 模块把 message / session /
run 三个仓储分开是同一条理由。

⚠️ 事务边界：画像行与水位**必须在同一个 scope 里**
------------------------------------------
本仓储（与 L1 那个一样）**从不** commit —— 边界由调用方的作用域决定。

而`advance_l2_cursor` 只在画像**确实写完**之后、且在同一个 scope 里被调用：

    profile 写了、水位没推   下一轮重抽同一段 —— 幂等（按主键 upsert），
                            只是白跑一次模型
    水位推了、profile 没写   那批 L1 **永久跳过**（下次按 `> 水位` 取增量）

第二条没有任何症状，所以它是这条链上唯一不能让步的地方 —— 与 L1 的
"记录 + 挂载表 + 水位同事务"同一纪律。

⚠️ 乐观锁是本契约的一部分
-------------------
`update_scene` 的 `expected_version` 不是可选的优化，是**并发正确性**的
落点：同一个 scope 被多个会话共同书写（画像按 scope、水位按会话），两个
L2 任务可以同时读到同一份场景、各自调模型、各写各的。谁后写谁覆盖 ——
而"覆盖"在这里意味着**丢掉另一次整合的全部结论**（不是丢一行数据，
是丢一次模型调用的结果）。

所以写回带 `WHERE version = :expected`，影响行数为 0 就是"这中间有人改过"，
调用方抛 `ConflictError` 交给任务重入（`RetryPolicy` 默认会重试）。
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities.scene import (
    SceneCursorState,
    SceneScope,
    SceneRow,
)

__all__ = ["MemorySceneRepository"]


class MemorySceneRepository(BaseRepository):

    # ========================================================
    # 水位（会话级）
    # ========================================================

    @abstractmethod
    async def get_l2_cursor(
            self,
            session_id: UUID,
    ) -> SceneCursorState:
        """
        取该会话的整合水位；没有行时返回**默认值**（水位 None、空 id 集、
        `last_l2_at` None）。

        行是惰性创建的，所以"还没整合过"不表现为 None 而表现为那一组默认
        值 —— 调用方不必把"没有行"和"有行但为空"当成两件事来判（它们对
        整合路径是同一件事：没有增量）。

        ⚠️ 但两者对**触发 A** 不是同一件事：`last_l2_at` 有值就意味着
        "至少隔 900 秒"那条底线已经生效过。所以它单独一个字段，见
        `SceneCursorState` 的说明。
        """

        raise NotImplementedError

    @abstractmethod
    async def advance_l2_cursor(
            self,
            session_id: UUID,
            *,
            last_atom_updated_time: datetime | None,
            last_atom_ids: Sequence[str],
            last_l2_at: datetime,
            user_id: str | None = None,
    ) -> None:
        """
        推进水位：记下"整合到哪条 L1 了"与"这次是什么时候跑的"。

        ⚠️ `last_atom_ids` 必须与 `last_atom_updated_time` 一起写
        ---------------------------------------------------
        它们是**一个**水位（时间戳 + 落在该时间戳上已消费的 id 集），分两次
        写会造出"时间是新的、id 集是旧的"这种组合，而下一轮读增量时它会
        按错的边界排除 —— 症状是同一微秒的那批**多读一遍**（模型重复看到
        同一批记忆）或**漏读**（永久丢）。见 memory_scene_cursor.py 的模块
        docstring。

        ⚠️ 空转那一轮也要调它（水位保持不变、只把 `last_l2_at` 戳一下）
        --------------------------------------------------------
        触发 A 算的是 `T = max(now + 90s, lastL2 + 900s)`。一个反复空转的
        会话如果不戳 `last_l2_at`，就会每 90 秒白跑一次模型 —— 而"空转"
        恰恰是常态（那批 L1 已经被别的任务整合走了）。

        ⚠️ 单调性由调用方保证
        -----------------
        这里**不**做 `GREATEST(...)`：往回退正是"重抽一段"的手段（将来要
        支持的话）。挡掉它会让那条路永远走不通，而回退本身不危险（重抽 =
        幂等，模型看到同一批记忆会得出大致相同的场景）。与 L1 的
        `advance_cursor` 同一条取舍。
        """

        raise NotImplementedError

    # ========================================================
    # 场景文件（memory_scene）
    # ========================================================

    @abstractmethod
    async def list_scenes(
            self,
            scope: SceneScope,
    ) -> list[SceneRow]:
        """
        按 scope 取**全部**场景（含正文），按 `filename` 升序。

        ⚠️ 为什么一次全取，不分批
        --------------------
        提示词要的就是"现有场景文件清单 + 全文"：模型必须看到全部现有场景
        才能判断新的记忆该 UPDATE 进哪一个、以及该不该 MERGE。分页会让它
        在看不见的部分上做决定，而那个决定会**删掉**它没看到的东西。

        行数按构造很小（上限见服务层的 `DEFAULT_MAX_SCENES`，实际十几个），
        所以全取不心疼。

        ⚠️ 升序是给**提示词缓存**用的
        ------------------------
        同一份清单每次读出来的顺序必须一致，否则模型的输入前缀每次都变，
        缓存命中率掉到零。按 filename 排是最稳定的选择（`updated_at` 每次
        写入都会变）。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_scene_delta(
            self,
            scope: SceneScope,
            *,
            updated_after: datetime | None,
            consumed_ids: Sequence[str],
            limit: int,
    ) -> list[SceneRow]:
        """
        按 scope 取**变化过的**场景（L3 画像生成的输入），按 `updated_at` **升序**。

        ⚠️ 为什么这个方法住在这里，而不是在 persona 仓储上
        ----------------------------------------
        `memory_scene` 是**本仓储的表** —— 与 L2 读 L1 的增量走
        `MemoryAtomRepository.list_session_delta`（而不是自己写 SQL）是同一条
        归属规则。跨表查会让"这两列的口径是什么"在两处各有一份答案。

        ⚠️ 水位按 `updated_at` 而不是场景 id
        ------------------------------
        与本仓储的写入语义配对：`update_scene` 原地改写 + `delete_scenes`
        真删，所以按 id 记水位会随删除失真。`updated_at` 天然覆盖 —— 被改写
        的场景一定前移（`values(updated_at=func.now())`），于是重新进入下一批
        增量。代价：**被删掉的场景不会出现**，画像里由它衍生的内容只能靠模型
        这一次的生成覆盖掉（服务侧会把现有 persona 全文喂给它，所以它有机会
        删掉那一段）。

        ⚠️ `consumed_ids` 不是冗余参数
        --------------------------
        与 `list_session_delta` 同一个成因：一次 L2 整合写的所有场景共享一个
        `now()`（Postgres 取事务开始时刻）。只按 `updated_at > :t` 取的话，
        水位推到 t 之后、恰好落在同一微秒的另一批会被 `>` **静默跳过**。所以
        条件是

            updated_at > :t OR (updated_at = :t AND id NOT IN :consumed)

        `consumed_ids` 就是"t 这个时间戳上已经抽象过的那批"。

        ⚠️ 升序 + `limit` 的组合是**安全**的
        ------------------------------
        升序取最早的一批，水位只推到最后一条**真的消费掉的**，所以超限的
        剩下的下一轮接着来。反过来（降序取最新的）会把中间那段永久跳过 ——
        那是静默丢数据。`limit <= 0` 返回空列表（Postgres 把负的 LIMIT 当
        "不限量"，见 `list_session_delta` 里同一处的说明）。
        """

        raise NotImplementedError

    @abstractmethod
    async def count_scenes(
            self,
            scope: SceneScope,
    ) -> int:
        """
        该 scope 下**一共有几个**场景（含没变化过的）。

        ⚠️ 为什么不复用 `list_scenes` 数一下长度
        --------------------------------
        `list_scenes` 会把**每篇场景的全文**都读出来（一次十几行 × 每篇
        上千字）。L3 的统计块要的是一个数，为它把全文搬进内存没有道理 ——
        而这个数恰恰是提示词里"场景总数"那一项，每一轮都要。

        ⚠️ 它与"变化场景数"是**两个**数，不通用
        --------------------------------
        变化的是增量（这一轮要抽象什么），总数是覆盖面（画像的依据有多厚）。
        两者在提示词里各有用途 —— 见 `render_user_message` 的说明。
        """

        raise NotImplementedError

    @abstractmethod
    async def insert_scenes(
            self,
            rows: Sequence[SceneRow],
    ) -> int:
        """
        插入新场景，返回**真的插进去**的行数。

        ⚠️ 主键冲突**不覆盖**（`ON CONFLICT DO NOTHING`）
        ---------------------------------------------
        这正是乐观锁在"新建"这一侧的形态：id 是内容派生的，所以冲突意味着
        **另一个 L2 任务刚刚为同一个 scope + 同一个文件名建了行**。此时正确
        的动作是放弃本次并重来（调用方会看到返回的行数少于传入的行数），
        而不是覆盖别人的结果 —— 覆盖等于悄悄丢掉另一次整合的结论。

        空序列是合法的 no-op（返回 0），与 L1 的 `upsert_atoms` 一致。
        """

        raise NotImplementedError

    @abstractmethod
    async def update_scene(
            self,
            scene_id: str,
            *,
            content: str,
            content_md5: str,
            expected_version: int,
    ) -> int:
        """
        原地改写一个场景（带乐观锁），返回影响行数（0 或 1）。

        `WHERE id = :id AND version = :expected_version` —— 返回 0 就是冲突，
        调用方据此抛 `ConflictError`（见模块 docstring）。

        ⚠️ `created_at` 不在更新的列里
        -------------------------
        它是"这一行第一次写入的时间"，改写不该动它。`updated_at` 相反，
        每次改写都前移。

        ⚠️ version 由本方法自增（`version = version + 1`），不是调用方传
        --------------------------------------------------------
        传进来的话，"忘了 +1"与"故意不 +1"长得一样，而前者的后果是乐观锁
        **永远不冲突**（基线不动）—— 那是静默失效，比冲突频繁得多要糟。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_scenes(
            self,
            scope: SceneScope,
            *,
            filenames: Sequence[str],
    ) -> int:
        """
        删除该 scope 下这几个场景文件，返回删掉的行数。

        ⚠️ `scope` 是**必填**，不是可选过滤器
        -------------------------------
        与 L1 的 `delete_batch` 同一条纪律，而且触发它的东西一样不可信：
        文件名来自**模型的输出**（`content` 写成 `[DELETED]` 的那些）。没有
        归属校验的话，一个幻觉出来的路径就能删掉别的 scope（乃至别的租户）
        的画像，而且**不可逆** —— 画像装的是 LLM 的结论，重跑一次得到的是
        **不同**的文档。

        签名里刻意不给"不带 scope 的批量删"留口子。删除是按 `filename`
        而不是按主键做的，也正是为了让这条 scope 校验落在**每一条** DELETE
        上（按主键删的话，"这个 id 属于谁"要先查一次，那就多了一个可以被
        绕过的中间步骤）。
        """

        raise NotImplementedError
