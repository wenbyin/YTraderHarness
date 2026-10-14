"""
MemoryAtomRepository —— L1 的持久化契约（memory_atom + 两张挂载表 + 水位）。

与 `MemoryIndexRepository`（L0）分开的理由：两者的**读写形状**不同。
L0 是"按 run 全量重写一段索引"，L1 是"按会话增量抽取 + 原地改写 + 有水位"。
合成一个契约只会让两边互相牵制 —— 与 session 模块把 message / session /
run 三个仓储分开是同一条理由。

⚠️ 事务边界：写入与推水位**必须在同一个 scope 里**
----------------------------------------------
本仓储（与 L0 那个一样）**从不** commit —— 边界由调用方的作用域决定
（`ServiceFactory.scope(契约)`，退出时提交）。

而水位与记录必须**一起提交**，这是整条链上最硬的一条约束：

    记录写了、水位没推   下一轮重抽同一段 —— 幂等（upsert），只是白跑
    水位推了、记录没写   那段**永久跳过**（下次按 `seq > 水位` 取增量）

所以 `advance_cursor` 只该在**记录确实写完**之后、且在同一个 scope 里被
调用。分开两个事务的话，第二条失效模式就成立了，而它没有任何症状。

⚠️ 当前只有"写"这一半
------------------
召回（两路 + RRF）与 `delete_batch`（多对一合并时删掉被吃掉的记录）属于
**去重**那一步，等它落地时加在这个契约上。当前抽取路径只需要写。
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from datetime import datetime
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities.atom import (
    AtomFtsRow,
    AtomRecall,
    AtomVecRow,
    MemoryAtom,
    MemoryAtomQueryScope,
    AtomCursorState,
)

__all__ = ["MemoryAtomRepository"]


class MemoryAtomRepository(BaseRepository):

    # ========================================================
    # 水位
    # ========================================================

    @abstractmethod
    async def get_cursor(
            self,
            session_id: UUID,
    ) -> AtomCursorState:
        """
        取该会话的水位；没有行时返回**默认值**（`last_seq=0`、情境为空串）。

        行是惰性创建的，所以"还没抽过"不表现为 None 而表现为 0 ——
        这样调用方不必把"没有行"和"水位是 0"当成两件事（它们本来就是
        同一件事，见 memory_atom_cursor.py 的类 docstring）。
        """

        raise NotImplementedError

    @abstractmethod
    async def advance_cursor(
            self,
            session_id: UUID,
            *,
            last_seq: int,
            last_scene_name: str,
            user_id: str | None = None,
    ) -> None:
        """
        把水位推到 `last_seq`，并记下当前情境。

        ⚠️ `last_scene_name` 与 `last_seq` 一起写，不能分两次
        ------------------------------------------------
        两次写之间会存在"抽到哪了"与"上一个情境是什么"不匹配的状态，而
        下一轮抽取读的正是这两个值 —— 拿到不匹配的组合会让它错误地判断
        「继承还是切换」。

        ⚠️ 单调性由调用方保证
        -----------------
        这里**不**做 `GREATEST(...)`：把水位往回退正是重抽一段的手段
        （将来要支持的话）。挡掉它会让那条路永远走不通，而它本身不是
        危险操作（回退 = 重抽 = 幂等）。
        """

        raise NotImplementedError

    # ========================================================
    # 记忆本体与两张挂载表
    # ========================================================

    @abstractmethod
    async def upsert_atoms(
            self,
            atoms: Sequence[MemoryAtom],
    ) -> int:
        """
        写入 / 原地改写记忆本体。返回写了几条。

        `record_id` 冲突时**原地更新**（不换 id、不新增行）—— 这是全库
        唯一的 L1 更新语义，`version` 由调用方在实体上算好带进来。

        ⚠️ `created_time` 不在更新的列里
        ----------------------------
        它是"这一行第一次写入的时间"，改写不该动它。`updated_time` 相反，
        每次改写都要前移（它是增量对账的水位）。

        空序列是合法的 no-op。
        """

        raise NotImplementedError

    @abstractmethod
    async def upsert_fts(
            self,
            rows: Sequence[AtomFtsRow],
    ) -> int:
        """
        写入 / 改写关键词索引行。

        ⚠️ 合并改写了 content 之后，这里**必须**跟着重写
        ------------------------------------------
        tokens 与 original 都是 content 的派生。不重写的话，搜到的还是
        旧内容，而这种不一致**不报错**（检索侧只会觉得"记忆没更新"）。
        调用方因此总是把 upsert_atoms / upsert_fts / upsert_vec 放在同一次
        调用序列里。
        """

        raise NotImplementedError

    @abstractmethod
    async def upsert_vec(
            self,
            rows: Sequence[AtomVecRow],
    ) -> int:
        """写入 / 改写向量索引行。与 `upsert_fts` 同一条理由，同样成组调用。"""

        raise NotImplementedError

    # ========================================================
    # 召回与删除（去重那一步用）
    # ========================================================

    @abstractmethod
    async def recall_keyword(
            self,
            scope: MemoryAtomQueryScope,
            *,
            tsquery: str,
            limit: int,
    ) -> list[AtomRecall]:
        """
        按关键词召回候选，按 `ts_rank` 降序。

        `scope` 决定过滤（会话 / 用户），`tsquery` 由**调用方**分词产出
        （`tokenization.tsquery_text`）—— 与 L0 的 `search_keyword` 同一条
        分工：仓储只管发 SQL，"怎么切词"只有一处。切不出词元时调用方
        **不要**调这里（那一路缺席），而不是传空串进来（`to_tsquery('')`
        会报错，或更糟：匹配一切）。
        """

        raise NotImplementedError

    @abstractmethod
    async def recall_vector(
            self,
            scope: MemoryAtomQueryScope,
            *,
            embedding: Sequence[float],
            limit: int,
    ) -> list[AtomRecall]:
        """
        按向量召回候选，按余弦距离升序（即相似度降序）。

        ⚠️ 分数是 `1 - distance`，在 Python 侧算
        ------------------------------------
        排序表达式必须与索引的操作符**逐字一致**（HNSW 的
        `vector_cosine_ops` 对 `<=>`），而"相似度"是给调用方看的另一个
        量。混在一条 SQL 里会让优化器认不出索引可用的形状 —— 结果一样、
        只是慢，而且不报错（与 L0 的 `search_vector` 同一条取舍）。
        """

        raise NotImplementedError

    @abstractmethod
    async def load(
            self,
            record_ids: Sequence[UUID],
    ) -> dict[UUID, MemoryAtom]:
        """
        按 id 批量取记忆本体，返回 `record_id -> 实体`。

        取不到的不在返回里（不是错误）—— 去重的候选池是按 id 召回的，
        而两条并发任务之间可能有记录被删掉，那种情况下"少一条候选"远好于
        "整批失败"。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_batch(
            self,
            record_ids: Sequence[UUID],
            *,
            session_id: UUID,
    ) -> int:
        """
        删除这批记忆：**主表 + 两张挂载表**，返回主表删掉的行数。

        ⚠️ `session_id` 是**必填**，不是可选过滤器
        -------------------------------------
        多对一合并会删掉"被吃掉的那几条"，而 `target_ids` 来自模型的
        输出 —— 那是**不可信输入**。没有归属校验的话，一个幻觉出来的 id
        就能删掉别的会话（乃至别的租户）的记忆，而且**不可逆**：L0 的索引
        删错了可以从消息重建，L1 删掉的是 LLM 的结论，重跑一次得到的是
        **不同**的记忆。

        ⚠️ 粒度与候选池**必须一致**（都是会话）
        ----------------------------------
        池子是按会话召回的，删除就按会话放行。两者一旦分叉，会得到
        "池子里明明有这条、删的时候却没删掉"的**半截合并**：目标行留着、
        新行又没写进去 —— 两边都不报错。要改粒度就两处一起改。

        签名里刻意不给"不带 scope 的批量删"留口子 —— 与 L0 的 `_conditions`
        必带 `user_id` 是同一条纪律，但这边更硬（那边是读，这边是删）。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_session_delta(
            self,
            session_id: UUID,
            *,
            updated_after: datetime | None,
            consumed_ids: Sequence[UUID],
            limit: int,
    ) -> list[MemoryAtom]:
        """
        按会话取**增量**记忆（L2 整合的输入），按 `updated_time` **升序**。

        ⚠️ 为什么水位用 `updated_time` 而不是 record_id
        ----------------------------------------
        `update` / `merge` 是**破坏性**的：多对一合并里存活那条保留 id、
        被吃掉的几条整行消失。按 id 记水位会随合并失真。而 `updated_time`
        天然覆盖这件事 —— 存活者一定被改写、一定前移，于是重新进入下一批
        增量（内容变了，画像该重看）。代价：被删掉的记忆不再出现，画像里
        由它衍生的内容只能靠模型这一次的整合覆盖掉。

        ⚠️ `consumed_ids` 不是冗余参数
        --------------------------
        Postgres 的 `now()` 取**事务开始时刻**，L1 一次运行写的所有 atom
        因此共享同一个 `updated_time`。只按 `updated_time > :t` 取的话，
        水位推到 t 之后、恰好落在同一微秒的另一批（并发提交）会被 `>`
        **静默跳过** —— 漏一条记忆、不报错、也永远补不回来。

        所以条件是

            updated_time > :t OR (updated_time = :t AND record_id NOT IN :consumed)

        `consumed_ids` 就是"t 这个时间戳上已经消费过的那批"。见
        models/sqlalchemy/memory_scene_cursor.py 的模块 docstring。

        ⚠️ 升序 + limit：**取最早的那一批**，不是最新的
        --------------------------------------
        调用方拿到多少就把水位推到多少（推到最后一条的 `updated_time`），
        所以

            升序 + limit    没取到的下一轮接着取 —— 一条都不会丢，只是慢
            降序 + limit    水位直接跳到最新，中间那段**永远不再出现**

        后者是静默丢数据，所以顺序在这里不是风格问题。

        `updated_after` 为 None（水位还没有值）时**不过滤时间**：那是
        "这个会话还没整合过任何 L1"，全量就是它的增量。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_by_source_message(
            self,
            message_id: UUID,
            *,
            session_id: UUID,
            limit: int,
    ) -> list[MemoryAtom]:
        """
        反查：引用了这条 L0 消息的记忆，按 `updated_time` 降序。

        给管理端的跨层溯源用（`memory.query.l0_atoms`）—— 与
        `list_session_delta` 是同一个列的两个方向，但**取数口径完全不同**：
        那条是增量（升序、按水位、要求一条不丢），这条是**看当下**
        （降序 = 最近改过的在前，取到 limit 就够）。

        ⚠️ `session_id` 是必填的收窄，不是可选过滤
        -----------------------------------
        `source_message_ids` 是一列 JSONB，**没有 GIN 索引**，包含查询
        （`@>`）单独用会全表扫。带上会话之后计划器先用
        `idx_memory_atom_session` 把候选收窄到这一个会话，包含判断只在
        那一小撮行上做。

        代价说清楚：**跨会话反查做不到**（要一张 GIN 索引，那意味着每次
        落库都多维护一棵索引 —— 写入侧的决定，不在这里顺手做掉）。

        ⚠️ 包含的是**字符串**
        ------------------
        `source_message_ids` 里存的是 message_id 的字符串形式（落库前由
        L1 的服务侧过滤过，解析不出 UUID 的已经剔掉，见
        `domain/schemas/atom.py` 那个字段的说明）。拿 UUID 直接比会一条
        都匹配不上，而且**不报错** —— 这是这条查询最容易写错的一处。

        `limit <= 0` 返回空列表（Postgres 把负的 LIMIT 当"不限量"，见
        `list_session_delta` 里的同一处说明）。
        """

        raise NotImplementedError

    @abstractmethod
    async def list_page(
            self,
            *,
            user_id: str,
            record_type: str | None,
            priority: int | None,
            scene_name: str | None,
            updated_from: datetime | None,
            updated_to: datetime | None,
            text_contains: str | None,
            order: str,
            limit: int,
            after: Sequence[Any] | None,
    ) -> tuple[list[MemoryAtom], bool]:
        """
        管理端浏览：按 scope + 过滤条件取**一页**记忆，返回 `(items, has_more)`。

        这是本契约里**唯一**一个"不服务于抽取/合并链路"的读方法 —— 它给
        `memory.query.l1` 用（只读视图），所以它的排序与过滤是**给人看**的，
        与 `list_session_delta`（增量，一条都不能丢）目的不同、形状也不同。

        ⚠️ 可见性口径与召回**逐字一致**
        -------------------------
        `user_id` 那条是 `= :uid OR IS NULL`，不是等值 —— NULL 是加这一列
        之前的存量语义（那批行对谁都可见，见 `_conditions` 与
        `agent_session` 的同一条规则）。浏览与召回用不同口径的话，"搜得到
        但列表里没有"会变成一件查不出原因的事。

        ⚠️ `text_contains` 是 `ILIKE '%…%'`，通配符会被转义
        -------------------------------------------
        它匹配 `content` 整串（中文没有分词配置，只能整串匹配 —— 与 L2 那边
        同一条取舍）。**不执行**语义或向量召回。

        ⚠️ `order` 决定 `after` 的形状与 keyset 条件
        -------------------------------------
            updated_at_desc / updated_at_asc    after = (updated_time, record_id)
            priority_desc / priority_asc        after = (priority, record_id)

        第二项都是 `record_id`：`updated_time` 会被合并**原地改写**、
        `priority` 大量重复 —— 没有唯一键做 tie-breaker，翻页会漏行或重复
        行（两个方向都会）。`after` 为 None 表示第一页。

        ⚠️ 返回 `has_more` 而不是总数
        ------------------------
        游标分页给不出 total（要算就得再扫一遍），而且调用方真正需要的
        是"还要不要继续翻"。实现上多取一条（`LIMIT limit + 1`）来判。
        """

        raise NotImplementedError
