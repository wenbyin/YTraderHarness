"""
MemoryPersonaRepository —— L3 画像与它的 scope 级水位的持久化契约。

⚠️ 为什么与 `MemorySceneRepository` 分开，而不是把四个方法塞进去
-------------------------------------------------------
两者的**键不同**，这是本质的：

    MemorySceneRepository    按 scope 读写**多行**（一个话题一行）+ 会话级水位
    MemoryPersonaRepository  按 scope 读写**一行**（一整篇）+ scope 级水位

合成一个契约会让"这个 scope 下有几行"变成调用方每次要自己想的事 —— 而
L2 是"一次十几行"，L3 是"恒为一行"，两者的乐观锁用法、幂等写法（upsert
多行 vs 单行）都不同。与 session 模块把 message / session / run 分开是同
一条理由。

⚠️ 但 L3 **读**场景走 `MemorySceneRepository.list_scene_delta`
----------------------------------------------------
那不是本契约的事：`memory_scene` 是那张表的仓储，增量口径（按 updated_at、
边界 id 的排除）只有它说了算。本契约只管 `memory_persona` 与
`memory_persona_cursor` 两张表。与 L2 读 L1 增量走
`MemoryAtomRepository.list_session_delta` 是同一条归属规则。

⚠️ 事务边界：画像行与水位**必须在同一个 scope 里**
------------------------------------------
本仓储**从不** commit —— 边界由调用方的作用域决定。

    persona 写了、水位没推   下一轮重抽同一批场景 —— 幂等（按主键 upsert），
                            只是白跑一次模型
    水位推了、persona 没写   那批场景**永久跳过**（下次按 `> 水位` 取增量）

第二条没有任何症状，所以它是这条链上唯一不能让步的地方 —— 与 L1 / L2
的"记录 + 水位同事务"同一纪律。

⚠️ 乐观锁是本契约的一部分
-------------------
`update_persona` 的 `expected_version` 不是可选的优化，是**并发正确性**的
落点：同一个 scope 可以被多个会话的 L2 完成各排一条 L3（见
application/trigger.py 的 `submit_persona_extract`），两条可以同时读到同一
篇画像、各自调模型、各写各的。谁后写谁覆盖 —— 而"覆盖"在这里意味着丢掉
另一次生成的**全部结论**（整篇画像，不是一行数据）。

所以写回带 `WHERE version = :expected`，影响行数为 0 就是"这中间有人改过"，
调用方抛 `ConflictError` 交给任务重入（`RetryPolicy` 默认会重试）。
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Sequence
from datetime import datetime

from app.common.repository.base import BaseRepository

from ..domain.entities.persona import (
    PersonaCursorState,
    PersonaRow,
)
from ..domain.entities.scene import SceneScope

__all__ = ["MemoryPersonaRepository"]


class MemoryPersonaRepository(BaseRepository):

    # ========================================================
    # 画像本体（memory_persona）
    # ========================================================

    @abstractmethod
    async def get_persona(
            self,
            scope: SceneScope,
    ) -> PersonaRow | None:
        """
        取该 scope 的那**一篇**画像；没有就返回 None。

        ⚠️ 这里返回 None 而不是默认值 —— 与两个 `get_*_cursor` 相反
        ------------------------------------------------
        水位的"没有行"与"有行但为空"对调用方是同一件事（没有增量）；而
        画像的"没有行"是**一个有意义的状态**：触发条件 1（该 scope 有场景
        但没有画像记录）就是靠它判的，而且它决定提示词进入"首次生成"模式。

        折成默认值会让"还没生成过"与"生成过但内容是空的"长得一样 ——
        后者在本仓库**不可能存在**（`ck_memory_persona_content_not_blank`
        挡着），所以这里干脆不做那个区分，返回 None 就是"还没有"。

        ⚠️ 一次主键查询，读出的是**整篇 content**（≤2000 字符）
        --------------------------------------------
        调用方（触发判定与生成）两处都要正文，所以不另做"只查存在性"的
        方法 —— 那样会有两个回答"这个 scope 有没有画像"的地方，而它们的
        答案可能不同（一个查了 `id IS NOT NULL`、一个读了行）。
        """

        raise NotImplementedError

    @abstractmethod
    async def insert_persona(
            self,
            row: PersonaRow,
    ) -> int:
        """
        插入一篇画像，返回**真的插进去**的行数（0 或 1）。

        ⚠️ 主键冲突**不覆盖**（`ON CONFLICT DO NOTHING`）
        -------------------------------------------
        与 `insert_scenes` 同一条：id 是 scope 派生的，所以冲突意味着
        **另一个 L3 任务刚刚为同一个 scope 建了行**（两个会话的 L2 完成
        各排了一条，而它们都读到"还没有画像"）。此时正确的动作是放弃本次
        并重来，而不是覆盖别人的结果 —— 覆盖等于悄悄丢掉另一次生成的结论。

        返回 0 时调用方抛 `ConflictError`（任务重入后会读到那一行，转入
        迭代更新那条路）。
        """

        raise NotImplementedError

    @abstractmethod
    async def update_persona(
            self,
            persona_id: str,
            *,
            content: str,
            content_md5: str,
            expected_version: int,
    ) -> int:
        """
        原地改写画像（带乐观锁），返回影响行数（0 或 1）。

        `WHERE id = :id AND version = :expected_version` —— 返回 0 就是冲突，
        调用方据此抛 `ConflictError`（见模块 docstring）。

        ⚠️ `created_at` 不在更新的列里
        -------------------------
        它是"这一行第一次写入的时间"，改写不该动它。`updated_at` 相反，
        每次改写都前移。

        ⚠️ version 由本方法自增（`version = version + 1`），不是调用方传
        --------------------------------------------------------
        与 `update_scene` 同一条取舍：传进来的话，"忘了 +1"与"故意不 +1"
        长得一样，而前者的后果是乐观锁**永远不冲突**（基线不动）—— 那是
        静默失效，比冲突频繁得多要糟。
        """

        raise NotImplementedError

    # ========================================================
    # 水位（scope 级）
    # ========================================================

    @abstractmethod
    async def get_l3_cursor(
            self,
            scope: SceneScope,
    ) -> PersonaCursorState:
        """
        取该 scope 的生成水位；没有行时返回**默认值**（两个水位 None）。

        行是惰性创建的，所以"还没生成过"不表现为 None 而表现为那一组默认值
        —— 与两个 `get_*_cursor` 同一条取舍。

        ⚠️ `last_l3_at` 有值时它单独有意义：触发条件 2（距上次生成 >= 50
        轮次）要靠它算基准，而它与"场景水位"是**两件事**（一个是墙钟、一个
        是数据的写入时间）。见 `PersonaCursorState` 的说明。
        """

        raise NotImplementedError

    @abstractmethod
    async def advance_l3_cursor(
            self,
            scope: SceneScope,
            *,
            last_scene_updated_time: datetime | None,
            last_scene_ids: Sequence[str],
            last_l3_at: datetime,
    ) -> None:
        """
        推进水位：记下"抽象到哪个场景了"与"这次是什么时候跑的"。

        ⚠️ `last_scene_ids` 必须与 `last_scene_updated_time` 一起写
        ---------------------------------------------------
        它们是**一个**水位（时间戳 + 落在该时间戳上已消费的 id 集），分两次
        写会造出"时间是新的、id 集是旧的"这种组合，而下一轮读增量时它会按
        错的边界排除 —— 症状是同一微秒的那批**多读一遍**（模型重复看到同一
        个场景）或**漏读**（永久丢）。见 memory_persona_cursor.py 的模块
        docstring。

        ⚠️ 空转那一轮也要调它（水位保持不变、只把 `last_l3_at` 戳一下）
        ---------------------------------------------------
        它是条件 2 的时间基准。不戳的话，一个反复空转的 scope 会被每一个
        L2 完成重新算成"距上次生成很久了"，于是每次 L2 完成都白跑一次 L3。

        ⚠️ 单调性由调用方保证
        -----------------
        与 `advance_l2_cursor` 同一条：这里**不**做 `GREATEST(...)`。往回退
        正是"重抽一段"的手段（全量重跑 L3 靠它），挡掉会让那条路永远走不通，
        而回退本身不危险（重抽 = 幂等）。
        """

        raise NotImplementedError
