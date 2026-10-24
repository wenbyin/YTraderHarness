"""
Repository contract for the Skill ↔ capability binding.

只管 skill_capability 一张关联表：
    读：某个 skill 绑了哪些能力（详情页）
    写：整体替换某个 skill 的绑定集合

⚠️ 能力**本体**的读写在 app/services/capability/。关联的存在性校验
由服务层先做：那个判断要查能力表，属于 capability 模块，不该在这里
再开一条跨模块的写路径。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository
from app.services.capability.domain.entities import Capability

__all__ = ["SkillCapabilityRepository"]


class SkillCapabilityRepository(BaseRepository):

    @abstractmethod
    async def list_for_skill(
            self,
            skill_id: UUID,
    ) -> list[Capability]:
        """
        返回 skill 绑定的能力实体，按能力的 sort_order / code 排序。

        返回实体而不是 id 列表：详情页要展示 code / name，
        而排序也要用能力表自己的字段 —— 一次 join 取回来，
        比先取 id 再逐个查少一轮往返。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_for_skill(
            self,
            skill_id: UUID,
            capability_ids: list[UUID],
    ) -> None:
        """
        整体替换一个 Skill 的能力绑定。

        语义是替换不是追加：没列出来的会被解除绑定。
        重复 id 由实现去重（双列主键不接受重复行）。
        """

        raise NotImplementedError
