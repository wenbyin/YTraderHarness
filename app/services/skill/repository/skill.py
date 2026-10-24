"""
Repository contract for Skill.

只管 skill 表本身；ability 绑定是另一个契约
（SkillCapabilityRepository），因为两者的读写时机不同 ——
列表页只读 skill，详情页才需要 join 能力。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import Skill
from ..domain.schemas.enum import SkillStatus

__all__ = ["SkillRepository"]


class SkillRepository(BaseRepository):

    @abstractmethod
    async def add(
            self,
            skill: Skill,
    ) -> Skill:
        """新建一个 Skill。skill_key 冲突由调用方先查过。"""

        raise NotImplementedError

    @abstractmethod
    async def update(
            self,
            skill: Skill,
    ) -> Skill:
        """
        更新可变字段。

        **不改 skill_key** —— 它是稳定标识，后续表按它引用本行。
        实现必须忽略实体里可能变了的 skill_key。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete(
            self,
            skill_id: UUID,
    ) -> None:
        """
        物理删除一个 Skill。

        绑定行由数据库的 ON DELETE CASCADE 一并清掉。
        将来有别的表 ref 本表时，外键会让删除失败 —— 这是预期的；
        常规退役应改 status=DISABLED。
        """

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            skill_id: UUID,
    ) -> Skill | None:
        raise NotImplementedError

    @abstractmethod
    async def get_by_key(
            self,
            skill_key: str,
    ) -> Skill | None:
        raise NotImplementedError

    @abstractmethod
    async def list_paged(
            self,
            *,
            status: SkillStatus | None = None,
            keyword: str | None = None,
            capability_id: UUID | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[Skill], int]:
        """
        分页列出 Skill，返回 (本页数据, 总数)。

        keyword 同时匹配 skill_key / name / description；
        capability_id 用 EXISTS 过滤 —— 一个 skill 可能绑多个能力，
        join 会让它在结果里重复出现，分页与 total 就都错了。
        """

        raise NotImplementedError
