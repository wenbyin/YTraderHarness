"""
Skill 的读写契约。

对应 skill.yaml：
    GET    /skills
    POST   /skills
    GET    /skills/{skill_id}
    PATCH  /skills/{skill_id}
    DELETE /skills/{skill_id}
    GET    /skills/{skill_id}/capabilities
    PUT    /skills/{skill_id}/capabilities

能力绑定与 Skill 本体放在同一个服务里：两条路径的主语都是 Skill，
绑定表也只属于本模块，没有拆成"工具侧关联"那种独立契约的理由。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    CapabilityRef,
    PagedSkillResponse,
    SkillCapabilityUpdate,
    SkillCreate,
    SkillDetail,
    SkillStatus,
    SkillUpdate,
)

__all__ = ["SkillService"]


class SkillService(BaseService):

    # ========================================================
    # Skill
    # ========================================================

    @abstractmethod
    async def list_skills(
            self,
            *,
            status: SkillStatus | None = None,
            keyword: str | None = None,
            capability_id: UUID | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> PagedSkillResponse:
        """
        分页列出 Skill。

        主 agent 读 skill-list 时传 status=ACTIVE；管理页不传
        就是全部状态。capability_id 用于反查"绑了某个能力的 skill"。
        """

        raise NotImplementedError

    @abstractmethod
    async def create_skill(
            self,
            request: SkillCreate,
    ) -> SkillDetail:
        """
        Raises:
            ConflictError: skill_key 已被占用。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_skill(
            self,
            skill_id: UUID,
    ) -> SkillDetail:
        """
        Raises:
            NotFoundError: Skill 不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def update_skill(
            self,
            skill_id: UUID,
            request: SkillUpdate,
    ) -> SkillDetail:
        """
        局部更新。skill_key 不可改 —— 它是稳定标识。

        Raises:
            NotFoundError: Skill 不存在。
            ValueError: 请求为空，或对非空字段显式传了 null。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete_skill(
            self,
            skill_id: UUID,
    ) -> None:
        """
        物理删除；能力绑定随 CASCADE 消失。

        将来有表 ref 本表时删除会失败（外键），常规退役请改
        status=DISABLED。

        Raises:
            NotFoundError: Skill 不存在。
        """

        raise NotImplementedError

    # ========================================================
    # 能力绑定
    # ========================================================

    @abstractmethod
    async def list_skill_capabilities(
            self,
            skill_id: UUID,
    ) -> list[CapabilityRef]:
        """
        Raises:
            NotFoundError: Skill 不存在。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_skill_capabilities(
            self,
            skill_id: UUID,
            request: SkillCapabilityUpdate,
    ) -> list[CapabilityRef]:
        """
        全量替换绑定；空数组表示清空。

        Raises:
            NotFoundError: Skill 不存在，或某个 capability_id 不存在。
        """

        raise NotImplementedError
