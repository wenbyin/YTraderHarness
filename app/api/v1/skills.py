"""
Skill 管理接口。

对应 skill.yaml 的 tags：
    Skills               Skill 本体的查询与维护
    Skill Capabilities   Skill ↔ 能力绑定

规格与实现在 app/services/skill/ 下，本文件只做 HTTP 层的转发。
鉴权不在这里建模：路由挂在 protected_router 上时，沿用平台登录；
Skill 本身没有额外的权限模型。

⚠️ 路由顺序：/skills/{skill_id} 与 /skills/{skill_id}/capabilities
段数不同，不存在宽匹配抢先的问题，不需要刻意排序。
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Query, status as http_status

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.skill.application.skill_service import (
    SkillService,
)
from app.services.skill.domain.schemas import (
    CapabilityRef,
    PagedSkillResponse,
    SkillCapabilityUpdate,
    SkillCreate,
    SkillDetail,
    SkillStatus,
    SkillUpdate,
)

router = APIRouter()


# ============================================================
# Skill
# ============================================================


@router.get(
    "/skills",
    response_model=PagedSkillResponse,
    tags=["Skills"],
)
async def list_skills(
        status: SkillStatus | None = Query(
            default=None,
            description=(
                "按生命周期状态过滤。主 agent 读 skill-list 时传 "
                "ACTIVE；不传表示全部状态。"
            ),
        ),
        q: str | None = Query(
            default=None,
            description=(
                "按 skill_key / name / description 模糊匹配。"
            ),
        ),
        capability_id: UUID | None = Query(
            default=None,
            description="反查绑定了指定能力的 Skill。",
        ),
        page: int = Query(default=1, ge=1),
        page_size: int = Query(default=20, ge=1, le=200),
        service: SkillService = Service[SkillService],
):
    """分页列出 Skill。"""

    try:
        return await service.list_skills(
            status=status,
            keyword=q,
            capability_id=capability_id,
            page=page,
            page_size=page_size,
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/skills",
    response_model=SkillDetail,
    status_code=http_status.HTTP_201_CREATED,
    tags=["Skills"],
)
async def create_skill(
        request: SkillCreate,
        service: SkillService = Service[SkillService],
):
    """新建 Skill。初始没有能力绑定。"""

    try:
        return await service.create_skill(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.get(
    "/skills/{skill_id}",
    response_model=SkillDetail,
    tags=["Skills"],
)
async def get_skill(
        skill_id: UUID,
        service: SkillService = Service[SkillService],
):
    """取单个 Skill（含 instructions 与能力绑定）。"""

    try:
        return await service.get_skill(skill_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.patch(
    "/skills/{skill_id}",
    response_model=SkillDetail,
    tags=["Skills"],
)
async def update_skill(
        skill_id: UUID,
        request: SkillUpdate,
        service: SkillService = Service[SkillService],
):
    """
    局部更新 Skill。

    skill_key 不在可改字段里 —— 它是稳定标识，后续 task board /
    run 记录按它引用。没有版本表，更新就地生效；运行时应在 spawn
    子 agent 时取快照。
    """

    try:
        return await service.update_skill(skill_id, request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.delete(
    "/skills/{skill_id}",
    status_code=http_status.HTTP_204_NO_CONTENT,
    tags=["Skills"],
)
async def delete_skill(
        skill_id: UUID,
        service: SkillService = Service[SkillService],
):
    """
    物理删除 Skill。能力绑定随 CASCADE 消失。

    常规退役请改 status=DISABLED：将来有表 ref 本表后，硬删除会
    被外键挡住。
    """

    try:
        await service.delete_skill(skill_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 能力绑定
# ============================================================


@router.get(
    "/skills/{skill_id}/capabilities",
    response_model=list[CapabilityRef],
    tags=["Skill Capabilities"],
)
async def list_skill_capabilities(
        skill_id: UUID,
        service: SkillService = Service[SkillService],
):
    """列出 Skill 已绑定的能力。"""

    try:
        return await service.list_skill_capabilities(skill_id)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.put(
    "/skills/{skill_id}/capabilities",
    response_model=list[CapabilityRef],
    tags=["Skill Capabilities"],
)
async def replace_skill_capabilities(
        skill_id: UUID,
        request: SkillCapabilityUpdate,
        service: SkillService = Service[SkillService],
):
    """
    全量替换 Skill 的能力绑定。空数组表示清空。

    绑定只决定工具可见性，不是授权：工具按能力树推导，
    见 skill 模块的 models.sql。
    """

    try:
        return await service.replace_skill_capabilities(
            skill_id, request
        )

    except ValueError as exc:
        raise to_http_error(exc) from exc
