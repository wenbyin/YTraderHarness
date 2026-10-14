"""
实体 -> 对外契约的组装。

纯函数、不碰数据库 —— 放在这里而不是各服务里各写一份，是因为
"能力长什么样"必须**只有一份**：本模块的接口、tool 模块列表页
里嵌的那份能力（ToolSummary.classifications）、详情页里的那份，
三者不一致的话前端会看到"同一个能力在两处显示不一样"。
"""

from __future__ import annotations

from ..domain.entities import Capability as CapabilityEntity
from ..domain.schemas import Capability

__all__ = [
    "build_capability",
]


def build_capability(
        entity: CapabilityEntity,
) -> Capability:
    return Capability(
        id=entity.capability_id,
        code=entity.code,
        name=entity.name,
        description=entity.description,
        parent_id=entity.parent_id,
        sort_order=entity.sort_order,
        is_active=entity.is_active,
    )
