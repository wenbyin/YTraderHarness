"""
工具发现与同步的契约。

对应 tool.yaml：
    POST /tools/discovery/sync

这是 definition 表**唯一**的写入口。它整行覆盖代码侧事实，
但永不触碰说明与分类（见 domain/entities/tool_definition.py）。
"""

from __future__ import annotations

from abc import abstractmethod

from app.common.service.base import BaseService

from ..domain.schemas import ToolSyncRequest, ToolSyncResult

__all__ = ["ToolDiscoveryService"]


class ToolDiscoveryService(BaseService):

    @abstractmethod
    async def sync(
            self,
            request: ToolSyncRequest | None = None,
    ) -> ToolSyncResult:
        """
        扫描代码里的工具声明，与库里的行对齐。

        比对按 tool_key，判据是 manifest_hash：

            库里没有              -> 新建
            有、摘要不同           -> 覆盖
            有、摘要相同           -> 原样跳过（只刷新 last_synced_at）
            库里 ACTIVE 但没扫到   -> 转 MISSING（或按请求物理删除）

        幂等：同样的代码跑第二次，结果一定是
        created=0 / updated=0 / missing=0，全部 unchanged。
        这条是判断"同步是否正常工作"最直接的信号 ——
        如果第二次还报 updated，说明摘要不稳定，
        每次同步都会无谓地重写整张表。
        """

        raise NotImplementedError
