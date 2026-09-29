"""
流水线可观测性的查询契约。

为什么单独一个服务而不是塞进 DocumentRuntimeService
--------------------------------------------------
DocumentRuntimeService 已经 665 行，职责是文档/版本的生命周期。
可观测性是**只读的另一个视图**，两者的读者、变更频率、
权限模型都不同。

拆开还有一个实际好处：文档运行时服务的依赖全是写路径的仓储，
查询服务只依赖观测两张表 + 图注册表，注入干净得多。

非职责
------
    - 触发运行（那是 ProcessingDispatcher）
    - 写任何东西

节点兼容性
----------
本服务的 detail 响应返回该次运行**自己的** topology_snapshot，
并给出 unknown_nodes 标志。前端据此渲染历史运行，
不受当前图版本影响。详见 domain/schemas/observability.py。
"""

from __future__ import annotations

from abc import abstractmethod
from datetime import datetime
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas.observability import (
    PipelineNodeCatalogResponse,
    PipelineNodeRunListResponse,
    PipelineRunDetailResponse,
    PipelineRunListResponse,
)


class PipelineQueryService(BaseService):
    """运行 / 节点 / 节点目录的只读查询。"""

    @abstractmethod
    async def list_runs(
            self,
            *,
            document_id: UUID | None = None,
            document_version_id: UUID | None = None,
            status: str | None = None,
            trigger_source: str | None = None,
            triggered_by: UUID | None = None,
            started_from: datetime | None = None,
            started_to: datetime | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> PipelineRunListResponse:
        raise NotImplementedError

    @abstractmethod
    async def get_run(
            self,
            pipeline_run_id: UUID,
    ) -> PipelineRunDetailResponse:
        """
        运行详情：run 行 + 全部节点行 + 拓扑快照 + unknown_nodes。

        Raises:
            NotFoundError: 运行不存在。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_node_runs(
            self,
            pipeline_run_id: UUID,
            *,
            page: int = 1,
            page_size: int = 200,
    ) -> PipelineNodeRunListResponse:
        """深扇出时单独翻节点行。"""
        raise NotImplementedError

    @abstractmethod
    async def get_node_catalog(
            self,
    ) -> PipelineNodeCatalogResponse:
        """
        当前图版本的节点目录。

        node_id 是契约键：前端用它把节点行翻译成展示信息。
        退役节点永久保留并标 deprecated，历史运行里的旧
        node_id 因此永远查得到解释。
        """
        raise NotImplementedError
