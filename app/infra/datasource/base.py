from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class DataSource(ABC):
    """
    所有数据源的统一生命周期抽象。

    DataSource 负责：
        - 初始化资源
        - 健康检查
        - 获取底层 client/resource
        - 关闭资源

    不负责：
        - 业务逻辑
        - Repository
        - Service
    """

    def __init__(
            self,
            name: str,
    ) -> None:
        self.name = name

    @abstractmethod
    async def health_check(self) -> bool:
        """
        检查数据源是否可用。
        """
        raise NotImplementedError

    @abstractmethod
    async def close(self) -> None:
        """
        释放资源。
        """
        raise NotImplementedError

    @abstractmethod
    def resource(self) -> Any:
        """
        返回底层客户端/资源。
        """
        raise NotImplementedError
