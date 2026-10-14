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

    async def initialize(self) -> None:
        """
        异步初始化钩子。**默认什么都不做。**

        给"构造客户端本身就是阻塞 I/O"的数据源用。典型是 Milvus：
        pymilvus 的 `MilvusClient(...)` 在构造时就去 gRPC 建连（连不上
        会重试着等），实测把事件循环按住 **3.3 秒** —— 而 `__init__`
        是同步的，没有地方 await，所以只能挪到这里包 `to_thread`。

        为什么值得为它加一个钩子
        ------------------------
        被按住的那几秒里，所有请求、SSE 心跳、以及 Redis 的读超时定时器
        一起停摆；后者恢复时会**谎报**成网络故障（"Timeout reading from
        localhost:6379"），把排查方向整个带偏。启动顺序是
        `create -> initialize -> health_check -> register`，
        这段活儿唯一的落点就是这里。

        ⚠️ 不是"每个数据源都要实现的仪式"：绝大多数客户端的构造是惰性的
        （redis / sqlalchemy / elasticsearch 都不在建构造时连接），
        它们什么都不用写。
        """

        return None

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
