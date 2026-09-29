"""
Repository contract for tool processor mounts.

一张表，但**所有写操作都以整条链为单位**
------------------------------------------
    tool_processor    某个 Tool 的链上的行，按 sequence 排序

链的顺序就是语义（谁先谁后决定执行顺序），因此任何一次写都必然
动到"顺序"这件事。把"读一行 / 写一行 / 挪一下"拆成三个接口交给
调用方组合，等于把唯一性约束的维护责任散到服务层 ——
挪位会撞 `uq_tool_processor_tool_sequence`（非延迟索引，逐行立即
检查），而调用方根本不知道有这回事。

所以这里只给"整条链"级别的入口，挪位与重编号都在实现里完成。

sequence 的不变式：**写完永远是 1..N，连续**
--------------------------------------------
前端可以传带空洞的 sequence（它只是位置提示），落库前一律重编号。
因此读出来的 sequence 永远是 1..N —— 调用方不必处理"1, 3, 7"这种链。
"""

from __future__ import annotations

from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import ToolProcessor

__all__ = ["ToolProcessorRepository"]


class ToolProcessorRepository(BaseRepository):

    # ========================================================
    # Read
    # ========================================================

    @abstractmethod
    async def list_for_tool(
            self,
            tool_id: UUID,
            *,
            enabled_only: bool = False,
    ) -> list[ToolProcessor]:
        """
        取一条链，按 sequence 升序。

        **不筛选 processor_key 是否还在代码里** —— 那要看注册表，
        仓储不该知道注册表。未注册的行照常返回，由组装层决定
        怎么呈现（见 application/_assembly.py）。

        enabled_only=True 时只返回启用的行：给"执行时真正会跑的
        那几条"用。管理面读链一律用 False —— 关掉的行也要看得见。
        """

        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            tool_id: UUID,
            processor_id: UUID,
    ) -> ToolProcessor | None:
        """
        取链上的一行。

        ⚠️ 必须同时按 tool_id 过滤：processor_id 是全局唯一的，
        但接口的路径是 /tools/{tool_id}/processors/{processor_id} ——
        只按 id 查会让"用 A 工具的 id 去改 B 工具的行"变成一次
        成功的越界写。
        """

        raise NotImplementedError

    @abstractmethod
    async def count_for_tool(self, tool_id: UUID) -> int:
        """链上现有多少行（含 disabled）。插入位置要按它钳制。"""

        raise NotImplementedError

    # ========================================================
    # Write
    # ========================================================

    @abstractmethod
    async def insert(
            self,
            entity: ToolProcessor,
            *,
            position: int | None = None,
    ) -> ToolProcessor:
        """
        往链上插一行，并把整条链重编号成 1..N。

        position 是**最终的 1-based 位置**（1 = 插到最前），
        None = 追加到末尾。调用方负责钳制到 [1, count + 1] ——
        仓储不做这个判断：钳制是业务决定（"传 99 算插到末尾"），
        不是存储层的规则。
        """

        raise NotImplementedError

    @abstractmethod
    async def update(self, entity: ToolProcessor) -> ToolProcessor:
        """
        覆盖一行（不改 sequence，也不改 processor_key ——
        那两个字段各有专门的操作）。

        行不存在时抛 NotFoundError，而不是静默 upsert：
        走到这里说明服务层刚查过它，不存在就是并发删掉了，
        悄悄建一行回来会让调用方以为改成功了。
        """

        raise NotImplementedError

    @abstractmethod
    async def delete(
            self,
            tool_id: UUID,
            processor_id: UUID,
    ) -> int:
        """
        删一行，并把剩下的重编号成 1..N（补洞）。

        返回删除的行数（0 = 本来就没有）。**不报错**：
        删除幂等，调用方据此决定要不要回 404。
        """

        raise NotImplementedError

    @abstractmethod
    async def replace_all(
            self,
            tool_id: UUID,
            entities: list[ToolProcessor],
    ) -> list[ToolProcessor]:
        """
        整条链替换：先全删，再按 entities 的顺序插入并编号 1..N。

        entities 里的 sequence 会被**忽略并重算** —— 顺序由列表
        下标决定，不让两个来源打架。

        ⚠️ 行是**全删重建**的：processor_id 全部重新生成。
        同一个 processor_key 可以在一条链里出现多次，按 key 复用
        id 本质上是歧义的（哪个 config 归哪一行？）。
        """

        raise NotImplementedError

    @abstractmethod
    async def reorder(
            self,
            tool_id: UUID,
            ordered_ids: list[UUID],
    ) -> int:
        """
        按给定顺序重编号整条链，返回行数。

        ordered_ids 必须是该 Tool **现有行的全集**（含 disabled）——
        实现要校验这一点并在不匹配时报错，而不是"把给到的排前面、
        剩下的接后面"：部分重排会让"剩下的排哪"变成实现细节，
        而调用方以为自己指定了完整顺序。
        """

        raise NotImplementedError
