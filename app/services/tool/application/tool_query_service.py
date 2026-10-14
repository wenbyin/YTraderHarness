"""
Tool 定义的查询契约。

对应 tool.yaml：
    GET /tools            列表
    GET /tools/{tool_id}  详情

只读，无副作用。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas import (
    PagedToolSummaryResponse,
    SourceState,
    SourceType,
    ToolDefinitionDetail,
)

__all__ = ["ToolQueryService"]


class ToolQueryService(BaseService):

    @abstractmethod
    async def list_tools(
            self,
            *,
            keyword: str | None = None,
            source_type: SourceType | None = None,
            classification_code: str | None = None,
            source_state: SourceState | None = None,
            page: int = 1,
            page_size: int = 20,
    ) -> PagedToolSummaryResponse:
        """
        分页列出工具定义。

        classification_code 按分类的**稳定标识**筛（而不是 id）：
        分类 id 会在重建分类树时变，code 不会，而调用方
        （前端 / Agent 配置）记住的是 code。

        description 回的是**生效值**（人工覆盖优先），
        与详情页保持一致。
        """

        raise NotImplementedError

    @abstractmethod
    async def get_tool(
            self,
            tool_id: UUID,
    ) -> ToolDefinitionDetail:
        """
        取一个工具的全部信息：事实 + 说明 + 分类。

        Raises:
            NotFoundError: 该 tool_id 不存在。
        """

        raise NotImplementedError

    # ========================================================
    # 解析（不是 HTTP 契约）
    # ========================================================
    #
    # ⚠️ 下面三个方法**不**对应 tool.yaml 里的任何路由，也不返回
    # DTO。它们的调用方是 Agent 侧（Agent Builder 与工具执行节点），
    # 要的是"哪个 key / 哪个 id"，而不是给人看的响应体。
    #
    # 刻意不放进 DTO 层：`/tools` 的响应形状由 tool.yaml 冻结，
    # 往里加字段是改对外契约；而这里返回的东西（一批 key、一张
    # key->id 表）是过程量，前端没有消费它的场景。

    @abstractmethod
    async def tool_keys_for_capabilities(
            self,
            codes: list[str],
    ) -> list[str]:
        """
        能力 code 列表 -> 挂在它们下面的 ACTIVE 工具的 tool_key。

        去重且保序（按 tool_key 排序，与仓储的 order_by 一致）——
        同一个工具挂多个能力时只出现一次；调用方直接把它当
        `context_policy.tool_keys` 用，顺序稳定才能让两次同样的
        分类得到同样的请求体。

        **不抛未知 code**：分类模型给出的 code 一定在候选集里，
        但候选集本身可能变了（还挂着旧 code 的能力被停用）。
        "某些 code 下没有工具"是正常结果，不是错误。
        """

        raise NotImplementedError

    @abstractmethod
    async def tool_ids_for_keys(
            self,
            tool_keys: list[str],
    ) -> dict[str, UUID]:
        """
        tool_key 列表 -> `{tool_key: tool_id}`。

        执行侧要它：处理器链是按 `tool_id` 挂的（tool_processor 表），
        而模型只给得出 tool_key。

        **软缺失**：查不到的 key 不在结果里，不抛 —— 调用方的 key
        已经过 registry 的 `select_specs` 筛过，这里的缺口只可能来自
        "代码里有、库里没同步过"，那是部署顺序问题，应该在执行那一步
        显式记录下来（没有 tool_id 就是没有处理器链，工具照样能跑）。
        """

        raise NotImplementedError

    @abstractmethod
    async def tool_documentation_overrides(
            self,
            tool_keys: list[str],
    ) -> dict[str, dict[str, Any]]:
        """
        tool_key 列表 -> 人工写的说明覆盖，形状：

            {tool_key: {"description": str | None,
                        "parameters": {"/parent_id": str | None, ...}}}

        Agent 主循环要它：模型看到的工具说明由**代码侧事实**（docstring /
        声明）与**人工覆盖**叠成，而后者住在库里。读接口那条路
        （`list_tools` → `build_summary`）早就在做同一件事，但模型那条
        路走的是注册表（纯内存），够不着库 —— 于是同一份覆盖"列表页
        看得到、模型看不到"。这个方法就是补上那一格。

        ⚠️ 它**只给覆盖值**，不给生效值。合并成"生效值"是消费方的事
        （`_assembly.build_summary` 与 `infra/agent/tool_exposure.for_wire`
        各按自己的上下文回落到代码侧事实），而这里若先合并，两处就会
        有两份不同的生效口径。

        ⚠️ 空串与 None 都算"没有覆盖"，与读接口的 `override or source`
        判据一致。两处口径不同的话，"改完没生效"会以两种方式各出现一次。

        **裁掉没有任何真值覆盖的 key**：这个 dict 会进 `context_policy`
        并被 checkpoint，没有覆盖的工具塞进去只是让它白白变大。

        ⚠️ `return_description_override` **不在**返回值里，这是刻意的：
        provider 的函数 schema 只有 name / description / parameters 三格，
        没有"返回值说明"的位置，带上去也到不了模型。它只由
        `GET /tools/{id}` 呈现。写在这里是因为"为什么这个字段不在"
        必须在读代码时看得见，而不是让人以为漏了。

        **软缺失**：查不到的 key（代码里有、库里没同步过）直接不在
        结果里，不抛 —— 与 `tool_ids_for_keys` 同一条。
        """

        raise NotImplementedError
