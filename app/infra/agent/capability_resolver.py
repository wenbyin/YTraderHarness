"""能力 code -> tool_key 的解析实现。

SkillLoader 需要把 `skill_capabilities ∪ task.capabilities` 变成
`context_policy.tool_keys`。这里复用工具模块的
`ToolQueryService.tool_keys_for_capabilities` —— 与 Agent Builder 的
能力->工具解析是同一个来源，避免两条路给出不同集合。

⚠️ 只做“看得到哪些工具”
----------------------
它不判权限、不执行。工具执行仍走 ToolExecutor 的处理器链
（checkauth / approval）。
"""

from __future__ import annotations

from collections.abc import Sequence

from app.agent.context.processor.sources import CapabilityResolver

__all__ = ["ServiceCapabilityResolver"]


class ServiceCapabilityResolver(CapabilityResolver):
    """按 capability code 查可绑定的 tool_key。"""

    async def tool_keys(self, codes: Sequence[str]) -> list[str]:

        usable = [
            str(code).strip()
            for code in codes
            if str(code).strip()
        ]

        if not usable:
            return []

        # 与 agent_builder_node._tool_keys_for_capabilities 同一套
        # mock 开关：前端联调不保证 capability/tool 种子齐全。
        from app.mock.settings import is_mock_enabled

        if is_mock_enabled():
            from app.mock.capability import select_mock_tool_keys

            return list(select_mock_tool_keys(usable))

        from app.common.factories import get_service_factory
        from app.services.tool.application import ToolQueryService

        factory = get_service_factory()

        async with factory.scope(
            ToolQueryService, begin=False
        ) as service:
            keys = await service.tool_keys_for_capabilities(usable)

        return list(keys or ())
