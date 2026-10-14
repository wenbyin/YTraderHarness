"""
职责提示词的取用契约。

⚠️ 本契约**没有**对应的 HTTP 路由，也不返回 DTO
--------------------------------------------
唯一调用方是 Agent 侧（`app/agent/prompt/provider.py`），它要的是一段文本，
不是给人看的响应体。这与 `ToolQueryService.tool_keys_for_capabilities`
（app/services/tool/application/tool_query_service.py）是同一种形状：服务
不必总要对应一个端点。**没有端点，所以没有 prompt.yaml。**

这不代表它"不该是服务"：它将来会读库（那时才有仓储与事务），而"经
ServiceFactory 造出来"这件事现在就成立，调用方到那时一行都不用改。
"""

from __future__ import annotations

from abc import abstractmethod

from app.common.service.base import BaseService

__all__ = ["PromptService"]


class PromptService(BaseService):

    @abstractmethod
    async def get_duty_prompt(self, duty: str | None = None) -> str:
        """
        取一份职责提示词。

        Args:
            duty: 职责标识（如 `stock_research`）。**None = 默认职责** ——
                  这不是"省略参数"的语法糖，而是把"由谁来选职责"这件事
                  留在服务侧：当前主循环就是传 None 的，等职责有了选择依据，
                  改的是这里的解析，不是每个调用方。

        Raises:
            NotFoundError: 这个职责标识没有登记。
        """

        raise NotImplementedError

    @abstractmethod
    async def default_duty_key(self) -> str:
        """
        这次运行该用哪个职责 —— **「选职责」的唯一落点**。

        为什么单独一个方法，而不是让调用方自己拼 `DEFAULT_DUTY`
        ------------------------------------------------------
        调用方（`AgentRunServiceImpl.create_run`）要的是"给我一个 duty_key
        存起来"，它**不该知道**这个 key 是怎么来的。今天它恒等于默认职责，
        将来会按某个依据选（会话配置 / Agent 目录 / 别的），而那次改动应该
        只发生在**这里**，不在每个建运行的地方。

        这个名字也刻意不叫 `duty_for(org_id=...)`：org_id 与职责无关
        （前者是团队归属，见 agent_session.org_id）。真要有入参，加在这里，
        但别为了"看起来可扩展"先放一个没人用的参数。

        Returns:
            一个已登记的职责标识（可以直接喂给 `get_duty_prompt`）。

        Raises:
            NotFoundError: 选出来的 key 没登记（说明目录与选择逻辑漂了）。
        """

        raise NotImplementedError

    @abstractmethod
    async def duty_keys(self) -> tuple[str, ...]:
        """
        已登记的职责标识（排序稳定）。

        给两处用：观测/排查时"到底有哪些职责"，以及将来做选择时拿它当
        取值域 —— 与分类提示词从枚举派生取值域是同一条思路：**取值域不
        手抄**，抄一份就会漂移。
        """

        raise NotImplementedError
