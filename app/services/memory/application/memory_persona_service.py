"""
MemoryPersonaService —— L3（memory_persona）：把变化的场景抽象成一整篇画像。

**没有 HTTP 端点。** 这个服务只有一条路：`memory.l3.extract` 任务的
handler。它的调用方是 Worker，不是请求（见 app/bootstrap/task.py 的注册处）。

与 `MemorySceneService`（L2）分开而不是并进去
--------------------------------------
两者的**键与产物都不同**：

    L2   输入一个会话里还没整合过的那批 L1 记忆，产物是**十几篇**场景
         文档，按 scope（跨会话）聚合、写回带乐观锁，水位按**会话**
    L3   输入一个 scope 下变化过的那批场景，产物是**一篇**画像，
         水位按 **scope**

并在一起会让 L2 的整合路径依赖"画像已经存在"这一整套概念，而那条路
（`memory.l2.extract`）的产物是场景 —— 它连画像表都不该知道（反过来也不
成立：L3 确实要读场景，那是它的输入）。

⚠️ 两份系统提示词，一个 model_role
-----------------------------
scope 带 org 的走团队 Operating Doctrine，不带 org 的走个人 User Narrative
Profile。选哪一份的唯一落点见 `prompts/memory_persona_extract.py` 的
`_render` —— 这里只把 `is_team` 传进去。

⚠️ 也是"没有检索"的一层
-------------------
与 L2 一样：画像当前**没有任何按内容搜的入口**，它整段拼进提示词（那是本
服务在做的另一半）。真要做（给主模型一个"这个用户是谁"的工具）是另一条
改动，落点在这里加一个 `search` —— 而不是在生成路径上预留参数。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.common.service.base import BaseService

from ..domain.tasks import MemoryPersonaExtractParams

__all__ = ["MemoryPersonaService"]


class MemoryPersonaService(BaseService):

    @abstractmethod
    async def memory_persona_extract(
            self,
            params: MemoryPersonaExtractParams,
            ctx: Any,
    ) -> None:
        """
        `memory.l3.extract` 的 handler 主体。

        ⚠️ 签名必须是 `(params, ctx)` 两个位置参数 —— 注册期会校验，
        少了它在**启动时**报错（见 app/infra/task/registry.py）。这一条
        当前用不到 ctx（不派生子任务），但契约里必须留着它。

        这一步做完什么
        -----------
            按 scope 水位取变化场景 -> 读现有画像与场景总数
            -> 渲染提示词（按 is_team 选系统提示词）-> 调模型
            -> 解析 {no_change, content} -> 乐观锁写回画像
            -> **推进水位**（与上面同一个事务）

        什么算成功、什么算失败
        ------------------
            没有变化的场景            正常结束（只把 last_l3_at 戳一下）
            模型判 no_change          不写库，但**推场景水位**（这批已经看过）
            内容与库里完全一致        跳过写入，不推 version，**推场景水位**
            模型调用抛                抛出去，让任务重试
            输出解析不了              抛出去，让任务重试
            乐观锁冲突                抛 ConflictError，让任务重入
            一切成功                  写画像 + 推水位，提交

        ⚠️ 水位**只在最后推**，且与画像行同事务 —— 提前推会让那批场景
        永远进不了画像，而它没有任何症状（下次按 `> 水位` 取增量）。
        """

        raise NotImplementedError
