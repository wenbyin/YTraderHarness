"""
MemorySceneService —— L2（memory_scene）：把 L1 记忆整合成场景文件。

**没有 HTTP 端点。** 这个服务只有一条路：`memory.l2.extract` 任务的
handler。它的调用方是 Worker，不是请求（见 app/bootstrap/task.py 的注册处）。

与 `MemoryAtomService`（L1）分开而不是并进去
--------------------------------------
两者的**输入与产物都不同**：

    L1   输入一个会话里还没抽过的那一段消息，产物是结构化的原子记忆，
         按会话收窄、**原地改写 + 破坏性删除**
    L2   输入一个会话里还没整合过的那一批 L1 记忆，产物是**整篇 markdown**
         的场景文件，按 scope（跨会话）聚合、写回带乐观锁

并在一起会让 L1 的抽取路径依赖"场景文件"这一整套概念，而那一条路
（`memory.l1.extract`）是纯抽取 —— 它连画像表都不该知道。

⚠️ 也是"没有检索"的一层
-------------------
L0/L1 的服务都有一对 `search`（HTTP + 工具）。L2 没有：场景文件当前是
**整段拼进提示词**的（那正是本服务做的事），还没有任何按内容搜画像的
入口。真要做（给主模型一个"这个用户的画像是什么"的工具）是另一条改动，
落点在这里加一个 `search` —— 而不是在抽取路径上预留参数。
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from app.common.service.base import BaseService

from ..domain.tasks import MemorySceneExtractParams

__all__ = ["MemorySceneService"]


class MemorySceneService(BaseService):

    @abstractmethod
    async def memory_scene_extract(
            self,
            params: MemorySceneExtractParams,
            ctx: Any,
    ) -> None:
        """
        `memory.l2.extract` 的 handler 主体。

        ⚠️ 签名必须是 `(params, ctx)` 两个位置参数 —— 注册期会校验，
        少了它在**启动时**报错（见 app/infra/task/registry.py）。这一条
        当前用不到 ctx（不派生子任务），但契约里必须留着它。

        这一步做完什么
        -----------
            按水位取 L1 增量 -> 解析 scope -> 读该 scope 的现有场景
            -> 渲染提示词 -> 调模型 -> 解析写入清单 -> 归一化文件名
            -> 乐观锁写回 + 删掉标记为 [DELETED] 的
            -> **推进水位**（与上面同一个事务）
            -> 让 `trigger.submit_persona_extract` 排一条 L3 生成任务
               （**不判定**：该不该真生成由执行侧判；**失败不抛**，见那里）

        什么算成功、什么算失败
        ------------------
            没有 L1 增量            正常结束（只把 last_l2_at 戳一下）
            会话没有归属            正常结束（打印一行说明，见实现）
            模型调用抛              抛出去，让任务重试
            输出解析不了            抛出去，让任务重试
            内容与库里完全一致      跳过写入，不推 version
            乐观锁冲突              抛 ConflictError，让任务重入
            一切成功                推水位，提交

        ⚠️ 水位**只在最后推**，且与画像行同事务 —— 提前推会让那批 L1
        永远进不了画像，而它没有任何症状（下次按 `> 水位` 取增量）。
        """

        raise NotImplementedError
