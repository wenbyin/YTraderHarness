"""
TaskSubmitter 的运行时适配器：让 Handler 能派生子任务（spec §8）。

为什么需要一层适配
----------------
TaskContext 里放的是 `TaskSubmitter`（窄接口，只有 submit），而真正的
实现是 TaskService —— 但 Handler 执行时**手里没有** TaskService 实例：
Worker 是按需装配的（用 ServiceFactory.scope 现开事务），不存在一个
长期活着的全局 TaskService。

所以这里做一个薄适配：每次 submit 现开一个自己的作用域，拿一个
TaskService，提交完就还给工厂。

⚠️ 每次 submit 各自开事务，而不是复用 Handler 的那一个
---------------------------------------------------
Handler 跑在 `ServiceFactory.scope(业务契约)` 的事务里。子任务的提交
**不能**复用那个事务，两个理由：

1. 会话不同。Handler 的事务绑在它自己的 session 上，而 TaskService
   需要的是 TaskStore / TaskOutboxStore —— 硬把两者塞进一个事务要穿透
   工厂的装配，代价远大于收益。

2. 语义上本就该分开。子任务是"这次执行产生的后续动作"，它应当在
   **父任务执行成功之后**独立存在。

代价是一个已知的小口子：父任务的事务若在提交子任务之后回滚（Handler
随后抛错），子任务已经落库、会被执行 —— 于是留下一条"父任务失败了、
但它的后续动作跑了"的孤儿。它不会造成数据不一致（子任务本身是完整的），
但确实会比预期多跑一次。彻底消除它需要跨事务协调（两阶段提交之类），
而 spec §16 明确第一版不要那些；这里选择把它写清楚，而不是假装没有。
"""

from __future__ import annotations

import logging
from uuid import UUID

from app.common.service.factory import ServiceFactory
from app.common.task.models import Task
from app.services.task.application import TaskService

__all__ = ["ScopedTaskSubmitter"]

logger = logging.getLogger(__name__)


class ScopedTaskSubmitter:
    """
    实现 app.common.task.protocols.TaskSubmitter（结构上满足即可）。

    不做成 TaskService 的代理：它只需要 submit 一个方法，而"能取消别人
    的任务"恰恰是 Handler **不该**有的能力 —— 见 TaskContext 的说明。
    """

    def __init__(
            self,
            *,
            services: ServiceFactory,
    ) -> None:

        self._services = services

    async def submit(
            self,
            task: Task,
    ) -> UUID:
        """提交一条子任务，返回它的 task_id。

        没挂 dispatcher：投递靠 outbox relay 与兜底扫描（它们本来就要
        覆盖"提交后没人投递"的情况）。在这里再推一次属于优化，
        而它会额外增加一层依赖，不值得。
        """

        async with self._services.scope(TaskService) as service:

            return await service.submit(task)
