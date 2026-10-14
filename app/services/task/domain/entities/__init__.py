"""
Task 模块的领域实体。

⚠️ Task 实体**不在这里定义**，它由 app/common/task/models.py 提供
------------------------------------------------------------------
其余模块（如 capability）的实体定义在 `domain/entities/` 下，本模块
刻意不照办：Task 同时是

    - 业务构造后提交的对象
    - Worker 交给 Handler 的对象
    - TaskContext 持有的对象

而 Worker 住在 app/infra/task，它**不允许** import app.services.*
（依赖方向见 app/common/task/__init__.py）。Task 若定义在 services 下，
要么 infra 反向依赖 services，要么每个跨界处都做一次无意义的类型转换。

所以 Task 定义在共同抽象层，这里只把它**再导出**一次 —— 于是
`from ..domain.entities import Task` 这种与其余模块一致的写法依然成立，
不需要谁去记住"task 模块的特殊情况"。

TaskOutbox 是本模块自己的实体：投递记录只有本模块与 infra 的 relay
关心，没有跨层的共同性，按惯例留在 domain/entities 下。
"""

from app.common.task.models import RawTaskParams, Task

from .task_outbox import TaskOutbox

__all__ = [
    "RawTaskParams",
    "Task",
    "TaskOutbox",
]
