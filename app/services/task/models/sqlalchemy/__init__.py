"""
Task 模块的 SQLAlchemy 模型。

import 本包 = tasks 与 task_outbox 两张表注册完毕。

⚠️ 新增本包时**必须**在 app/infra/datasource/initializer.py 的
import_all_models() 里加一行 —— 那个函数硬编码了要 import 的模型包列表
（它自己的 docstring 里也承认了这点）。漏加的症状是：表不会被建出来，
而且**启动不报错**，直到第一条 Task 提交时才炸出
"relation \"tasks\" does not exist"。
"""

from .task import Task, TaskModel
from .task_outbox import OutboxModel, TaskOutbox

__all__ = [
    "OutboxModel",
    "Task",
    "TaskModel",
    "TaskOutbox",
]
