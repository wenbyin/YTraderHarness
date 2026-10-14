"""
Task 基础设施自己的异常。

为什么在 app.common 而不是 infra/task
------------------------------------
app.common.task 是三个方向共同的依赖终点：

    common/task  ←  services/task  ←  infra/task

TaskStore / Dispatcher / Worker / Registry 都会抛这里的异常，而它们分处
两层。异常若定义在下游，上游就不得不反向 import —— 而
app/common 必须保持纯叶子（见 app/common/exceptions.py 的模块说明）。

与 app.common.exceptions 的分工
------------------------------
那边的异常描述"这次**请求**怎么了"（404 / 409 / 401），由 HTTP 层映射成
状态码；这里的异常描述"这条 **Task** 怎么了"，由 Worker 用来决定
"再试一次还是判死"。

两者会在一个地方相遇：Task 查不到时查询接口照样要给 404。所以那种情况
直接复用那边的 NotFoundError，不在这里另起一个同名概念。
"""

from __future__ import annotations

from app.common.exceptions import ApplicationError, ConflictError

__all__ = [
    "TaskError",
    "TaskParamsError",
    "TaskPermanentError",
    "TaskStateError",
    "UnknownHandlerError",
]


class TaskError(ApplicationError, ValueError):
    """
    Task 基础设施的异常基类。

    同时继承 ValueError 的理由与 app.common.exceptions 里那批一样：
    路由层与既有调用方的 `except ValueError` 继续可用。
    """


class UnknownHandlerError(TaskError):
    """
    Task.handler 指向一个从未注册过的 Handler。

    这是**配置错误**，不是暂时性故障：注册表在启动时就固定了，
    重试多少次结果都一样。Worker 因此直接判 FAILED，不消耗 attempt。

    只在"确实没注册"时抛。Handler 自己执行时抛出的异常不算 ——
    那属于执行失败，见 TaskRetryableError。
    """


class TaskParamsError(TaskError):
    """
    提交进来的参数与 Handler 注册的参数类型对不上。

    与 pydantic 的 ValidationError 分开，是因为它抓的是**更隐蔽**的一类：
    字段**没被拒绝**，但也没被接收。参数类是普通 BaseModel 时默认
    extra="ignore"，于是多出来的字段被静默丢弃，Handler 收到一份缺了
    参数的输入 —— 不报错，只是行为不对。

    这类问题与 UnknownHandlerError 同性质：**提交方与注册方对不上**，
    属于配置/调用错误，重试多少次都一样。Worker 因此直接判 FAILED，
    不消耗 attempt（见 infra/task/retry.py 的致命异常名单）。
    """


class TaskStateError(TaskError, ConflictError):
    """
    当前状态不允许这个操作。

    cancel 一条已经 SUCCESS 的 Task、claim 一条正在 RUNNING 的 Task、
    retry 一条已经到 max_attempts 的 Task，都会抛它。

    继承 ConflictError 而不是只继承 TaskError：对 HTTP 层来说它就是 409，
    与能力树里"还有子节点所以不能删"是同一类冲突。查询接口因此不必为它
    单写一个 except 分支，而 `except TaskError` 也照样接得住。
    """


class TaskPermanentError(TaskError):
    """
    Handler 显式声明"这次失败是永久性的，别浪费重试次数"。

    为什么是"声明永久"而不是"声明可重试"
    ----------------------------------
    Worker 的默认是**重试**（受 attempt < max_attempts 约束，见
    infra/task/retry.py）：Handler 的作者没想到重试这回事时，至少还能
    拿到几次重试，而不是静默地一次就死。

    在那个默认下，"请求重试"的异常是**空的 API** —— 抛它和不抛它
    行为一样。真正缺的是反方向：Handler 明明知道这次不该再试
    （上游返回 400、单据已经作废、状态机不允许），却只能眼看着它
    重试三次、每次间隔指数增长，把终态推迟好几分钟。

    所以这里提供一个说"不"的口子，而不是一个说"是"的。
    """
