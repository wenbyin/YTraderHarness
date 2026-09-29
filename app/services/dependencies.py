"""
泛型服务依赖：一个入口取代 N 个几乎一样的方法。

原来每个服务都要在各自模块的 dependencies.py 里写一个

    async def document_runtime_service(request: Request):
        factory = service_factory_of(request)
        async with factory.scope(DocumentRuntimeService) as service:
            yield service

8 个服务就是 8 份同样的样板 —— 除了契约类型以外没有任何区别，
而且每加一个服务都要再抄一遍（漏抄一个就是运行期 500）。

现在用泛型取代它们：

    from app.services.dependencies import Service

    @router.post("/documents")
    async def create_document(
            request: DocumentCreateRequest,
            service: DocumentRuntimeService = Service[
                DocumentRuntimeService
            ],
    ):
        return await service.create_document(request)

`Service[契约]` 展开成一个 `Depends(...)`，FastAPI 据此解析 ——
与原来的 `Depends(document_runtime_service)` 逐字等价，
只是那个函数不必再由人抄一遍。

为什么不再需要 Request
----------------------
工厂是**应用级**的（见 common/factories.py），一个应用一份，
不随请求变化。原来要从 `request.app.state` 里摸，现在直接
`get_service_factory()`。

这不只是少写一个参数 —— 它让同一套取用方式在**没有 Request 的
地方**也成立：流水线节点、后台任务、Agent 工具。那些地方原来
各自想办法，现在与请求路径走同一个入口。

为什么下标结果是个 Depends 而不是 Annotated
------------------------------------------
写成注解（`service: Service[契约]`）更省字，但**行不通**：
Python 不允许"无默认值的参数跟在有默认值的参数后面"，而这些
路由函数里 service 前面普遍是 `page_size: int = Query(...)` 之类。
把它挪到最前面要改 30 多处签名，且每个函数形状都得跟着变。

所以走默认值这条老路 —— 与既有写法一模一样：

    service: FileService = Depends(file_service)      # 改造前
    service: FileService = Service[FileService]       # 改造后

契约类型同样写了两遍，但这正是它改造前的样子（那时是类型一遍、
函数名一遍），而且注解保持为**契约本身**，IDE 与类型检查器看到的
就是 DocumentRuntimeService，补全与检查都不打折。

要开事务吗
----------
由工厂判断（ServiceFactory.needs_session）：实现里有仓储参数就开，
没有就不开。因此能力目录那种不碰库的服务不会白占一个连接，
而其余服务照旧拿到请求级事务。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Generic, TypeVar

from fastapi import Depends

from app.common.factories import get_service_factory
from app.common.service.base import BaseService

__all__ = [
    "Service",
    "service_dependency",
]

#: 服务契约的类型变量。上界是 BaseService —— 传进来的必须是服务契约，
#: 否则工厂那一步会以一句看不懂的报错失败。
_T = TypeVar("_T", bound=BaseService)


def service_dependency(
        contract: type[_T],
) -> Callable[[], AsyncIterator[_T]]:
    """
    产出一个"按契约取服务"的 FastAPI 依赖。

    这是底层原语；日常写法用 Service[契约] 泛型别名。

    开不开事务由工厂决定（见 ServiceFactory.needs_session），
    因此不碰库的服务不会被白开一段会话。
    """

    async def dependency() -> AsyncIterator[_T]:

        factory = get_service_factory()

        if factory.needs_session(contract):
            # 有仓储：开一段事务，退出时提交（异常则回滚）。
            async with factory.scope(contract) as service:
                yield service

        else:
            # 不碰库：只造实例，不占连接。
            yield factory.create(contract)

    return dependency


class Service(Generic[_T]):
    """
    泛型服务依赖：

        service: DocumentRuntimeService = Service[DocumentRuntimeService]

    下标求值返回一个 `Depends(...)`，因此它是**默认值**而不是注解 ——
    理由见模块 docstring（Python 的签名语法限制）。

    为什么不直接继承 Generic 让 typing 自己处理下标：
    默认行为返回一个泛型别名，FastAPI 会把它当成请求体模型而不是
    依赖。这里覆盖 __class_getitem__ 换成 Depends，依赖才挂得上去。
    """

    def __class_getitem__(cls, contract: type[_T]) -> object:
        return Depends(service_dependency(contract))
