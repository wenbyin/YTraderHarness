"""
@tool —— 打在 service 方法上的自注册装饰器。

用法
----
    class DocumentQueryService(BaseService):        # 契约
        @abstractmethod
        async def list_documents(self, req): ...

    class DocumentQueryServiceImpl(DocumentQueryService):
        @tool(key="document.list")
        async def list_documents(self, req: ListDocumentsRequest):
            \"\"\"列出文档。\"\"\"
            ...

声明完就**已经注册好了** —— 没有人需要再去某个清单里补一行。

一个对象，两种身份
------------------
    langchain 工具   它是 StructuredTool 的实例 —— bind_tools() / ToolNode
                     直接认，模型看到的就是它
    描述符           它的类型定义了 __get__ —— instance.method 照常是
                     绑定方法，既有调用点一行都不用改

两件事必须同时成立，否则要么模型用不了，要么代码调不了。

为什么不能只用 langchain 的 @tool
---------------------------------
langchain 的 @tool 打在方法上会**吃掉这个方法**：`s.method` 变成
StructuredTool，不再是绑定方法（实测 BaseTool 既没有 __get__ 也没有
__set_name__）。那会打断全部走 `service.method(...)` 的既有调用点 ——
API 路由与服务间调用都在这条路上。

所以这里把它做成**描述符**：

    __set_name__   装饰器执行时**类还不存在**，只有这一步能拿到
                   owner（Impl 类）—— 契约要从它的 MRO 推
    __get__        实例访问返回绑定方法（调用点不变）
                   类访问返回工具本身（bind / 自省用）

顺带解决了身份推导：`source_ref`（业务模块归属）与 `contract`
（工厂要的抽象基类）都只能从 owner 推，而 langchain 的 @tool
不参与描述符协议，拿不到 owner。

⚠️ 代价：声明层现在硬依赖 langchain
------------------------------------
本模块以前只依赖注册表（"任何 service 模块 import 它都很轻"）。
现在它必须继承 StructuredTool，于是**每个带工具的 service 模块
import 时都会构造 pydantic 工具模型**。

这是刻意的取舍：模型的工具体系就是 langchain 那套，声明不变成工具，
就得在执行期另造一个影子工具（那样 schema 与声明会漂成两份）。

schema 从哪来
-------------
`args_schema` 喂的是**算好的 JSON Schema dict**（来自方法的请求模型），
不是让 langchain 从签名推。langchain 对 dict 有专门的路径：
它原样返回、不构造子集模型、也不剔注入参数（见
langchain_core/tools/base.py 的 tool_call_schema）。

⚠️ **不喂会静默出错**：args_schema 为 None 时它会从 `_run` 的签名
瞎推，产出 `{args, config, kwargs}` 那种垃圾 schema —— 模型据此
填参数，而且没有任何报错。因此这个参数是必填的，不是可选的。

人工覆盖（tool_documentation）**不在这里合并**
----------------------------------------------
装饰发生在 import 期，那时读不到库。这里只用**代码侧事实**
（docstring / 声明）。人工覆盖由准备工具清单的那一层叠上来
（`tool.description` 与 `tool.args_schema` 构造后仍可赋值，
`tool_call_schema` 因此自动跟着变）。

执行不走这里
------------
`_run` / `_arun` 是**明确的占位**（抛 NotImplementedError 并指向
该去的地方）。真正的执行在 `app/infra/agent/tool_executor.py`：
它按 tool_id 取处理器链、驱动链、再用 factory.create(contract)
造出 service 调方法。

为什么不让描述符自己执行
------------------------
声明层要能反向依赖 service 工厂与处理器链，就等于把"声明一个工具"
的代价从"建一个 pydantic 模型"变成"拖进整个执行栈"。而且工具要跑的
不只是方法本身 —— 还有链上的拦截 / 转换 / 观测，那些不属于工具。
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import PrivateAttr

from .registry import get_tool_registry
from .schema import resolve_request_model, synthesize_schema
from .spec import ToolSpec, derive_source_ref

__all__ = [
    "ToolDescriptor",
    "tool",
]


def _input_schema_of(func: Callable[..., Any]) -> dict[str, Any]:
    """
    方法 -> 它收的那个请求模型的 JSON Schema。

    与 scanner.py 里 ServiceMethodScanner 用的是**同一套**推导
    （resolve_request_model / synthesize_schema）：两处必须一致，
    否则"绑定给模型的 schema"与"同步进库的 schema"会漂成两份，
    而它们本该是同一个东西。
    """

    request = resolve_request_model(func)

    if request is not None:
        return request[1].model_json_schema()

    return synthesize_schema(func)


class ToolDescriptor(StructuredTool):
    """
    一个被 @tool 标记的方法 —— 同时是 langchain 工具与描述符。

    构造走的是 `tool()`，不直接实例化：KeyError 之类的参数校验在
    那边做，这里只管"成为工具"与"成为描述符"。
    """

    #: 声明里给的三个值。存成私有属性 —— 它们是**声明**，
    #: 不是工具对外暴露的字段（工具那边叫 name / description）。
    _key: Any = PrivateAttr(default=None)
    _declared_name: Any = PrivateAttr(default=None)
    _declared_description: Any = PrivateAttr(default=None)

    #: owner 与 ToolSpec 在 __set_name__ 里才有。
    _owner: Any = PrivateAttr(default=None)
    _attr: Any = PrivateAttr(default=None)
    _spec: Any = PrivateAttr(default=None)

    # ========================================================
    # 描述符协议
    # ========================================================

    def __set_name__(self, owner: type, name: str) -> None:
        """
        类体执行完的那一刻被调用 —— 这是唯一能拿到 owner 的时机。

        注册就发生在这里：import 这个模块 = 工具进了注册表。
        """

        self._owner = owner
        self._attr = name

        target = self._method()

        assert target is not None, (
            f"{owner.__name__}.{name} 的 @tool 没有收到函数 —— "
            "请确认 @tool 是直接打在方法上的。"
        )

        # source_ref 一律**推导**，不接受声明 —— 它是"这个工具
        # 从哪来"的事实，由来源决定，不是作者的选择。
        source_ref = derive_source_ref(owner.__module__)

        self._spec = ToolSpec(
            tool_key=self._key or f"{source_ref}.{name}",
            name=self._declared_name or name,
            owner=owner,
            method_name=name,
            func=target,
            description=self._declared_description,
            source_ref=source_ref,
        )

        # 工具名 = tool_key。它是模型看到的函数名，也是链与
        # tool_definition 的查找键 —— 三者必须是同一个字符串，
        # 否则"模型调用的工具"与"库里配了处理器的工具"对不上。
        self.name = self._spec.tool_key

        # 描述**合成一处**：取 spec 的生效值，而不是在这里再算一遍
        # "声明优先、否则 docstring"。两处各算一遍的话，声明为空串
        # 时两边会给出不同结果（一处回落 docstring、一处停在空串），
        # 而"模型看到的描述"与"接口返回的描述"不该有两个来源。
        self.description = self._spec.effective_description or ""

        get_tool_registry().register(self._spec)

    def __get__(
            self,
            obj: Any,
            objtype: type | None = None,
    ) -> Any:

        # 类访问：给扫描、自省、以及 bind_tools / ToolNode 用。
        if obj is None:
            return self

        # 实例访问：返回绑定方法，服务方法照常可用。
        return self._method().__get__(obj, objtype)

    # ========================================================
    # 访问器
    # ========================================================

    @property
    def spec(self) -> ToolSpec | None:
        return self._spec

    @property
    def owner(self) -> type | None:
        return self._owner

    @property
    def attr(self) -> str | None:
        return self._attr

    def _method(self) -> Callable[..., Any] | None:
        """
        被装饰的那个**未绑定**方法。

        同步方法进 `func`、异步进 `coroutine` —— StructuredTool 按
        这个分了两条执行路径，我们两边都得认。服务方法实际都是
        async，同步那条只是不让人写出来就炸。
        """

        return self.coroutine or self.func

    # ========================================================
    # 执行入口（占位）
    # ========================================================

    async def _arun(self, **kwargs: Any) -> Any:
        """
        本描述符**不负责执行**，见模块 docstring 的"执行不走这里"。

        刻意显式抛错而不是留空：留空的话基类会拿 kwargs 去调那个
        未绑定方法，报的是"缺少参数 self"—— 那句话指向不了真正
        的原因，排查要从这里绕一圈。

        ⚠️ 但这**不是**"工具还不能跑"：能跑，只是入口不在描述符上。
        直接 `await tool.ainvoke(...)`（例如被 langgraph 的 ToolNode
        驱动）会撞到这里，绕开处理器链 —— 而那正是我们要避免的：
        链上可能挂着审批与脱敏，绕过它执行比失败更糟。
        """

        raise NotImplementedError(
            f"{self.name} 不通过描述符执行。"
            "工具的执行入口是 app/infra/agent/tool_executor.py："
            "它按 tool_id 取处理器链（拦截 / 转换 / 观测）、驱动链，"
            "再用 factory.create(contract) 造出 service 调方法。"
            "直接 ainvoke 会绕过整条链。"
        )

    def _run(self, **kwargs: Any) -> Any:
        raise NotImplementedError(
            f"{self.name} 不通过描述符执行（同步路径同理，"
            "见 _arun 的说明）。"
        )

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return (
            f"<tool {self.name!r} "
            f"{self._owner.__name__ if self._owner else '?'}"
            f".{self._attr}>"
        )


def tool(
        func: Callable[..., Any] | None = None,
        /,
        *,
        key: str | None = None,
        name: str | None = None,
        description: str | None = None,
) -> Any:
    """
    把一个 service 方法声明成工具。

    两种写法都支持：

        @tool                                 key 由 模块.方法名 推出
        @tool(key="document.list")            显式声明

    key 建议显式给：它是跨版本稳定的业务标识，同步按它比对，
    而推导出来的 key 会随模块改名/挪包而变 —— 那会让一个
    原本 unchanged 的工具突然变成"消失 + 新建"。

    只有三个参数，都是"工具是什么"：身份、显示名、说明。
    tags / 分类 / 读写标记不在这里 —— 见模块 docstring。
    读写区分请用分类体系（建一个 read / write 分类挂上去），
    那样它有列、能筛选、能被接口读出来。
    """

    if func is not None:
        return _build(func, key, name, description)

    def decorate(inner: Callable[..., Any]) -> ToolDescriptor:
        return _build(inner, key, name, description)

    return decorate


def _build(
        func: Callable[..., Any],
        key: str | None,
        name: str | None,
        description: str | None,
) -> ToolDescriptor:
    """
    造出"既是工具又是描述符"的那个对象。

    直接构造而不是走 `StructuredTool.from_function`：那个 classmethod
    还要处理 infer_schema / parse_docstring / response_format 一堆
    我们不需要的推断，而 args_schema 是我们算好的，推断一律不要。
    """

    # 同步 / 异步分两条字段 —— StructuredTool 就是这么分的。
    callable_field = (
        {"coroutine": func}
        if inspect.iscoroutinefunction(func)
        else {"func": func}
    )

    descriptor = ToolDescriptor(
        **callable_field,
        # 先占一个名字，__set_name__ 里会换成 tool_key（那时才知道
        # source_ref）。name 是必填字段，没有"先留空"这个选项。
        name=name or func.__name__,
        # description 也是必填（str，不是 str | None）。声明优先，
        # 其次 docstring；两样都没有就是空串 —— 那确实是个漏写，
        # 但报错的时机应该由工具目录那边决定，不是 import 期。
        description=(
            description
            if description is not None
            else inspect.getdoc(func)
        ) or "",
        # ⚠️ 必填。不喂的话 langchain 会从 _run 的签名瞎推出一份
        # 垃圾 schema，而且不报错（见模块 docstring）。
        args_schema=_input_schema_of(func),
    )

    descriptor._key = key
    descriptor._declared_name = name
    descriptor._declared_description = description

    return descriptor
