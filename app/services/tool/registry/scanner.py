"""
扫描器：一条来源 = 一个扫描器。

为什么要拆成多个
----------------
"扫描"不是一个动作而是**一类**动作。本进程里的工具至少有两种
写法，取数方式完全不同：

    @tool 服务方法        注册表里已经有（import 时自注册的）
    langchain BaseTool    要遍历模块属性去找

将来还有 MCP server、OpenAPI spec —— 那些要发网络请求、读文件。
把它们塞进一个函数里，很快就会变成一堆 if。

拆开之后，新增一种来源 = 加一个扫描器，注册表与同步逻辑都不动。
这与 domain/schemas/enum.py 的 SourceType 是同一个切分轴：
枚举先占好 mcp / openapi 的值，就是为了这里。

统一出口是 ToolManifestData（纯数据）—— 扫描器不碰数据库、
不知道有没有同步过。
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from ..domain.schemas.enum import SourceType
from .docstring import parse_docstring, resolve_returns
from .registry import TOOL_SOURCE_PACKAGES, ToolRegistry, iter_package_modules
from .schema import (
    extract_parameter_descriptions,
    output_schema_of,
    resolve_request_model,
    strip_injected_args,
    synthesize_schema,
)
from .spec import (
    SOURCE_REF_VARIABLE,
    ToolManifestData,
    ToolSpec,
    derive_source_ref,
)

__all__ = [
    "LangchainToolScanner",
    "ServiceMethodScanner",
    "ToolScanner",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class ToolScanner(Protocol):
    """
    一条工具来源。

    实现只需回答一个问题："现在这个进程里，你这条来源有哪些工具？"
    过滤（按 source_type / source_ref）由编排层统一做，
    扫描器不必各自实现一遍。
    """

    def scan(self) -> list[ToolManifestData]:
        ...


# ============================================================
# 来源 1：@tool 声明的服务方法
# ============================================================


class ServiceMethodScanner:
    """
    从 ToolRegistry 里取 —— 那些 @tool 方法在 import 时就自注册好了。

    这里调 scan() 是为了触发**包遍历**：某个 service 模块可能恰好
    没被任何路径 import 到，那样它就不在注册表里，而"进程 import 过
    什么"是运行期状态，不该影响"代码库里有哪些工具"这个事实。
    """

    source_type = SourceType.LOCAL_METHOD

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    def scan(self) -> list[ToolManifestData]:
        return [
            self._to_manifest(spec)
            for spec in self._registry.scan()
            # ⚠️ 模块级 langchain 工具**跳过**：它们由
            # LangchainToolScanner 负责产出 manifest，而两个扫描器
            # 产出的是同一个 tool_key —— `ToolDiscoverer.discover()`
            # 对"同一个 key 被两条来源声明"是**直接抛错**的，
            # 不跳过就会让同步接口整个崩掉。
            #
            # 注册表里两种都收（select_specs 需要它们都在），
            # 但"同步的产出"只能由一条来源负责。
            if spec.owner is not None
        ]

    @staticmethod
    def _to_manifest(spec: ToolSpec) -> ToolManifestData:
        """一个 ToolSpec -> 一份事实快照。"""

        request = resolve_request_model(spec.func)

        input_schema = (
            request[1].model_json_schema()
            if request is not None
            else synthesize_schema(spec.func)
        )

        output_schema = output_schema_of(spec.func)

        return ToolManifestData(
            tool_key=spec.tool_key,
            name=spec.name,
            source_type=SourceType.LOCAL_METHOD,
            source_ref=spec.source_ref,
            entrypoint=spec.entrypoint,
            input_schema=input_schema,
            output_schema=output_schema,
            source_docstring=_docstring(
                spec.effective_description,
                input_schema,
                output_schema,
            ),
        )


# ============================================================
# 来源 2：langchain 原生的工具
# ============================================================


class LangchainToolScanner:
    """
    扫模块级的 langchain `BaseTool` 实例。

    覆盖两种常见写法 —— 它们在运行时是同一种东西：

        @tool                                   # langchain 的
        def search(query: str) -> str: ...

        search = StructuredTool.from_function(...)

    两者在模块命名空间里都是 BaseTool 实例，因此一个 isinstance
    就能都捞到。

    ⚠️ 扫不到"挂在类方法上的 langchain @tool"
    那种写法本身是坏的（见 decorator.py 的说明：它会把方法变成
    StructuredTool，`self.method` 不再可调用），所以不是漏扫。

    ⚠️ 与 @tool 服务方法的区别在**执行**，不在发现
    这类工具没有 owner 类，也就没有可以交给工厂去建的 self ——
    调用时直接调它自己即可。执行差异由 app/infra/agent 那层适配，
    扫描阶段两者产出同一种快照。
    """

    source_type = SourceType.LOCAL_METHOD

    def __init__(
            self,
            packages: tuple[str, ...] | None = None,
    ) -> None:
        self._packages = (
            packages if packages is not None else TOOL_SOURCE_PACKAGES
        )

    def scan(self) -> list[ToolManifestData]:

        found: dict[str, ToolManifestData] = {}

        for value, module, attr_name in iter_langchain_tools(
                self._packages
        ):
            manifest = self._to_manifest(value, module, attr_name)

            existing = found.get(manifest.tool_key)

            if existing is not None:
                # 两个不同的对象抢同一个 key —— 静默取其一比报错
                # 危险，与 ToolRegistry.register 同一套取舍。
                raise ValueError(
                    f"tool_key {manifest.tool_key!r} 被两个 langchain "
                    f"工具占用：{existing.entrypoint} 与 "
                    f"{manifest.entrypoint}。请给其中一个改 name。"
                )

            found[manifest.tool_key] = manifest

        return list(found.values())

    @staticmethod
    def _to_manifest(
            value: Any,
            module: Any,
            attr_name: str,
    ) -> ToolManifestData:
        """一个 BaseTool -> 一份事实快照。"""

        name = getattr(value, "name", None) or attr_name
        description = getattr(value, "description", None) or None

        module_name = module.__name__

        # 模块顶部写了 TOOL_SOURCE_REF 就用它。
        #
        # 这是组合工具唯一能指定 source_ref 的途径 —— langchain 的
        # @tool 不接受我们的任何参数，声明不到装饰器上去。
        # 不写就推导（见 spec.derive_source_ref）。
        declared = getattr(module, SOURCE_REF_VARIABLE, None)

        source_ref = (
            declared
            if isinstance(declared, str) and declared
            else derive_source_ref(module_name)
        )

        input_schema = _args_schema_of(value)

        return ToolManifestData(
            tool_key=f"{source_ref}.{name}",
            name=name,
            source_type=SourceType.LOCAL_METHOD,
            source_ref=source_ref,
            entrypoint=f"{module_name}:{attr_name}",
            input_schema=input_schema,
            # langchain 没有统一的"输出 schema"概念（返回值往往是
            # 字符串），因此留空而不是编一个 —— returns 于是只能
            # 回落到 description 里的 Returns 段（通常也没有）。
            output_schema=None,
            source_docstring=_docstring(
                description, input_schema, None
            ),
        )


# ============================================================
# 共用
# ============================================================


def _docstring(
        raw: str | None,
        input_schema: dict[str, Any],
        output_schema: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """
    整理成结构，而不是留一整段字符串。

    参数说明取自 input_schema（pydantic 字段说明 / langchain 的
    args_schema），这样它与文档覆盖用的 JSON Pointer 天然对得上。

    description / returns 由 docstring 切段得到（见 docstring.py），
    是前端直接要的两个键；returns 还要过一道"DTO 优先"。

    raw 一直保留：它是原文，人工覆盖与排查都以它为准，切段只是
    在它旁边多给几个**读得出来**的入口。
    """

    parameters = extract_parameter_descriptions(input_schema)

    sections = parse_docstring(raw)

    returns = resolve_returns(
        sections.get("returns"), output_schema
    )

    if not raw and not parameters and not returns:
        return None

    payload: dict[str, Any] = {
        "raw": raw,
        "parameters": parameters,
    }

    if sections.get("description"):
        payload["description"] = sections["description"]

    if returns:
        payload["returns"] = returns

    return payload


def _iter_modules(packages: tuple[str, ...]):
    for package_name in packages:
        yield from iter_package_modules(package_name)


#: 供 `ToolRegistry.scan()` 复用同一份发现逻辑 —— 它是**运行时**那一侧，
#: 本模块 import 了 registry，所以那边只能**惰性** import 它。
def iter_langchain_tools(
        packages: tuple[str, ...] | None = None,
) -> Iterator[tuple[Any, Any, str]]:
    """
    遍历包，产出模块级的 langchain 工具：(工具本体, 所属模块, 属性名)。

    ⚠️ 两个消费方，必须是**同一份**发现逻辑
    -------------------------------------
        LangchainToolScanner   同步：产出 manifest 写进 tool_definition
        ToolRegistry.scan()    运行时：登记进注册表，select_specs 才看得见

    各写一份遍历的话，两处对"哪些算模块级工具"的判定早晚会分叉 ——
    而分叉的症状正是这个函数出现之前那个 bug：库里明明有这条工具、
    模型却永远拿不到（同步看得到，运行时看不到）。

    同一个对象被两个名字引用（别名、re-export）时只产出一次，
    否则一个工具会在库里出现两行。

    没装 langchain 就安静跳过 —— 管理功能（同步、文档、分类）不该
    因为少一个可选依赖就跑不起来。
    """

    base_tool = _import_base_tool()

    if base_tool is None:
        return

    seen: set[int] = set()

    for module in _iter_modules(
            packages if packages is not None else TOOL_SOURCE_PACKAGES
    ):

        for attr_name, value in vars(module).items():

            if not isinstance(value, base_tool):
                continue

            if id(value) in seen:
                continue

            seen.add(id(value))

            yield value, module, attr_name


def _import_base_tool() -> type | None:
    """
    延迟取 langchain 的 BaseTool。

    延迟是为了让 app/services/tool 在**没有 langchain** 的环境里
    照样能 import —— 同步、文档、分类这些管理功能不该被一个
    可选依赖绑住。取不到就返回 None，扫描器据此跳过。
    """

    try:
        from langchain_core.tools import BaseTool

    except ImportError:
        logger.info(
            "未安装 langchain_core，跳过 langchain 工具扫描。"
        )
        return None

    return BaseTool


def _args_schema_of(value: Any) -> dict[str, Any]:
    """
    取 langchain 工具**给模型看**的参数 schema。

    ⚠️ 优先用 `tool_call_schema`，不是 `args_schema`
    --------------------------------------------
    `args_schema` 是**执行侧**的完整模型，包含 `ToolRuntime` /
    `InjectedState` 这类注入参数；它甚至无法生成 JSON Schema
    （ToolRuntime 内部有 Callable 字段，pydantic 会直接抛）。

    `tool_call_schema` 是 LangChain 为"模型调用"准备的那一份，
    已经剔除了直接注入参数。这里再做一次 `strip_injected_args` 兜底
    （例如 service 方法的合成 schema），保证落库 / 展示的那份
    里没有 `runtime`。

    schema 有三种形态，都要认：
        pydantic 模型类   -> model_json_schema()
        dict              -> 原样用
        None              -> 空对象（无参数工具）
    """

    func = (
        getattr(value, "func", None)
        or getattr(value, "coroutine", None)
    )

    schema = _schema_dict(getattr(value, "tool_call_schema", None))

    if schema is None:
        schema = _schema_dict(getattr(value, "args_schema", None))

    if schema is None:
        return {}

    return strip_injected_args(schema, func) if func else schema


def _schema_dict(schema: Any) -> dict[str, Any] | None:
    """pydantic 模型类 / dict -> JSON Schema dict；认不出返回 None。"""

    if isinstance(schema, dict):
        return schema

    if isinstance(schema, type):
        try:
            return schema.model_json_schema()
        except Exception:
            return None

    return None
