"""
schema 推导：从"代码里的一个工具"算出它的输入 / 输出 / 参数说明。

纯函数，不依赖注册表、不依赖具体来源 —— 服务方法与外部工具
（langchain BaseTool 等）都走这一套，算出来的东西才能一致。

单独一个模块是为了避开循环：discoverer 要用它，scanner 也要用它，
而 discoverer 要 import scanner。
"""

from __future__ import annotations

import inspect
import typing
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

__all__ = [
    "extract_parameter_descriptions",
    "output_description",
    "output_schema_of",
    "resolve_request_model",
    "synthesize_schema",
]


#: schema 递归的深度上限。防御自引用的模型（树形结构很常见），
#: 也能避免一个巨大 schema 拖慢同步。
MAX_SCHEMA_DEPTH = 4

#: Python 类型 -> JSON Schema 类型。只覆盖合成 schema 时需要的那几种。
_JSON_TYPES: dict[Any, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def resolve_request_model(
        func: Callable[..., Any],
) -> tuple[str, type[BaseModel]] | None:
    """
    方法是不是"只收一个 pydantic 模型参数"？是则返回 (参数名, 模型)。

    两个地方要它：
        scanner   取模型的 JSON Schema 当输入 schema
        invoke    把 Agent 传来的扁平参数包成模型实例

    后者是必须的：langchain 之类的调用方给的是
    `{"text": "hi"}`，而服务方法的签名是 `(self, req: EchoRequest)`，
    直接透传会报 unexpected keyword argument。
    """

    try:
        hints = typing.get_type_hints(func)
    except Exception:
        return None

    parameters = [
        p for p in inspect.signature(func).parameters.values()
        if p.name != "self"
    ]

    if len(parameters) != 1:
        return None

    annotation = hints.get(parameters[0].name)

    if isinstance(annotation, type) and issubclass(
            annotation, BaseModel
    ):
        return parameters[0].name, annotation

    return None


def synthesize_schema(func: Callable[..., Any]) -> dict[str, Any]:
    """
    给"没有单一 pydantic 参数"的方法合成一个输入 schema。

    这类方法不该出现在工具里（Agent 需要结构化参数），
    但发现阶段不该直接抛 —— 扫到一个不合格的方法就中断整次同步
    太苛刻。合成一个能看的 schema，让它在列表里出现、属性清楚，
    由人决定要不要改签名。
    """

    try:
        hints = typing.get_type_hints(func)
    except Exception:
        hints = {}

    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in inspect.signature(func).parameters.values():

        if parameter.name == "self":
            continue

        if parameter.kind in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            continue

        annotation = hints.get(parameter.name)

        schema: dict[str, Any] = {
            "title": parameter.name.replace("_", " ").title(),
        }

        json_type = _JSON_TYPES.get(annotation)

        if json_type:
            schema["type"] = json_type
        elif isinstance(annotation, type) and issubclass(
                annotation, BaseModel
        ):
            schema.update(annotation.model_json_schema())
        else:
            # 认不出来就别猜类型，留空比猜错好。
            schema["type"] = "string"

        if parameter.default is not inspect.Parameter.empty:
            schema["default"] = parameter.default
        else:
            required.append(parameter.name)

        properties[parameter.name] = schema

    result: dict[str, Any] = {
        "title": func.__name__,
        "type": "object",
        "properties": properties,
    }

    if required:
        result["required"] = required

    return result


def output_schema_of(
        func: Callable[..., Any],
) -> dict[str, Any] | None:
    """返回注解是 pydantic 模型时给出 schema，否则 None。"""

    try:
        hints = typing.get_type_hints(func)
    except Exception:
        return None

    annotation = hints.get("return")

    if isinstance(annotation, type) and issubclass(
            annotation, BaseModel
    ):
        return annotation.model_json_schema()

    return None


def output_description(
        output_schema: dict[str, Any] | None,
) -> str | None:
    """
    输出模型自带的描述 —— pydantic 把模型类的 docstring 放进了
    schema 的 description，所以这里能白拿一份。

    两个地方要用，因此抽出来只留一份：
        扫描期    填 source_docstring.returns（见 docstring.py）
        组装期    算 documentation 的返回值说明来源
    """

    if not output_schema:
        return None

    description = output_schema.get("description")

    return description if isinstance(description, str) else None


def extract_parameter_descriptions(
        schema: dict[str, Any],
        *,
        prefix: str = "",
        defs: dict[str, Any] | None = None,
        depth: int = 0,
) -> dict[str, str]:
    """
    从 JSON Schema 里抽出 "参数路径 -> 说明"。

    路径用 JSON Pointer，与 ToolParameterDocumentation.parameter_path
    同一套写法（例如 /query、/filters/date_from）。

    pydantic 会把嵌套模型放进 $defs 并用 $ref 引用，因此这里要
    解析 $ref —— 否则嵌套对象的字段说明一个都拿不到。
    """

    defs = defs if defs is not None else schema.get("$defs", {})

    if depth > MAX_SCHEMA_DEPTH:
        return {}

    found: dict[str, str] = {}

    for name, raw in (schema.get("properties") or {}).items():

        node = _resolve_ref(raw, defs)

        path = f"{prefix}/{name}"

        description = node.get("description")

        if description:
            found[path] = description

        # 嵌套对象：继续往下，让 /filters/date_from 这种路径能被定位到。
        if node.get("type") == "object" or "properties" in node:
            found.update(
                extract_parameter_descriptions(
                    node,
                    prefix=path,
                    defs=defs,
                    depth=depth + 1,
                )
            )

    return found


def _resolve_ref(
        node: dict[str, Any],
        defs: dict[str, Any],
) -> dict[str, Any]:
    """
    把 {"$ref": "#/$defs/X"} 换成 X 的定义；其它原样返回。

    两种形态都要处理，第二种容易漏：

        {"$ref": "#/$defs/Filters"}                       必填的嵌套模型
        {"anyOf": [{"$ref": "#/$defs/Filters"},
                   {"type": "null"}]}                      Optional[Filters]

    pydantic 把可选嵌套模型包在 anyOf 里，只看顶层的 $ref 会一个
    嵌套字段都拿不到 —— 而 JSON Pointer 这套设计本来就是为了
    定位嵌套字段，漏掉它等于白做。
    """

    ref = node.get("$ref")

    if isinstance(ref, str):
        target = _lookup_ref(ref, defs)

        if target is not None:
            # $ref 同一层可能还挂了别的键（例如 description），合并进去。
            merged = dict(target)
            merged.update(
                {k: v for k, v in node.items() if k != "$ref"}
            )
            return merged

        return node

    for keyword in ("anyOf", "oneOf", "allOf"):

        branches = node.get(keyword)

        if not isinstance(branches, list):
            continue

        for branch in branches:

            if not isinstance(branch, dict):
                continue

            branch_ref = branch.get("$ref")

            if not isinstance(branch_ref, str):
                continue

            target = _lookup_ref(branch_ref, defs)

            if target is None:
                continue

            merged = dict(target)
            merged.update(
                {k: v for k, v in node.items() if k != keyword}
            )
            return merged

    return node


def _lookup_ref(
        ref: str,
        defs: dict[str, Any],
) -> dict[str, Any] | None:
    """从 "#/$defs/Name" 取出定义。取不到返回 None。"""

    name = ref.rsplit("/", 1)[-1]
    target = defs.get(name)

    return target if isinstance(target, dict) else None
