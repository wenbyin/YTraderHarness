"""
工具暴露源：`tool_keys -> 可 bind 的工具对象`。

它是 `app/agent/context/processor/sources.py::ToolExposureSource` 的
**唯一实现**，被 capability 处理器（上下文流水线第 7 阶段）调用。

为什么是纯内存的
----------------
"能 bind 的工具对象"只可能来自**代码注册表**（`@tool` 描述符在
import 期自注册，见 registry/decorator.py）。数据库那边存的是
`tool_definition` —— 一行事实快照，没有 Python 对象，也没法 bind。
所以这里不查库：查了也变不出工具对象来。

这条很要紧：本类跑在**每一次模型调用的上下文构建里**，多一次
数据库往返就是每次调用都多一次。而它要的东西进程里本来就有。

tool_id 不在这里
--------------
处理器链是按 `tool_id`（UUID）挂的，而本类只认 tool_key。这个映射
是**数据库事实**，由执行侧（tools 节点）单独查一次 —— 那里每轮只
查一次，且只在真的有待执行调用时才查。放进本类会让"装配给模型看
什么"这件纯代码的事拖上一条 DB 依赖。

conversation_id 本期不用
-----------------------
协议里带着它是为了将来"按会话/租户过滤工具"，现在没有任何一条
工具来源支持按会话隔离。签名里保留它，实现里忽略 —— 但**显式写在
docstring 里**，免得下一个人以为它在生效。

工具名在这里转成 provider 能接受的形式
------------------------------------
`tool_key` 形如 `capability.list_capabilities`，而 OpenAI 兼容端点要求
函数名匹配 `^[a-zA-Z0-9_-]+$` —— 点号会让整个请求 400。转义规则落在
`wire_name`（与 `select_specs` 同住 `registry/selection.py`，理由见那里）。

⚠️ 转的是**副本**，不是注册表里那个描述符本体
-------------------------------------------
`spec.descriptor` 每次拿到的都是同一个对象（描述符协议，见 spec.py），
就地改它的 `name` / `description` / `args_schema` 会把注册表里那份也
改掉 —— 进程级单例，一次改动影响此后所有调用，而且没有任何地方会报错。
所以这里**一律**取一份副本再改（`for_wire` 里那条"名字已经合法就直接
返回本体"的捷径因此删掉了）。

反向的翻译（模型回传的函数名 -> tool_key）在 tools 节点，用的是同一个
`wire_name` —— 出站与入站必须是同一套规则，否则模型调了它看得见的工具，
却在允许集里对不上。

人工写的说明覆盖在这一层叠上来
----------------------------
工具说明的**代码侧事实**（docstring / 声明）在 import 期就钉在描述符上了，
而**人工覆盖**住在库里 —— 装饰期读不到库（见 decorator.py 的说明），
所以由本层在出站前合并。数据不在这里查：由调用方（capability 处理器，
源头是 Agent Builder 那一轮预载进 `context_policy.tool_overrides` 的
结果）传进来，本层仍然**零 DB 往返**。

不合并的话症状是"同一份覆盖在 GET /tools 里生效、模型却看不到" ——
两条路各自都工作正常，只有对着请求体才看得出来。
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Mapping, Sequence
from typing import Any

from app.agent.context.processor.sources import ToolExposureSource
from app.services.tool.registry import select_specs, wire_name
from app.services.tool.registry.schema import (
    injected_args,
    strip_injected_args,
)

__all__ = ["RegistryToolExposureSource", "for_wire"]

logger = logging.getLogger(__name__)


class RegistryToolExposureSource(ToolExposureSource):
    """
    按 key 从进程级工具注册表取工具。

    选取规则完全落在 `select_specs` 里 —— 那是"哪些工具能用"的**唯一**
    判定，执行侧（tools 节点）用的是同一个函数、同一个输入，因此
    "模型看到的工具"与"允许执行的工具"是两个相等的集合。
    """

    async def resolve(
            self,
            *,
            tool_keys: Sequence[str],
            conversation_id: str | None,
            overrides: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> Sequence[Any]:

        # 无状态、无 IO，async 只是为了满足协议 —— 将来某条来源
        # （MCP）真的要发请求时，这个签名不用改。
        #
        # overrides 由调用方（capability 处理器，数据来自 Agent Builder
        # 那一轮预载的 context_policy）传进来。本层因此**仍然不查库**，
        # 见模块 docstring 的"为什么是纯内存的"。
        by_key = overrides or {}

        return [
            for_wire(spec, by_key.get(spec.tool_key))
            for spec in select_specs(tool_keys)
        ]


def for_wire(
        spec: Any,
        overrides: Mapping[str, Any] | None = None,
) -> Any:
    """
    描述符的一份**副本**：名字取 provider 能接受的形式，说明按人工覆盖叠加。

    `select_specs` 已经保证 `spec.descriptor` 非空（没有描述符的 key
    根本不进结果集），所以这里不做判空。

    公开而不是私有：**出站与入站必须用同一份名字**。出站是这里
    （bind 给模型看），入站是 tools 节点（模型回传的名字 -> tool_key），
    两边都得拿到"模型实际看到的那个名字" —— 各写一份改名逻辑，早晚
    会在某一条分支上漂移，而症状是"模型调了它看得见的工具却被判未暴露"。

    人工覆盖（overrides）
    --------------------
    形状与 `ToolQueryService.tool_documentation_overrides` 的返回一致：

        {"description": "人工写的整工具说明",      可选
         "parameters": {"/parent_id": "人工写的参数说明"}}   可选

    ⚠️ **合并只发生在这里**，这就是 decorator.py 说的"人工覆盖由准备
    工具清单的那一层叠上来"。它落在这一层而不是 decorator：装饰发生在
    import 期，那时读不到库。

    判据是**真值**（空串与 None 都算"没写覆盖"），与读接口
    `_assembly.build_summary` 的 `override or source` 是同一个口径 ——
    两处不同的话，同一份覆盖会在列表页生效、在模型那里不生效。

    ⚠️ **一律返回副本**，包括"名字本来就合法"那一种
    ----------------------------------------------
    曾经有一条捷径：`descriptor.name` 已经等于目标名时直接返回它自己。
    那时本函数只改名，返回本体无害。现在它还会写 description 与
    args_schema —— 而**模块级 langchain 工具的 `spec.descriptor` 就是
    模块命名空间里的那个对象本体**（进程级单例，见 spec.py 的说明），
    往上面写一次，此后所有调用都用着那份人工覆盖，且没有任何地方会
    报错。所以那条捷径必须删掉，代价是每次多一个浅拷贝对象。
    """

    descriptor = spec.descriptor

    update: dict[str, Any] = {"name": wire_name(spec.tool_key)}

    # ⚠️ 模型侧 schema 用 tool_call_schema，并显式剔除注入参数
    # ----------------------------------------------------
    # `args_schema` 是执行侧的完整模型，包含 ToolRuntime /
    # InjectedState；直接拿它 bind 给模型，模型的参数说明里会出现一个
    # 它填不了的对象。
    #
    # 模块级 langchain 工具的 `tool_call_schema` 已经由 LangChain 剔除
    # 了注入参数；这里再走一遍 `strip_injected_args`，覆盖 service
    # 方法那种由我们合成 schema 的情形。
    func = (
        getattr(descriptor, "func", None)
        or getattr(descriptor, "coroutine", None)
    )

    schema_source = (
        getattr(descriptor, "tool_call_schema", None)
        or descriptor.args_schema
    )

    if func and injected_args(func):

        schema_dict = _schema_dict(schema_source)

        if schema_dict is not None:
            schema_dict = strip_injected_args(schema_dict, func)
            update["args_schema"] = schema_dict
            schema_source = schema_dict

    if overrides:

        description = overrides.get("description")

        if description:
            update["description"] = description

        parameters = overrides.get("parameters")

        if isinstance(parameters, Mapping) and parameters:

            patched = _apply_parameter_overrides(
                schema_source, parameters
            )

            # None = 一个路径都没落上（参数被改名/删掉了，或路径写错）。
            # 那时**不带这个键**：保持原 schema，而不是塞一份看不出
            # 差别的副本。
            if patched is not None:
                update["args_schema"] = patched

    return descriptor.model_copy(update=update)


# ============================================================
# 参数说明：JSON Pointer -> JSON Schema
# ============================================================


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


def _apply_parameter_overrides(
        args_schema: Any,
        overrides: Mapping[str, str],
) -> dict[str, Any] | None:
    """
    把 `{"/parent_id": "说明"}` 落进 args_schema，返回**改过的副本**。

    一个都没落上时返回 None。

    两种 args_schema 形状都要认
    --------------------------
    `ModelMetaclass`    模块级 langchain 工具（`agent/tools` 那批）——
                        langchain 自己从签名建的 pydantic 模型
    `dict`              `@tool` 服务方法 —— decorator 喂的就是算好的
                        JSON Schema dict（见 decorator._input_schema_of）

    前者先 `model_json_schema()` 摊平成同一种形状，此后只有一条路。
    摊平后塞回 `model_copy(update={"args_schema": ...})` 是可行的：
    出站（`bind_tools` -> `convert_to_openai_tool`）认 dict 形态的
    args_schema，而**执行**根本不走这个副本（走 `spec.tool` 或
    `factory.create(contract)`，见 tool_executor.py），所以这里把它
    降成 dict 没有别的后果。

    ⚠️ 先深拷贝
    ----------
    `args_schema` 是描述符（进程级单例）身上的那个 dict，就地改嵌套的
    `properties` 会污染注册表 —— 改了没有报错，而是此后所有调用都带着
    这份覆盖。`model_copy` 是浅拷贝，挡不住这个。

    路径走不到时**只告警、不造属性**：那说明参数被改名或删了（库里
    留着一条人工说明），或者路径本身就写错了。凭空补一个 property 会
    让模型看见一个执行时不存在的参数 —— 与 select_specs 的"跳过要留痕"
    是同一条立场。
    """

    if isinstance(args_schema, dict):
        schema: dict[str, Any] = copy.deepcopy(args_schema)

    elif hasattr(args_schema, "model_json_schema"):
        schema = copy.deepcopy(args_schema.model_json_schema())

    else:
        # 既不是 dict 也不是 pydantic 模型 —— 认不出来就别猜
        # （猜错的后果是覆盖悄悄消失，而那正是本次要修的症状）。
        logger.warning(
            "工具的参数 schema 类型不认识（%s），"
            "参数说明覆盖本次不生效。",
            type(args_schema).__name__,
        )
        return None

    applied = 0

    for path, description in overrides.items():

        if not description:
            continue

        node = _resolve_pointer(schema, path)

        if node is None:
            logger.warning(
                "参数说明覆盖的路径 %r 在工具的 args_schema 里走不到"
                "（参数可能已改名或删除），已跳过。",
                path,
            )
            continue

        node["description"] = description

        applied += 1

    return schema if applied else None


def _resolve_pointer(
        schema: dict[str, Any],
        pointer: str,
) -> dict[str, Any] | None:
    """
    JSON Pointer（`/filters/date_from`）-> schema 里那个节点。

    走不到时返回 None。认三种中转，因为 input_schema 是一棵真的树
    （见 domain/entities/tool_documentation.py 里为什么用 path 而不是
    参数名）：

        $ref        pydantic 把嵌套模型提到 `$defs` 下，此处是引用
        anyOf       `X | None` 会摊成一个分支列表，挑**认得参数名**那支
        items       数组元素（`/filters/ranges/0/field` 里那个数字段）

    `~1` / `~0` 的转义按 RFC 6901 还原 —— 参数名里带 `/` 或 `~`
    虽然罕见，但转义是这一层的定义的一部分，不做就是在窄路上悄悄
    认错一个参数。
    """

    if not pointer.startswith("/"):
        logger.warning(
            "参数说明覆盖的路径 %r 不是绝对 JSON Pointer，已跳过。",
            pointer,
        )
        return None

    tokens = [
        token.replace("~1", "/").replace("~0", "~")
        for token in pointer[1:].split("/")
    ]

    node: Any = schema

    for token in tokens:

        node = _deref(node, schema)

        if not isinstance(node, dict):
            return None

        # 数组下标：所有元素共用同一份 `items` schema
        # （JSON Schema 里没有"第 3 个元素的 schema"这种东西）。
        # 所以 `0` 这一步不改变节点，只是往下走 `items`。
        if token.isdigit() and "items" in node:
            node = node["items"]
            continue

        properties = node.get("properties")

        if not isinstance(properties, dict) or token not in properties:
            return None

        node = properties[token]

    node = _deref(node, schema)

    return node if isinstance(node, dict) else None


def _deref(
        node: Any,
        root: dict[str, Any],
        depth: int = 0,
) -> Any:
    """
    解开 `$ref` 与 `anyOf`，给出一个"有 properties 的节点"（拿不到就原样返回）。

    `anyOf` 挑**第一个解开之后带 properties 的分支**。注意是"解开之后"：
    pydantic 为 `X | None` 生成的是

        {"anyOf": [{"$ref": "#/$defs/X"}, {"type": "null"}]}

    分支上**没有**字面量 `properties`（在 $ref 后面），直接看分支本身
    会把 `X | None` 这种最常见的情形全部漏掉 —— 而 `retrieve_processor`
    这类可选嵌套参数正好都是这个形状。

    `X | Y`（同一字段两种形态）本来就没有唯一答案，挑第一个是确定的，
    且与 pydantic 生成的顺序一致。

    ⚠️ 认不出 `additionalProperties` 这种**映射**参数
    ----------------------------------------------
    `dict[Key, Value]` 生成的形状是 `{"additionalProperties": {...}}`，
    键不是 `properties` 里的条目（`document.retrieve_documents` 的
    `retrieval_config` 就是它）。JSON Pointer 指不到"映射里的某一个键"，
    因为那个键集是自由/半自由的 —— 硬认会让一个**拼错的键**也解析成功，
    比跳过更糟。这类参数今天只能靠代码侧说明，走不到时上面的告警会说话。

    深度上限是防御性的：`$ref` 成环时不要转到天荒地老（库里不该有环，
    但人工编辑过的 schema 什么形状都可能）。
    """

    if depth > 8:
        return node

    if not isinstance(node, dict):
        return node

    ref = node.get("$ref")

    if isinstance(ref, str) and ref.startswith("#/$defs/"):

        target = root.get("$defs", {}).get(ref[len("#/$defs/"):])

        if target is None:
            return node

        return _deref(target, root, depth + 1)

    branches = node.get("anyOf") or node.get("oneOf")

    if isinstance(branches, list):

        for item in branches:

            candidate = _deref(item, root, depth + 1)

            if isinstance(candidate, dict) and "properties" in candidate:
                return candidate

        return node

    return node
