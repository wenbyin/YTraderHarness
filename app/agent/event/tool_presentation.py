"""
展示描述符 -> chat 事件里的 `display` 载荷。

只做三件事：

    1. 渲染 summary 模板（{input.x} / {output.y}）
    2. 按 props.fields 的 JSON Pointer 白名单裁剪入参 / 结果
    3. 组装 renderer / name / summary / props

⚠️ 不 import 服务层的 DTO
------------------------
本模块是投影层的纯函数，只按属性名读取（renderer / display_name /
summary / show_input / show_output / props），不依赖
app.services.tool 的任何类型 —— 投影层不该被服务层的 import 图
牵住。

⚠️ 模板与白名单都**不抛**
------------------------
工具参数是模型给的、结果是业务给的，形状不可控。解析不到就当空，
一次展示配置写错不能把一次工具调用变成失败。
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ShapedToolOutput",
    "build_display",
    "decode_output",
    "fields_of",
    "output_of",
    "parse_tool_output",
    "render_summary",
    "restrict_fields",
    "shape_tool_output",
]

_MISSING = object()

#: 只认 input / output 两个根；其它花括号原样保留（文案里可能有正常的大括号）。
_TEMPLATE = re.compile(r"\{(input|output)\.([^{}]+)\}")


def _lookup(value: Any, path: str) -> Any:
    """点路径取值：支持 Mapping 键与列表下标。取不到返回哨兵。"""

    current = value

    for part in path.split("."):

        if part == "":
            continue

        if isinstance(current, Mapping):
            if part not in current:
                return _MISSING
            current = current[part]

        elif isinstance(current, (list, tuple)) and not isinstance(
            current, (str, bytes)
        ):
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return _MISSING

        else:
            return _MISSING

    return current


def render_summary(
        template: str | None,
        *,
        input_value: Any = None,
        output_value: Any = None,
) -> str | None:
    """
    渲染短句模板。没有模板返回 None；模板本身没有占位符时原样返回。

    解析不到的路径渲染成空串 —— 与「缺字段」的展示口径一致，
    而不是把 `{input.foo}` 原样漏给用户。
    """

    if not template:
        return None

    # 模板要 output，而这一刻还没有（tool.called 阶段）：
    # 宁可不发 summary，也不要发一句「共  条」。completed 会带上完整那句。
    if output_value is None and "{output." in template:
        return None

    def replace(match: re.Match[str]) -> str:

        root_name, path = match.group(1), match.group(2)

        root = input_value if root_name == "input" else output_value

        if root is None:
            return ""

        found = _lookup(root, path)

        if found is _MISSING or found is None:
            return ""

        if isinstance(found, (Mapping, list, tuple)):
            return json.dumps(
                found, ensure_ascii=False, default=str
            )

        return str(found)

    return _TEMPLATE.sub(replace, template)


def _get_pointer(value: Any, pointer: str) -> Any:
    """JSON Pointer 取值。取不到返回哨兵。"""

    if pointer in ("", "/"):
        return value

    current = value

    for raw in pointer.lstrip("/").split("/"):

        token = raw.replace("~1", "/").replace("~0", "~")

        if isinstance(current, Mapping):
            if token not in current:
                return _MISSING
            current = current[token]

        elif isinstance(current, (list, tuple)) and not isinstance(
            current, (str, bytes)
        ):
            try:
                current = current[int(token)]
            except (ValueError, IndexError):
                return _MISSING

        else:
            return _MISSING

    return current


def restrict_fields(
        value: Any,
        fields: Sequence[str] | None,
) -> Any:
    """
    只保留 `fields` 里的 JSON Pointer。

    没有白名单（None / 空）时原样返回 —— show_input / show_output
    是粗粒度开关，fields 是细粒度裁剪，两者正交。

    数组下标路径会落成嵌套 dict（例如 `/items/0/price` ->
    `{"items": {"0": {"price": ...}}}`）。展示层只读不重建，
    这个形状足够；不做成数组是为了不替客户端猜长度。
    """

    if not fields:
        return value

    if isinstance(fields, (str, bytes)) or not isinstance(
        fields, Sequence
    ):
        return value

    result: dict[str, Any] = {}

    for pointer in fields:

        if not isinstance(pointer, str) or not pointer.startswith("/"):
            continue

        found = _get_pointer(value, pointer)

        if found is _MISSING:
            continue

        tokens = [
            token.replace("~1", "/").replace("~0", "~")
            for token in pointer.lstrip("/").split("/")
        ]

        if not tokens:
            continue

        node = result

        for token in tokens[:-1]:
            child = node.get(token)
            if not isinstance(child, dict):
                child = {}
                node[token] = child
            node = child

        node[tokens[-1]] = found

    return result


def decode_output(value: Any) -> Any:
    """
    custom stream 的 `output` 是字符串：工具节点用 `_render()` 把非 str
    结果 JSON 化之后才写进事件（见 graph/react/tools.py）。

    展示层要读结构（`{output.rowCount}`、`output_path=/bars`），所以先解
    一层。解析不了就原样返回 —— 工具本来就返回纯文本的情况很常见。

    ⚠️ 历史通道不需要这一步：`message_recorder._tool_output()` 已经从
    ToolMessage 的正文里解过一层。这里补的是**实时**通道。
    """

    if not isinstance(value, str):
        return value

    text = value.strip()

    if not text:
        return value

    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return value


def parse_tool_output(text: str) -> Any:
    """
    工具结果的**文本** -> 可展示的值。

    真实的 ToolMessage 成功结果是 `<ref:tool>` 包起来的一段（带引用规则
    尾注，见 context/processor/reference.py）：先取正文，再解 JSON。
    解不开就原样返回文本（工具返回纯文本很常见）。

    ⚠️ 不要在调用方各自写一遍
    ------------------------
    写路径（message_recorder）与读路径（messages API 的旧行自愈）必须
    用同一套解析，否则会出现「实时对了、历史还错」这种分叉。
    """

    from app.agent.context.processor.reference import unwrap_reference

    source = str(text or "").strip()

    if not source:
        return None

    unwrapped = unwrap_reference(source)
    candidate = source if unwrapped is None else unwrapped

    try:
        return json.loads(candidate)
    except (TypeError, ValueError):
        return candidate


def output_of(presentation: Any, value: Any) -> Any:
    """
    `props.output_path` + `props.max_rows`：决定展示用的 `output`。

    output_path
        从工具结果里取一段。典型用途：MCP 行情工具返回
        `{bars: [...], rowCount: N, ...}`，表格要的是 `bars` 数组本身 ——
        不取的话前端会把整个包装对象当成一行。

        ⚠️ 路径不存在时返回 `[]`，不是退回整个包装对象：退回会让表格画出
        一行 `{bars:..., rowCount:...}`，比不画更糟。空数组会让前端
        `buildTable` 返回 null、退化成 card。

    max_rows
        数组只留前 N 条。表格只画 5 行，一次查询却可能带 200 行 ——
        不压的话 SSE 载荷与历史 JSONB 都在为不显示的行付费。摘要里的
        总量（如 `{output.rowCount}`）不受影响。
    """

    props = getattr(presentation, "props", None)

    if not isinstance(props, Mapping):
        return value

    path = props.get("output_path")

    if isinstance(path, str) and path.startswith("/"):
        found = _get_pointer(value, path)
        value = [] if found is _MISSING else found

    max_rows = props.get("max_rows")

    if (
            isinstance(max_rows, int)
            and not isinstance(max_rows, bool)
            and max_rows >= 0
            and isinstance(value, list)
    ):
        value = value[:max_rows]

    return value


def fields_of(presentation: Any) -> list[str] | None:
    """取描述符的 props.fields；形状不对就当没配。"""

    props = getattr(presentation, "props", None)

    if not isinstance(props, Mapping):
        return None

    fields = props.get("fields")

    if not isinstance(fields, list):
        return None

    return [item for item in fields if isinstance(item, str)]


def build_display(
        presentation: Any,
        *,
        input_value: Any = None,
        output_value: Any = None,
) -> dict[str, Any]:
    """
    描述符 -> 事件里的 `display` 载荷。

    ⚠️ input / output **不在这里**
    ----------------------------
    它们留在事件载荷的顶层（由投影层按 show_input / show_output
    决定出不出键），display 只放「怎么画、写什么」。两边都放一份
    会让订阅端不知道该信哪个。
    """

    renderer = getattr(presentation, "renderer", None)
    renderer_value = (
        renderer.value if hasattr(renderer, "value") else str(renderer)
    )

    display: dict[str, Any] = {"renderer": renderer_value}

    name = getattr(presentation, "display_name", None)

    if name:
        display["name"] = name

    summary = render_summary(
        getattr(presentation, "summary", None),
        input_value=input_value,
        output_value=output_value,
    )

    if summary:
        display["summary"] = summary

    props = getattr(presentation, "props", None)

    if isinstance(props, Mapping) and props:
        display["props"] = dict(props)

    return display


@dataclass(frozen=True, slots=True)
class ShapedToolOutput:
    """一次工具结果的展示形态：display 必有；output 只在 show_output 时。"""

    display: dict[str, Any]

    output: Any = None

    has_output: bool = False


def shape_tool_output(
        presentation: Any,
        raw_output: Any,
) -> ShapedToolOutput:
    """
    工具结果 -> `(display, output?)` 的**唯一**组合点。

    三处调用方（实时投影、落库、读时重算）共用这一份：
    `decode_output` / `parse_tool_output` 之后把 raw_output 传进来，
    不要各自再拼 `output_of` + `restrict_fields` + `build_display` ——
    那正是「实时对了、历史还错」的来源。

    `raw_output` 必须是**结构化的**值（不是 custom stream 的 JSON 字符串、
    也不是带 `<ref:tool>` 包装的正文）；那两层的剥离在调用方各自完成，
    因为它们拿到的输入形态不同。
    """

    display = build_display(
        presentation,
        output_value=raw_output,
    )

    if not getattr(presentation, "show_output", False):
        return ShapedToolOutput(display=display)

    return ShapedToolOutput(
        display=display,
        output=restrict_fields(
            output_of(presentation, raw_output),
            fields_of(presentation),
        ),
        has_output=True,
    )
