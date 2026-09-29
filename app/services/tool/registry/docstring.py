"""
docstring -> 适配前端的结构化段。

为什么需要它
------------
`source_docstring` 落库时原本只有两个键：raw（整段原文）与
parameters（来自 input_schema 的 JSON Pointer -> 说明）。前端要的是
**切好的段** —— 它按 `description` 与 `returns` 取整体描述与返回值
说明，直接读 raw 只能拿到一大坨，读不出边界。

所以切分放在**扫描期**，与 raw 一起落库：读接口不必各自再解析一遍，
前端也不必自己切字符串。raw 保留不删 —— 它是原文，覆盖与排查都要用。

支持的范围
----------
只认 Google 风格的节标题（Args / Returns / Raises …），且必须
**顶格**：本仓库的 docstring 经 inspect.getdoc() 去掉公共缩进后，
节标题一律落在第 0 列。识别不出来的内容一律留在 description 里，
不猜 —— 猜错会把正文吃掉，而"少切一段"只是退回原样。

刻意不认 `标题\\n------` 那种下划线风格：本仓库用它写的是模块
docstring 里的小标题（"为什么不是 X"这类），不是参数说明，把它
当成节会把正文拦腰截断。
"""

from __future__ import annotations

import re
import textwrap

from .schema import output_description

__all__ = [
    "parse_docstring",
    "resolve_returns",
    "split_sections",
]


#: 认得的节标题。只用于**切段**：除 Returns 外都不产出键，
#: 其余各节的内容仍原样留在 raw 里。
_SECTION_TITLES: tuple[str, ...] = (
    "Args",
    "Arguments",
    "Returns",
    "Return",
    "Raises",
    "Yields",
    "Examples",
    "Notes",
    "Attributes",
)

#: 节标题行：顶格、以冒号结尾、冒号后没有内容。
#:
#: 要求冒号后为空是有意的 —— `Returns: 总数` 这种同行写法一旦认了，
#: 就得再处理"同行内容 + 后续缩进行"的拼接，而本仓库没有这种写法。
_SECTION_RE = re.compile(
    rf"^({'|'.join(_SECTION_TITLES)}):\s*$"
)

#: 产出 returns 的节标题，按优先级排。
_RETURNS_TITLES: tuple[str, ...] = ("Returns", "Return")


def split_sections(raw: str) -> tuple[str, dict[str, str]]:
    """
    按节标题把 docstring 切成 (正文, {节标题: 内容})。

    正文 = 第一个节标题**之前**的全部内容，多段保留换行。
    节内容 = 标题行之后到下一个标题行之前的全部内容，去掉公共缩进。
    """

    description_lines: list[str] = []
    collected: dict[str, list[str]] = {}

    current: str | None = None

    for line in raw.splitlines():

        matched = _SECTION_RE.match(line)

        if matched is not None:
            current = matched.group(1)
            collected.setdefault(current, [])
            continue

        if current is None:
            description_lines.append(line)
        else:
            collected[current].append(line)

    sections = {
        title: _clean(lines) for title, lines in collected.items()
    }

    # 空节（标题写了但没内容）不算数：留着会让下游拿到一个空串，
    # 分不清"没写"和"写了但空"。
    return (
        _clean(description_lines),
        {
            title: text
            for title, text in sections.items()
            if text
        },
    )


def parse_docstring(raw: str | None) -> dict[str, str]:
    """
    -> {"description": ..., "returns": ...}

    取不到的键**不出现**，而不是给一个 None：前端判空只写一种写法
    （`if (doc.description)`），不必同时防 undefined 与 null。
    """

    if not raw or not raw.strip():
        return {}

    description, sections = split_sections(raw)

    parsed: dict[str, str] = {}

    if description:
        parsed["description"] = description

    for title in _RETURNS_TITLES:
        if sections.get(title):
            parsed["returns"] = sections[title]
            break

    return parsed


def resolve_returns(
        section_returns: str | None,
        output_schema: dict | None,
) -> str | None:
    """
    返回值说明：谁有取谁，**DTO 优先**。

    输出模型自己的 docstring 就是一份现成的返回值描述（pydantic 会把它
    放进 schema 的 description），方法里再写一遍 Returns 段是重复的。
    两边都写了以 DTO 为准 —— 类型定义比方法注释更靠近事实，方法改了
    返回值而注释忘了改的情况太常见。

    被 DTO 盖住的那段不会丢：它仍在 source_docstring.raw 里。
    """

    return output_description(output_schema) or section_returns or None


def _clean(lines: list[str]) -> str:
    """去公共缩进、去首尾空白。空行组不留下空串。"""

    return textwrap.dedent("\n".join(lines)).strip()
