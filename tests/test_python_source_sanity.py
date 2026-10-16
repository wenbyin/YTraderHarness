"""
源码体检：**每个 .py 都能编译**，以及一个反复踩到的引号陷阱。

为什么值得单列一条
------------------
这两条都不是"某个功能对不对"，而是"代码根本收不起来"：

    ast.parse 失败       pytest 在**收集阶段**就中断，报的是一行
                         `SyntaxError`，而看不出还有没有别处也坏了。
                         更糟的是**没有任何测试 import 的文件坏了也照样
                         静默**——直到某个入口恰好 import 它。
    中文引号截断字符串   `"…"这是中文引号"…"` 里的半角引号会把字面量
                         提前截断，而**报错行指向字符串结尾**，不是出问题
                         的那对引号。

后一条是**本仓库的高频坑**：docstring 与 `description` 全是中文散文，写
「引号」时很容易顺手打成半角。实测同一个坑连踩了 4 次，每次都是整套测试
收不起来才发现——所以它不该靠"记得"。

扫描规则（**只报一件事**）
--------------------
用 `tokenize` 分词，只查**单引号字符串字面量内部**有没有 `汉字"汉字`。

⚠️ 为什么不能只按文本扫：**注释与 docstring 里的半角引号是无害的**
----------------------------------------------------------------
docstring 是三个引号包的，里面单个 `"` 不会截断；注释更是惰性的。仓库
里这类写法有几十处（中文散文里顺手用半角引号），它们是**正常风格**。

第一版就是纯文本扫的，结果把仓库扫出一大片 —— 那样的用例只会被人关掉。

致命的只有一种：`"…中文"中文"…"` 这种**单引号字面量**里的中文引号 ——
它会在中间把字面量截断。分词之后这条边界是精确的：

    COMMENT / docstring(三引号)  -> 跳过（无害）
    单引号 STRING                -> 查里面的 `汉字"汉字`
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent

#: 要体检的源码根。
SOURCE_ROOTS = ("app", "tests")

#: `汉字"汉字` —— 半角引号左右都是汉字。
_CJK_QUOTE = re.compile(r"[一-鿿]\"[一-鿿]")


def _source_files() -> list[Path]:
    """全部 .py（跳过虚拟环境与缓存）。"""

    files: list[Path] = []

    for root in SOURCE_ROOTS:

        base = REPO_ROOT / root

        if not base.exists():
            continue

        files.extend(
            path
            for path in base.rglob("*.py")
            if "__pycache__" not in path.parts
        )

    return sorted(files)


def test_every_source_file_parses() -> None:
    """
    每个 .py 都能解析。

    ⚠️ **包括不 import 到任何测试的文件** —— 那正是这条的价值：一个没人
    import 的模块坏掉时，别的检查一条都不会红（pytest 只收集测试文件），
    直到某个入口恰好碰到它。
    """

    broken: list[str] = []

    for path in _source_files():

        try:
            ast.parse(
                path.read_text(encoding="utf-8"),
                filename=str(path),
            )

        except SyntaxError as exc:
            broken.append(f"{path.relative_to(REPO_ROOT)}:{exc.lineno} {exc.msg}")

    assert not broken, (
        "这些源文件解析不了（它们会让 import 与测试收集一起中断）：\n  "
        + "\n  ".join(broken)
    )


def test_no_chinese_quotes_inside_string_literals() -> None:
    """
    **单引号字面量内部**没有 `汉字"汉字` —— 它会在中间把字面量截断。

    详见模块说明里的扫描规则。命中时给出文件、行号与那段字面量，因为
    `SyntaxError` 的报出行号指向的是**字符串结尾**，不是出问题的那对引号。
    """

    offenders: list[str] = []

    for path in _source_files():

        source = path.read_text(encoding="utf-8")

        try:
            tokens = list(
                tokenize.generate_tokens(io.StringIO(source).readline)
            )

        except (tokenize.TokenError, IndentationError):
            # 坏到分不了词 —— 交给上面那条 ast.parse 去报，这里不重复喊。
            continue

        for token in tokens:

            if token.type != tokenize.STRING:
                continue

            # 三引号（docstring）里面的引号无害，跳过。
            if token.string.startswith(("\"\"\"", "'''")):
                continue

            if _CJK_QUOTE.search(token.string):
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{token.start[0]}"
                    f"\n      {token.string.strip()[:80]}"
                )

    assert not offenders, (
        "这些**单引号字面量**里混进了半角中文引号（会把字面量截断，"
        "而报错行指向结尾）：\n  "
        + "\n  ".join(offenders)
        + "\n\n  改用「」或全角引号。"
    )
