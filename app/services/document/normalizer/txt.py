"""
通用文本归一化实现（具体层）。

继承层级（严格对应 parser / cleaner / chunker）
----------------------------------------------
    DocumentProcessor
      └── DocumentNormalizer          normalizer/base.py  role 基类
            └── TextDocumentNormalizer normalizer/text.py  中间基类（模板）
                  └── TxtCommonNormalizer                  具体实现 ← 本文件

处理顺序是设计的一部分
----------------------
下面 13 步的执行顺序**不能随便调**，每一步都有前置依赖：

    1  统一编码     孤立代理项 -> U+FFFD   必须最先：否则后面任何一次
                                           encode 都可能直接抛异常
    2  BOM          移除 U+FEFF
    3  换行         CRLF / CR / U+2028/29 -> LF   必须在空行处理之前
    4  换页符       \\f -> LF              先变行，才能参与空行合并
    5  特殊空格     NBSP / U+3000 等 -> 空格   必须在空白折叠之前
                                           （否则 NBSP 会躲过折叠）
    6  控制字符     删噪声、留语义
    7  Unicode 形式 NFC                    放在全角折叠之前：NFC 是无损
                                           的规范合并，不会与兼容折叠打架
    8  缩进         行首 tab -> 4 空格
    9  空白         行内连续空格 -> 1（保留行首缩进）
    10 空行         连续 3+ 换行 -> 2
    11 数字字母形式 全角字母数字 -> 半角
    12 标点形式     中文标点两侧的多余空白
    13 全角半角     全角**标点** -> 半角（默认关闭，见配置说明）

一步都不删"内容"
----------------
唯一会被删掉的字符是**控制字符与空白**这两类非内容物。CJK 字符、
字母、数字一个都不会消失 —— 全角字母数字只是折叠成半角，字符本身
还在。这条性质由测试守着（幂等 + 实体不丢）。

⚠️ 本文件里所有不可见字符一律写成 \\uXXXX 转义，**不要**直接粘贴
字面量 —— 字面量在编辑器里完全看不出差别，改动和 review 都会失守。
"""

from __future__ import annotations

import re
import unicodedata

from pydantic import BaseModel, Field

from app.services.document.domain.schemas import ProcessorRequest

from .text import TxtNormalizationResult, TextDocumentNormalizer

__all__ = [
    "NormalizationConfig",
    "TxtCommonNormalizer",
]


# ============================================================
# 配置项名 / 步骤名（也用于审计）
# ============================================================

_STEP_ENCODING = "encoding"
_STEP_BOM = "bom"
_STEP_NEWLINE = "newline"
_STEP_FORM_FEED = "form_feed"
_STEP_SPECIAL_SPACES = "special_spaces"
_STEP_CONTROL_CHARS = "control_chars"
_STEP_UNICODE_FORM = "unicode_form"
_STEP_INDENT = "indent"
_STEP_WHITESPACE = "whitespace"
_STEP_BLANK_LINES = "blank_lines"
_STEP_DIGIT_LETTER = "digit_letter"
_STEP_PUNCTUATION = "punctuation"
_STEP_FULLWIDTH_PUNCT = "fullwidth_punct"


# ============================================================
# 字符集合
# ============================================================
#
# 本文件所有不可见字符一律写 \uXXXX 转义，**不要**直接粘贴字面量 ——
# 字面量在编辑器里完全看不出差别，review 与改动都会失守。

#: 孤立代理项（U+D800–U+DFFF）。
#:
#: 合法的 Unicode 文本里不该出现它们；一旦出现，text.encode("utf-8")
#: 会直接抛 UnicodeEncodeError —— 而下游每次算 checksum / 落盘都要
#: encode。所以这一步必须最先做，把不合法序列变成合法的替换字符。
_LONE_SURROGATE = re.compile("[\ud800-\udfff]")

#: 控制字符 —— 删掉。
#:
#: 刻意**不含** TAB(\u0009) 与 LF(\u000a)：它们是结构，不是噪声。
#: 也刻意不含换页符(\u000c) —— 那是独立一步，单独可控。
#:
#: **保留** ZWNJ(\u200c) 与 ZWJ(\u200d)：emoji 组合序列靠 ZWJ 连接，
#: 删了就散架；ZWNJ 在波斯语 / 阿拉伯语 / 印度语系里承载实际语义。
#: 看不见 ≠ 没意义。
_CONTROL_CHARS = re.compile(
    "["
    "\u0000-\u0008"      # C0（跳过 TAB \u0009、LF \u000a）
    "\u000b"             # VT（\u000c 是换页符，另有一步）
    "\u000e-\u001f"      # C0 其余
    "\u007f"             # DEL
    "\u0080-\u009f"      # C1
    "\u200b"             # 零宽空格 —— 真·噪声
    "\u200e\u200f"       # LRM / RLM 方向标记
    "\u202a-\u202e"      # 方向嵌入 / 覆盖
    "\u2060-\u2064"      # word joiner / 不可见运算符
    "\u2066-\u2069"      # 方向隔离
    "]"
)

#: 特殊空格 —— 全部归到普通空格 U+0020。
#: 不含零宽空格(\u200b) —— 那个在上一步删掉，不是"变成空格"。
_SPECIAL_SPACES = re.compile(
    "[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]"
)

#: 行内的连续空白（2 个以上）折叠成一个。
#: (?<=\S) 保证**行首缩进不被吃掉** —— 缩进是结构信息。
_INLINE_SPACES = re.compile(r"(?<=\S)[ \t]{2,}")

#: 连续 3 个以上换行 -> 2 个（即最多保留一个空行）。
_BLANK_RUN = re.compile(r"\n{3,}")

#: 全角字母数字的码位区间。
#: 刻意**只取字母数字**，不含 U+FF01-FF0F / U+FF1A-FF20 等
#: 全角**标点**区间 —— 那些属于第 13 步，默认关闭。
_HALFWIDTH_ALNUM_RANGES = (
    (0xFF10, 0xFF19),    # 全角 0-9
    (0xFF21, 0xFF3A),    # 全角 A-Z
    (0xFF41, 0xFF5A),    # 全角 a-z
)

#: 全角标点区间（第 13 步，默认关闭时才用）。
_FULLWIDTH_PUNCT_RANGES = (
    (0xFF01, 0xFF0F),    # ！＂＃＄％＆＇（）＊＋，－．／
    (0xFF1A, 0xFF20),    # ：；＜＝＞？＠
    (0xFF3B, 0xFF40),    # ［＼］＾＿｀
    (0xFF5B, 0xFF5E),    # ｛｜｝～
)

#: 全角与半角之间的固定码位差。
_HALFWIDTH_OFFSET = 0xFEE0

#: 中文标点的「左侧不该有空格」集合。
_CJK_TRAILING = "\uff0c\u3002\uff01\uff1f\uff1b\uff1a\u3001\uff09\u300b\u300d\u300f\u3011\u2026"

#: 中文标点的「右侧不该有空格」集合。
_CJK_LEADING = "\uff08\u300a\u300c\u300e\u3010"


# ============================================================
# 参数结构
# ============================================================
#
# 每一项都会经 get_config_schema() 变成前端表单，
# 所以 Field 的 title / description / default 是**对外契约**的一部分。


class NormalizationConfig(BaseModel):
    """归一化的完整参数结构：一步一个开关。"""

    encoding: bool = Field(
        default=True,
        title="统一编码",
        description=(
            "把孤立代理项（不合法 Unicode 序列）替换为 U+FFFD。"
            "不合法序列会让后续所有编码操作抛异常，建议常开。"
        ),
    )

    bom: bool = Field(
        default=True,
        title="移除 BOM",
        description=(
            "去掉 U+FEFF，包含出现在文本中间的零宽不换行空格。"
        ),
    )

    newline: bool = Field(
        default=True,
        title="统一换行",
        description="CRLF / CR / U+2028 / U+2029 统一为换行符。",
    )

    form_feed: bool = Field(
        default=True,
        title="换页符转换",
        description=(
            "换页符转为换行，**保留页边界** —— 直接删掉会丢掉分页信息，"
            "将来做页码定位就用不上了。"
        ),
    )

    special_spaces: bool = Field(
        default=True,
        title="特殊空格归一",
        description=(
            "不换行空格(NBSP)、全角空格、窄空格等各种 Unicode 空格"
            "统一为普通空格。"
        ),
    )

    control_chars: bool = Field(
        default=True,
        title="清理控制字符",
        description=(
            "删除 C0/C1 控制字符、零宽空格与双向文本标记。"
            "**保留**制表符、换行，以及零宽连接符（emoji 组合依赖它）。"
        ),
    )

    unicode_form: bool = Field(
        default=True,
        title="Unicode 形式",
        description=(
            "规范化到 NFC。**不做** NFKC 兼容折叠 —— 那会把带圈数字变成"
            "普通数字、把单位符号拆成字母，属于改内容而非改形。"
        ),
    )

    indent: bool = Field(
        default=True,
        title="缩进统一",
        description="行首的制表符展开为 4 个空格。只动行首，不动行内。",
    )

    whitespace: bool = Field(
        default=True,
        title="行内空白折叠",
        description=(
            "把行内连续的空格折叠成一个。**保留行首缩进** —— "
            "那是 Markdown 代码块、嵌套列表的结构信息。"
        ),
    )

    blank_lines: bool = Field(
        default=True,
        title="空行合并",
        description=(
            "连续 3 个以上换行合并为 2 个（最多保留一个空行）。"
            "**不删空行** —— 段落边界是切分阶段的最高级分隔符。"
        ),
    )

    digit_letter: bool = Field(
        default=True,
        title="全角字母数字转半角",
        description="全角英文字母与数字转半角。中文标点不受影响。",
    )

    punctuation: bool = Field(
        default=True,
        title="标点旁空白清理",
        description=(
            "去掉中文标点两侧的多余空格（「你好 ， 世界」→「你好，世界」）。"
            "**只动空白，标点本身一个字符都不改** —— 改成 ASCII 标点会让"
            "中文句子边界识别失效。"
        ),
    )

    fullwidth_punct: bool = Field(
        default=False,
        title="全角标点转半角（有风险）",
        description=(
            "默认**关闭**。开启后全角逗号句号括号会被转成 ASCII，"
            "后果是中文句子边界识别失效（下游切分靠中文句末标点认句子）、"
            "且标点语义改变。除非下游确实是纯英文管道，否则不要开。"
        ),
    )


# ============================================================
# 各步骤实现
# ============================================================


def _fix_encoding(text: str) -> str:
    return _LONE_SURROGATE.sub("\ufffd", text)


def _strip_bom(text: str) -> str:
    return text.replace("\ufeff", "")


def _unify_newline(text: str) -> str:
    return (
        text
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")    # LINE SEPARATOR
        .replace("\u2029", "\n")    # PARAGRAPH SEPARATOR
    )


def _convert_form_feed(text: str) -> str:
    return text.replace("\u000c", "\n")


def _unify_special_spaces(text: str) -> str:
    return _SPECIAL_SPACES.sub(" ", text)


def _strip_control_chars(text: str) -> str:
    return _CONTROL_CHARS.sub("", text)


def _to_nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def _fix_indent(text: str) -> str:
    """只展开**行首**的制表符；行内的留给空白折叠那一步。"""

    out: list[str] = []

    for line in text.split("\n"):

        index = 0
        while index < len(line) and line[index] in " \t":
            index += 1

        out.append(line[:index].replace("\t", "    ") + line[index:])

    return "\n".join(out)


def _collapse_inline_spaces(text: str) -> str:
    return _INLINE_SPACES.sub(" ", text)


def _merge_blank_lines(text: str) -> str:
    return _BLANK_RUN.sub("\n\n", text)


def _fold_halfwidth_alnum(text: str) -> str:
    out: list[str] = []

    for char in text:
        code = ord(char)
        if any(lo <= code <= hi for lo, hi in _HALFWIDTH_ALNUM_RANGES):
            out.append(chr(code - _HALFWIDTH_OFFSET))
        else:
            out.append(char)

    return "".join(out)


def _strip_spaces_around_cjk_punct(text: str) -> str:
    """
    去掉中文标点两侧的多余空格。

    只处理**中文标点**：ASCII 的逗号句号不在集合里，所以英文文本
    （含 "hello, world" 这种惯用空格）完全不受影响。
    """

    text = re.sub(rf"[ \t]+(?=[{_CJK_TRAILING}])", "", text)
    text = re.sub(rf"(?<=[{_CJK_LEADING}])[ \t]+", "", text)

    return text


def _fold_fullwidth_punct(text: str) -> str:
    out: list[str] = []

    for char in text:
        code = ord(char)
        if any(lo <= code <= hi for lo, hi in _FULLWIDTH_PUNCT_RANGES):
            out.append(chr(code - _HALFWIDTH_OFFSET))
        else:
            out.append(char)

    return "".join(out)


def _changed(before: str, after: str) -> int:
    """改动量：等长时数不同的位置，变短时数净减少的字符数。"""

    if before == after:
        return 0

    if len(before) == len(after):
        return sum(1 for a, b in zip(before, after) if a != b)

    return abs(len(before) - len(after))


#: (步骤名, 配置项名, 处理函数)。**顺序即执行顺序**，不要重排 ——
#: 依赖关系见模块 docstring。
_STEPS: tuple[tuple[str, str, object], ...] = (
    (_STEP_ENCODING, "encoding", _fix_encoding),
    (_STEP_BOM, "bom", _strip_bom),
    (_STEP_NEWLINE, "newline", _unify_newline),
    (_STEP_FORM_FEED, "form_feed", _convert_form_feed),
    (_STEP_SPECIAL_SPACES, "special_spaces", _unify_special_spaces),
    (_STEP_CONTROL_CHARS, "control_chars", _strip_control_chars),
    (_STEP_UNICODE_FORM, "unicode_form", _to_nfc),
    (_STEP_INDENT, "indent", _fix_indent),
    (_STEP_WHITESPACE, "whitespace", _collapse_inline_spaces),
    (_STEP_BLANK_LINES, "blank_lines", _merge_blank_lines),
    (_STEP_DIGIT_LETTER, "digit_letter", _fold_halfwidth_alnum),
    (_STEP_PUNCTUATION, "punctuation", _strip_spaces_around_cjk_punct),
    (_STEP_FULLWIDTH_PUNCT, "fullwidth_punct", _fold_fullwidth_punct),
)


# ============================================================
# 实现
# ============================================================


class TxtCommonNormalizer(TextDocumentNormalizer):
    """通用文本归一化器：13 步，每步可独立开关。"""

    config_model = NormalizationConfig

    async def execute_normalize(
            self,
            request: ProcessorRequest,
    ) -> TxtNormalizationResult:

        config = self.config_model.model_validate(request.config)

        text = request.text or ""

        steps: dict[str, int] = {}

        for name, option, handler in _STEPS:

            if not getattr(config, option):
                continue

            before = text
            text = handler(text)  # type: ignore[operator]

            changed = _changed(before, text)
            if changed:
                steps[name] = changed

        return TxtNormalizationResult(text=text, steps=steps)
