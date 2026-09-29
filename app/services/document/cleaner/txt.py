"""
.txt 的清洗实现（具体层）。

继承层级（严格对应 parser）
--------------------------
    DocumentProcessor
      └── DocumentCleaner          cleaner/base.py   role 基类
            └── TextDocumentCleaner cleaner/text.py  中间基类（模板）
                  └── TxtDocumentCleaner              具体实现 ← 本文件

本文件只做两件事：
    1. 定义这个实现的**参数结构**（下面那组 Config）
    2. 实现 execute_clean()：把配置编译成正则，逐行判定并剔除

包装（checksum / ProcessorResult / metadata 透传）全部由
TextDocumentCleaner.clean() 这个模板方法负责，这里不重复。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

from pydantic import BaseModel, Field

from app.services.document.domain.schemas import (
    ProcessorRequest,
)

from .text import (
    TextDocumentCleaner,
    TxtCleaningResult,
)

__all__ = ["TxtDocumentCleaner"]


# ============================================================
# 参数结构
# ============================================================
#
# 每个 Config 都会经 get_config_schema() 变成前端表单，
# 所以 Field 的 title / description / default 是**对外契约**的一部分。


class PageNumberConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="是否删除页码。",
    )
    patterns: list[str] = Field(
        default=[
            r"^\s*[-—–\[\(【]*\s*\d{1,4}\s*[-—–\]\)】]*\s*$",
            r"^\s*第\s*\d+\s*页(\s*[/共]\s*\d+\s*页)?\s*$",
            r"(?i)^\s*Page\s+\d+(\s+of\s+\d+)?\s*$",
            r"(?i)^\s*[ivxlcdm]{2,7}\s*$",
        ],
        description="页码正则列表，按 fullmatch 匹配整行。",
    )

    @classmethod
    def _patterns_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("page_number.patterns 不能为空")
        return v


class WatermarkConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="是否删除整行水印。",
    )
    patterns: list[str] = Field(
        default=[
            r"草稿",
            r"机密",
            r"内部资料",
            r"样本",
            r"复印无效",
            r"禁止外传",
            r"confidential",
            r"draft",
            r"sample",
            r"internal use only",
        ],
        description="整行匹配的水印正则，默认忽略大小写。",
    )

    @classmethod
    def _patterns_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("watermark.patterns 不能为空")
        return v


class SeparatorConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="是否删除重复分隔线（----、====、**** 等）。",
    )

    pattern: str = Field(
        default=r"^([-=_*~#·•])\1*$",
        description="重复分隔线正则，按 fullmatch 匹配整行。",
    )

    min_length: int = Field(
        default=3,
        ge=1,
        description="分隔线最短长度，低于此长度不视为分隔线。",
    )


class TableSeparatorConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="是否删除表格线残留（+----+----+、|----|----| 等）。",
    )

    pattern: str = Field(
        default=r"^\s*[+|][\s\-+|:=]{3,}[+|]\s*$",
        description="表格线残留正则，按 fullmatch 匹配整行。",
    )


class OcrGarbageConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="是否删除 OCR 乱码片段。",
    )
    pattern: str = Field(
        default=r"^[\W_]{3,}$",
        description=(
            "整行乱码正则，按 fullmatch 匹配整行即判定为乱码。"
            "补充 garbage_chars / 占比两条规则。"
        ),
    )
    garbage_chars: str = Field(
        default="�□■◆●○◇¤§¶†‡※…",
        description="被视作乱码的字符集合。",
    )
    normal_punct: str = Field(
        default="，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-",
        description=(
            "「正常标点」，不计入乱码符号占比。"
            "不在这个集合里的符号才会被算作噪声。"
        ),
    )
    min_length: int = Field(
        default=4,
        ge=1,
        description="判定为乱码片段的最短长度。",
    )
    bad_char_ratio: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="乱码字符占比达到该值即判定为乱码行。",
    )
    symbol_ratio: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="非正常标点符号占比达到该值即判定为乱码行。",
    )


class RepeatedBoilerplateConfig(BaseModel):
    enabled: bool = Field(
        default=True,
        description="是否删除重复出现的短行。",
    )
    min_count: int = Field(
        default=3,
        ge=2,
        description="重复出现次数达到该值即视为页眉/页脚/水印。",
    )
    max_length: int = Field(
        default=80,
        ge=1,
        description="参与重复统计的行最大长度。",
    )


class LineNumberConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description="是否删除行首行号。默认关闭，可能误删有序列表。",
    )
    pattern: str = Field(
        default=r"^\s*\d{1,4}[.、)）]\s*",
        description=(
            "行首行号正则，按 match 匹配行首。"
            "默认要求行号后跟分隔符，避免误删「2024 年」这类正文。"
        ),
    )


class TxtCleaningConfig(BaseModel):
    """TxtDocumentCleaner 的完整参数结构。"""

    page_number: PageNumberConfig = Field(
        default_factory=PageNumberConfig,
        description="页码",
    )
    watermark: WatermarkConfig = Field(
        default_factory=WatermarkConfig,
        description="水印",
    )
    separator: SeparatorConfig = Field(
        default_factory=SeparatorConfig,
        description="分隔符",
    )
    table_separator: TableSeparatorConfig = Field(
        default_factory=TableSeparatorConfig,
        description="表格",
    )
    ocr_garbage: OcrGarbageConfig = Field(
        default_factory=OcrGarbageConfig,
        description="OCR乱码",
    )
    repeated_boilerplate: RepeatedBoilerplateConfig = Field(
        default_factory=RepeatedBoilerplateConfig,
        description="重复行",
    )
    line_number: LineNumberConfig = Field(
        default_factory=LineNumberConfig,
        description="行码",
    )


# ============================================================
# 实现
# ============================================================


class TxtDocumentCleaner(TextDocumentCleaner):
    """纯文本清洗器。

    键（name）继承自 TextDocumentCleaner —— 与解析器那边
    get_parser_key() 放在 TextDocumentParser 上同理，
    具体实现不重复声明。
    """

    config_model = TxtCleaningConfig

    async def execute_clean(
            self,
            request: ProcessorRequest,
    ) -> TxtCleaningResult:
        """按配置清洗正文，返回清洗后的文本。"""

        config = self.config_model.model_validate(
            request.config,
        )

        result = self.clean_txt(
            text=request.text or "",
            config=config,
        )

        return result

    @staticmethod
    def clean_txt(
            text: str,
            config: TxtCleaningConfig,
    ) -> TxtCleaningResult:
        # 按配置编译正则
        page_patterns = [
            re.compile(p) for p in config.page_number.patterns
        ]
        sep_re = re.compile(config.separator.pattern)
        table_sep_re = re.compile(config.table_separator.pattern)
        garbage_re = re.compile(config.ocr_garbage.pattern)
        line_num_re = re.compile(config.line_number.pattern)
        normal_punct = set(config.ocr_garbage.normal_punct)

        lines = text.splitlines(keepends=True)

        repeated = (
            _detect_repeated_boilerplate(
                lines, config.repeated_boilerplate
            )
            if config.repeated_boilerplate.enabled
            else set()
        )

        kept: list[str] = []
        removed: list[tuple[str, str]] = []
        stats: Counter[str] = Counter()

        for raw in lines:
            stripped = raw.strip()
            reason: str | None = None

            if not stripped:
                pass  # 空行交给 Normalization

            elif config.page_number.enabled and _is_page_number(
                    stripped, page_patterns
            ):
                reason = "page_number"

            elif config.separator.enabled and _is_separator(
                    stripped, sep_re, config.separator.min_length
            ):
                reason = "separator"

            elif config.table_separator.enabled and _is_table_separator(
                    stripped, table_sep_re
            ):
                reason = "table_separator"

            elif config.watermark.enabled and _is_watermark(
                    stripped, config.watermark.patterns
            ):
                reason = "watermark"

            elif config.ocr_garbage.enabled and _is_ocr_garbage(
                    stripped,
                    config.ocr_garbage,
                    garbage_re,
                    normal_punct,
            ):
                reason = "ocr_garbage"

            elif (
                    config.repeated_boilerplate.enabled
                    and stripped in repeated
            ):
                reason = "repeated_boilerplate"

            elif config.line_number.enabled and _is_line_number(
                    stripped, line_num_re
            ):
                reason = "line_number"

            if reason is not None:
                removed.append((raw, reason))
                stats[reason] += 1
                continue

            kept.append(raw)

        return TxtCleaningResult(
            text="".join(kept),
            removed_lines=removed,
            stats=dict(stats),
        )


# ============================================================
# 单行判定辅助
# ============================================================


def _is_page_number(stripped: str, patterns: list[re.Pattern]) -> bool:
    if not stripped:
        return False
    return any(p.fullmatch(stripped) for p in patterns)


def _is_separator(
        stripped: str,
        pattern: re.Pattern,
        min_len: int,
) -> bool:
    if len(stripped) < min_len:
        return False
    return bool(pattern.fullmatch(stripped))


def _is_table_separator(stripped: str, pattern: re.Pattern) -> bool:
    return bool(pattern.fullmatch(stripped))


def _is_watermark(stripped: str, patterns: list[str]) -> bool:
    if not stripped:
        return False
    return any(
        re.fullmatch(pat, stripped, flags=re.I)
        for pat in patterns
    )


def _is_ocr_garbage(
        stripped: str,
        cfg: OcrGarbageConfig,
        garbage_re: re.Pattern,
        normal_punct: set[str],
) -> bool:
    if not stripped:
        return False

    if garbage_re.fullmatch(stripped):
        return True

    garbage_chars = set(cfg.garbage_chars)

    bad = sum(
        1 for ch in stripped
        if ch in garbage_chars or ord(ch) == 0xFFFD
    )
    if len(stripped) >= cfg.min_length and (
            bad / len(stripped) >= cfg.bad_char_ratio
    ):
        return True

    symbols = sum(
        1 for ch in stripped
        if not ch.isalnum()
        and not ch.isspace()
        and ch not in normal_punct
    )
    if len(stripped) >= cfg.min_length and (
            symbols / len(stripped) >= cfg.symbol_ratio
    ):
        return True

    return False


def _is_line_number(stripped: str, pattern: re.Pattern) -> bool:
    return bool(pattern.match(stripped))


def _detect_repeated_boilerplate(
        lines: Iterable[str],
        cfg: RepeatedBoilerplateConfig,
) -> set[str]:
    counts: Counter[str] = Counter()

    for raw in lines:
        key = raw.strip()
        if not key or len(key) > cfg.max_length:
            continue
        if re.search(r"[。！？.!?]", key):
            continue
        counts[key] += 1

    return {k for k, c in counts.items() if c >= cfg.min_count}
