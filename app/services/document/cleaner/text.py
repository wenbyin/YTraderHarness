"""
文本清洗器的中间基类（.txt 家族）。

与 parser 的层级一一对应
------------------------
    parser/base.py    DocumentParser         role 基类：process() -> parse()
    parser/text.py    TextDocumentParser     中间基类：模板 parse() -> execute_parse()
    parser/txt.py     CommonTxtDocumentParser 具体实现：execute_parse()

    cleaner/base.py   DocumentCleaner        role 基类：process() -> clean()
    cleaner/text.py   TextDocumentCleaner    中间基类：模板 clean() -> execute_clean()
    cleaner/txt.py    TxtDocumentCleaner     具体实现：execute_clean()

这一层存在的理由与解析器那边完全一样：把**共性**（算 checksum、
装 ProcessorResult、带回累积的 metadata）留在模板方法里，具体实现
只写"怎么清洗"，避免每个实现各写一遍包装逻辑然后各自漂移。

键也放在这一层，与 get_parser_key() 放在 TextDocumentParser 上同理。

与 TextDocumentParser 的差别
----------------------------
清洗器**没有** support_types / modality_types / __init__ —— 那是
loader 专属的"按文件后缀自选"概念（见 parser/base.py 的说明），
清洗器是被配置显式指定的，不做自选。

get_modality() 仍然要在这里给出来 —— 它是**所有 role 都有的适用域**
（见 DocumentProcessor 与 capability.py 的前端契约第 2 条），
不是 loader 专属概念。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.services.document.domain.schemas import (
    ProcessorRequest,
    ProcessorResult,
)

from .base import DocumentCleaner

__all__ = [
    "TextDocumentCleaner",
    "TxtCleaningResult",
]


class TxtCleaningResult(BaseModel):
    """
    清洗结果（含审计信息）。

    为什么不让 execute_clean() 直接返回 str
    --------------------------------------
    "清洗后的正文"只是结果的一半 —— **清掉了什么、各条规则各命中
    多少次**是观测与调参最需要的信息。如果直接返回 str，这些信息在
    实现内部就被丢掉了，外部再也拿不到。

    为什么放在中间层而不是具体实现里
    --------------------------------
    模板方法 clean() 与抽象钩子 execute_clean() 都在这一层声明，
    返回类型必须跟着契约一起在这一层 —— 否则具体层要从中间层
    import 契约、中间层又要从具体层 import 类型，直接成环。

    模板方法只取 text 去算 checksum；removed_lines / stats 保留给
    调用方按需消费（当前节点还没读，见文件末尾的说明）。
    """

    text: str

    removed_lines: list[tuple[str, str]] = Field(
        default_factory=list,
        description="被删除的行与原由，例如 ('第 3 页', 'page_number')。",
    )

    stats: dict[str, int] = Field(
        default_factory=dict,
        description="各条规则命中次数。",
    )


class TextDocumentCleaner(DocumentCleaner, ABC):
    """纯文本清洗器的中间基类。"""

    #: 与解析器的 TEXT_TXT_COMMON_PARSER 并列的键。
    #: 一旦发布**永不复用** —— 它会冻结进 PipelineConfigSnapshot，
    #: 历史运行靠它解释"当时用的是哪个清洗器"。
    name = "TEXT_TXT_COMMON_CLEANER"

    async def clean(
            self,
            request: ProcessorRequest,
    ) -> ProcessorResult:
        """
        对外暴露的 clean 方法（模板）。

        具体实现只写 execute_clean()，包装与 checksum 在这里统一做。
        """
        result = await self.execute_clean(request)

        cleaned_text = result.text

        checksum = hashlib.sha256(
            cleaned_text.encode("utf-8"),
        ).hexdigest()

        return ProcessorResult(
            document_id=request.document_id,
            document_version_id=request.document_version_id,
            text=cleaned_text,
            text_checksum=checksum,
            # metadata 是累积的 —— 这一阶段不增不减，必须原样带回
            # **上游那份**，否则下游再也看不到文档级元数据。
            #
            # 注意不是 dict(result)：那是清洗结果（text/removed_lines/
            # stats），不是文档元数据。混进去会把审计信息当成业务
            # 元数据一路带到 chunk 上。
            metadata=dict(request.metadata),
        )

    @abstractmethod
    async def execute_clean(
            self,
            request: ProcessorRequest,
    ) -> TxtCleaningResult:
        """
        真正清洗文本，返回带审计信息的结果。

        模板方法只取 result.text 去算 checksum 并装成 ProcessorResult；
        removed_lines / stats 由调用方按需消费。
        """
        raise NotImplementedError

