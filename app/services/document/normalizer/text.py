"""
文本归一化器的中间基类（.txt 家族）。

与 parser / cleaner / chunker 的层级一一对应
--------------------------------------------
    parser/base.py      DocumentParser        role 基类：process() -> parse()
    parser/text.py      TextDocumentParser    中间基类：模板 parse() -> execute_parse()
    parser/txt.py       CommonTxtDocumentParser

    cleaner/base.py     DocumentCleaner       role 基类：process() -> clean()
    cleaner/text.py     TextDocumentCleaner   中间基类：模板 clean() -> execute_clean()
    cleaner/txt.py      TxtDocumentCleaner

    chunker/base.py     DocumentChunker       role 基类：process() -> chunk()
    chunker/text.py     TextDocumentChunker   中间基类：模板 chunk() -> execute_chunk()
    chunker/txt.py      TxtRecursiveChunker

    normalizer/base.py  DocumentNormalizer      role 基类：process() -> normalize()
    normalizer/text.py  TextDocumentNormalizer  中间基类：模板 normalize() -> execute_normalize()
    normalizer/txt.py   TxtCommonNormalizer     ← 具体实现

这一层做的是**包装**：算 checksum、装 ProcessorResult、原样带回累积
的 metadata。具体实现只回答"怎么归一化"。

键（name）也放在这一层，与 get_parser_key() 放在 TextDocumentParser
上同理 —— 但注意：**同一家族有第二个具体实现时，它必须自己覆盖
name**，否则两个实现会拿到同一个键，在 PipelineRegistry 里静默互相
覆盖（注册表现在会显式报冲突，见 registry.py）。
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

from app.services.document.domain.schemas import (
    ProcessorRequest,
    ProcessorResult,
)

from .base import DocumentNormalizer

__all__ = [
    "TextDocumentNormalizer",
    "TxtNormalizationResult",
]


class TxtNormalizationResult(BaseModel):
    """
    归一化结果（含审计信息）。

    为什么不让 execute_normalize() 直接返回 str
    -------------------------------------------
    与 cleaner 的 TxtCleaningResult 同一个理由：**哪一步改动了多少**
    是排查"文本怎么变成这样了"时唯一有用的线索。直接返回 str 的话，
    这些信息在实现内部就被丢掉了。

    模板方法只取 text 去算 checksum；steps 保留给调用方按需消费
    （当前节点还没读，与 TxtCleaningResult 的处境相同）。
    """

    text: str

    steps: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "每一步的改动量：键是步骤名，值是该步改动的字符数"
            "（替换类按「有多少个位置不同」计，删除类按净减少数计）。"
            "没触发改动的步骤不出现在这里。"
        ),
    )


class TextDocumentNormalizer(DocumentNormalizer, ABC):
    """纯文本归一化器的中间基类。"""

    #: 与 TEXT_TXT_COMMON_PARSER / CLEANER 等并列的键。
    #: 一旦发布**永不复用** —— 它会冻结进 PipelineConfigSnapshot。
    name = "TEXT_TXT_COMMON_NORMALIZER"

    async def normalize(
            self,
            request: ProcessorRequest,
    ) -> ProcessorResult:
        """
        对外暴露的 normalize 方法（模板）。

        具体实现只写 execute_normalize()，包装在这里统一做。
        """

        result = await self.execute_normalize(request)

        text = result.text

        return ProcessorResult(
            document_id=request.document_id,
            document_version_id=request.document_version_id,
            text=text,
            text_checksum=hashlib.sha256(
                text.encode("utf-8")
            ).hexdigest(),
            # metadata 是累积的 —— 这一阶段**不抽元数据**（那是阶段 6
            # 的活），因此原样带回上游那份，不增不减。
            metadata=dict(request.metadata),
        )

    @abstractmethod
    async def execute_normalize(
            self,
            request: ProcessorRequest,
    ) -> TxtNormalizationResult:
        """真正做归一化，返回结果与逐步审计信息。"""
        raise NotImplementedError

