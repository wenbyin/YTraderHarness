from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod

from pydantic import BaseModel

from app.services.document.domain.schemas import (
    ParseRequest,
    ParseResult,
)
from app.services.document.parser.base import DocumentParser


class TextDocumentParserConfig(BaseModel):
    pass


class TextDocumentParser(DocumentParser, ABC):
    """
    文件解析器。

    职责：
        File bytes → normalized text / structured content

    不负责：
        Chunk
        Index
        Storage
    """

    config_model = TextDocumentParserConfig

    def __init__(self) -> None:
        super().__init__(
            support_types=[
                ".txt",
                ".md"
            ]
        )

    """
       对外暴露的parse方法
       """

    async def parse(
            self,
            request: ParseRequest,
    ) -> ParseResult:
        parsed_text = await self.execute_parse(request)

        checksum = hashlib.sha256(
            parsed_text.encode("utf-8"),
        ).hexdigest()

        return ParseResult(
            document_id=request.document_id,
            document_version_id=request.document_version_id,
            text=parsed_text,
            text_checksum=checksum,
        )

    @abstractmethod
    async def execute_parse(self, request: ParseRequest) -> str:
        raise NotImplementedError

    def get_parser_key(self) -> str:
        return "TEXT_TXT_COMMON_PARSER"

