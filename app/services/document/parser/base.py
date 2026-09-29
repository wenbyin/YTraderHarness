from __future__ import annotations
from abc import ABC, abstractmethod

from app.services.document.domain.schemas import (
    ParseRequest,
    ParseResult,
)
from app.services.document.domain.schemas.enum import ProcessorRole
from app.services.document.parser.processor import (
    TextDocumentProcessor,
)


class DocumentParser(TextDocumentProcessor):
    """
    文件解析器（阶段 3，role=loader）。

    职责：
        File bytes → normalized text / structured content

    不负责：
        Chunk
        Index
        Storage

    为什么 support_types / modality_types / supports() 在这里
    --------------------------------------------------------
    它们是 **loader 专属**概念：只有"按文件后缀自动选一个解析器"
    这件事才需要"我认哪些后缀""我是哪个模态"。

    放在公共基类 DocumentProcessor 上会连累其他 role ——
    清洗器 / 切分器被迫实现自己用不到的 get_modality()，
    还得在构造时传一个无意义的 support_types。
    """

    #: loader 是写入流水线的阶段 3（见 ProcessorRole 的说明）。
    role = ProcessorRole.LOADER

    def __init__(
            self,
            support_types: list[str],
    ) -> None:
        self.support_types = support_types
        self.modality_types: list[str] = [
            "text",
            "image",
            "audio",
            "video",
        ]

    def get_key(self) -> str:
        """
        沿用既有的 get_parser_key()，不引入第二套身份。

        键的取值例如 "TEXT_TXT_COMMON_PARSER"，它同时是
        DocumentProcessorDefinition.loader_name 的取值。
        """
        return self.get_parser_key()

    async def process(
            self,
            request: ParseRequest,
    ) -> ParseResult:
        """统一入口：节点只认 process()。"""
        return await self.parse(request)

    def get_process_type(self) -> str:
        return "PROCESSOR_PARSER"

    @abstractmethod
    async def parse(
            self,
            request: ParseRequest,
    ) -> ParseResult:
        raise NotImplementedError

    @abstractmethod
    def get_parser_key(self) -> str:
        raise NotImplementedError

    def is_default(self) -> bool:
        """同一后缀有多个候选 loader 时，是否为默认选中的那个。"""
        return False

    def supports(
            self,
            suffix: str,
    ) -> bool:
        return (
            self.get_modality() in self.modality_types
            and suffix in self.support_types
        )
