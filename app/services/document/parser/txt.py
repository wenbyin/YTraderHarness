from __future__ import annotations

from pydantic import BaseModel, Field

from app.services.document.domain.schemas import (
    ParseRequest,
    ParseResult,
)
from app.services.document.parser.text import TextDocumentParser


class CommonTxtDocumentParser(TextDocumentParser):
    """
    通用纯文本解析器。

    它不负责"文件字节 -> 文本"的解码 —— 上游（ingestion）已经把源文件的
    文本读出来了，这里直接取用。真正按格式解码的 loader 将来另开实现。

    本实现目前**没有参数**。要加参数时，声明一个 config_model
    （一个 pydantic BaseModel 的子类），get_config_schema() 会自动把它
    变成前端表单 —— 与 cleaner / chunker 的写法一致。
    ⚠️ 不要在 config_model 为 None 时调 model_validate：那是拿 None 调
    方法，必 AttributeError。
    """

    async def execute_parse(self, request: ParseRequest) -> str:
        # origin_text 在 ParseRequest 里是可选的，而模板 parse() 紧接着
        # 就要 .encode("utf-8") 算 checksum —— 这里兜成 "" 而不是把
        # None 漏下去。
        return request.origin_text or ""

    def is_default(self) -> bool:
        return True
