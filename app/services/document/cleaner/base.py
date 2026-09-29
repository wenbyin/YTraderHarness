"""
清洗能力契约（阶段 5，role=clean）。

职责：
    去页眉页脚、水印、重复行等噪声。

不负责：
    编码 / 换行 / 空白 / 全半角归一（那是 normalize，阶段 4）
    文档级元数据抽取（那是 enrich_metadata，阶段 6）
    Chunk / Index / Storage

与 DocumentParser 的关系
------------------------
两者都继承 DocumentProcessor，但**互不继承** —— 清洗器不是解析器。

共性（key / role / config_model / config_schema）留在
DocumentProcessor 上；差异留在各自的 role 基类里：

    DocumentParser    输入是**文件**（storage_key / raw）  -> parse()
    DocumentCleaner   输入是**文本**（上游产物）           -> clean()

因此 loader 专属的 support_types / modality_types / supports()
不出现在这里，清洗器也不需要实现 get_modality()、
不需要在构造时传 support_types。
"""

from __future__ import annotations

from abc import abstractmethod

from app.services.document.domain.schemas import (
    ProcessorRequest,
    ProcessorResult,
)
from app.services.document.domain.schemas.enum import ProcessorRole

# ⚠️ DocumentProcessor 目前放在 parser/ 包里（parser/processor.py）。
# 它是所有 role 的公共基类，住在某个 role 的包内属于历史错位；
# 搬出去要动 parser/__init__.py、registry.py、capability_service_impl.py
# 等多处，因此暂留原地 —— 但继承关系本身是对的，这里只是 import 路径难看。
from app.services.document.parser.processor import (
    TextDocumentProcessor,
)

__all__ = ["DocumentCleaner"]


class DocumentCleaner(TextDocumentProcessor):
    """
    清洗实现契约。

    只有清洗这一件事：文本进、文本出。切分与向量化是后面的阶段，
    不属于这里。
    """

    role = ProcessorRole.CLEAN

    def get_process_type(self) -> str:
        return "PROCESSOR_CLEANER"

    async def process(
            self,
            request: ProcessorRequest,
    ) -> ProcessorResult:
        """统一入口：节点只认 process()。"""
        return await self.clean(request)

    @abstractmethod
    async def clean(
            self,
            request: ProcessorRequest,
    ) -> ProcessorResult:
        """
        清洗一段文本。

        实现约定：
            - **只改 text**，不要动 metadata —— 那是 enrich 阶段的活，
              在这里改会让两个阶段的职责互相污染
            - 上游 metadata 应当原样带回（它是累积的）
            - 清洗后确实没有内容时**显式报错**，不要返回空串 ——
              空产物会让下游切分安静地产出零个 chunk，极难追
        """
        raise NotImplementedError
