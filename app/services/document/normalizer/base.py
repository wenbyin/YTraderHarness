"""
归一化能力契约（阶段 4，role=normalize）。

职责：
    统一文本的**形** —— 编码合法性、Unicode 形式、换行、空白、
    全角半角、控制字符等。

不负责（这是本 role 的硬边界）：
    不删内容、不抽元数据、不切分、不索引。

"只改形不改质" 怎么落实
-----------------------
这句话如果只是写在文档里，实现时一定会被人越界（顺手"清洗"掉点东西
看起来总是更"干净"）。所以边界定成两条可检验的性质：

    1. **幂等**：normalize(normalize(x)) == normalize(x)
    2. **实体不丢**：CJK 字符与字母数字一个不少（全角字母数字折叠成
       半角属于"形"，但字符类别与其身份不变）

    —— 换句话说，这个阶段**只会让字符变样或消失于"空白/控制字符"
    这两类非内容物**，不会让一个有语义的字符凭空不见。

清洗是下一个阶段（role=clean）的事：去页眉页脚、水印、重复行那些
**确实要删东西**的处理在那边，两边不要互相兼职。

与其它 role 的关系
------------------
与 DocumentParser / DocumentCleaner / DocumentChunker / DocumentEmbedder
平级：都继承 DocumentProcessor，彼此互不继承。
"""

from __future__ import annotations

from abc import abstractmethod

from app.services.document.domain.schemas import (
    ProcessorRequest,
    ProcessorResult,
)
from app.services.document.domain.schemas.enum import ProcessorRole

# ⚠️ DocumentProcessor 目前住在 parser/ 包里，与 cleaner/base.py 等
# 同样的历史错位。继承关系本身是对的，只是 import 路径难看。
from app.services.document.parser.processor import (
    TextDocumentProcessor,
)

__all__ = ["DocumentNormalizer"]


class DocumentNormalizer(TextDocumentProcessor):
    """
    归一化实现契约。

    文本进、文本出（基数 1→1），因此产物走 ProcessorResult.text，
    不像 chunker / embedder 那样需要额外的列表通道。
    """

    role = ProcessorRole.NORMALIZE

    def get_process_type(self) -> str:
        return "PROCESSOR_NORMALIZER"

    async def process(
            self,
            request: ProcessorRequest,
    ) -> ProcessorResult:
        """
        统一入口：节点只认 process()。

        这里只做转发 —— 真正的包装（算 checksum、装 ProcessorResult、
        带回 metadata）在中间层的模板方法里，与 cleaner 的分工一致。
        """
        return await self.normalize(request)

    @abstractmethod
    async def normalize(
            self,
            request: ProcessorRequest,
    ) -> ProcessorResult:
        """
        归一化，返回 ProcessorResult。

        实现必须满足 base 模块 docstring 里那两条性质：
        幂等、且不丢有语义的字符。
        """
        raise NotImplementedError
