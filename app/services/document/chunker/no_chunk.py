"""
不分块实现。

用途
----
有些 retrieval 就是要"整篇当一个块" —— 例如 retrieval_type=document 的
直读场景，或者文档本来就很短。与其在节点里写 if 判断"这个 retrieval 要
不要切分"，不如把它做成一个**正常的能力项**：前端在能力目录里选它即可，
节点侧一行分支都不用加。

这正是把能力暴露成"普通 processor"的价值：**选择由配置表达，不由代码
分支表达**。节点永远只做一件事 —— 按 key 取能力、调 process()。

无参数
------
它没有任何可调项，config_model 保持基类的 None，
get_config_schema() 会返回一个空的 object，前端不渲染表单。

⚠️ 与 512 token 的 embedder 搭配时要当心
-----------------------------------------
它产出的**唯一一个 chunk 就是整篇正文**。而
TEXT_BGE_SMALL_ZH_EMBEDDER 的输入上限是 512 token —— 超出的部分会被
**静默截断**（不报错），也就是一篇长文只有开头一小段真正进了向量。

所以：
    - 与这个 embedder 搭配时，只适合短文档
    - 长文档请用带 size 的 chunker，或先换成支持长上下文的 embedder
"""

from __future__ import annotations

from app.services.document.domain.schemas import ChunkRequest

from .text import ChunkSegment, TextDocumentChunker

__all__ = ["NoChunkChunker"]


class NoChunkChunker(TextDocumentChunker):
    """不分块：整篇正文作为单个 chunk。"""

    #: 键一旦发布**永不复用** —— 它会冻结进 PipelineConfigSnapshot。
    #: 与其他键的命名逻辑一致（末尾是 role 名词）。
    name = "TEXT_NO_CHUNK_CHUNKER"

    async def execute_chunk(
            self,
            request: ChunkRequest,
    ) -> list[ChunkSegment]:
        """返回唯一一段 —— 整篇正文。"""

        text = request.text or ""

        # 空文本不产出 chunk。模板本来也会丢掉空段，这里直接不给，
        # 省一次组装，语义也更直白：没内容就没块。
        if not text:
            return []

        # 偏移必须是相对输入正文的完整区间 ——
        # content == text[start_index:end_index] 这条不变式靠它成立，
        # 下游落库的 start/end 才指得回原文。
        return [
            ChunkSegment(
                text=text,
                start_index=0,
                end_index=len(text),
            )
        ]
