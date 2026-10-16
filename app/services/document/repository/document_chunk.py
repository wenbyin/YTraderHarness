from abc import abstractmethod

from typing import Any
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import DocumentChunk


class DocumentChunkRepository(BaseRepository):
    """
    DocumentChunk 持久化抽象。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()。
    """

    @abstractmethod
    async def add_many(
            self,
            chunks: list[DocumentChunk],
    ) -> None:
        """
        批量写入 Chunk。

        供后续接入真正的切分流水线时使用。
        当前版本没有调用方。
        """
        raise NotImplementedError

    @abstractmethod
    async def delete_by_identity(
            self,
            document_version_id: UUID,
            chunker_key: str,
            chunker_config: dict[str, Any],
    ) -> int:
        """
        删掉该版本下**这一份切分产出**的全部 Chunk，返回删除行数。

        只给**重跑**用
        --------------
        重跑会重新生成 document_chunk_id（chunker 里是 uuid4），而
        chunk_index 从 0 重来，于是
        `(document_version_id, chunker_key, chunker_config, chunk_index)`
        这一组键必然与上一次留下的行重复 —— 表上的唯一约束
        uq_document_chunk_version_chunker_index 会**直接拒绝插入**。
        所以重跑要先清场。

        过滤条件就是那个唯一约束去掉 chunk_index，因此删掉的恰好是
        "这次插入会撞上"的那些行，不多不少。

        ⚠️ 刻意**不**提供"按版本全删"
        ----------------------------
        同一版本下可能有多份切分产出（不同 chunker / 不同配置，见
        chunker/text.py 开头）。按版本删会让**并发的另一个分支**刚写
        进去的行被这里抹掉 —— 而扇出的 N 路是并发跑的，那种丢失不会
        报错，只表现为"某一类 chunk 少了一批"。
        """
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            document_chunk_id: UUID,
    ) -> DocumentChunk | None:
        raise NotImplementedError

    @abstractmethod
    async def list(
            self,
            *,
            document_version_id: UUID | None = None,
            chunker_key: str | None = None,
            chunker_config: dict[str, Any] | None = None,
            processor_definition_id: UUID | None = None,
            chunk_index: int | None = None,
            keyword: str | None = None,
            offset: int = 0,
            limit: int = 20,
    ) -> tuple[list[DocumentChunk], int]:
        """
        分页查询 Chunk。

        返回 (当前页实体列表, 满足条件的总数)。

        这里按 **chunker 身份** 过滤而不是按 retrieval
        ---------------------------------------------
        document_chunk 上存的就是 (chunker_key, chunker_config)，没有
        retrieval 列 —— 切分产出只取决于 chunker，见实体的说明。
        调用方（服务层）负责先把 retrieval 解析成身份：
        `chunker/identity.py` 的 chunker_identity() 给出同样的两项。

        仓储不自己做这个解析 —— 它没有注册表，也拿不到 chunker 实现的
        config_model，自己拼一份归一化逻辑必然与写入侧漂移。

        排序固定为 document_version_id, chunker_key, chunk_index。
        带上 chunker_key 是必要的：一个版本下可能有多份切分产出，
        只按 chunk_index 排会让它们交错在一起。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_ids(
            self,
            document_version_id: UUID,
            processor_definition_ids: list[UUID],
    ) -> list[UUID]:
        """
        取出指定版本下、由指定 Pipeline 产出的 Chunk ID。

        用于 DocumentDetail.canonical_chunk_ids。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_by_ids(
            self,
            document_chunk_ids: list[UUID],
    ) -> list[DocumentChunk]:
        """
        按主键批量取 Chunk。

        查询期回表用：索引召回只回 chunk_id / score / metadata，
        正文与归属（版本 / 文档 / 类型）只有 PostgreSQL 有。

        返回顺序**不保证**与入参一致 —— 调用方按 chunk_id 自建
        映射对齐。顺序是融合阶段的职责，不是这里的。
        入参为空时返回空列表，不要退化成全表扫描。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_samplable_chunk_ids(
            self,
            *,
            document_type_definition_ids: list[UUID],
            chunker_keys: list[str],
            max_pool: int,
    ) -> list[dict]:
        """
        取出这些 DTD 下**可以拿来出题**的 chunk 的**身份**（不带正文）。

        返回元素的形状固定为：

            {
              "chunk_id":                    UUID,
              "document_version_id":         UUID,
              "document_type_definition_id": UUID,
              # 源文档身份 —— 出题**必需**（正文常写「本公司」）
              "document_id":                 UUID,
              "document_title":              str | None,
              "document_key":                str | None,
            }

        ⚠️ 文档身份三列不是附赠
        --------------------
        本语料是"一家公司一份文档"，而事件正文里公司名常常不出现 ——
        它写的是「本公司」（实测 1477/39697 个 chunk 含这三个字）。出题
        时必须把主体从**文档**取来写进问题；否则 100 家公司会出 100 条
        逐字相同的题，而「本公司」在索引里毫无区分度。

        它们不要额外的查询代价：DTD 在 `document` 上，两级 join 本来
        绕不过去，只是原先没 select 出这几列。

        返回这个形状而不是实体，是为了能直接喂给
        `eval/sampling.py::sample_chunks`（它按 `chunk_id` 与
        `document_type_definition_id` 分组抽名额）。

        ⚠️ **只回身份，不回正文，是刻意的**
        ---------------------------------
        语料可能有几万块（实测某 DTD 下 39697 块）。抽样只需要"有哪些
        候选"，而**正文只在抽中之后才需要** —— 那时用 `list_by_ids` 按
        选出的那一小批取。一次把几万条正文拉进内存，只是为了从里面挑
        100 条。

        ⚠️ `max_pool` 是有意设的上界，但它会**被日志说出来**
        --------------------------------------------------
        池子被截断会让抽样只在"前 max_pool 个"里进行，而那是**按
        chunk_id 排序**的前若干个、与内容质量无关。所以调用方看到
        `len(result) == max_pool` 时必须记一条 warning（见
        application/retrieval_eval_service_impl.py）—— 静默截断会让
        "数据集只覆盖了语料的一小部分"这件事没有任何痕迹。

        ⚠️ 排序是**必需**的，不是好看
        ---------------------------
        带 `LIMIT` 而不带 `ORDER BY` 时，PostgreSQL 返回的是**任意**子集
        （取决于物理存储与计划）。那会让"同种子 + 同语料 = 同一批 chunk"
        这条可断言的复现性彻底失效 —— 而且它每次跑都可能抽到不同的
        样本，看起来像模型或检索不稳定。

        过滤条件：文档与版本都未删除、`chunker_key` 在给定集合内、
        正文非空。**正文为空与偏移缺失的块不参与出题** —— 前者没有可
        问的内容，后者的区间对不上（求交会错位，且不报错）。
        """
        raise NotImplementedError
