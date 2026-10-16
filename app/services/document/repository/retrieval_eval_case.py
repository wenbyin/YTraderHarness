from abc import abstractmethod
from uuid import UUID

from app.common.repository.base import BaseRepository

from ..domain.entities import (
    RetrievalEvalCase,
    RetrievalEvalCaseGoldChunk,
)
from ..domain.schemas.enum import EvalCaseValidationStatus

__all__ = ["RetrievalEvalCaseRepository"]


class RetrievalEvalCaseRepository(BaseRepository):
    """
    QA case 与它的金标集合的持久化抽象。

    两张表一个仓储：金标离开 case 没有意义（它们只被一起读、一起写），
    而导出与评测都要"case + 它的全部金标"这一个整体。

    Transaction rule
    ----------------
    实现不得调用 session.commit() / session.rollback()，
    事务生命周期由 Service 层拥有。
    """

    @abstractmethod
    async def add_cases(
            self,
            cases: list[RetrievalEvalCase],
    ) -> int:
        """
        批量新增 case，返回**实际写入**的条数。

        ⚠️ **冲突即跳过，不报错**（`ON CONFLICT DO NOTHING`），这是重试
        幂等的落点。

        一个窗口失败后重试时，那个窗口的 case 会被重造一遍。而"重造"
        有两种情形，必须分开看待：

            首轮就没成功 -> 一条都没写过 -> 正常插入
            首轮写成功过、窗口状态被误判成失败 -> 唯一约束挡住

        第二种情形就是"重试不能造出重复"这条要求的全部内容。靠应用层
        "先查有没有、再决定插不插"是**不够的**：两个 worker 同时重试
        同一个窗口时，两边都会查到"没有"，然后都插 —— 而库上的唯一约束
        是唯一能真正排掉这个竞态的地方。

        因此返回值是"这次真的插了几条"，而它**可能小于**入参长度。
        调用方拿它当 `valid_case_count` 是对的（那正是"这个窗口贡献了
        多少条"）。
        """
        raise NotImplementedError

    @abstractmethod
    async def add_gold_chunks(
            self,
            gold_chunks: list[RetrievalEvalCaseGoldChunk],
    ) -> int:
        """
        批量新增金标，返回实际写入条数。

        与 `add_cases` 同样的 ON CONFLICT DO NOTHING 语义：跨 chunk 裁定
        可能被重跑（换提示词、换模型），重跑不该在同一个 (case, chunk)
        上留下两行。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_cases(
            self,
            *,
            bundle_id: UUID,
            document_type_definition_id: UUID | None = None,
            validation_status: EvalCaseValidationStatus | None = None,
            offset: int = 0,
            limit: int = 50,
    ) -> tuple[list[RetrievalEvalCase], int]:
        """
        分页查询某个 bundle 的 case。返回 (当前页, 总数)。

        排序固定 `document_type_definition_id, source_chunk_id,
        question_order` —— 与导出的口径**一致**。两处顺序不同的后果是
        "预览看到的第 3 条"与"导出文件里的第 3 条"不是同一个东西，
        而复核一个可疑 case 时正是靠这个对应关系。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_all_validated_cases(
            self,
            bundle_id: UUID,
    ) -> list[RetrievalEvalCase]:
        """
        取一个 bundle 里**全部校验通过**的 case，按导出顺序排好。

        **不分页**，因为导出物必须是完整的、且顺序确定 —— 分页读出来再
        拼装会让"确定性导出"依赖调用方每次都拼对，而 `dataset_sha256`
        是建在这个确定性上的。
        """
        raise NotImplementedError

    @abstractmethod
    async def count_cases(
            self,
            bundle_id: UUID,
            *,
            validation_status: EvalCaseValidationStatus | None = None,
    ) -> int:
        """
        数一个 bundle 的 case。

        单列一个方法而不是复用 `list_cases(...)[1]`：计数要读**整个**
        bundle 的行（可能上千），而 `list_cases` 为了拿总数会把当页实体
        也构造出来 —— 那是一次白做的映射。
        """
        raise NotImplementedError

    @abstractmethod
    async def list_gold_chunks(
            self,
            eval_case_ids: list[UUID],
    ) -> dict[UUID, list[RetrievalEvalCaseGoldChunk]]:
        """
        批量取这些 case 的金标，按 case 分组。

        批量而不是逐个：导出时几百条 case 逐个查就是几百次往返（N+1），
        而它们的金标本来可以一次查完。

        返回 case_id -> 金标列表。**没有金标的 case 不出现在结果里**
        （用 `.get(id, [])` 取值）。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_many(
            self,
            eval_case_ids: list[UUID],
    ) -> list[RetrievalEvalCase]:
        """
        按主键批量取 case。

        ⚠️ 存在的理由：`list_cases` 是**按 bundle 分页**的，而调用方常常
        手上只有一批 case id（评测结果行、逐条诊断、导出）。没有这个方法
        就只能"先查 bundle、再把整个数据集的 case 读一遍，只为挑出那几条"
        —— 而那在 1000 条的数据集上是 20 倍的浪费，且看起来完全正常。

        入参为空时返回空列表，**不要退化成全表扫描**。
        返回顺序不保证与入参一致，调用方按 id 自建映射。
        """
        raise NotImplementedError

    @abstractmethod
    async def count_distinct_source_chunks(
            self,
            bundle_id: UUID,
    ) -> int:
        """
        数这个 bundle 涉及多少个**不同的**源 chunk。

        它是"实际抽中的 chunk 数"的口径 —— 不是 case 数除以 M。同一个
        源 chunk 出的题数可能少于 M（内容不够），所以 `case_count / M`
        会低估，而那个低估会让"数据集覆盖面"看起来比实际小。
        """
        raise NotImplementedError
