"""
业务级文档检索的应用层契约。

对应 app/services/document/document_retrieve.yaml。

一次调用 = 一轮检索
-------------------
本服务不**自动**做：查询改写、扩召回、重试、迭代检索、agent loop。
调用方拿到结果后自己决定要不要换个说法再调一次 ——
规格把这条写死了（"The service does not automatically expand recall,
rewrite queries, retry retrieval, or start an agent loop."）。

查询改写是唯一的例外，且是**显式**的：请求里传了
`query_rewriter` 才改，改一次，改失败就退回原文（见
`DocumentRetrieveResponse.query_rewrite` 的 `degraded`）。
没传 = 与从前逐字节相同。

因此本服务也不返回"还需要继续找"之类的信号，只回一轮的结果。

Recaller 不在契约里
-------------------
规格：Recaller 是内部实现细节，不通过公开 API 暴露。因此
DocumentRetrieveRequest 上没有 recaller 配置，本契约也不暴露
"用哪个召回器"。候选集由实现内联调用 IndexBackend.search 得到。
"""

from abc import abstractmethod

from app.common.service.base import BaseService

from ..domain.schemas import (
    DocumentRetrieveRequest,
    DocumentRetrieveResponse,
)

__all__ = ["DocumentRetrieveService"]


class DocumentRetrieveService(BaseService):
    """
    跨 DocumentTypeDefinition 的业务检索。

    装配路径由 document_type_definition_ids 的个数决定：

        1 个   -> 该 DTD 的 RetrieveProcessor
        >1 个  -> 各 DTD 各自跑完，再过 COMMON_RETRIEVE_PROCESSOR_CHAIN

    两次调用之间没有任何状态：同样的入参应当得到同样的结果
    （除了 retrieve_id 与底层索引的实时性）。
    """

    @abstractmethod
    async def retrieve_documents(
            self,
            request: DocumentRetrieveRequest,
    ) -> DocumentRetrieveResponse:
        """
        执行一轮检索。

        Args:
            request:
                目标 DTD、查询文本，以及可选的 RetrieveProcessor 覆盖。

        Returns:
            DocumentRetrieveResponse:
                本轮结果 + 实际生效的 RetrieveProcessor 回显。
                没有任何召回命中时 results 为空列表，**不是**错误。

        Raises:
            NotFoundError:
                任一 document_type_definition_id 不存在或已删除。
                整体失败而不是跳过它 —— 静默少查一个类型会让调用方
                把"这个类型没数据"和"这个类型没查"混为一谈。
            ValueError:
                DTD 上的 fusion / rerank 配置非法（例如 top_k 越界）。
        """
        raise NotImplementedError
