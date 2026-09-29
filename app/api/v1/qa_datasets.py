"""
QA 测试数据集生成端点。

**手动调用**：人工触发一次、看返回的统计、再去磁盘上取产物。它不挂在
任何管线步骤上，调度器也不会自动拉起它。

因为是同步接口，一次调用会**阻塞到生成结束**（窗口数 × 单次 LLM 耗时），
所以请求体里有 `max_windows` 且默认只有 2 —— 详见 schema 里的说明。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.common.utils import to_http_error

from ...services.document.application.qa_dataset_service import (
    QADatasetService,
)
from ...services.dependencies import Service
from ...services.document.domain.schemas.qa_dataset import (
    QADatasetGenerateRequest,
    QADatasetGenerateResponse,
)

router = APIRouter(
    tags=["QA Dataset"],
)


@router.post(
    "/qa-datasets:generate",
    response_model=QADatasetGenerateResponse,
)
async def generate_qa_dataset(
        request: QADatasetGenerateRequest,
        service: QADatasetService = Service[QADatasetService],
) -> QADatasetGenerateResponse:
    """
    对某个文档版本生成一份 QA 测试数据集。

    产物**不在响应里**，而在返回的 `dataset_path` 指向的文件里 ——
    数据集可能有几百条，塞进 JSON 响应既没必要也难用。
    """

    try:
        return await service.generate(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc
