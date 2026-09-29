from abc import abstractmethod
from typing import Any
from uuid import UUID

from app.common.service.base import BaseService

from ..domain.schemas.file import (
    DeleteFileRequest,
    DeleteFileResponse,
    GetFileContentRequest,
    GetFileRequest,
    GetFileResponse,
    FileContentResponse,
    FileResponse,
    ListFilesRequest,
    ListFilesResponse,
    UploadFileRequest,
)


class FileService(BaseService):

    @abstractmethod
    async def upload_file(
            self,
            request: UploadFileRequest,
    ) -> FileResponse:
        raise NotImplementedError

    @abstractmethod
    async def store_artifact(
            self,
            *,
            kind: str,
            content: bytes,
            content_type: str | None = None,
            tenant_id: UUID | None = None,
            metadata: dict[str, Any] | None = None,
    ) -> FileResponse:
        """
        写入一个流水线中间产物。

        与 upload_file 的区别：

            - 按 (tenant_id, checksum) 去重，命中则复用既有 File
            - file_name 填 kind（内部标识，非展示名）
            - 不做面向用户的上传校验

        产物同样是 file 行，但语义是系统生成物。
        当前没有字段区分两者（已知待办，见实现处的说明）。

        Raises:
            ValueError: kind 为空或 content 为空。
        """
        raise NotImplementedError

    @abstractmethod
    async def get_file(
            self,
            request: GetFileRequest,
    ) -> GetFileResponse:
        raise NotImplementedError

    @abstractmethod
    async def get_file_content(
            self,
            request: GetFileContentRequest,
    ) -> FileContentResponse:
        raise NotImplementedError

    @abstractmethod
    async def delete_file(
            self,
            request: DeleteFileRequest,
    ) -> DeleteFileResponse:
        raise NotImplementedError

    @abstractmethod
    async def list_files(
            self,
            request: ListFilesRequest,
    ) -> ListFilesResponse:
        """
        分页查询 File 列表。

        Raises:
            ValueError:
                created_from 晚于 created_to。
        """
        raise NotImplementedError
