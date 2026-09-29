from abc import ABC, abstractmethod
from typing import BinaryIO

from ..domain.schemas.file import UploadFileRequest


class FileStorage(ABC):

    @abstractmethod
    async def put(
            self,
            file_id,
            file_name: str,
            content: bytes,
            content_type: str | None,
    ) -> str:
        """
        保存文件并返回 storage_key。
        """
        raise NotImplementedError

    @abstractmethod
    async def get(
            self,
            storage_key: str,
    ) -> bytes:
        raise NotImplementedError

    @abstractmethod
    async def delete(
            self,
            storage_key: str,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    async def exists(
            self,
            storage_key: str,
    ) -> bool:
        raise NotImplementedError
