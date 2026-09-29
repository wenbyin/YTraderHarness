"""
处理能力目录的实现。

数据源只有一个：进程启动时由 PipelineRegistry 反射出来的能力表。
它是**不可变的**（registry 构造后即冻结），因此这里没有任何缓存
需求 —— 每次请求重建一遍 DTO 的成本远低于一次数据库往返。

一个刻意的失败模式
------------------
selector.modality_type 用的是 ModalityType 枚举，pydantic 会校验。
如果某个 loader 的 get_modality() 拼错（例如返回 "txt"），
本接口会直接报错，而不是静默给前端一个过滤不出来的值。

这正是想要的：DocumentParser.supports() 里
`get_modality() in self.modality_types` 那一句对拼写错误是**静默
返回 False** 的（该 parser 会从选择里凭空消失），而能力目录会把
同一个 bug 显式暴露出来。
"""

from __future__ import annotations

from app.common.exceptions import NotFoundError

from app.common.service.base import ServiceContext

from ..domain.schemas.capability import (
    ProcessorCapabilityCatalogResponse,
    ProcessorCapabilityEntry,
    ProcessorCapabilitySelector,
)
from ..domain.schemas.enum import ModalityType, ProcessorRole
from ..parser.processor import DocumentProcessor
from ..pipeline.registry import PipelineRegistry
from .capability_service import ProcessorCapabilityService

__all__ = ["ProcessorCapabilityServiceImpl"]


class ProcessorCapabilityServiceImpl(ProcessorCapabilityService):
    """只读能力目录。持有的是 registry，不是 session。"""

    def __init__(
            self,
            registry: PipelineRegistry,
            *,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._registry = registry

    # ========================================================
    # 查询
    # ========================================================

    async def list_capabilities(
            self,
            *,
            role: ProcessorRole | None = None,
            modality_type: ModalityType | None = None,
            file_extension: str | None = None,
    ) -> ProcessorCapabilityCatalogResponse:

        entries = self._build_catalog()

        if role is not None:
            entries = [
                e for e in entries if e.role == role
            ]

        if modality_type is not None:
            # 对所有 role 都成立 —— modality 是能力的适用域，
            # 不是 loader 专属（见 capability.py 的前端契约第 2 条）。
            entries = [
                e for e in entries
                if e.modality_type == modality_type
            ]

        if file_extension is not None:
            wanted = self._normalize_suffix(file_extension)

            entries = [
                e for e in entries
                if (
                    e.selector is not None
                    and wanted in e.selector.support_suffix
                )
            ]

        return ProcessorCapabilityCatalogResponse(
            items=entries,
            support_suffix=self._suffix_union(entries),
            total=len(entries),
        )

    async def get_capability(
            self,
            processor_key: str,
    ) -> ProcessorCapabilityEntry:

        for entry in self._build_catalog():
            if entry.processor_key == processor_key:
                return entry

        raise NotFoundError(
            f"Unknown processor capability: {processor_key!r}. "
            f"Registered: "
            f"{sorted(e.processor_key for e in self._build_catalog())}."
        )

    # ========================================================
    # 组装
    # ========================================================

    def _build_catalog(self) -> list[ProcessorCapabilityEntry]:
        """
        把注册表里的能力实例转成目录条目。

        尚未归类（role 为 None）的能力被跳过 —— 宁可它不出现，
        也不要给一个猜出来的 role。
        """

        entries: list[ProcessorCapabilityEntry] = []

        for capability in self._registry.all_capabilities():
            entry = self._to_entry(capability)

            if entry is not None:
                entries.append(entry)

        return entries

    @classmethod
    def _to_entry(
            cls,
            capability: DocumentProcessor,
    ) -> ProcessorCapabilityEntry | None:

        capability_type = type(capability)

        role = capability_type.role

        if role is None:
            return None

        selector: ProcessorCapabilitySelector | None = None

        if role is ProcessorRole.LOADER:
            # 只有 loader 有选择器：文件刚进来、还没查任何配置时，
            # 它靠 supports() 按后缀自选。其他 role 由配置显式点名。
            selector = ProcessorCapabilitySelector(
                support_suffix=sorted(
                    cls._normalize_suffix(suffix)
                    for suffix in capability.support_types
                ),
                is_default=capability.is_default(),
            )

        return ProcessorCapabilityEntry(
            processor_key=capability.get_key(),
            role=role,
            # 全员都有：这个能力处理什么模态的内容。
            modality_type=capability.get_modality(),
            selector=selector,
            config_schema=capability_type.get_config_schema(),
        )

    # ========================================================
    # Helpers
    # ========================================================

    @staticmethod
    def _suffix_union(
            entries: list[ProcessorCapabilityEntry],
    ) -> list[str]:
        """
        本次返回的 loader 条目里全部后缀的去重并集，已排序。

        受过滤影响是刻意的：过滤到 role=chunk 时它就该是空的，
        否则前端会把一个没有任何 loader 能处理的格式当成可上传。
        """

        return sorted({
            suffix
            for entry in entries
            if entry.selector is not None
            for suffix in entry.selector.support_suffix
        })

    @staticmethod
    def _normalize_suffix(value: str) -> str:
        """
        归一化为小写 + 带前导点。

        规格承诺 support_suffix 就是这个形态，而 file_extension
        过滤也走同一个函数 —— 否则 ".TXT" 与 "txt" 会匹配不上。
        """

        suffix = value.strip().lower()

        if not suffix or suffix.startswith("."):
            return suffix

        return f".{suffix}"
