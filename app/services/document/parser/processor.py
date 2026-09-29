"""
Processor 能力契约 —— 所有 role 的公共基类。

只放**每个 role 都有**的东西
----------------------------
    key / role / config_model        身份与配置
    get_key() / get_config_schema()  能力目录用的访问器
    process()                        统一的执行入口

role 专属的概念**不在这里**
---------------------------
loader 的 support_types / modality_types / supports() 属于
DocumentParser（见 parser/base.py）；chunker 的 chunk() 属于
ChunkStrategy。混进这个基类，会逼着每个 role 实现自己用不到的
东西 —— 一个清洗器为什么要有"我是哪个 modality"？

继承结构
--------
    DocumentProcessor                  ← 本模块，只放共性
      ├── DocumentParser               ← parser/base.py    role=loader
      │     └── TextDocumentParser
      │           └── CommonTxtDocumentParser
      └── DocumentCleaner              ← cleaner/base.py   role=clean
            └── TxtDocumentCleaner

新增一个 role：建一个同级包，写 role 基类 + 具体实现，再把包名
登记进 registry.py 的 CAPABILITY_PACKAGES —— 能力目录自动包含它。

不负责：
    Chunk（那是 ChunkStrategy）
    Index（那是 IndexBackend）
    Storage（那是 FileStorage）
"""

from __future__ import annotations

from abc import ABC, abstractmethod, abstractclassmethod
from typing import Any, ClassVar, Literal

from langchain_core.tools import tool
from pydantic import BaseModel

from app.services.document.domain.schemas import (
    ProcessorRequest,
    ProcessorResult,
)
from app.services.document.domain.schemas.enum import ProcessorRole

__all__ = ["DocumentProcessor", "TextDocumentProcessor"]


class DocumentProcessor(ABC):
    """
    Processor 实现契约的公共部分。

    name 是它在 DocumentProcessorDefinition.processor_config 与
    PipelineRegistry 里的注册名 —— 节点按名字查表，不直接 import
    具体实现，因此换实现不需要改节点。
    """

    #: 注册名，例如 "unicode_normalizer" / "boilerplate_cleaner"。
    #: 仅作 get_key() 的兜底 —— 能力目录统一走 get_key()。
    name: str = ""

    #: 该能力接受的参数结构。无参数的实现留 None。
    config_model: ClassVar[type[BaseModel] | None] = None

    #: 在写入流水线里承担的阶段。
    #: None 表示尚未归类 —— 这类能力不进能力目录。
    role: ClassVar[ProcessorRole | None] = None

    #: get_config_schema() 的缓存。纯函数，按类缓存一份即可。
    config_model_schema: ClassVar[dict[str, Any] | None] = None

    @abstractmethod
    async def process(
            self,
            request: ProcessorRequest,
    ) -> ProcessorResult:
        """
        统一执行入口。

        各 role 基类把它覆写成自己的语义方法：
            DocumentParser   -> parse()
            DocumentCleaner  -> clean()
        节点只认这一个入口，因此换 role 基类不影响节点。
        """
        raise NotImplementedError

    @abstractmethod
    def get_process_type(self) -> str:
        """
        历史遗留的类型标记，取值是魔法字符串。

        已被 role 取代 —— 新代码用 `type(capability).role`。
        保留是因为还没有任何地方在读它，删掉会是一次无谓的
        破坏性改动；等确认无人使用后一并清理。
        """
        raise NotImplementedError

    def get_key(self) -> str:
        """
        能力的全局唯一注册键。

        能力目录与注册表都走这里，避免出现第二套身份。
        默认取 name；DocumentParser 覆写成 get_parser_key()。
        """
        return self.name

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        if cls.config_model is None:
            return {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }

        if cls.config_model_schema is None:
            cls.config_model_schema = cls.config_model.model_json_schema()

        return cls.config_model_schema

    @abstractmethod
    def get_modality(self) -> str:
        raise NotImplementedError


class TextDocumentProcessor(DocumentProcessor, ABC):
    """
    服务**文本模态**的 Processor 基类。

    它只做一件事：把 `get_modality()` 定死成 `"text"`。

    为什么值得单独抽一层
    --------------------
    `get_modality()` 原本是 `DocumentProcessor` 上的**抽象**方法，
    于是**每一个** role 的**每一个**实现都得写一遍
    `return "text"` —— 而现实里这个项目处理的**全是文本**。

    这件事在代码里早就被记为痛点了（见 parser/base.py 与
    cleaner/base.py 的模块说明："放在公共基类上会连累其他 role ——
    清洗器 / 切分器被迫实现自己用不到的 get_modality()"）。
    这一层就是那个问题的解：把"我是文本"这个事实**收敛到一处**，
    各 role 基类继承它，具体实现什么都不用写。

    ⚠️ 它不改变 `DocumentProcessor` 的契约
    -------------------------------------
    `get_modality()` 仍然是 `DocumentProcessor` 上的抽象方法 ——
    将来要接图片/表格模态时，那个实现继承 `DocumentProcessor` 直接、
    自己提供 `get_modality()` 即可，这条口子是留着的。**不要**为了
    省事把它从基类上删掉：删了之后，一个忘了声明模态的实现会静默
    按"文本"处理，而模态错配的症状（索引 schema 对不上、检索排序
    没有意义）不会报错。

    ⚠️ 它必须是抽象的（ABC）
    ------------------------
    能力注册表按"叶子子类"发现实现（见 pipeline/registry.py 的
    `_leaf_subclasses`），**只跳过抽象类**。这一层如果变成具体类，
    一旦哪天它的子类都被删光，它自己就会被当成一个可注册的能力
    实例化出来 —— 一个没有 name、没有 role 的空壳。
    """

    def get_modality(self) -> str:
        return "text"

