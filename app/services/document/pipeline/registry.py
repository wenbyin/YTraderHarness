"""
能力注册表：把配置里的**名字**解析成**实现**。

为什么需要它
------------
配置里存的是字符串：

    DocumentProcessorDefinition.loader_name      = "pdf_loader"
    DocumentRetrievalDefinition.chunker_name     = "recursive_chunker"

节点拿到的是名字，需要一个地方把它换成对象。让节点直接
import 具体实现会把图绑死在实现上，换实现就得改节点。

注册表是可替换能力的唯一汇合点。

现在全是空的
------------
index/es.py、index/milvus.py、index/fiass.py 目前是 0 字节，
parser/、policy/ 下也没有任何具体实现。因此默认构造出来的
注册表是**空的**，节点会在查表时抛 KeyError。

这是刻意的：让"能力还没实现"在第一次运行时立刻暴露，
而不是悄悄跳过某个阶段假装成功。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.common.utils import (
    import_package_modules,
    leaf_subclasses,
)
from app.services.document.domain.schemas.enum import (
    EmbeddingOutput, StorageBackend, ModalityType, ProcessorRole,
)
from app.services.document.index.base import IndexBackend
from app.services.document.parser.base import DocumentParser
from app.services.document.parser.processor import (
    DocumentProcessor,
)
from app.services.document.enricher import (
    DocumentChunkEnricher,
)

__all__ = ["PipelineRegistry", "CapabilityNotFound"]

#: 索引后端的存放位置。
#:
#: 与 CAPABILITY_PACKAGES 分开：那一套按**名字**注册（键来自配置的
#: 字符串），索引后端按 **StorageBackend 枚举**注册，而且需要外部注入
#: 客户端。两套注册模型不同，硬塞进一个发现循环只会两边都别扭。
INDEX_PACKAGE = "app.services.document.index"


#: 各 role 实现所在的包 —— 反射发现的根。
#:
#: 新增一个 role：建一个同级包（role 基类 + 具体实现），再把包名
#: 加进这个元组即可。注册表、能力目录、目录接口都不用改。
CAPABILITY_PACKAGES = (
    "app.services.document.parser",
    "app.services.document.cleaner",
    "app.services.document.chunker",
    "app.services.document.embedder",
    "app.services.document.normalizer",
    # 阶段 8 的增强能力。原本它的契约住在 policy/ —— 而 policy 不在
    # 这个列表里、ChunkEnricher 也不是 DocumentProcessor 的子类，
    # 于是 registry.enrichers 一直是空表。见 enricher/base.py。
    "app.services.document.enricher",
)


class CapabilityNotFound(LookupError):
    """
    按名字查不到能力。

    继承 LookupError 而不是 ValueError，是为了不被
    to_http_error 当成 400 —— 这是服务端配置/实现缺失，
    不是调用方的错。
    """


@dataclass
class PipelineRegistry:
    """
    能力注册表。

    构造后即可冻结使用；图运行期间不修改。
    """

    def __init__(
            self,
            *,
            index_clients: dict[StorageBackend, Any] | None = None,
    ):
        capabilities = self.found_capabilities()

        # loader：按 supports() 自选，**顺序即优先级**，因此存列表。
        self.parsers = [
            capability
            for capability in capabilities
            if type(capability).role is ProcessorRole.LOADER
        ]

        # 其余 role：由配置按 key 显式指定，因此存字典。
        # role 为 None 的尚未归类，两边都不进。
        #
        # ⚠️ 键必须全局唯一，因此这里**显式检查**冲突而不是用字典推导。
        # 推导会让后一个实现静默覆盖前一个 —— 表现为"某个能力在目录里
        # 凭空消失"，没有任何报错，极难查。
        #
        # 最容易撞键的情形：多个具体实现继承了**同一个中间基类**的
        # name。中间基类只能给一个默认键，第二个具体实现必须自己覆盖
        # （见 chunker/graph.py 的说明）。
        self.processors: dict[str, DocumentProcessor] = {}

        for capability in capabilities:

            role = type(capability).role

            if role is None or role is ProcessorRole.LOADER:
                continue

            key = capability.get_key()

            existing = self.processors.get(key)

            if existing is not None:
                raise ValueError(
                    f"能力键冲突：{key!r} 同时被 "
                    f"{type(existing).__name__} 与 "
                    f"{type(capability).__name__} 占用。"
                    "每个具体实现都要声明自己的 name / get_key()，"
                    "不能沿用中间基类的默认值 —— 否则先注册的那个"
                    "会被静默顶掉。"
                )

            self.processors[key] = capability

        self.index_backends = self.found_index_backends(
            index_clients or {}
        )

    #: Parser 按 supports() 自选，因此存列表而不是字典。
    parsers: list[DocumentParser] = field(
        default_factory=list
    )

    processors: dict[str, DocumentProcessor] = field(
        default_factory=dict
    )

    # 注意：**没有** chunkers 这张表。
    # chunker 已并入 DocumentProcessor 体系（role=CHUNK），落在
    # 上面的 processors 里 —— 与 chunker/ 包、CAPABILITY_PACKAGES 一致。
    # 原来的 ChunkStrategy + registry.chunkers 已退役。

    enrichers: dict[str, DocumentChunkEnricher] = field(
        default_factory=dict
    )

    # 注意：**没有** embedders 这张表。
    # embedder 已并入 DocumentProcessor 体系（role=EMBED），落在上面的
    # processors 里 —— 与 embedder/ 包、CAPABILITY_PACKAGES 一致。
    # 原来的 policy/embedder.py 的 Embedder + registry.embedders 已退役。

    #: 按 StorageBackend 注册，而不是按名字 —— 一个后端类型只有一个实现。
    index_backends: dict[StorageBackend, IndexBackend] = field(
        default_factory=dict
    )

    def find_default_parser(
            self,
            suffix: Any = None,
    ) -> DocumentParser:
        for parser in self.parsers:
            if parser.supports(
                    suffix=suffix,
            ) and parser.is_default():
                return parser

        return None

    # ========================================================
    # Parser
    # ========================================================

    def find_parser_by_key(
            self,
            parser_key: Any = None,
    ) -> DocumentParser:
        """
        按 content_type / file_name 选出能处理该文件的 Parser。

        第一个 supports() 返回 True 的胜出 —— 注册顺序即优先级，
        因此专用 Parser 要注册在通用 Parser 之前。
        """

        for parser in self.parsers:
            if parser.get_parser_key() == parser_key:
                return parser

        raise CapabilityNotFound(
            f"No DocumentParser registered under key "
            f"{parser_key!r}. Registered: "
            f"{sorted(p.get_parser_key() for p in self.parsers)}."
        )

    # ========================================================
    # 能力目录
    # ========================================================

    def all_capabilities(self) -> list[DocumentProcessor]:
        """
        能力目录接口的发现源：把所有 role 的能力拼成一个列表。

        覆盖 parsers（loader）与 processors（其余已归类 role，
        目前是 cleaner）。chunker / enricher / embedder 的 ABC
        尚未加入 DocumentProcessor 的继承链（见 policy/ 下的三个
        ABC），等它们并进来后会自然成为 processors 的一部分，
        这里不用改。
        """

        capabilities: list[DocumentProcessor] = []

        # loader：按 supports() 自选，因此存的是列表
        capabilities.extend(self.parsers)

        # 其余 role：按名字注册
        capabilities.extend(self.processors.values())

        return capabilities

    def processor_by_role(
            self,
            role: ProcessorRole,
    ) -> DocumentProcessor | None:
        """
        按 role 取该阶段的默认能力；该阶段还没实现时返回 None。

        为什么返回 None 而不是抛 CapabilityNotFound
        ------------------------------------------
        _require() 的显式失败针对的是"配置点名要的东西没了" ——
        那是服务端配置错误，必须炸。

        这里是"该阶段还没实现"（阶段 4 normalize、阶段 6
        enrich_metadata 目前都还没有实现）。调用方据此走透传分支，
        而不是让整条流水线失败 —— 与各节点现有的占位行为一致。
        """

        for capability in self.all_capabilities():
            if type(capability).role is role:
                return capability

        return None

    def embedder_for(
            self,
            kind: EmbeddingOutput,
    ) -> DocumentProcessor | None:
        """
        按**向量形态**取 embedder；该形态没有实现时返回 None。

        为什么不能只用 processor_by_role(EMBED)
        --------------------------------------
        EMBED 这个 role 下有多个实现，而它们**不可互换**：稠密向量与
        稀疏向量的数学形态不同，索引侧存的字段类型也不同（Milvus 的
        FLOAT_VECTOR vs SPARSE_FLOAT_VECTOR，ES 的 dense_vector vs
        sparse_vector）。

        拿稠密模型去服务 sparse_vector **不会报错** —— 维度对得上、
        检索照常返回结果，只是排序没有意义。所以必须按形态精确匹配，
        而不是"有 embedder 就用"。

        返回 None 的语义与 processor_by_role 一致：该形态还没实现，
        调用方据此显式失败，而不是随手找个 embedder 顶上。

        ⚠️ 同一形态有**两个**实现时直接报错，不挑第一个
        ---------------------------------------------------
        `processors` 那张表按**键**查重（见上面的能力键冲突检查），
        但两个 embedder 可以键不同、形态相同 —— 那一层查不出来。而
        写入期与查询期都只按形态取（没有按名字指定的入口），所以
        "返回第一个"等于**按模块名排序静默二选一**：两个模型各有各的
        向量空间，混用不会报错，只表现为召回率莫名下降。

        这与能力键冲突、以及 retrieve/rerank.py 里 reranker_types 的
        冲突检查是同一套立场：能静态发现的问题不要留到运行期靠观察。
        """

        matched: list[DocumentProcessor] = [
            capability
            for capability in self.all_capabilities()
            if type(capability).role is ProcessorRole.EMBED
            and getattr(type(capability), "output_kind", None) is kind
        ]

        if len(matched) > 1:
            names = "、".join(
                f"{type(c).__name__}({c.get_key()})"
                for c in matched
            )
            raise ValueError(
                f"向量形态 {kind.value} 有 {len(matched)} 个 embedder："
                f"{names}。"
                "写入期与查询期都只按形态取 embedder，同时存在多个会"
                "静默选中其中一个 —— 而不同模型的向量空间不可比，"
                "混用不报错、只让召回率下降。"
                "请保证每个形态只有一个实现（换模型 = 删掉旧的）。"
            )

        return matched[0] if matched else None

    # ========================================================
    # Named capabilities
    # ========================================================

    def processor(self, name: str) -> DocumentProcessor:
        return self._require(
            self.processors, name, "DocumentProcessor"
        )

    # chunker 走 processor(name) 或 processor_by_role(CHUNK)，
    # 不再有独立的 chunker() —— 见上面关于 chunkers 表的说明。

    def enricher(self, name: str) -> ChunkEnricher:
        return self._require(
            self.enrichers, name, "DocumentChunkEnricher"
        )

    # embedder 走 processor(name) 或 processor_by_role(EMBED)，
    # 不再有独立的 embedder() —— 见上面关于 embedders 表的说明。

    def index_backend(
            self,
            backend: StorageBackend,
    ) -> IndexBackend:
        return self._require(
            self.index_backends, backend, "IndexBackend"
        )

    # ========================================================

    @staticmethod
    def _require(
            table: dict[Any, Any],
            name: Any,
            kind: str,
    ) -> Any:

        try:
            return table[name]

        except KeyError as exc:
            raise CapabilityNotFound(
                f"No {kind} registered under {name!r}. "
                f"Registered: {sorted(map(str, table))}. "
                "实现缺失时应当显式失败，而不是跳过该阶段。"
            ) from exc

    @staticmethod
    def found_index_backends(
            clients: dict[StorageBackend, Any],
    ) -> dict[StorageBackend, IndexBackend]:
        """
        反射发现 IndexBackend 实现，按各自声明的 backend_type 注册。

        客户端**由外部注入**（见 IndexBackend 的说明）：客户端的生命
        周期、连接池、凭据都由 infra 的 datasource 层管，业务层再建一套
        会多出第二份连接池与第二处凭据。

        某个后端的客户端没给（数据源被禁用，或者连不上而它是
        required: false）时，**该后端不注册**。之后配置里用到它会在
        查表时抛 CapabilityNotFound，明确告知"这个后端当前不可用" ——
        比让某个分支悄悄跳过写入要好。
        """

        import_package_modules(INDEX_PACKAGE)

        backends: dict[StorageBackend, IndexBackend] = {}

        for backend_cls in leaf_subclasses(
                IndexBackend, skip_abstract=True
        ):

            # 没声明 backend_type 的不会被注册：注册表按它索引，
            # 留空等于"这个实现不知道该服务哪个后端"。
            backend_type = getattr(
                backend_cls, "backend_type", None
            )

            if backend_type is None:
                continue

            client = clients.get(backend_type)

            if client is None:
                continue

            backends[backend_type] = backend_cls(client)

        return backends

    @staticmethod
    def found_capabilities() -> list[DocumentProcessor]:
        """
        反射发现 DocumentProcessor 的所有实现。

        约定：
            实现放在 CAPABILITY_PACKAGES 列出的包里，
            继承 DocumentProcessor 的某个 role 基类
            （DocumentParser / DocumentCleaner / …）。

        从 DocumentProcessor 出发深度优先，只收"最小子类"（自己
        不再有子类的那些），依次实例化。因此新增一个实现只要丢进
        对应 role 的包目录即可；抽象类（没写完的中间基类）自动跳过。

        顺序即优先级：loader 会用这个顺序做后缀自选，深度优先保证
        叶子实现排在同层通用实现之前，同层内按模块名排序。
        """

        import_package_modules(*CAPABILITY_PACKAGES)

        result: list[DocumentProcessor] = []

        for capability_cls in leaf_subclasses(
                DocumentProcessor, skip_abstract=True
        ):
            result.append(capability_cls())

        return result
