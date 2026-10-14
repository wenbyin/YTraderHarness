"""
工具声明的数据结构。

    ToolSpec          代码里"声明了什么"（不可变，注册表的条目）
    ToolManifestData  扫描出来的"事实快照"（交给同步去比对与落库）
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from app.common.utils import canonical_json, sha256_text

from ..domain.schemas.enum import SourceType

__all__ = [
    "ToolManifestData",
    "ToolSpec",
    "derive_source_ref",
]


#: 模块级可选变量名：写了就用它当 source_ref，不写就推导。
#:
#: 给**组合工具**用 —— langchain 的 @tool 不接受我们的任何参数，
#: 因此它声明不了 source_ref，只能靠模块里这一行来指定：
#:
#:     # app/agent/tools/quote_tool.py
#:     TOOL_SOURCE_REF = "agent.market"
#:
#:     @tool
#:     def quote(symbol: str) -> dict: ...
#:
#: 不改任何装饰器签名，对我那个 @tool 与 langchain 的 @tool 一视同仁。
SOURCE_REF_VARIABLE = "TOOL_SOURCE_REF"


def derive_source_ref(module_name: str) -> str:
    """
    从模块路径推出所属业务模块名。

        app.services.document.application.document_service -> document
        app.agent.tools.quote_tool                         -> agent
        app.infra.graph.runner                             -> infra

    规则是"`app.` 之后的第二段"，只有 services 例外 —— 那里第三段
    才是业务模块名（app.services.<模块>.…）。这样每种包都得到一个
    **成组**的 source_ref，而不是退化成模块名（那会让每个工具各占
    一个"来源"，按 source_ref 过滤就废了）。

    放这里而不是各扫描器自己写一份：装饰器与 scanner 都要用它，
    而 spec.py 是两者共同的祖先 —— 两处各写一份，早晚会漂移成
    两个工具算出不同的 source_ref，然后同步把它们当成两个来源。

    推不出来时退回最后一段，至少不会让 source_ref 是空的。
    """

    parts = module_name.split(".")

    if len(parts) >= 2 and parts[0] == "app":

        if parts[1] == "services" and len(parts) >= 3:
            return parts[2]

        return parts[1]

    return parts[-1]


@dataclass(frozen=True)
class ToolSpec:
    """
    一个可执行的工具 —— 身份 + 签名 + 说明。

    两种执行入口（**二选一**，见 __post_init__ 的校验）
    -------------------------------------------------
        @tool 服务方法        owner 类 + method_name。invoke 时用
                              `factory.create(契约)` 造出实例，再 getattr
                              到这个方法上。
        模块级 langchain 工具  `tool` —— 一个 BaseTool **本体**。它没有
                              owner 类，没有可以交给工厂去造的 self，
                              调用时直接调它自己。

    刻意不存"字符串入口"去 import —— 那样每次调用都要重新解析字符串、
    还得处理 import 失败，而进程里本来就有对象。

    为什么两种入口必须住在同一个类里
    ------------------------------
    `select_specs` 是"给模型看什么"与"允许执行什么"的**唯一**判定，而
    它只读注册表。两种工具若各用一套注册机制，就会出现"同步那条路
    （写 tool_definition）看得到它、运行时看不到"—— 症状是库里明明有
    这条工具、模型却永远拿不到，每一轮只留一条容易被忽略的 WARNING。

    entrypoint 字符串（见 ToolManifestData）只用于展示与审计。

    刻意**不含** tags / 分类 / 读写标记这类属性
    ------------------------------------------
    那些是"这个工具被谁怎么用了"的关联，是**连接者**的事，
    不是工具自己的事。它们由外部系统挂载，落在
    tool_classification 与 tool_definition_classification 上
    （有完整的 CRUD 与关联接口，也有列可以筛选）。

    曾经这里放过 tags 与 danger，两个都没有对应的 ORM 列却参与
    manifest_hash —— 结果是"改一下 tags 就报一次 updated、整行重写，
    而那个值根本没地方存"。制造变更噪音却不产生任何可观测效果，
    比"多余"更糟。
    """

    tool_key: str

    name: str

    #: 执行入口**之一**：服务方法（`tool` 为空时必填）。
    owner: type | None = None

    method_name: str | None = None

    func: Callable[..., Any] | None = None

    #: 执行入口**之二**：模块级 langchain 工具本体。
    #:
    #: 这类工具**没有 owner 类** —— 它是模块命名空间里的一个 BaseTool
    #: 实例，不是某个类的方法，因此没有可以交给工厂去造的 self，调用时
    #: 直接调它自己（见 infra/agent/tool_executor.py 的分支）。
    #:
    #: ⚠️ 为什么两种入口必须在**同一个** ToolSpec 里
    #: 因为 `select_specs` 是"给模型看什么"与"允许执行什么"的**唯一**
    #: 判定，而它只读注册表。两种工具若各用一套注册机制，就会出现
    #: "同步（写 tool_definition 那条路）看得到它、运行时看不到"——
    #: 症状是库里明明有这条工具，模型却永远拿不到，而每一轮都只留一条
    #: 容易被忽略的 WARNING。
    tool: Any = None

    #: 声明里的描述，为空时回退到 docstring。
    description: str | None = None

    #: 所属业务模块，例如 document。写进 source_ref。
    #:
    #: 不由作者在装饰器里写 —— 从 owner.__module__ 推导，
    #: 或由扫描器按来源填（MCP server 标识、OpenAPI spec 位置）。
    source_ref: str | None = None

    def __post_init__(self) -> None:

        # 两种入口都没有 = 这个 spec 谁都执行不了，而它会一路混过
        # select_specs（那些检查只问 descriptor 非空）直到执行期才炸。
        # 构造期就拦住，报错点离原因最近 —— 与 BaseRepository 那条同。
        if self.tool is None and (
                self.owner is None or self.method_name is None
        ):
            raise ValueError(
                f"ToolSpec({self.tool_key!r}) 既没有 owner/method_name，"
                "也没有 tool —— 两种执行入口必须有一个。"
            )

    # ========================================================
    # Derived
    # ========================================================

    @property
    def contract(self) -> type:
        """
        该工具所属的**抽象契约**，供工厂构造实例。

        owner 是 Impl 类（例如 DocumentRuntimeServiceImpl），
        而工厂要的是契约（DocumentRuntimeService）—— 在 MRO 里
        找第一个抽象基类即可，与 @tool 描述符推导契约是同一套。

        ⚠️ 只对"服务方法"这类入口有意义。模块级 langchain 工具没有
        owner、也不经工厂构造，走到这里说明调用方**没有先看 `tool`**
        —— 那是执行器的分支写错了，明确报错而不是给出一个误导的契约。
        """

        if self.owner is None:
            raise ValueError(
                f"ToolSpec({self.tool_key!r}) 是模块级 langchain 工具，"
                "没有 owner 也没有契约 —— 它不经工厂构造，"
                "调用方应当直接调 `spec.tool`。"
            )

        import inspect

        for candidate in self.owner.__mro__:
            if (
                candidate is not self.owner
                and inspect.isabstract(candidate)
            ):
                return candidate

        return self.owner

    @property
    def descriptor(self) -> Any:
        """
        这个工具**本体**（bind / 执行入口用）—— 一个 langchain
        `BaseTool`。

        两种入口的来源不同：

            服务方法          owner 上那个 `ToolDescriptor`。它既是
                              `StructuredTool`，也是描述符 —— **类访问**
                              `getattr(owner, method_name)` 时，`__get__`
                              的 `obj is None` 分支会把它自己返回
                              （见 decorator.py）。
            模块级 langchain   `self.tool` 就是本体，直接返回。

        为什么**服务方法**那一路不直接持有它
        ---------------------------------
        spec 是"声明了什么"的不可变事实，注册表只存这一份。多存一个
        指回描述符的字段，等于让同一条事实有两个来源 —— 而它们会在
        `importlib.reload` 之类的场景下漂成两份（描述符是新的、
        spec 里的引用是旧的）。

        （`tool` 那一路没有这个问题：它不是描述符，没有"类上那一份"
        与"spec 里那一份"的区别。）

        取不到时返回 None 而不是抛 AttributeError
        ---------------------------------------
        调用方（select_specs）本来就要处理"这个 key 取不到可执行对象"
        这件正常事（MCP / OpenAPI 那类来源没有描述符），让它显式判断
        比让它去 catch 一个异常干净。

        langchain 延迟 import：spec.py 是声明层，不该在 import 期
        就把 langchain 拖上（decorator.py 已经拖了它，但那是"定义
        工具"必须付的代价，读 spec 不需要）。
        """

        if self.tool is not None:
            return self.tool

        if self.owner is None or self.method_name is None:
            return None

        target = getattr(self.owner, self.method_name, None)

        if target is None:
            return None

        from langchain_core.tools import BaseTool

        # 类访问拿到的不一定是描述符 —— owner 上那个名字也可能被
        # 别的东西占着（子类覆盖、或 @tool 已经不在）。只认"它确实
        # 是个 langchain 工具"，这是 bind / 执行共同的前提。
        return target if isinstance(target, BaseTool) else None

    @property
    def entrypoint(self) -> str:
        """展示用入口，形如 python.module:Class.method。"""

        if self.owner is None:

            # 模块级 langchain 工具：拿它底层那个函数的定义位置。
            # 取不到就退回工具名 —— 它只用于展示与审计，编一个
            # 看起来像入口的假字符串比给个朴素的值更糟。
            underlying = (
                getattr(self.tool, "func", None)
                or getattr(self.tool, "coroutine", None)
            )
            module = getattr(underlying, "__module__", "")
            qualname = getattr(underlying, "__qualname__", "")

            return (
                f"{module}:{qualname}" if module and qualname
                else self.name
            )

        return (
            f"{self.owner.__module__}:"
            f"{self.owner.__qualname__}.{self.method_name}"
        )

    @property
    def effective_description(self) -> str | None:
        """生效描述：声明优先，否则回退到 docstring 摘要。"""

        if self.description:
            return self.description

        import inspect

        # 无 owner 那一路 `func` 可能是 None（入口是 `tool`）——
        # 那是"没有可读的 docstring"，不是错误。langchain 工具自己的
        # description 由描述符负责（见 _to_manifest）。
        if self.func is None:
            return None

        return inspect.getdoc(self.func)


@dataclass(frozen=True)
class ToolManifestData:
    """
    一次发现产出的**事实快照** —— 对应 tool.yaml 的 ToolManifest，
    只是还没有 id 与时间戳（那些由落库时补）。

    manifest_hash 是它的稳定摘要，同步靠它判断
    created / updated / unchanged。
    """

    tool_key: str

    name: str

    source_type: SourceType

    source_ref: str | None

    entrypoint: str

    input_schema: dict[str, Any]

    output_schema: dict[str, Any] | None

    source_docstring: dict[str, Any] | None

    # ========================================================
    # Hash
    # ========================================================

    def compute_hash(self) -> str:
        """
        算摘要。

        ⚠️ 参与哈希的字段是**刻意挑选**的，不是"整个对象"。
        判据只有一条：**这个字段有对应的列吗？**

        纳入的字段（都有列，能被读出来、能用上）：身份、来源、
        schema、说明文本。

        不纳入的两类：
            discovered_at / last_synced_at 这类时间戳
                每次同步都变，纳进去等于永远"有变化"，
                unchanged 计数永远是 0
            没有列的属性（曾经的 tags / danger）
                改了会触发 updated 与整行重写，但那个值根本
                没地方存 —— 制造噪音却不产生任何可观测效果
        """

        payload = {
            "tool_key": self.tool_key,
            "name": self.name,
            "source_type": self.source_type.value,
            "source_ref": self.source_ref,
            "entrypoint": self.entrypoint,
            "input_schema": self.input_schema,
            "output_schema": self.output_schema,
            "source_docstring": self.source_docstring,
        }

        return sha256_text(canonical_json(payload))
