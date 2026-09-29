"""
工具声明的数据结构。

    ToolSpec          代码里"声明了什么"（不可变，注册表的条目）
    ToolManifestData  扫描出来的"事实快照"（交给同步去比对与落库）
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Callable

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
    一个被 @tool 声明的**服务方法**。

    owner 与 method_name 是**执行入口**：invoke 时用
    `factory.create(契约)` 造出实例，再 getattr 到这个方法上。
    刻意不存"字符串入口"去 import —— 那样每次调用都要重新解析
    字符串、还得处理 import 失败，而进程里本来就有对象。

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

    所以本类只描述工具**是什么**：身份 + 签名 + 说明。
    """

    tool_key: str

    name: str

    owner: type

    method_name: str

    func: Callable[..., Any]

    #: 声明里的描述，为空时回退到 docstring。
    description: str | None = None

    #: 所属业务模块，例如 document。写进 source_ref。
    #:
    #: 不由作者在装饰器里写 —— 从 owner.__module__ 推导，
    #: 或由扫描器按来源填（MCP server 标识、OpenAPI spec 位置）。
    source_ref: str | None = None

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
        """

        import inspect

        for candidate in self.owner.__mro__:
            if (
                candidate is not self.owner
                and inspect.isabstract(candidate)
            ):
                return candidate

        return self.owner

    @property
    def entrypoint(self) -> str:
        """展示用入口，形如 python.module:Class.method。"""

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

        canonical = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()
