"""
Tool 模块冒烟测试。

不需要数据库 —— 仓储用内存假实现（与 retrieve_smoke.py 同一套做法）。
这样能让"同步的状态机"可确定地验证：

    created -> unchanged -> updated -> MISSING

这四种转移是同步的全部价值所在，而它们靠连真库来测反而更难
构造（要反复改代码、重启进程）。

跑法：
    python -m app.agent.tool_smoke
"""

from __future__ import annotations

import asyncio
import contextlib
import copy
import dataclasses
import pathlib
import shutil
import sys
import tempfile
import textwrap
from abc import abstractmethod
from datetime import datetime, timezone
from uuid import UUID

from pydantic import BaseModel, Field

from app.common.exceptions import ConflictError, NotFoundError
from app.common.factories import (
    build_factories,
    get_repository_factory,
    get_service_factory,
    reset_factories,
)
from app.common.service.base import BaseService
from app.services.capability.application.capability_service_impl import (
    CapabilityServiceImpl,
)
from app.services.capability.domain.entities import (
    Capability,
)
from app.services.capability.domain.schemas import (
    CapabilityCreate,
    CapabilityReplace,
    CapabilityUpdate,
)
from app.services.capability.domain.schemas import (
    Capability as CapabilitySchema,
)
from app.services.tool.application.tool_classification_service_impl import (
    ToolClassificationServiceImpl,
)
from app.services.tool.application.tool_discovery_service_impl import (
    ToolDiscoveryServiceImpl,
)
from app.services.tool.application.tool_documentation_service_impl import (
    ToolDocumentationServiceImpl,
)
from app.common.context import ContextBuilderType
from app.services.tool.application.tool_query_service_impl import (
    ToolQueryServiceImpl,
)
from app.services.tool.domain.entities import (
    ToolDefinition,
)
from app.services.tool.domain.schemas.enum import SourceType
from app.services.tool.domain.schemas import (
    ParameterDocumentationUpdate,
    SourceState,
    ToolClassificationUpdate,
    ToolDocumentationUpdate,
    ToolSyncRequest,
)
from app.services.tool.registry import (
    LangchainToolScanner,
    ServiceMethodScanner,
    ToolDiscoverer,
    ToolScanner,
    get_tool_registry,
    resolve_returns,
    select_specs,
    tool,
    tool_key_of,
    wire_name,
)


# ============================================================
# 示例服务：声明即注册
# ============================================================


class Filters(BaseModel):
    date_from: str | None = Field(
        default=None, description="起始日期（含）。",
    )


class ListDocumentsRequest(BaseModel):
    keyword: str = Field(default="", description="按标题模糊搜索。")
    page: int = Field(default=1, description="页码，从 1 开始。")
    filters: Filters | None = Field(
        default=None, description="附加过滤条件。",
    )


class DocumentQueryService(BaseService):
    """契约。"""

    @abstractmethod
    async def list_documents(
            self, req: ListDocumentsRequest,
    ) -> dict:
        ...

    @abstractmethod
    async def count_documents(self, tenant: str) -> int:
        ...


class DocumentQueryServiceImpl(DocumentQueryService):
    """实现。@tool 打在这里。"""

    def __init__(self, ctx=None):
        super().__init__(ctx)

    @tool(key="document.list")
    async def list_documents(
            self, req: ListDocumentsRequest,
    ) -> dict:
        """列出文档，支持关键词与分页。"""
        return {"keyword": req.keyword, "page": req.page}

    # 没有 pydantic 参数的方法 —— 走"合成 schema"那条路。
    # 现实里不该这么写，这里刻意留一个来固定这条兜底路径的行为。
    #
    # docstring 写成**分节**的：返回注解是 int 而不是 pydantic 模型，
    # 因此没有输出模型可回落，returns 只能来自这里的 Returns 段 ——
    # 恰好把"docstring 切段"这条路径也固定住。
    @tool(key="document.count")
    async def count_documents(self, tenant: str) -> int:
        """统计文档数量。

        Args:
            tenant: 租户标识。

        Returns:
            该租户下的文档总数。
        """
        return 0


# ============================================================
# 内存假仓储
# ============================================================


class FakeDefinitionRepository:
    """只实现同步用得到的那几个方法。"""

    def __init__(self):
        self.rows: dict[UUID, ToolDefinition] = {}

        #: tool_key -> 它挂着的能力 code。真实库里这是
        #: tool_definition_classification 连 capability 两跳；
        #: 假仓储没有那张表，用一个显式映射代替（smoke 自己往里放）。
        self.capability_codes: dict[str, set[str]] = {}

    async def add(self, definition):
        self.rows[definition.tool_id] = definition
        return definition

    async def update(self, definition):
        self.rows[definition.tool_id] = definition
        return definition

    async def get(self, tool_id):
        return self.rows.get(tool_id)

    async def get_by_key(self, tool_key):
        for row in self.rows.values():
            if row.tool_key == tool_key:
                return row
        return None

    async def list_by_keys(self, tool_keys):
        wanted = set(tool_keys)
        return [
            r for r in self.rows.values()
            if r.tool_key in wanted
        ]

    async def list_active(self):
        return [
            r for r in self.rows.values()
            if r.source_state == SourceState.ACTIVE
        ]

    async def list_all(self, *, source_type=None):
        return list(self.rows.values())

    async def list_paged(
            self, *, keyword=None, source_type=None,
            source_ref=None,
            source_state=None, classification_code=None,
            offset=0, limit=20,
    ):
        rows = sorted(self.rows.values(), key=lambda r: r.tool_key)

        if keyword:
            rows = [
                r for r in rows
                if keyword.lower() in r.tool_key.lower()
                or keyword.lower() in r.name.lower()
            ]

        if source_ref is not None:
            rows = [
                r for r in rows if r.source_ref == source_ref
            ]

        if source_state is not None:
            rows = [
                r for r in rows if r.source_state == source_state
            ]

        return rows[offset:offset + limit], len(rows)

    async def list_active_by_classification_codes(self, codes):
        if not codes:
            return []

        wanted = set(codes)

        return [
            r for r in sorted(
                self.rows.values(), key=lambda r: r.tool_key
            )
            if r.source_state == SourceState.ACTIVE
            and self.capability_codes.get(r.tool_key, set()) & wanted
        ]

    async def mark_missing(self, tool_ids, *, now):
        count = 0
        for tool_id in tool_ids:
            row = self.rows.get(tool_id)
            if row is not None:
                self.rows[tool_id] = row.model_copy(
                    update={
                        "source_state": SourceState.MISSING,
                        "last_synced_at": now,
                        "updated_at": now,
                    }
                )
                count += 1
        return count

    async def touch_synced_at(self, tool_ids, *, now):
        for tool_id in tool_ids:
            row = self.rows.get(tool_id)
            if row is not None:
                self.rows[tool_id] = row.model_copy(
                    update={"last_synced_at": now}
                )
        return len(tool_ids)

    async def delete_many(self, tool_ids):
        count = 0
        for tool_id in tool_ids:
            if self.rows.pop(tool_id, None) is not None:
                count += 1
        return count


class FakeDocumentationRepository:
    def __init__(self):
        self.docs: dict[UUID, dict] = {}

    async def upsert(
            self, tool_id, *, description_override,
            return_description_override, updated_by, now,
    ):
        self.docs[tool_id] = {
            "description_override": description_override,
            "return_description_override": (
                return_description_override
            ),
            "parameters": self.docs.get(tool_id, {}).get(
                "parameters", []
            ),
            "updated_by": updated_by,
            "updated_at": now,
        }
        return await self.get(tool_id)

    async def replace_parameter_documentations(
            self, tool_id, entries, *, updated_by, now,
    ):
        self.docs.setdefault(tool_id, {})["parameters"] = list(
            entries
        )

    async def get(self, tool_id):
        from app.services.tool.domain.entities import (
            ParameterDocumentation,
            ToolDocumentation,
        )

        raw = self.docs.get(tool_id)
        if raw is None:
            return None

        return ToolDocumentation(
            tool_id=tool_id,
            description_override=raw["description_override"],
            return_description_override=(
                raw["return_description_override"]
            ),
            updated_by=raw["updated_by"],
            updated_at=raw["updated_at"],
            parameter_documentations=[
                ParameterDocumentation(
                    parameter_documentation_id=UUID(int=i + 1),
                    tool_id=tool_id,
                    parameter_path=path,
                    description_override=override,
                    updated_at=raw["updated_at"],
                )
                for i, (path, override) in enumerate(
                    raw.get("parameters", [])
                )
            ],
        )

    async def list_parameter_documentations(self, tool_id):
        doc = await self.get(tool_id)
        return [] if doc is None else doc.parameter_documentations

    async def map_description_overrides(self, tool_ids):
        return {
            t: self.docs[t]["description_override"]
            for t in tool_ids if t in self.docs
        }

    async def map_parameter_overrides(self, tool_ids):
        # 与真实现一致：**不在这里滤 None**（"有行但没写"与"没有行"
        # 要分得开），由上层按自己的口径判"算不算覆盖"。
        return {
            t: {
                path: override
                for path, override in self.docs[t].get("parameters", [])
            }
            for t in tool_ids if t in self.docs
        }

    async def count_for_tools(self, tool_ids):
        return sum(
            1 + len(self.docs[t].get("parameters", []))
            for t in tool_ids if t in self.docs
        )


class FakeCapabilityRepository:
    """
    同时充任两个契约的内存假实现：

        CapabilityRepository            能力树本体（含 count_links）
        ToolClassificationRepository    tool ↔ 能力关联

    两边关心的数据是同一份（树 + 关联），拆成两个假实现只会让
    "关联指向的能力必须存在"这类跨契约的一致性无处安放。
    """

    def __init__(self):
        self.rows: dict[UUID, Capability] = {}
        self.links: dict[UUID, list[UUID]] = {}

    # --------------------------------------------------------
    # 能力树本体（CapabilityRepository）
    # --------------------------------------------------------

    async def add(self, entity):
        self.rows[entity.capability_id] = entity
        return entity

    async def update(self, entity):
        self.rows[entity.capability_id] = entity
        return entity

    async def get(self, capability_id):
        return self.rows.get(capability_id)

    async def get_by_code(self, code):
        for row in self.rows.values():
            if row.code == code:
                return row
        return None

    async def list(
            self, *, parent_id=None, keyword=None, is_active=None,
    ):
        rows = sorted(
            self.rows.values(),
            key=lambda r: (r.sort_order, r.code),
        )
        if parent_id is not None:
            rows = [r for r in rows if r.parent_id == parent_id]
        if is_active is not None:
            rows = [r for r in rows if r.is_active == is_active]
        return rows

    async def delete(self, capability_id):
        self.rows.pop(capability_id, None)
        # 真库里这一步由 ON DELETE CASCADE 完成，假实现自己来。
        for tool_id, ids in self.links.items():
            self.links[tool_id] = [
                i for i in ids if i != capability_id
            ]

    async def count_children(self, capability_id):
        return sum(
            1 for row in self.rows.values()
            if row.parent_id == capability_id
        )

    # --------------------------------------------------------
    # tool ↔ 能力关联（ToolClassificationRepository）
    # --------------------------------------------------------

    async def list_for_tool(self, tool_id):
        return [
            self.rows[cid]
            for cid in self.links.get(tool_id, [])
            if cid in self.rows
        ]

    async def list_for_tools(self, tool_ids):
        return {t: await self.list_for_tool(t) for t in tool_ids}

    async def replace_tool_classifications(
            self, tool_id, classification_ids,
    ):
        self.links[tool_id] = list(classification_ids)

    async def count_links(self, classification_ids):
        wanted = set(classification_ids)
        return sum(
            1 for ids in self.links.values()
            if wanted & set(ids)
        )

    async def count_skill_links(self, capability_ids):
        # 本冒烟不建 Skill 绑定；契约新增的这条反查返回 0。
        return 0


# ============================================================
# 断言工具
# ============================================================


def check(label, condition):
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title):
    print(f"\n{title}")


def _raises(exception, fn):
    """fn() 抛出了指定异常吗。"""

    try:
        fn()
    except exception:
        return True
    except Exception:
        return False
    return False


def _fake_session_factory():
    """
    给"装配应用级工厂"用的最小 session_factory。

    本节只验证装配与取用，不真的查库，因此会话是空壳。
    """

    class _Session:
        @contextlib.asynccontextmanager
        async def begin(self):
            yield self

    @contextlib.asynccontextmanager
    async def _cm():
        yield _Session()

    return _cm()


def _make_langchain_fixture():
    """
    造一个临时包，里面放一个 langchain 原生的工具。

    用临时包而不是往仓库里塞 fixture：扫描器按 __path__ 遍历，
    必须有个真实的包才验证得了 walk_packages + isinstance 这条路径。
    """

    root = tempfile.mkdtemp(prefix="toolsmoke_")
    package = pathlib.Path(root) / "smokepkg"
    package.mkdir()

    (package / "__init__.py").write_text("", encoding="utf-8")

    (package / "kit.py").write_text(
        textwrap.dedent(
            '''
            from langchain_core.tools import tool
            from pydantic import BaseModel, Field


            class LookupRequest(BaseModel):
                symbol: str = Field(description="股票代码。")


            @tool(args_schema=LookupRequest)
            def lookup_stock(symbol: str) -> str:
                """按代码查股票。"""
                return symbol
            '''
        ).lstrip(),
        encoding="utf-8",
    )

    sys.path.insert(0, root)

    return root


def _cleanup_langchain_fixture(root):
    if root in sys.path:
        sys.path.remove(root)

    shutil.rmtree(root, ignore_errors=True)

    for name in [
        n for n in list(sys.modules) if n.startswith("smokepkg")
    ]:
        sys.modules.pop(name, None)


class _OwnToolsScanner(ServiceMethodScanner):
    """
    只交出本文件声明的工具。

    光传一个收窄的扫描器列表**不够**：ServiceMethodScanner.scan()
    内部会调 registry.scan()，而后者按 TOOL_SOURCE_PACKAGES 把仓库里
    那些包整个走一遍（那是刻意的 —— "进程 import 过什么"不该影响
    "代码库里有哪些工具"）。副作用是 app.services.document.application
    里真实的 @tool 也会被带进注册表。

    本用例要的是"固定住自己那两个"，所以在出口处按 source_ref 滤一道：
    fixture 定义在本文件里，source_ref 推出来就是 __main__。
    """

    def scan(self):
        return [
            m for m in super().scan()
            if m.source_ref == "__main__"
        ]


# ============================================================
# 用例
# ============================================================


async def main() -> int:

    registry = get_tool_registry()

    # 本用例只关心自己声明的两个工具。
    #
    # 不用 ToolDiscoverer.default()：那是**全局**扫描，会把仓库里
    # 所有工具（含 app/agent/tools/ 下的组合工具）都扫进来 ——
    # 那样这个用例会随"别人在别处加了个工具"而失败，而失败信息
    # 指向不了真正的原因。
    #
    # 收窄靠 _OwnToolsScanner —— 理由见它的 docstring：只换扫描器
    # 列表挡不住 ServiceMethodScanner 内部的包遍历。
    #
    # 全局扫描本身另有用例覆盖（见本文件末尾的"全局扫描"一节）。
    discoverer = ToolDiscoverer([_OwnToolsScanner(registry)])
    definitions = FakeDefinitionRepository()
    documentations = FakeDocumentationRepository()

    # 一份假数据同时喂给两个契约：能力树本体（capability 模块）与
    # 关联（tool 模块）。见 FakeCapabilityRepository 的 docstring。
    store = FakeCapabilityRepository()

    discovery = ToolDiscoveryServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=documentations,
        discoverer=discoverer,
    )
    query = ToolQueryServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=documentations,
        tool_classification_repository=store,
    )
    docs = ToolDocumentationServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=documentations,
    )
    # 能力树本体：只有能力仓储，不认识工具。
    taxonomy = CapabilityServiceImpl(capability_repository=store)
    # 工具 ↔ 能力关联：留在 tool 模块，校验存在性要读能力表，
    # 因此两个仓储都注入。
    links = ToolClassificationServiceImpl(
        tool_classification_repository=store,
        tool_definition_repository=definitions,
        capability_repository=store,
    )

    # --------------------------------------------------------
    section("自注册（没有任何 register() 调用）")

    # ⚠️ 这里断言**包含**，不是相等。
    #
    # `registry` 是 `get_tool_registry()` —— **进程级全局**的那一个，而
    # `@tool` 是**导入时自注册**的：本进程只要 import 过任何一个带 @tool 的
    # 模块（例如 app/services/capability 的实现类），那个 key 就在表里。
    #
    # 因此"表里恰好只有我这两个"不是这个测试能断言的性质 —— 它会随
    # **别人**在别处加一个 @tool 而变红，而失败信息指向不了真正的原因
    # （与本文件开头那段"不用全局扫描"的取舍是同一类问题：那次收窄的是
    # **扫描器**，挡不住装饰器的自注册）。
    #
    # 标签说的是"两个都进了"，包含断言正是它的意思。
    check(
        "两个 @tool 方法都进了注册表",
        {"document.list", "document.count"} <= set(registry.keys()),
    )
    check(
        "key 来自显式声明",
        "document.list" in registry,
    )

    spec = registry.get("document.list")
    check(
        "source_ref 由模块路径推导（不由作者声明）",
        spec.source_ref == "__main__",
    )
    check(
        "声明里没有 tags / 分类 / 读写标记这些关联属性",
        not any(
            hasattr(spec, name)
            for name in ("tags", "danger", "classifications")
        ),
    )
    check(
        "契约从 MRO 推出",
        spec.contract is DocumentQueryService,
    )
    check(
        "entrypoint 是 module:Class.method",
        spec.entrypoint.endswith(
            "__main__:DocumentQueryServiceImpl.list_documents"
        ),
    )

    # --------------------------------------------------------
    section("方法照常可用（没有打断既有调用点）")

    service = DocumentQueryServiceImpl()
    result = await service.list_documents(
        ListDocumentsRequest(keyword="年报", page=3)
    )
    check(
        "实例访问拿到的是绑定方法，调用正常",
        result == {"keyword": "年报", "page": 3},
    )
    check(
        "类访问拿到的是描述符（扫描/自省用）",
        type(
            DocumentQueryServiceImpl.__dict__["list_documents"]
        ).__name__ == "ToolDescriptor",
    )

    # 同 key 换 entrypoint 必须报错（静默取其一更危险）
    try:
        registry.register(
            dataclasses.replace(spec, owner=DocumentQueryService)
        )
    except ValueError:
        check("同 key 不同 entrypoint -> 报错", True)
    else:
        check("同 key 不同 entrypoint -> 报错", False)

    # --------------------------------------------------------
    section("发现：schema 与摘要")

    manifests = discoverer.discover()
    check("扫出 2 个工具", len(manifests) == 2)

    by_key = {m.tool_key: m for m in manifests}
    listing = by_key["document.list"]

    check(
        "输入 schema 来自 pydantic 模型",
        set(listing.input_schema["properties"])
        == {"keyword", "page", "filters"},
    )
    check(
        "嵌套参数的说明被提取成 JSON Pointer",
        listing.source_docstring["parameters"].get(
            "/filters/date_from"
        ) == "起始日期（含）。",
    )
    check(
        "docstring 摘要进了 source_docstring",
        listing.source_docstring["raw"]
        == "列出文档，支持关键词与分页。",
    )
    check(
        "没有分节时 description 就是全文",
        listing.source_docstring["description"]
        == "列出文档，支持关键词与分页。",
    )
    check(
        "返回注解不是输出模型时不产出 returns",
        "returns" not in listing.source_docstring,
    )

    counted = by_key["document.count"]
    check(
        "description 只留第一个节标题之前的正文",
        counted.source_docstring["description"] == "统计文档数量。",
    )
    check(
        "Returns 段被切出来",
        counted.source_docstring["returns"] == "该租户下的文档总数。",
    )
    check(
        "returns：两边都写了以 DTO 为准",
        resolve_returns(
            "方法里的 Returns 段。", {"description": "DTO 的描述。"}
        )
        == "DTO 的描述。",
    )
    check(
        "returns：DTO 没写才回落到方法里的 Returns 段",
        resolve_returns("方法里的 Returns 段。", {"type": "object"})
        == "方法里的 Returns 段。",
    )
    check(
        "returns：两边都没有时为空",
        resolve_returns(None, None) is None,
    )

    check(
        "无 pydantic 参数的方法走合成 schema",
        counted.input_schema["properties"]["tenant"]["type"]
        == "string",
    )
    check(
        "合成 schema 标出了必填",
        counted.input_schema.get("required") == ["tenant"],
    )

    h1 = listing.compute_hash()

    def hash_of(tool_key):
        return {
            m.tool_key: m for m in discoverer.discover()
        }[tool_key].compute_hash()

    check("摘要稳定（同一份代码两次一致）", h1 == hash_of("document.list"))

    # --------------------------------------------------------
    section("扫描器：一条来源一个，互不干扰")

    check(
        "两个扫描器都满足 ToolScanner 协议",
        isinstance(ServiceMethodScanner(registry), ToolScanner)
        and isinstance(LangchainToolScanner(), ToolScanner),
    )

    fixture_root = _make_langchain_fixture()

    try:
        found = LangchainToolScanner(
            packages=("smokepkg",)
        ).scan()

        check("扫到了模块级的 langchain 工具", len(found) == 1)

        lc = found[0]
        check("name 取自 BaseTool.name", lc.name == "lookup_stock")
        check(
            "描述取自 BaseTool.description",
            lc.source_docstring["raw"] == "按代码查股票。",
        )
        check(
            "参数 schema 取自 args_schema",
            "symbol" in lc.input_schema["properties"],
        )
        check(
            "参数说明进了 JSON Pointer",
            lc.source_docstring["parameters"].get("/symbol")
            == "股票代码。",
        )
        check(
            "entrypoint 是 module:属性名",
            lc.entrypoint == "smokepkg.kit:lookup_stock",
        )
        check(
            "与 @tool 同属一个 source_type（都在本进程代码里）",
            lc.source_type.value == "local_method",
        )

        # 按 source_ref 收窄：只要 @tool 那一份时，langchain 的不该混进来
        only_methods = discoverer.discover(
            source_refs=["__main__"]
        )
        check(
            "按 source_ref 收窄有效",
            bool(only_methods)
            and all(
                m.source_ref == "__main__" for m in only_methods
            )
            and all(
                m.tool_key != lc.tool_key for m in only_methods
            ),
        )
    finally:
        _cleanup_langchain_fixture(fixture_root)

    # --------------------------------------------------------
    section("同步：created -> unchanged")

    first = await discovery.sync()
    check("首次全部新建", first.created == 2 and first.scanned == 2)
    check("首次没有 MISSING", first.missing == 0)

    second = await discovery.sync()
    check(
        "再同步全部 unchanged（幂等）",
        second.unchanged == 2
        and second.created == 0
        and second.updated == 0,
    )
    check(
        "unchanged 也刷新了 last_synced_at",
        all(
            r.last_synced_at >= r.discovered_at
            for r in definitions.rows.values()
        ),
    )

    # --------------------------------------------------------
    section("同步：updated（同一 entrypoint、内容变了）")

    # 直接改注册表里那条 spec 的描述 —— entrypoint 没变，
    # 因此注册表允许覆盖，正是"代码改了、工具还是同一个"的情形。
    registry.register(
        dataclasses.replace(
            registry.get("document.list"),
            description="改过的描述。",
        )
    )

    third = await discovery.sync()
    check(
        "只有那一个被覆盖",
        third.updated == 1 and third.unchanged == 1,
    )
    check(
        "摘要确实变了",
        definitions.rows[
            next(
                r.tool_id for r in definitions.rows.values()
                if r.tool_key == "document.list"
            )
        ].manifest_hash != h1,
    )

    # --------------------------------------------------------
    section("人工说明不被同步覆盖")

    target = next(
        r for r in definitions.rows.values()
        if r.tool_key == "document.list"
    )

    written = await docs.replace_documentation(
        target.tool_id,
        ToolDocumentationUpdate(
            description_override="人工写的描述。",
            parameter_descriptions=[
                ParameterDocumentationUpdate(
                    parameter_path="/keyword",
                    description_override="人工写的参数说明。",
                )
            ],
        ),
        updated_by="tester",
    )
    check(
        "生效描述取人工覆盖",
        written.effective_description == "人工写的描述。",
    )

    para = {
        p.parameter_path: p for p in written.parameter_descriptions
    }
    check(
        "参数说明三态齐全（来源/覆盖/生效）",
        para["/keyword"].source_description == "按标题模糊搜索。"
        and para["/keyword"].description_override
        == "人工写的参数说明。"
        and para["/keyword"].effective_description
        == "人工写的参数说明。",
    )
    check(
        "没被覆盖的参数回退到代码里的说明",
        para["/page"].effective_description == "页码，从 1 开始。",
    )

    await discovery.sync()
    after = await docs.get_documentation(target.tool_id)
    check(
        "同步之后人工内容原样还在",
        after.effective_description == "人工写的描述。"
        and after.updated_by == "tester",
    )

    # --------------------------------------------------------
    section("同步：MISSING（代码里没有了）")

    saved_count_spec = registry.get("document.count")
    registry._specs.pop("document.count")
    fourth = await discovery.sync()
    check("扫到的只剩 1 个", fourth.scanned == 1)
    check("缺席的被标成 MISSING", fourth.missing == 1)

    gone = next(
        r for r in definitions.rows.values()
        if r.tool_key == "document.count"
    )
    check(
        "行还在，只是状态变了（没被物理删除）",
        gone.source_state == SourceState.MISSING,
    )

    fifth = await discovery.sync()
    check(
        "已经是 MISSING 的不重复计数",
        fifth.missing == 0,
    )

    # 把手头那条**原样**放回去（不是重建 —— 重建会让字段漂移，
    # 也就测不出下面这个"内容没变但状态要恢复"的分支）
    registry.register(saved_count_spec)
    sixth = await discovery.sync()

    check(
        "重新出现时从 MISSING 恢复成 ACTIVE",
        next(
            r for r in definitions.rows.values()
            if r.tool_key == "document.count"
        ).source_state == SourceState.ACTIVE,
    )
    check(
        "内容一模一样也要恢复（不能因摘要相同就落进 unchanged）",
        sixth.updated == 1 and sixth.unchanged == 1,
    )

    # --------------------------------------------------------
    section("物理删除（remove_missing=true）")

    registry._specs.pop("document.count")
    seventh = await discovery.sync(
        ToolSyncRequest(remove_missing=True)
    )
    check("缺席的被物理删除", seventh.missing == 1)
    check(
        "报出了被级联带走的说明行数",
        seventh.orphaned_documentations == 0,
    )
    check(
        "行确实没了",
        not any(
            r.tool_key == "document.count"
            for r in definitions.rows.values()
        ),
    )

    # --------------------------------------------------------
    section("查询服务")

    page = await query.list_tools(page=1, page_size=10)
    check("分页返回 1 条", page.total == 1 and len(page.items) == 1)
    check(
        "列表里的描述是生效值（与详情一致）",
        page.items[0].description == "人工写的描述。",
    )

    detail = await query.get_tool(target.tool_id)
    check(
        "详情里的描述与列表一致",
        detail.documentation.effective_description
        == page.items[0].description,
    )
    check(
        "详情带上了 manifest",
        detail.manifest.tool_key == "document.list",
    )

    try:
        await query.get_tool(UUID(int=999))
    except NotFoundError:
        check("查不存在的工具 -> NotFoundError", True)
    else:
        check("查不存在的工具 -> NotFoundError", False)

    # --------------------------------------------------------
    # 能力树本体：由 capability 模块的 CapabilityService 负责，
    # 它只认识能力仓储，不碰工具。
    section("能力：能力树")

    root = await taxonomy.create_capability(
        CapabilityCreate(code="retrieval", name="检索类")
    )
    child = await taxonomy.create_capability(
        CapabilityCreate(
            code="retrieval.semantic",
            name="语义检索",
            parent_id=root.id,
        )
    )

    try:
        await taxonomy.create_capability(
            CapabilityCreate(code="retrieval", name="重名")
        )
    except ConflictError:
        check("code 冲突 -> ConflictError", True)
    else:
        check("code 冲突 -> ConflictError", False)

    try:
        await taxonomy.update_capability(
            root.id, CapabilityUpdate(parent_id=child.id)
        )
    except ConflictError:
        check("把自己挂到后代下面 -> 成环被拒", True)
    else:
        check("把自己挂到后代下面 -> 成环被拒", False)

    # --------------------------------------------------------
    # 关联：留在 tool 模块，路径与术语都是 classifications。
    section("能力：工具 ↔ 能力关联")

    linked = await links.replace_tool_classifications(
        target.tool_id,
        ToolClassificationUpdate(classification_ids=[child.id]),
    )
    check("关联成功", len(linked) == 1 and linked[0].code
          == "retrieval.semantic")

    # 返回的必须是 **DTO** 而不是实体：两个接口的 response_model 是
    # schemas.Capability（有 id），而仓储交出来的是实体（有
    # capability_id、没有 id）。少了组装这一步，FastAPI 会在响应校验
    # 时抛 'response.0.id Field required' —— 报错指向响应体，离原因很远。
    # 所以这里断言类型，而不是只断言 .code（两者都有 code，断言不出来）。
    check(
        "关联接口回的是 DTO 不是实体",
        isinstance(linked[0], CapabilitySchema)
        and not isinstance(linked[0], Capability),
    )

    fetched = await links.get_tool_classifications(target.tool_id)
    check(
        "读关联接口回的是 DTO 不是实体",
        len(fetched) == 1
        and isinstance(fetched[0], CapabilitySchema)
        and not isinstance(fetched[0], Capability),
    )
    check(
        "DTO 带 id 且值等于实体的 capability_id",
        fetched[0].id == child.id,
    )

    relisted = await query.list_tools(page=1, page_size=10)
    check(
        "列表页带出了分类（批量查询，不是 N+1）",
        len(relisted.items[0].classifications) == 1,
    )

    try:
        await links.replace_tool_classifications(
            target.tool_id,
            ToolClassificationUpdate(
                classification_ids=[UUID(int=888)]
            ),
        )
    except NotFoundError:
        check("关联不存在的分类 -> NotFoundError", True)
    else:
        check("关联不存在的分类 -> NotFoundError", False)

    emptied = await links.replace_tool_classifications(
        target.tool_id,
        ToolClassificationUpdate(classification_ids=[]),
    )
    check("传空数组即全部解除", emptied == [])

    # --------------------------------------------------------
    section("能力：PUT 整体替换")

    # root 下这时还挂着 child。替换语义是"没传的可空字段清零"，
    # 所以只报必填字段就该把 parent_id 清掉。
    replaced = await taxonomy.replace_capability(
        child.id,
        CapabilityReplace(
            name="语义检索（改名）", sort_order=3, is_active=False
        ),
    )
    check(
        "只报必填字段 -> description / parent_id 被清空",
        replaced.parent_id is None and replaced.description is None,
    )
    check(
        "name / sort_order / is_active 按请求落库",
        replaced.name == "语义检索（改名）"
        and replaced.sort_order == 3
        and replaced.is_active is False,
    )
    check("code 不被 PUT 改动", replaced.code == "retrieval.semantic")

    try:
        await taxonomy.replace_capability(
            root.id,
            CapabilityReplace(
                name=root.name, sort_order=0, is_active=True,
                parent_id=root.id,
            ),
        )
    except ConflictError:
        check("PUT 把自己作为父分类 -> 成环被拒", True)
    else:
        check("PUT 把自己作为父分类 -> 成环被拒", False)

    # 复原，后面的删除用例还要用这棵树。
    await taxonomy.replace_capability(
        child.id,
        CapabilityReplace(
            name="语义检索", description="语义", sort_order=0,
            is_active=True, parent_id=root.id,
        ),
    )

    # --------------------------------------------------------
    section("能力：DELETE")

    try:
        await taxonomy.delete_capability(root.id)
    except ConflictError:
        check("有子分类 -> 拒绝", True)
    else:
        check("有子分类 -> 拒绝", False)

    try:
        await taxonomy.delete_capability(root.id, force=True)
    except ConflictError:
        check("force 也解不开子分类这关", True)
    else:
        check("force 也解不开子分类这关", False)

    await links.replace_tool_classifications(
        target.tool_id,
        ToolClassificationUpdate(classification_ids=[child.id]),
    )

    try:
        await taxonomy.delete_capability(child.id)
    except ConflictError as exc:
        check("仍被工具引用且没 force -> 拒绝", True)
        check("错误里给出引用数", "1" in str(exc))
    else:
        check("仍被工具引用且没 force -> 拒绝", False)

    await taxonomy.delete_capability(child.id, force=True)

    try:
        await taxonomy.get_capability(child.id)
    except NotFoundError:
        check("force=true 删除成功", True)
    else:
        check("force=true 删除成功", False)

    check(
        "删除连带解除了工具关联",
        not await links.get_tool_classifications(target.tool_id),
    )

    try:
        await taxonomy.delete_capability(child.id)
    except NotFoundError:
        check("删不存在的分类 -> NotFoundError", True)
    else:
        check("删不存在的分类 -> NotFoundError", False)

    await taxonomy.delete_capability(root.id)
    check("子节点删掉后父节点可删", True)

    # --------------------------------------------------------
    section("全局扫描：不关心位置，两种声明都扫得到")

    everything = ToolDiscoverer.default().discover()
    keys = {m.tool_key for m in everything}

    composite = [
        m for m in everything
        if m.entrypoint.startswith("app.agent.tools")
    ]

    check(
        "app/agent/tools/ 下的组合工具被全局扫到",
        bool(composite),
    )
    check(
        "组合工具的 source_ref 由模块路径推导；显式 TOOL_SOURCE_REF 优先",
        all(
            (
                m.source_ref == "agent.platform"
                if m.entrypoint.startswith(
                    "app.agent.tools.catalog_tool"
                )
                else (
                    m.source_ref == "web"
                    if m.entrypoint.startswith(
                        "app.agent.tools.web_tool"
                    )
                    else m.source_ref == "agent"
                )
            )
            for m in composite
        ),
    )
    check(
        "两种声明方式共存于同一份清单",
        not keys.isdisjoint(
            {"document.list", "document.count"}
        )
        and not keys.isdisjoint(
            {m.tool_key for m in composite}
        ),
    )
    # 全局清单现在可能包含 MCP 来源（mcp.yaml 存在时）。本地工具仍
    # 必须都是 local_method，外部工具必须标成 mcp —— 判据是 entrypoint
    # 前缀，而不是「清单里只有一种来源」那个已经过期的假设。
    check(
        "代码里的工具是 local_method；外部来源标成 mcp",
        all(
            m.source_type.value
            == ("mcp" if m.entrypoint.startswith("mcp:") else "local_method")
            for m in everything
        ),
    )

    # --------------------------------------------------------
    section("取回描述符：spec.descriptor")

    spec = registry.get("document.list")

    from langchain_core.tools import BaseTool

    check("拿到的是 langchain 工具",
          isinstance(spec.descriptor, BaseTool))
    check(
        "就是 owner 上那个描述符本体（不是另造一个）",
        spec.descriptor is DocumentQueryServiceImpl.__dict__["list_documents"],
    )
    check(
        "名字与 tool_key 一致（模型看到的函数名就是查找键）",
        spec.descriptor.name == spec.tool_key,
    )
    check(
        "认不出的 key 取不到描述符，且不抛",
        registry.get("nope.missing") is None,
    )

    # --------------------------------------------------------
    section("select_specs：能用哪些工具的唯一判定")

    # 键从注册表现取：这个用例要验的是"去重 / 保序 / 跳过认不出的"，
    # 与仓库里到底有哪些工具无关 —— 写死名字会让它随工具增减而失败。
    available = registry.keys()

    check(
        "注册表里至少两个工具（本用例的前提）",
        len(available) >= 2,
    )

    first_key, second_key = available[0], available[1]

    picked = select_specs([
        first_key,
        first_key,              # 重复
        "nope.missing",         # 注册表里没有
        "",                     # 空串
        second_key,
    ])

    check(
        "去重、保序、跳过认不出的 key",
        [s.tool_key for s in picked] == [first_key, second_key],
    )
    check(
        "空输入得到空集合（不是 None、也不抛）",
        select_specs([]) == [] and select_specs(None) == [],
    )

    # ⚠️ 这一条盯的是"两处漂移"那个坑：capability 装配（给模型看什么）
    # 与 tools 节点（允许执行什么）调的是同一个函数、同一个输入，
    # 所以两个集合恒等。这里验的是它**确实是纯函数**——同样的输入
    # 两次调用结果一致，且不依赖调用顺序。
    check(
        "同样的输入两次调用结果一致（纯函数）",
        [s.tool_key for s in select_specs([first_key])]
        == [s.tool_key for s in select_specs([first_key])],
    )

    # --------------------------------------------------------
    section("工具名转义：tool_key（带点）<-> provider 安全名")

    # ⚠️ 这一组盯的是一个**真发生过的** 400：
    #     Invalid 'tools[0].function.name': string does not match pattern
    #     ^[a-zA-Z0-9_-]+$
    # tool_key 形如 `source_ref.method`（见 decorator.py 的推导），点号
    # 是结构性的，而 OpenAI 兼容端点不接受。转义只在**发给 provider 的
    # 那个字符串**上做，tool_key 本身（契约字段、库里的列）一个字不动。

    check(
        "点号换成下划线",
        wire_name("capability.list_capabilities")
        == "capability_list_capabilities",
    )
    check(
        "已经合法的名字原样返回",
        wire_name("document_list") == "document_list",
    )
    check(
        "转出来的名字符合 provider 的字符集",
        all(
            all(c.isalnum() or c in "_-" for c in wire_name(k))
            for k in available
        ),
    )

    # 反向：只在**给定的那一批**里找，且与 wire_name 是一对。
    check(
        "反查能翻回来",
        tool_key_of(
            wire_name(first_key), [first_key, second_key]
        ) == first_key,
    )
    check(
        "不在给定集合里的名字 -> None（模型幻觉）",
        tool_key_of("nope_missing", [first_key, second_key]) is None,
    )
    check(
        "空集合 -> None",
        tool_key_of(wire_name(first_key), []) is None,
    )
    # 转义是**多对一**的（source_ref 与名称里都可能带下划线），集合内撞名
    # 时必须拒绝而不是静默取其一 —— 取错的后果是执行了另一个工具。
    check(
        "集合内撞名 -> None（不猜）",
        tool_key_of("a_b_c", ["a_b.c", "a.b_c"]) is None,
    )
    check(
        "只有一条匹配时照常翻回去",
        tool_key_of("a_b_c", ["a_b.c"]) == "a_b.c",
    )

    # --------------------------------------------------------
    section("暴露给模型的是改名副本，注册表本体不动")

    # 描述符协议的 `spec.descriptor` 每次拿到的是**同一个对象**（进程级
    # 单例）。就地改它的 name 会把注册表里那份也改掉，一次改动影响此后
    # 所有调用且没有任何地方会报错 —— 所以暴露源必须取副本。
    #
    # ⚠️ 这一段必须挑一个**服务方法**（owner 非空）：上面那个"同一个对象"
    # 说的是描述符协议，而模块级 langchain 工具的 descriptor 就是模块里
    # 那个 BaseTool 本身 —— 它也在下面单独验。
    from app.infra.agent.tool_exposure import RegistryToolExposureSource

    service_key = next(
        key for key in registry.keys()
        if registry.get(key).owner is not None
    )

    exposed = await RegistryToolExposureSource().resolve(
        tool_keys=[service_key], conversation_id=None,
    )

    check(
        "暴露出去的名字是 provider 安全形式",
        len(exposed) == 1 and exposed[0].name == wire_name(service_key),
    )
    check(
        "注册表里那份描述符的 name 没被改动",
        registry.get(service_key).descriptor.name == service_key,
    )

    # 对照：模块级 langchain 工具的 `descriptor.name` 是**函数名**，不等于
    # tool_key —— 只有 ToolDescriptor 才在 `__set_name__` 里把 name 设成
    # tool_key。所以它对 `for_wire` 走的是"必须改名"那条分支。拿它一起
    # 验，否则那条分支在这个仓库里永远没被覆盖到。
    ownerless = [
        key for key in registry.keys()
        if registry.get(key).owner is None
    ]

    # ⚠️ 硬断言，不是"有就顺便验一下"：这一条盯的是一个**真发生过的**
    # bug —— `scan()` 曾经只触发 @tool 描述符自注册，而模块级 langchain
    # 工具没有描述符协议，于是永远进不了注册表。后果是同步那条路
    # （写 tool_definition）看得到它、`select_specs` 看不到：库里明明有
    # 这条工具，模型却永远拿不到。
    #
    # `app/agent/tools/catalog_tool.py` 随仓库发布，所以这里的期望不是
    # "碰巧有"而是"必须有"。
    check(
        "注册表里有模块级 langchain 工具（scan 收了它们）",
        len(ownerless) >= 1,
    )

    bare_key = ownerless[0]

    exposed = await RegistryToolExposureSource().resolve(
        tool_keys=[bare_key], conversation_id=None,
    )

    check(
        "无 owner 的工具也暴露成安全名",
        len(exposed) == 1
        and exposed[0].name == wire_name(bare_key),
    )
    check(
        "原始的 BaseTool 没被就地改名",
        registry.get(bare_key).descriptor.name != wire_name(bare_key),
    )

    # --------------------------------------------------------
    section("策略里的 tool_keys")

    from app.agent.context.processor.policy import (
        ContextPolicy,
        DefaultContextPolicyResolver,
        merge_policy_state,
    )

    base_policy = ContextPolicy(source="smoke")

    check(
        "基线默认没有工具",
        base_policy.tool_keys == ()
        and base_policy.as_state()["tool_keys"] == [],
    )

    merged = merge_policy_state(
        base_policy, {"tool_keys": ["document.list"]}
    )

    check(
        "覆盖项进得了合并后的结构（Agent Builder 写的就是它）",
        merged["tool_keys"] == ["document.list"],
    )
    check(
        "未知键照旧保留（response_format 那条路不会被挤掉）",
        merge_policy_state(
            base_policy, {"response_format": "X", "tool_keys": []}
        ).get("response_format") == "X",
    )
    check(
        "from_state 把 list 收成元组（字段是不可变的）",
        ContextPolicy.from_state(
            merged, base=base_policy
        ).tool_keys == ("document.list",),
    )
    check(
        "非法的值被忽略而不是强转",
        ContextPolicy.from_state(
            {"tool_keys": [1, 2]}, base=base_policy
        ).tool_keys == (),
    )
    check(
        "classifier 的基线不带工具（那条 gate 是关的）",
        DefaultContextPolicyResolver()
        .resolve(
            agent_type="react",
            builder_type=ContextBuilderType.CLASSIFIER,
        )
        .tool_keys == (),
    )

    # --------------------------------------------------------
    section("工具说明覆盖：从库到 bind 的合并")

    # 这一节盯的是一条**已经断掉的路**：人工写的说明（tool_documentation
    # 与 tool_parameter_documentation）只被 GET /tools 读，模型那条路走
    # 注册表（纯内存）够不着库 —— 症状是"列表页看得到、模型看不到"，
    # 而两边各自都工作正常，只有对着请求体才看得出来。
    #
    # 合并点是 `for_wire`（decorator.py 说的"准备工具清单的那一层"），
    # 数据由 Agent Builder 那一轮预载、经 context_policy 传下来。
    from app.infra.agent.tool_exposure import (
        _apply_parameter_overrides,
        for_wire,
    )
    from langchain_core.utils.function_calling import (
        convert_to_openai_tool as _to_openai,
    )

    service_spec = registry.get(service_key)
    bare_spec = registry.get(bare_key)

    # 先记下"合并之前"的样子，最后验单例没被动过。
    pristine_description = service_spec.descriptor.description
    pristine_schema = copy.deepcopy(service_spec.descriptor.args_schema)

    # 参数名从 schema 里现取：这个用例要验的是"覆盖落到了哪个节点"，
    # 与具体是哪个工具无关（写死参数名会让它随工具改名而失败）。
    param_name = next(iter(pristine_schema["properties"]))

    wired = for_wire(service_spec, {
        "description": "人工写的整工具说明。",
        "parameters": {f"/{param_name}": "人工写的参数说明。"},
    })

    payload = _to_openai(wired)["function"]

    check(
        "人工写的工具说明覆盖了代码侧的",
        payload["description"] == "人工写的整工具说明。",
    )
    check(
        "人工写的参数说明落进了 parameters",
        payload["parameters"]["properties"][param_name].get(
            "description"
        ) == "人工写的参数说明。",
    )
    check(
        "覆盖只落在副本上：注册表里那份描述与 schema 都没动",
        service_spec.descriptor.description == pristine_description
        and service_spec.descriptor.args_schema == pristine_schema,
    )

    # 模块级 langchain 工具（owner 为空）：它的 `spec.descriptor` 就是
    # 模块命名空间里那个对象**本体**，且 `args_schema` 是 pydantic 模型
    # 而不是 dict —— for_wire 曾经有一条"名字已经合法就直接返回本体"
    # 的捷径，往上面写说明会污染进程级注册表。
    bare_pristine = bare_spec.descriptor.description
    bare_keys = list(
        bare_spec.descriptor.args_schema.model_json_schema()[
            "properties"
        ]
    )

    wired_bare = for_wire(bare_spec, {
        "description": "人工写的组合工具说明。",
        "parameters": (
            {f"/{bare_keys[0]}": "人工写的参数说明。"} if bare_keys
            else {}
        ),
    })

    check(
        "无 owner 的模块级工具也能被覆盖（走的是模型那条 args_schema 形状）",
        _to_openai(wired_bare)["function"]["description"]
        == "人工写的组合工具说明。",
    )
    check(
        "原始的 BaseTool 没被写进人工说明",
        bare_spec.descriptor.description == bare_pristine,
    )

    # 空串与 None 都算"没写覆盖" —— 判据必须与读接口
    # （_assembly.build_summary 的 `override or source_raw`）一致，
    # 否则同一份覆盖会"列表页生效、模型不生效"。
    untouched = for_wire(service_spec, {
        "description": "",
        "parameters": {f"/{param_name}": ""},
    })

    check(
        "空串覆盖 = 没覆盖（与读接口同一个口径）",
        untouched.description == pristine_description
        and untouched.args_schema == pristine_schema,
    )

    check(
        "没有覆盖时也返回副本，不返回注册表本体",
        for_wire(service_spec) is not service_spec.descriptor,
    )

    # 路径走不到：只告警、**绝不凭空造属性**。造出来的参数模型会去填，
    # 而执行时它并不存在。
    missing = _apply_parameter_overrides(
        pristine_schema, {"/nope_not_a_param": "x"}
    )

    check(
        "走不到的路径不造属性（整体返回 None）",
        missing is None,
    )

    # 嵌套与数组：input_schema 是一棵真的树，所以路径按 JSON Pointer
    # 走（见 domain/entities/tool_documentation.py 里为什么用 path）。
    nested = _apply_parameter_overrides(
        {
            "type": "object",
            "properties": {
                "filters": {
                    "type": "object",
                    "properties": {
                        "ranges": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "field": {"type": "string"}
                                },
                            },
                        }
                    },
                }
            },
        },
        {"/filters/ranges/0/field": "人工：字段名"},
    )

    check(
        "数组元素的路径走得到（下标不改变节点）",
        nested["properties"]["filters"]["properties"]["ranges"][
            "items"
        ]["properties"]["field"]["description"] == "人工：字段名",
    )

    # 映射参数（`dict[K, V]` -> additionalProperties）：它的键不是
    # properties 里的条目，JSON Pointer 指不过去 —— 硬认会让一个**拼错
    # 的键**也解析成功，比跳过更糟。跳过必须留痕（上面有 WARNING）。
    mapped = _apply_parameter_overrides(
        {
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "object",
                        "properties": {"top_k": {"type": "integer"}},
                    },
                }
            },
        },
        {"/config/top_k": "人工：映射里的参数"},
    )

    check(
        "映射参数（additionalProperties）走不到 -> 跳过而不是乱认",
        mapped is None,
    )

    # 服务层：键 -> 覆盖。它内部复用 tool_ids_for_keys 这条唯一
    # 的键/id 桥，且**裁掉没有任何真值覆盖的工具**（这个 dict 会进
    # context_policy 并被 checkpoint，没有覆盖的塞进去只是白白变大）。
    overridden_key = "smoke.overridden"

    # 时间戳只是为了让实体构造得起来（这几个字段没有默认值），
    # 与本节断言无关 —— 用一个定值而不是 datetime.now()，
    # 免得两次运行之间出现任何可比性。
    NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

    await definitions.add(ToolDefinition(
        tool_id=UUID(int=9001),
        tool_key=overridden_key,
        name="overridden",
        source_type=SourceType.LOCAL_METHOD,
        manifest_hash="smoke",
        discovered_at=NOW,
        last_synced_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    ))

    await documentations.upsert(
        UUID(int=9001),
        description_override="人工：整工具",
        return_description_override="人工：返回值",
        updated_by="smoke",
        now=NOW,
    )
    await documentations.replace_parameter_documentations(
        UUID(int=9001),
        [("/parent_id", "人工：参数")],
        updated_by="smoke",
        now=NOW,
    )

    # 对照：这个工具**没有**任何人工说明，应当被裁掉。
    await definitions.add(ToolDefinition(
        tool_id=UUID(int=9002),
        tool_key="smoke.untouched",
        name="untouched",
        source_type=SourceType.LOCAL_METHOD,
        manifest_hash="smoke",
        discovered_at=NOW,
        last_synced_at=NOW,
        created_at=NOW,
        updated_at=NOW,
    ))

    resolved = await query.tool_documentation_overrides(
        [overridden_key, "smoke.untouched", "smoke.not_in_db"]
    )

    check(
        "形状是 {tool_key: {description, parameters}}",
        resolved.get(overridden_key) == {
            "description": "人工：整工具",
            "parameters": {"/parent_id": "人工：参数"},
        },
    )
    check(
        "没有人工说明的工具、库里查不到的 key 都被裁掉",
        "smoke.untouched" not in resolved
        and "smoke.not_in_db" not in resolved,
    )
    check(
        "return_description_override 不进这个结构"
        "（provider 的函数 schema 没有那一格）",
        "return_description_override" not in resolved[overridden_key],
    )

    # 参数级覆盖**也**要能进 schema（只有工具级覆盖的那种工具靠这条）。
    param_only = for_wire(bare_spec, {
        "parameters": (
            {f"/{bare_keys[0]}": "只有参数级"} if bare_keys else {}
        ),
    })

    if bare_keys:
        check(
            "只有参数级覆盖时，工具描述保持空、参数说明照常进",
            _to_openai(param_only)["function"]["parameters"][
                "properties"
            ][bare_keys[0]].get("description") == "只有参数级",
        )

    # 传输层：覆盖随 context_policy 走，而 `merge_policy_state` 保留
    # 未知键 —— tool_overrides 全靠这条才到得了 capability 处理器。
    check(
        "tool_overrides 能穿过策略合并（它是个未知键）",
        merge_policy_state(
            base_policy, {"tool_overrides": {overridden_key: {}}}
        ).get("tool_overrides") == {overridden_key: {}},
    )

    # --------------------------------------------------------
    section("应用级工厂：没有 Request 也取得到")

    reset_factories()

    check(
        "没装配时明确报错（而不是现场造一个缺依赖的）",
        _raises(RuntimeError, get_service_factory),
    )

    factory = build_factories(
        _fake_session_factory,
        pool={"pipeline_registry": "FAKE_REGISTRY"},
    )

    check(
        "装配后无参可取 —— Agent 工具路径靠这条",
        get_service_factory() is factory,
    )
    check(
        "仓储工厂同样是应用级的",
        get_repository_factory() is factory.repositories,
    )
    check(
        "池子原样带进上下文（组合工具要用）",
        factory.pool.get("pipeline_registry") == "FAKE_REGISTRY",
    )

    reset_factories()

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
