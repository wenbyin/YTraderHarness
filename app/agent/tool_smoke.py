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
import dataclasses
import pathlib
import shutil
import sys
import tempfile
import textwrap
from abc import abstractmethod
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
from app.services.tool.application.classification_service_impl import (
    ClassificationServiceImpl,
)
from app.services.tool.application.tool_discovery_service_impl import (
    ToolDiscoveryServiceImpl,
)
from app.services.tool.application.tool_documentation_service_impl import (
    ToolDocumentationServiceImpl,
)
from app.services.tool.application.tool_query_service_impl import (
    ToolQueryServiceImpl,
)
from app.services.tool.domain.entities import (
    ToolClassification,
    ToolDefinition,
)
from app.services.tool.domain.schemas import (
    ClassificationCreate,
    ClassificationUpdate,
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
    tool,
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

        if source_state is not None:
            rows = [
                r for r in rows if r.source_state == source_state
            ]

        return rows[offset:offset + limit], len(rows)

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

    async def count_for_tools(self, tool_ids):
        return sum(
            1 + len(self.docs[t].get("parameters", []))
            for t in tool_ids if t in self.docs
        )


class FakeClassificationRepository:
    def __init__(self):
        self.rows: dict[UUID, ToolClassification] = {}
        self.links: dict[UUID, list[UUID]] = {}

    async def add(self, entity):
        self.rows[entity.classification_id] = entity
        return entity

    async def update(self, entity):
        self.rows[entity.classification_id] = entity
        return entity

    async def get(self, classification_id):
        return self.rows.get(classification_id)

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
    classifications = FakeClassificationRepository()

    discovery = ToolDiscoveryServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=documentations,
        discoverer=discoverer,
    )
    query = ToolQueryServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=documentations,
        tool_classification_repository=classifications,
    )
    docs = ToolDocumentationServiceImpl(
        tool_definition_repository=definitions,
        tool_documentation_repository=documentations,
    )
    taxonomy = ClassificationServiceImpl(
        tool_classification_repository=classifications,
        tool_definition_repository=definitions,
    )

    # --------------------------------------------------------
    section("自注册（没有任何 register() 调用）")

    check(
        "两个 @tool 方法都进了注册表",
        set(registry.keys()) == {"document.list", "document.count"},
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
    section("分类")

    root = await taxonomy.create_classification(
        ClassificationCreate(code="retrieval", name="检索类")
    )
    child = await taxonomy.create_classification(
        ClassificationCreate(
            code="retrieval.semantic",
            name="语义检索",
            parent_id=root.id,
        )
    )

    try:
        await taxonomy.create_classification(
            ClassificationCreate(code="retrieval", name="重名")
        )
    except ConflictError:
        check("code 冲突 -> ConflictError", True)
    else:
        check("code 冲突 -> ConflictError", False)

    try:
        await taxonomy.update_classification(
            root.id, ClassificationUpdate(parent_id=child.id)
        )
    except ConflictError:
        check("把自己挂到后代下面 -> 成环被拒", True)
    else:
        check("把自己挂到后代下面 -> 成环被拒", False)

    linked = await taxonomy.replace_tool_classifications(
        target.tool_id,
        ToolClassificationUpdate(classification_ids=[child.id]),
    )
    check("关联成功", len(linked) == 1 and linked[0].code
          == "retrieval.semantic")

    relisted = await query.list_tools(page=1, page_size=10)
    check(
        "列表页带出了分类（批量查询，不是 N+1）",
        len(relisted.items[0].classifications) == 1,
    )

    try:
        await taxonomy.replace_tool_classifications(
            target.tool_id,
            ToolClassificationUpdate(
                classification_ids=[UUID(int=888)]
            ),
        )
    except NotFoundError:
        check("关联不存在的分类 -> NotFoundError", True)
    else:
        check("关联不存在的分类 -> NotFoundError", False)

    emptied = await taxonomy.replace_tool_classifications(
        target.tool_id,
        ToolClassificationUpdate(classification_ids=[]),
    )
    check("传空数组即全部解除", emptied == [])

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
        "组合工具走的是 TOOL_SOURCE_REF 模块级变量",
        all(
            m.source_ref == "agent.platform" for m in composite
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
    check(
        "组合工具与 @tool 同属 local_method",
        len({m.source_type for m in everything}) == 1,
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
