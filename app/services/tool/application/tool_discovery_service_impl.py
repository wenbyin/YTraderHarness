"""
ToolDiscoveryService 的默认实现。

事务边界由调用方决定 —— 整次同步在**一个**事务里完成，
中途失败不会留下"新建了一半"的中间态。

⚠️ 已知取舍：一次同步可能扫出几百个工具，全都写在同一段事务里，
因此事务会比较大。当前工具数量（几十个）下没有影响；
真要上千个时，应改成按批提交（每批一个 scope），
代价是同步结果不再是原子的。
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from uuid import uuid4

from app.common.service.base import ServiceContext

from ..domain.entities import ToolDefinition
from ..domain.schemas import (
    SourceState,
    SourceType,
    ToolSyncRequest,
    ToolSyncResult,
)
from ..registry.discoverer import ToolDiscoverer
from ..registry.spec import ToolManifestData
from ..repository.tool_definition import ToolDefinitionRepository
from ..repository.tool_documentation import (
    ToolDocumentationRepository,
)
from .tool_discovery_service import ToolDiscoveryService


class ToolDiscoveryServiceImpl(ToolDiscoveryService):

    def __init__(
            self,
            *,
            tool_definition_repository: ToolDefinitionRepository,
            tool_documentation_repository: (
                ToolDocumentationRepository
            ),
            discoverer: ToolDiscoverer | None = None,
            ctx: ServiceContext | None = None,
    ) -> None:
        super().__init__(ctx)

        self._definitions = tool_definition_repository
        self._documentations = tool_documentation_repository

        # 生产上走 default()：扫描器都是进程级事实（"代码里声明了
        # 哪些工具"），不是"这一份运行时依赖"，因此不进池子。
        #
        # 但**留一个注入口**：同步是全局的（会把仓库里所有工具都扫
        # 进来），因此想单独验证同步逻辑的用例必须能把扫描范围收窄，
        # 否则它会随"别人在别处加了个工具"而失败 —— 那种测试很脆，
        # 而且失败信息指向不了真正的原因。
        #
        # 工厂不会注入它（不在池子里、也不是仓储契约），
        # 因此走默认值这条分支，见 ServiceFactory._build_kwargs 第 6 条。
        self._discoverer = (
            discoverer if discoverer is not None
            else ToolDiscoverer.default()
        )

    # ========================================================
    # 同步
    # ========================================================

    async def sync(
            self,
            request: ToolSyncRequest | None = None,
    ) -> ToolSyncResult:

        request = request or ToolSyncRequest()

        # ⚠️ 扫描是**同步**的，必须挪出事件循环
        # ------------------------------------
        # 本地扫描器要 import 包，MCP 扫描器要等 `tools/list` ——
        # `mcp_client.list_provider_tools` 虽然自己起线程跑一次性
        # 会话，但**等它结果**的这一侧仍然阻塞（`future.result`）。
        # 直接调用会把循环按住整段扫描时间（哨兵日志里那条
        # "事件循环被卡住 N 秒" 就是这个），同一时刻所有请求、SSE
        # 心跳、Redis 读超时定时器一起停摆。
        #
        # 线程里是安全的：扫描只读进程内注册表 + 发网络请求，
        # 不碰任何 async 资源（DB 从下面的 `_reconcile` 才开始用，
        # 仍留在循环上）；返回的是纯数据快照，跨线程传递没有
        # 循环亲和性。
        manifests = await asyncio.to_thread(
            self._discoverer.discover,
            source_types=self._parse_source_types(
                request.source_types
            ),
            source_refs=request.source_refs or None,
        )

        now = datetime.now(timezone.utc)

        created, updated, unchanged = await self._reconcile(
            manifests, now
        )

        missing, orphaned = await self._retire_absent(
            {m.tool_key for m in manifests},
            request=request,
            now=now,
        )

        return ToolSyncResult(
            scanned=len(manifests),
            created=created,
            updated=updated,
            unchanged=unchanged,
            missing=missing,
            orphaned_documentations=orphaned,
        )

    # ========================================================
    # internals
    # ========================================================

    async def _reconcile(
            self,
            manifests: list[ToolManifestData],
            now: datetime,
    ) -> tuple[int, int, int]:
        """把扫出来的快照与库里的行对齐。返回 (新建, 覆盖, 未变)。"""

        if not manifests:
            return 0, 0, 0

        existing = {
            d.tool_key: d
            for d in await self._definitions.list_by_keys(
                [m.tool_key for m in manifests]
            )
        }

        created = updated = 0
        unchanged_ids = []

        for manifest in manifests:

            digest = manifest.compute_hash()
            current = existing.get(manifest.tool_key)

            if current is None:
                await self._definitions.add(
                    self._to_entity(manifest, digest, now)
                )
                created += 1
                continue

            # 两种情况都要重写：
            #
            #   1. 内容变了（摘要不同）
            #   2. 它之前是 MISSING，现在代码里又出现了
            #
            # 第 2 条容易漏，而且漏了以后症状很隐蔽：方法被删掉再
            # 加回来（改回原名、重新挂上 @tool），内容与摘要是**一模一样**
            # 的 —— 只比摘要的话会落进"unchanged"，于是那一行永远停在
            # MISSING，一个已经恢复的工具在列表里一直显示"已消失"。
            #
            # 状态恢复也记入 updated：行确实被重写了。ToolSyncResult
            # 的六个字段是 tool.yaml 定死的，没有单独的 restored，
            # 硬塞进 unchanged 会让"我什么都没改"这句话变成假的。
            if (
                current.manifest_hash != digest
                or current.source_state != SourceState.ACTIVE
            ):
                await self._definitions.update(
                    self._to_entity(
                        manifest,
                        digest,
                        now,
                        tool_id=current.tool_id,
                        # 保留首次发现时间 —— 每次同步都刷成现在
                        # 等于这个字段没有意义。
                        discovered_at=current.discovered_at,
                        created_at=current.created_at,
                    )
                )
                updated += 1
                continue

            unchanged_ids.append(current.tool_id)

        # "内容没变"也要刷新 last_synced_at：这个字段的语义是
        # "最近一次同步时间"，不是"最近一次变化时间"。
        # 一条 IN 语句搞定，不是 N 次 update。
        await self._definitions.touch_synced_at(
            unchanged_ids, now=now
        )

        return created, updated, len(unchanged_ids)

    async def _retire_absent(
            self,
            scanned_keys: set[str],
            *,
            request: ToolSyncRequest,
            now: datetime,
    ) -> tuple[int, int]:
        """
        处理"库里有、代码里已经没有了"的行。

        返回 (本次判定为缺失的行数, 被级联删掉的说明行数)。

        两种模式下"缺失"的取法**不一样**，这不是笔误：

            remove_missing=false   只看 ACTIVE 的行（已经 MISSING 的
                                   不再重复标、也不再重复计数，否则
                                   每次同步都会把它们再报一遍）
            remove_missing=true    还要把**已经 MISSING 的**行一起捞
                                   出来删 —— 它们是上一轮标出来的，
                                   不在 ACTIVE 里，漏了这一步就出现
                                   "点了删除、missing 却恒为 0、
                                   刷新后缺失的工具还在"
        """

        # ⚠️ 按**本次同步的范围**对账，而不是全表
        # ------------------------------------
        # source_types / source_refs 是"这次只扫这一部分"的意思。
        # 若拿全表去比对，一次 `source_types=["mcp"]` 的同步会把所有
        # 本地工具判成「没扫到」并全部转 MISSING —— 它们只是这次
        # 没被扫，不是消失了。MCP 接入后这条路径会变得常见（前端
        # 刷新单个 provider），所以必须在这里收窄。
        #
        # 收窄对删除同样生效：删是**不可逆**的，范围写错一次就少一批行。
        wanted_types = set(
            self._parse_source_types(request.source_types) or []
        )
        wanted_refs = set(request.source_refs or [])

        def in_scope(definition: ToolDefinition) -> bool:
            return (
                (
                    not wanted_types
                    or definition.source_type in wanted_types
                )
                and (
                    not wanted_refs
                    or definition.source_ref in wanted_refs
                )
            )

        active = [
            d
            for d in await self._definitions.list_active()
            if in_scope(d)
        ]

        stale = [d for d in active if d.tool_key not in scanned_keys]

        if not request.remove_missing:
            # 默认路径：只标记，不删除。
            if not stale:
                return 0, 0

            await self._definitions.mark_missing(
                [d.tool_id for d in stale], now=now
            )

            return len(stale), 0

        # ---- remove_missing=true：物理删除 ----
        #
        # 除了"这一轮才缺失的"（stale），还要删**已经 MISSING** 的行。
        # 它们是上一次（remove_missing=false，默认值）同步标出来的：
        # source_state 已经不是 ACTIVE，`list_active()` 根本取不到，
        # 于是上一版实现里这一批行永远删不掉，missing 也永远报 0。
        #
        # `tool_key not in scanned_keys` 这一条是防御性的：理论上
        # _reconcile 已经把"重新出现"的行改回 ACTIVE 了，但漏掉它的
        # 代价是删掉一个**代码里还在**的工具 —— 这里宁可多判断一次。
        already_missing = [
            d
            for d in await self._definitions.list_missing()
            if in_scope(d) and d.tool_key not in scanned_keys
        ]

        doomed = stale + already_missing

        if not doomed:
            return 0, 0

        doomed_ids = [d.tool_id for d in doomed]

        # 物理删除前先数一下会连带丢掉多少人工写的说明 ——
        # CASCADE 是静默的，不报这个数字，用户事后才发现。
        orphaned = await self._documentations.count_for_tools(
            doomed_ids
        )

        await self._definitions.delete_many(doomed_ids)

        return len(doomed), orphaned

    @staticmethod
    def _parse_source_types(
            raw: list[str],
    ) -> list[SourceType] | None:
        """
        把请求里的字符串转成枚举。

        认不出来的值直接忽略而不是报错：这个参数是**收窄**扫描范围
        用的，调用方多写一个将来才支持的来源不该让整次同步失败。
        """

        if not raw:
            return None

        parsed = []

        for item in raw:
            try:
                parsed.append(SourceType(item))
            except ValueError:
                continue

        return parsed or None

    @staticmethod
    def _to_entity(
            manifest: ToolManifestData,
            digest: str,
            now: datetime,
            *,
            tool_id=None,
            discovered_at: datetime | None = None,
            created_at: datetime | None = None,
    ) -> ToolDefinition:
        """快照 -> 领域实体。新建时 id 与两个时间戳现场生成。"""

        return ToolDefinition(
            tool_id=tool_id or uuid4(),
            tool_key=manifest.tool_key,
            name=manifest.name,
            source_type=manifest.source_type,
            source_ref=manifest.source_ref,
            entrypoint=manifest.entrypoint,
            input_schema=manifest.input_schema,
            output_schema=manifest.output_schema,
            source_docstring=manifest.source_docstring,
            manifest_hash=digest,
            # 扫到了就是 ACTIVE —— 包括从 MISSING 恢复回来的。
            source_state=SourceState.ACTIVE,
            discovered_at=discovered_at or now,
            last_synced_at=now,
            created_at=created_at or now,
            updated_at=now,
        )
