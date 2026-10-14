"""
管理端记忆查询接口：提取进度水位 + 跨层溯源。

对应 memory_api_v2.yaml 的两个 tag：

    Memory Extraction    POST /memory/extraction/l1|l2|l3   读三张水位表
    Memory Query         POST /memory/query/l1/sources      记忆 -> 原话
                         POST /memory/query/l0/atoms        原话 -> 记忆

⚠️ 管理端门槛写在**注册那一行**，不在这里
------------------------------------
这五个端点要求 `Y-Trader-Admin`（见 memory_api_v2.yaml 的 security 段）。
那道门槛加在 `app/api/v1/router.py` 的 include 那一行上：

    protected_router.include_router(
        memory_query_router,
        dependencies=[Depends(require_admin)],
    )

与本仓库"接口注册在哪棵树上就是什么待遇，照着 include 那行读"那条规矩
一致（见 router.py 的模块 docstring 与 users.py 里 /users/search 的同一处
说明）。**不要**在这里的签名里再加一行 `Depends(require_admin)` 把它抄一
遍 —— 抄的那行看起来只是"取值"，被删掉时不会有人觉得删错了，而删掉之后
端点**看起来**只是少了个参数。

因此本文件的端点签名里**没有** `user`：身份（target）在请求体里，那是
管理端视图的语义（见 schemas/query.py 的模块 docstring）。已登录这件事
由注册处那道依赖顺带保证。

⚠️ 归属校验**没有**被管理端跳过
------------------------
"能看谁的记忆"由端点上的角色门槛决定，"这个会话/这条记忆是不是真属于
那个 target"由服务层照常校验（见 impl 的 `_require_session`）。
管理端跳过的只是"只能看自己"这一条。
"""

from __future__ import annotations

from fastapi import APIRouter

from app.common.utils import to_http_error
from app.services.dependencies import Service
from app.services.memory.application.memory_query_service import (
    MemoryQueryService,
)
from app.services.memory.domain.schemas.query import (
    L0AtomsRequest,
    L0AtomsResponse,
    L0QueryRequest,
    L0QueryResponse,
    L1ExtractionRequest,
    L1ExtractionResponse,
    L1QueryRequest,
    L1QueryResponse,
    L1SourcesRequest,
    L1SourcesResponse,
    L2DetailRequest,
    L2DetailResponse,
    L2ExtractionRequest,
    L2ExtractionResponse,
    L2QueryRequest,
    L2QueryResponse,
    L3ExtractionRequest,
    L3ExtractionResponse,
    L3QueryResponse,
    MemoryRetrievalPrompt,
    MemoryRetrievalRequest,
    TargetRequest,
)

router = APIRouter()


# ============================================================
# 上下文预览
# ============================================================


@router.post(
    "/memory/retrieval",
    response_model=MemoryRetrievalPrompt,
    tags=["Memory Retrieval"],
)
async def preview_memory_context(
        request: MemoryRetrievalRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    预览默认记忆上下文的组装结果（**不调模型**）。

    默认组装顺序是 `persona (L3) -> scenes (L2)`，而 L2 只带索引字段
    （`filename` / `summary` / `heat`），不带场景全文。

    ⚠️ 本端点**不执行** L0/L1 召回
    ------------------------
    那两层不属于默认上下文 —— Agent 在运行时按需调 `memory.search` /
    `memory_atom.search` 召回（见 memory.yaml 与 memory_atom.yaml）。
    `retrieve.l2` / `retrieve.l3` 只决定**这次预览**展示哪几层，
    不影响 Agent 运行时的任何决策。

    ⚠️ `prompt.system` 现在是 null
    -------------------------
    记忆上下文提示词的正文还没写（见
    `app/agent/prompt/prompts/memory_retrieval.py` 的 SYSTEM_PROMPT）。
    `prompt.user` 已经是按 sections 渲染出来的真实预览。

    某一层查不出来时那一层进 `degraded`，其余照常返回（部分降级）。
    """

    try:
        return await service.retrieval_preview(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 提取进度水位
# ============================================================


@router.post(
    "/memory/extraction/l1",
    response_model=L1ExtractionResponse,
    tags=["Memory Extraction"],
)
async def get_l1_extraction(
        request: L1ExtractionRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    这个会话的 L1（原子记忆）抽取到哪条消息了。

    读的是 `memory_atom_cursor` 那**一行权威水位**：`last_seq` 就是
    "抽到哪了"的全部答案（下一次抽取读 `seq > 它`）。不返回"还剩几轮没抽"
    那类派生值 —— 口径只有一处，见 schemas/query.py 的模块 docstring。

    ⚠️ `last_scene_name` 是 L1 的**情境名**，不是 L2 的场景名
    --------------------------------------------------
    它是提示词任务一「结合上一个情境」的输入。拿它当场景文件名去对 L2 的
    `/memory/query/l2` 是**对不上**的（本仓库这两者没有映射关系，见
    yaml 里"跨层溯源"那段的能力边界）。
    """

    try:
        return await service.l1_extraction(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/memory/extraction/l2",
    response_model=L2ExtractionResponse,
    tags=["Memory Extraction"],
)
async def get_l2_extraction(
        request: L2ExtractionRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    这个会话的 L2（场景整合）整合到哪条记忆了。

    三个字段要一起看：`last_atom_updated_time` 是进度，`last_l2_at` 是
    "上次跑完是什么时候"（**含空转那一次**），`last_atom_ids` 只是边界
    那一批 —— 它是取增量时排除重复用的，不是一份消费台账。
    """

    try:
        return await service.l2_extraction(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/memory/extraction/l3",
    response_model=L3ExtractionResponse,
    tags=["Memory Extraction"],
)
async def get_l3_extraction(
        request: L3ExtractionRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    这个 scope 的 L3（画像）抽象到哪一版场景了。

    ⚠️ 与上面两条**粒度不同**：水位的键是 scope（agent + user|team），
    不是会话 —— 画像本身就是跨会话聚合的产物。所以请求体里只有 target。

    这个 scope 从没生成过画像**不是错误**：返回一组默认值（水位 null、
    空 id 集），与 `/memory/query/l3` 那边"画像尚未生成时返回 null"是
    同一条取舍。
    """

    try:
        return await service.l3_extraction(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 浏览
# ============================================================
#
# 这一组**不做召回**：过滤只有等值 / 范围 / 字面包含，翻页只有游标。
# 语义召回走 memory.yaml 的 /memory/search 与 memory_atom.yaml 的
# /memory-atom/search（那两个同时是给主模型的工具）。


@router.post(
    "/memory/query/l0",
    response_model=L0QueryResponse,
    tags=["Memory Query"],
)
async def query_memory_l0(
        request: L0QueryRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    分页浏览 L0 原始对话记录（不执行 hybrid / 向量 / 关键词召回）。

    ⚠️ `text_contains` 匹配的是**文本块**，不是整条记录的 JSON
    ------------------------------------------------
    它等价于 `_assembly.message_text`（"这条消息的正文是什么"的唯一一份
    定义：只取 `type=text` 的块）。所以纯工具调用的消息搜不到东西 ——
    它们的 `text` 本来就是空串。通配符（`%` `_`）会被转义，按字面匹配。

    ⚠️ `session_id` 不传时是**跨该 user 的会话**查
    --------------------------------------
    返回的每条记录都带 `session_id`，据它去 `/memory/extraction/l1` 可以
    看那个会话抽到哪了。

    团队 target -> 400（`agent_message` 没有 org 列，归属在会话上）；
    会话不存在或不可见 -> 404。
    """

    try:
        return await service.l0_records(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/memory/query/l1",
    response_model=L1QueryResponse,
    tags=["Memory Query"],
)
async def query_memory_l1(
        request: L1QueryRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    分页浏览 L1 原子记忆记录（不执行 hybrid / 向量 / 关键词召回）。

    `text_contains` 是 `ILIKE` 整串匹配（中文没有分词配置，只能整串），
    与 `type` / `priority` / `scene_name` 的等值过滤、`updated_from` /
    `updated_to` 的范围过滤可以叠加。

    ⚠️ 时间范围走的是 `updated_from` / `updated_to`（系统写的
    `updated_time`），**不是** `timestamp_str` —— 那是 LLM 推出来的文本
    （可能是"2018 年"），拿它做范围过滤要么报错要么静默给错结果。

    团队 target -> 400（`memory_atom` 的可见性维度只有 `user_id`，
    `org_id` 是元数据、不参与过滤）。
    """

    try:
        return await service.l1_records(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/memory/query/l2",
    response_model=L2QueryResponse,
    tags=["Memory Query"],
)
async def query_memory_l2(
        request: L2QueryRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    分页列出 L2 场景索引（**不含**正文；要正文走 `/memory/query/l2/detail`）。

    ⚠️ 这是**列举**，不是检索
    --------------------
    `memory_scene` 没有挂载表（既没有 FTS 也没有向量索引）—— 一个 scope
    就十几篇，建索引买不到任何东西。`filename_contains` 只做文件名的字面
    包含筛选，而文件名是模型自己写的路径，按它筛的实际价值有限。

    `summary` / `heat` 是从场景正文的 META 块**按需解析**出来的（不是表上
    的列）；解析不出时给空串 / null，不报错。

    两种 scope 都支持（场景就是按 scope 聚合的）。
    """

    try:
        return await service.l2_scenes(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/memory/query/l2/detail",
    response_model=L2DetailResponse,
    tags=["Memory Query"],
)
async def query_memory_l2_detail(
        request: L2DetailRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    获取单篇场景全文（含 META 块）。精确读取，不执行语义检索。

    `filename` 用 `/memory/query/l2` 返回的那个（逐字匹配，不做归一化 ——
    归一化是写入侧的事）。取不到 -> 404。
    """

    try:
        return await service.l2_scene_detail(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/memory/query/l3",
    response_model=L3QueryResponse,
    tags=["Memory Query"],
)
async def query_memory_l3(
        request: TargetRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    读取该 scope 的画像（`persona.md`）。

    ⚠️ **尚未生成画像不是错误**
    ----------------------
    返回 `persona: null`（HTTP 200），不是 404 —— 与 `/memory/extraction/l3`
    那边"水位是一组默认值"同一条取舍。用 404 表达它会让调用方把"这个用户
    还没画像"与"agent_id 传错了"混成同一个响应。

    团队与个人 scope 都支持（一个 scope 至多一篇画像，所以请求体里只有
    `target`，没有第二个参数）。
    """

    try:
        return await service.l3_persona(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


# ============================================================
# 跨层溯源
# ============================================================


@router.post(
    "/memory/query/l1/sources",
    response_model=L1SourcesResponse,
    tags=["Memory Query"],
)
async def get_l1_sources(
        request: L1SourcesRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    这条 L1 记忆是从哪几句原话来的（`memory_atom.source_message_ids`）。

    ⚠️ 层间唯一一条**真落在库里**的边
    ----------------------------
    L2 -> L1、L3 -> L2 都没有存储映射（场景文件名是模型写的路径，画像
    那边只剩边界那一批 id），那两个方向的接口本组**不提供** —— 用近似
    匹配会给出看起来完整、实际缺一半的答案。

    ⚠️ 返回里 `missing_message_ids` 不是装饰
    --------------------------------
    这一列**没有外键**，也不保证消息还在。所以"记得有 5 条来源、只找到
    3 条"必须是看得见的事实 —— 只返回找到的那三条，读起来就是"这条记忆
    只有三条来源"。
    """

    try:
        return await service.l1_sources(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc


@router.post(
    "/memory/query/l0/atoms",
    response_model=L0AtomsResponse,
    tags=["Memory Query"],
)
async def get_l0_atoms(
        request: L0AtomsRequest,
        service: MemoryQueryService = Service[MemoryQueryService],
):
    """
    反查：这句原话进了哪些记忆。

    ⚠️ `session_id` 必填，而且消息必须**确实在这个会话里**
    ---------------------------------------------
    包含查询（`source_message_ids @> [...]`）上没有索引可用，靠会话把它
    收窄到这几条记忆上（见 `MemoryAtomRepository.list_by_source_message`）。
    跨会话反查要一张 GIN 索引 —— 那是写入侧的决定，不在这个只读接口里
    顺手做掉。

    消息不存在 -> 404（不是空列表）：后者读起来是"这条消息没有产生任何
    记忆"，而那是另一件事（空列表表示真没有记忆引用它）。
    """

    try:
        return await service.l0_atoms(request)

    except ValueError as exc:
        raise to_http_error(exc) from exc
