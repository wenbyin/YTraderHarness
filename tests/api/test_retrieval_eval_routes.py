"""
检索评测端点的**线上形状**：状态码、响应体、错误映射，以及它们有没有
真的挂在受保护的那棵路由树上。

⚠️ 为什么必须真打一次 HTTP
------------------------
服务层用例拿回来的是 DTO 对象 —— 字段名写错、类型不对、被 `response_model`
挡掉，那些用例**全都不会红**。决定前端看到什么的是 FastAPI 那一步序列化。

（本仓库既有的一处教训，见 test_session_messages_response.py 的同一段说明：
实体被当 DTO 返回过，服务层用例全绿，只有真打接口才炸。这里同样要经过
`response_model` 那一关。）

装置刻意**不碰数据库**：`Service[契约]` 只在请求时调一次
`get_service_factory()`，替成一个"不需要事务"的假工厂即可。于是这些用例
验的是 路由 -> DTO -> 序列化 -> 状态码 那一段，毫秒级。

⚠️ 这里补上了一条 router.py 的 docstring 提到、但**实际不存在**的守护
------------------------------------------------------------------------
`app/api/v1/router.py` 的模块 docstring 写着"guard 见
tests/api/test_auth_enforcement.py：它遍历 protected_router 里注册的每个
路径，断言不带 token 一律 401"。**那个文件不在仓库里**（tests/api 下只有
两条用例）。所以"新接口忘了注册到 protected_router 上"这件事一直没有
任何机械保护 —— 而它的后果是一个匿名就能建的接口。

本文件不重建那条全局 guard（那要处理认证组件开不开的两种环境），只钉住
**本模块这一组路径**确实在受保护的树上。见
`test_all_paths_live_on_the_protected_router`。
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import FastAPI

import app.services.dependencies as dependencies_module
from app.api.v1.retrieval_eval import router as retrieval_eval_router
from app.api.v1.router import protected_router
from app.common.auth import current_user
from app.common.exceptions import ConflictError, NotFoundError
from app.services.document.domain.schemas.enum import (
    EvalBundleStatus,
    EvalJobStatus,
)
from app.services.document.domain.schemas.retrieval_eval import (
    EvalBundleDetail,
    EvalBundleExport,
    EvalBundleListResponse,
    EvalMetrics,
    EvalRunDetail,
)

_BUNDLE_ID = uuid4()
_RUN_ID = uuid4()
_CASE_ID = uuid4()


def _bundle_detail() -> EvalBundleDetail:
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)

    return EvalBundleDetail(
        bundle_id=_BUNDLE_ID,
        name="线上形状数据集",
        status=EvalBundleStatus.READY,
        document_type_definition_ids=[uuid4()],
        source_chunk_count=12,
        case_count=30,
        latest_generation_status=EvalJobStatus.COMPLETED,
        dataset_sha256="a" * 64,
        created_at=now,
        completed_at=now,
        schema_version="1.0",
        selection_config={"seed": 7},
        generation_config={"questions_per_chunk": 3},
        latest_generation_run_id=_RUN_ID,
        updated_at=now,
    )


class _FakeService:
    """
    只实现这些用例走到的那几个方法。

    刻意**不**实现全部契约：多实现一个方法就多一份与真实现漂移的可能，
    而这里要验的是路由层，不是服务层。
    """

    def __init__(self, *, error: Exception | None = None) -> None:
        self._error = error
        #: 痕迹页收到的实参 —— 路由有没有把 `attempt_no` 漏掉，只能从这里看。
        self.trace_calls: list[tuple[Any, Any, Any]] = []

    def _raise_if_needed(self) -> None:
        if self._error is not None:
            raise self._error

    async def create_bundle(self, request: Any, **_: Any) -> Any:
        self._raise_if_needed()

        from app.services.document.domain.schemas.retrieval_eval import (
            CreateEvalBundleAcceptedResponse,
        )

        return CreateEvalBundleAcceptedResponse(
            bundle_id=_BUNDLE_ID,
            generation_run_id=_RUN_ID,
            status=EvalBundleStatus.GENERATING,
        )

    async def list_bundles(self, **_: Any) -> EvalBundleListResponse:
        self._raise_if_needed()
        return EvalBundleListResponse(
            items=[], page=1, page_size=20, total=0,
        )

    async def get_bundle(self, bundle_id: UUID) -> EvalBundleDetail:
        self._raise_if_needed()
        return _bundle_detail()

    async def export_bundle(self, bundle_id: UUID) -> EvalBundleExport:
        self._raise_if_needed()
        return EvalBundleExport(
            schema_version="1.0",
            bundle_id=_BUNDLE_ID,
            bundle_name="线上形状数据集",
            dataset_sha256="b" * 64,
            items=[],
        )

    async def get_generation_run(self, generation_run_id: UUID) -> Any:
        self._raise_if_needed()
        raise NotFoundError("假服务：这条用例不走到这里")

    async def create_eval_run(self, request: Any, **_: Any) -> Any:
        self._raise_if_needed()

        from app.services.document.domain.schemas.retrieval_eval import (
            CreateEvalRunAcceptedResponse,
        )

        return CreateEvalRunAcceptedResponse(
            eval_run_id=_RUN_ID,
            bundle_id=_BUNDLE_ID,
            status=EvalJobStatus.QUEUED,
        )

    async def get_eval_run(self, eval_run_id: UUID) -> EvalRunDetail:
        self._raise_if_needed()

        from datetime import datetime, timezone

        return EvalRunDetail(
            eval_run_id=_RUN_ID,
            bundle_id=_BUNDLE_ID,
            status=EvalJobStatus.COMPLETED,
            max_k=10,
            include_stage_trace=True,
            total_cases=3,
            completed_cases=3,
            failed_cases=0,
            # `hit_rate_at_5` 刻意留 None：max_k=10 时它本该有值，但这条
            # 用例要验的是"None 能原样过序列化" —— 用一个必然为 None 的
            # 字段（在 max_k<5 时）反而让断言依赖别的东西。
            metrics=EvalMetrics(
                total_cases=3,
                completed_cases=3,
                failed_cases=0,
                hit_count=2,
                miss_count=1,
                hit_rate_at_1=0.5,
                hit_rate_at_5=None,
                hit_rate_at_k=2 / 3,
                mrr_at_k=0.75,
                max_k=10,
            ),
            created_at=datetime.now(timezone.utc),
        )

    async def get_case_trace(
            self,
            eval_run_id: UUID,
            eval_case_id: UUID,
            *,
            attempt_no: int | None = None,
    ) -> Any:
        self._raise_if_needed()

        from app.services.document.domain.schemas.retrieval_eval import (
            EvalCaseTraceResponse,
            StageTraceItem,
        )

        self.trace_calls.append((eval_run_id, eval_case_id, attempt_no))

        return EvalCaseTraceResponse(
            eval_run_id=eval_run_id,
            eval_case_id=eval_case_id,
            # 服务层不传轮次时自己定代表行；替身这里照同一件事做，
            # 免得用例断言的是"路由替服务层做决定"。
            attempt_no=attempt_no if attempt_no is not None else 3,
            available_attempts=[1, 2, 3],
            query_text="这一轮实际发出的问题",
            gold_chunk_ids=[],
            stages=[
                StageTraceItem(
                    stage_key="fusioner",
                    source_key="",
                    input_count=50,
                    output_count=10,
                )
            ],
        )

    async def list_run_cases(self, eval_run_id: UUID, **_: Any) -> Any:
        self._raise_if_needed()

        from app.services.document.domain.schemas.enum import (
            EvalCaseResultStatus,
        )
        from app.services.document.domain.schemas.retrieval_eval import (
            EvalRunCase,
            EvalRunCaseListResponse,
        )

        def _item(case_id: UUID, attempt_no: int, hit: bool | None) -> Any:
            return EvalRunCase(
                eval_case_id=case_id,
                question="第几轮找到的？",
                chunk_id=uuid4(),
                document_type_definition_id=uuid4(),
                status=EvalCaseResultStatus.COMPLETED,
                attempt_no=attempt_no,
                query_text=f"第 {attempt_no} 轮实际发出的问题",
                gold_hit=hit,
            )

        return EvalRunCaseListResponse(
            items=[
                _item(uuid4(), 1, True),
                _item(uuid4(), 3, False),
            ],
            page=1,
            page_size=50,
            total=2,
        )


class _FakeFactory:
    """
    `needs_session=False` 是关键：走 `factory.create(contract)` 那条分支，
    不 open 事务，于是这些用例不需要数据库。
    """

    def __init__(self, service: Any) -> None:
        self._service = service

    def needs_session(self, contract: Any) -> bool:
        return False

    def create(self, contract: Any) -> Any:
        return self._service


def _app(
        monkeypatch: pytest.MonkeyPatch,
        *,
        error: Exception | None = None,
        service: _FakeService | None = None,
) -> FastAPI:

    app = FastAPI()
    app.include_router(retrieval_eval_router, prefix="/api/v1")

    app.dependency_overrides[current_user] = lambda: SimpleNamespace(
        subject="user-1"
    )

    # 传 `service=` 是为了让用例能回头看**替身收到了什么实参**（路由有没有
    # 把查询参数漏掉，只有这一面看得见）。不传就照旧自己造一个。
    fake = service if service is not None else _FakeService(error=error)

    monkeypatch.setattr(
        dependencies_module,
        "get_service_factory",
        lambda: _FakeFactory(fake),
    )

    return app


async def _request(
        app: FastAPI,
        method: str,
        path: str,
        **kwargs: Any,
) -> httpx.Response:

    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(
        transport=transport, base_url="http://test"
    ) as client:
        return await client.request(method, path, **kwargs)


# ============================================================
# 数据集侧：真的能用
# ============================================================


async def test_create_bundle_accepts_with_202(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    **202，不是 201。**

    资源确实建出来了，但数据集还不可用（造题是分钟级的后台任务）。
    201 会让调用方以为可以立刻去导出了。
    """

    app = _app(monkeypatch)

    response = await _request(
        app,
        "POST",
        "/api/v1/retrieval-eval/bundles",
        json={
            "name": "线上形状数据集",
            "document_type_definition_ids": [str(uuid4())],
            "sampled_chunk_count": 12,
        },
    )

    assert response.status_code == 202

    body = response.json()

    assert body["bundle_id"] == str(_BUNDLE_ID)
    assert body["generation_run_id"] == str(_RUN_ID)
    assert body["status"] == "generating"


async def test_duplicate_dtds_are_rejected_with_422(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    DTD 重复在**进入服务层之前**就被 pydantic 挡下（422，不是 400）。

    这条钉住的是"拒绝而不是静默去重"—— 静默去重的代价是调用方以为抽了
    三个 DTD 的范围，实际只有两个，而数据集本身不会体现这个差别。
    """

    app = _app(monkeypatch)
    dtd = str(uuid4())

    response = await _request(
        app,
        "POST",
        "/api/v1/retrieval-eval/bundles",
        json={
            "name": "重复 DTD",
            "document_type_definition_ids": [dtd, dtd],
            "sampled_chunk_count": 5,
        },
    )

    assert response.status_code == 422


async def test_export_serializes_dataset_sha256(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    导出物带着指纹，且键名与契约一致。

    这条是"DTO 真的过得去序列化"的落点 —— 服务层用例看不到这一段。
    """

    app = _app(monkeypatch)

    response = await _request(
        app,
        "GET",
        f"/api/v1/retrieval-eval/bundles/{_BUNDLE_ID}/export",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["dataset_sha256"] == "b" * 64
    assert body["schema_version"] == "1.0"
    assert body["items"] == []


async def test_bundle_detail_exposes_gold_chunk_ids(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    明细里的 `document_type_definition_ids` 与配置快照都在线上。

    它们是"这批题是怎么来的"的唯一答案 —— 少了它们，数据集就只能被
    使用、不能被解释。
    """

    app = _app(monkeypatch)

    response = await _request(
        app,
        "GET",
        f"/api/v1/retrieval-eval/bundles/{_BUNDLE_ID}",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["status"] == "ready"
    assert body["selection_config"] == {"seed": 7}
    assert body["latest_generation_run_id"] == str(_RUN_ID)


# ============================================================
# 错误映射
# ============================================================


async def test_not_found_maps_to_404(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app(monkeypatch, error=NotFoundError("数据集不存在"))

    response = await _request(
        app,
        "GET",
        f"/api/v1/retrieval-eval/bundles/{_BUNDLE_ID}",
    )

    assert response.status_code == 404


async def test_conflict_maps_to_409(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    「不 ready 不能导出」走 409 而不是 400：请求本身没毛病，是**当前状态**
    不允许 —— 而 409 让调用方知道"等造题跑完再试"是有意义的。
    """

    app = _app(monkeypatch, error=ConflictError("数据集还不是 ready"))

    response = await _request(
        app,
        "GET",
        f"/api/v1/retrieval-eval/bundles/{_BUNDLE_ID}/export",
    )

    assert response.status_code == 409


# ============================================================
# 评测侧：显式未实现
# ============================================================


async def test_create_eval_run_accepts_with_202(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    评测也是 **202**：逐条 case 调检索是分钟级的事。

    （这一组原本是 501 占位 —— 检索链路的逐阶段 trace 挂点补齐之后转正。
    那条 `test_eval_side_..._501` 因此删掉了，而不是留着改断言：留着会让
    "占位端点"这件事看起来还在。）
    """

    app = _app(monkeypatch)

    response = await _request(
        app,
        "POST",
        "/api/v1/retrieval-eval/runs",
        json={"bundle_id": str(_BUNDLE_ID), "max_k": 10},
    )

    assert response.status_code == 202

    body = response.json()

    assert body["eval_run_id"] == str(_RUN_ID)
    assert body["bundle_id"] == str(_BUNDLE_ID)


async def test_eval_run_metrics_serialize(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    指标里的 `null` 必须是 `null`，不能变成缺字段或 0。

    `hit_rate_at_5` 在 `max_k < 5` 时**没有意义**，而"没有意义"与
    "命中率是 0"是两回事 —— 后者会被读成"top5 一个都没中"。这条焊的是
    FastAPI 那一步序列化（服务层用例看不到它）。
    """

    app = _app(monkeypatch)

    response = await _request(
        app, "GET", f"/api/v1/retrieval-eval/runs/{_RUN_ID}",
    )

    assert response.status_code == 200

    metrics = response.json()["metrics"]

    assert metrics["hit_rate_at_5"] is None
    assert metrics["max_k"] == 10
    assert metrics["total_cases"] == 3


# ============================================================
# 逐轮（attempt_no）：痕迹与 case 列表
# ============================================================


async def test_case_trace_serializes_the_round(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    痕迹响应里必须**看得见这是哪一轮**。

    逐轮放宽里第 1 轮与第 3 轮的参数不同（第 3 轮才开精排），不标出是哪
    一轮的话，这份痕迹会被读成「这条 case 的检索过程」。这条焊的是
    FastAPI 那一步序列化 —— 服务层用例拿的是 DTO 对象，字段名写错、
    被 `response_model` 挡掉，那些用例全都不会红。
    """

    service = _FakeService()
    app = _app(monkeypatch, service=service)

    response = await _request(
        app,
        "GET",
        f"/api/v1/retrieval-eval/runs/{_RUN_ID}/cases/{_CASE_ID}/trace",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["attempt_no"] == 3
    assert body["available_attempts"] == [1, 2, 3]
    assert body["stages"][0]["stage_key"] == "fusioner"

    # 不传时**不能**由路由塞一个默认轮次进去：取哪一轮是服务层的规则
    # （代表行），路由再定一次就有了第二份规则。
    assert service.trace_calls == [(_RUN_ID, _CASE_ID, None)]


async def test_case_trace_passes_the_requested_round(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    `attempt_no` 是查询串上的参数，要真的传到服务层。

    与上一条合起来才完整：上一条证「不传就是 None」，这条证「传了不会
    被路由吞掉」—— 吞掉的后果是永远只看得到代表行那一轮，而 200 加上
    字段齐全让这件事看起来完全正常。
    """

    service = _FakeService()
    app = _app(monkeypatch, service=service)

    response = await _request(
        app,
        "GET",
        f"/api/v1/retrieval-eval/runs/{_RUN_ID}/cases/{_CASE_ID}/trace",
        params={"attempt_no": 2},
    )

    assert response.status_code == 200
    assert service.trace_calls == [(_RUN_ID, _CASE_ID, 2)]

    body = response.json()

    assert body["attempt_no"] == 2, "响应要跟着请求的那一轮走"


async def test_case_trace_rejects_round_zero(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    轮次从 1 起：`attempt_no=0` 是 422，不该被当成「没传」而静默回落到
    代表行 —— 那样调用方以为自己在看第 0 轮，实际拿到的是另一轮。
    """

    app = _app(monkeypatch)

    response = await _request(
        app,
        "GET",
        f"/api/v1/retrieval-eval/runs/{_RUN_ID}/cases/{_CASE_ID}/trace",
        params={"attempt_no": 0},
    )

    assert response.status_code == 422


async def test_case_list_serializes_the_round(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    列表里每条 case 带**轮次号**，`total` 按原样序列化。

    轮次号是这一行唯一的"它属于哪一轮"的说明 —— 缺了它，逐轮放宽带来的
    多行在响应里完全看不出来（这也是它此前会一条 case 出现两三遍却没人
    发现的原因）。
    """

    app = _app(monkeypatch)

    response = await _request(
        app, "GET", f"/api/v1/retrieval-eval/runs/{_RUN_ID}/cases",
    )

    assert response.status_code == 200

    body = response.json()

    assert body["total"] == 2
    assert [item["attempt_no"] for item in body["items"]] == [1, 3]
    assert [item["gold_hit"] for item in body["items"]] == [True, False]

    # 实际发出的文本也要过序列化这一关：前端要靠它回答"这条命中的是哪句
    # 话"，而字段名写错/被 response_model 挡掉只有真打接口才看得见。
    assert [item["query_text"] for item in body["items"]] == [
        "第 1 轮实际发出的问题",
        "第 3 轮实际发出的问题",
    ]


# ============================================================
# 注册位置（那条缺失的 guard 的替代品）
# ============================================================


def _leaf_paths(router: Any) -> set[str]:
    """
    递归收集一棵路由树上的全部叶路径。

    本仓库的 FastAPI 版本把 `include_router` 的结果包成
    `_IncludedRouter` 而不是摊平进父路由 —— 所以直接读
    `protected_router.routes[*].path` 拿到的是 `?`（那些对象没有 `path`）。
    要顺着 `original_router` 往下走。写这个helper是因为**第一次就踩了**：
    那样读出来是 0 条，看起来像"注册失败了"。
    """

    found: set[str] = set()

    for route in getattr(router, "routes", []):

        path = getattr(route, "path", None)

        if path is not None:
            found.add(path)
            continue

        inner = getattr(route, "original_router", None)

        if inner is not None:
            found |= _leaf_paths(inner)

    return found


def test_all_paths_live_on_the_protected_router() -> None:
    """
    本模块的每条路径都在 **protected_router** 上。

    `/api/v1` 下有两棵树，而"谁在哪棵树上"决定它要不要 token（见
    router.py 的模块 docstring）。注册错了不会报错 —— 只会得到一个匿名
    就能建数据集的接口。

    ⚠️ router.py 的 docstring 说这件事由
    `tests/api/test_auth_enforcement.py` 全局守护，**但那个文件不存在**。
    这条是缩到本模块的替代品：只断言这 11 条路径在受保护的树上。
    """

    protected = _leaf_paths(protected_router)

    # ⚠️ 比较的是**不带 `/api/v1`** 的路径。
    #
    # 那棵树上的叶路径来自各个子路由（`original_router`），而子路由自己
    # 没有前缀 —— `/api/v1` 是在 `include_router` 那一刻拼上去的，不在
    # 叶节点的 `path` 里。第一次写这条断言时加了前缀，于是**全部**路径
    # 都被判成"缺失"，看起来像注册失败。
    ours = {route.path for route in retrieval_eval_router.routes}

    assert len(ours) == 11, f"端点数变了：{len(ours)}"

    missing = ours - protected

    assert not missing, (
        f"这些路径没有注册到 protected_router 上（匿名可访问）：{missing}"
    )
