"""
检索延迟探针：**同一个查询反复打，看它是"特殊"还是"偶发"**。

跑法：
    .venv/Scripts/python.exe -m app.services.document.retrieve_latency_probe

要回答的问题
------------
线上有一条评测 case 的检索超过 60 秒没返回：

    ST康佳A于2022-12-06接待的机构调研，参与对象和参与方式分别是什么？

而同批另外 39 条都是 0.3–0.5 秒。这两种解释会导向完全不同的修法：

    **这个查询特殊**      某个词/某种长度让 ES 或 Milvus 走了慢路径
                          -> 要查查询形状、索引映射
    **偶发**              某一跳卡住（compaction / segment merge /
                          连接重连 / GC），恰好撞在这条上
                          -> 要查的是超时与降级，不是查询

所以本探针做两件事：

    同一查询 **跑 N 次**    每次都慢 = 特殊；时快时慢 = 偶发
    另一条**同类且成功**的查询做对照   排除"这一类问题都慢"

为什么必须连真后端
------------------
`app/agent/retrieve_smoke.py` 用的是内存假实现（它自己写着"不连 ES"）——
假实现里没有 compaction、没有连接池、没有段合并，**恰好是这条查询可能
卡住的那几层**。

⚠️ 只读，不写任何数据。
"""

from __future__ import annotations

import asyncio
import sys
import time

__all__ = ["main"]

#: 线上卡住的那条。
SLOW_QUERY = (
    "ST康佳A于2022-12-06接待的机构调研，参与对象和参与方式分别是什么？"
)

#: 对照：同批里跑成功过的同类问题。
CONTROL_QUERY = "合百集团2018年1月16日披露的机构调研事项中，参与对象是谁？"

#: 那条 case 所属的 DTD（COMPANY_EVENTS）。两个查询都在它上面。
DTD_ID = "28f35eaa-e276-42f5-a6ab-ee0c004a16d1"

#: 每条查询跑几次。
ROUNDS = 5


async def _one(
        retriever: object,
        query: str,
        *,
        route_timeout_ms: int | None,
) -> dict:
    from uuid import UUID

    from app.services.document.domain.schemas.enum import RetrievalType
    from app.services.document.domain.schemas.retrieve import (
        DocumentRetrieveRequest,
        RetrievalRecallConfig,
    )

    request = DocumentRetrieveRequest(
        document_type_definition_ids=[UUID(DTD_ID)],
        input_text=query,
        only_active=True,
        include_stage_trace=True,
        retrieval_config=(
            {
                RetrievalType.DENSE_VECTOR: RetrievalRecallConfig(
                    timeout_ms=route_timeout_ms
                ),
                RetrievalType.KEYWORD: RetrievalRecallConfig(
                    timeout_ms=route_timeout_ms
                ),
            }
            if route_timeout_ms
            else None
        ),
    )

    started = time.monotonic()

    try:
        response = await retriever.retrieve_documents(request)
        elapsed = time.monotonic() - started

    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "elapsed": time.monotonic() - started,
            "error": f"{type(exc).__name__}: {exc}",
        }

    stages = {
        stage.stage_key: stage.input_count
        for stage in (response.trace.stages if response.trace else [])
    }

    return {
        "ok": True,
        "elapsed": elapsed,
        "results": len(response.results),
        # 哪几路真的返回了 —— **挂住的那一路不会出现在这里**，
        # 这是"哪一路卡住"唯一的线索。
        "recall_routes": {
            key: value
            for key, value in stages.items()
            if key.startswith("recall")
        },
        "degraded": [
            f"{item.retrieval_type}: {item.reason[:60]}"
            for item in response.degraded_routes
        ],
    }


async def main() -> int:

    from app.main import app

    print("启动 lifespan（reranker 与云端模型预热会花几分钟）……", flush=True)

    async with app.router.lifespan_context(app):

        from app.common.factories import get_service_factory
        from app.services.document.application.document_retrieve_service import (
            DocumentRetrieveService,
        )

        factory = get_service_factory()

        async def run(label: str, query: str, *, timeout_ms: int | None) -> None:

            print(f"\n=== {label}（逐路超时 {timeout_ms}）===", flush=True)
            print(f"    {query[:50]}", flush=True)

            for index in range(1, ROUNDS + 1):

                outcome = await _one(
                    retriever, query, route_timeout_ms=timeout_ms,
                )

                if not outcome["ok"]:
                    print(
                        f"    #{index} 失败 {outcome['elapsed']:.1f}s "
                        f"{outcome['error'][:70]}",
                        flush=True,
                    )
                    continue

                print(
                    f"    #{index} {outcome['elapsed']:.2f}s "
                    f"结果 {outcome['results']} 条  "
                    f"返回的路 {outcome['recall_routes']}"
                    + (
                        f"  降级 {outcome['degraded']}"
                        if outcome["degraded"]
                        else ""
                    ),
                    flush=True,
                )

        # 一个 scope 复用（只读，不写）
        async with factory.scope(DocumentRetrieveService) as retriever:

            # 先不加逐路超时：看它**本来**会等多久
            await run("线上卡住的那条（无逐路超时）",
                      SLOW_QUERY, timeout_ms=None)

            await run("对照：同类但成功过的那条（无逐路超时）",
                      CONTROL_QUERY, timeout_ms=None)

            # 再加逐路超时：看是**哪一路**降级
            await run("线上卡住的那条（逐路 5 秒）",
                      SLOW_QUERY, timeout_ms=5000)

    print("\nlifespan 已干净退出。")
    return 0


if __name__ == "__main__":

    from app.serve import loop_factory

    raise SystemExit(
        asyncio.run(main(), loop_factory=loop_factory())
    )
