"""
评测执行冒烟：**跑一轮真的评测**（真库 + 真索引后端），打印指标与宏观丢因。

跑法：
    .venv/Scripts/python.exe -m app.services.document.retrieval_eval_evaluate_smoke

为什么必须走完整 lifespan
------------------------
评测执行里每一道题都要**真调一次检索**（`DocumentRetrieveService`），
而那个服务需要索引后端与 pipeline registry —— 后者由 lifespan 装配。
用假检索服务测等于把"评测到底测出什么"这件事整个跳过去。

它验的是那条最容易出错的缝
--------------------------
    纯函数（命中判定 / 漏斗归因 / 指标）  <- 已有单元用例
    这条缝：把 response.trace 映射成两张表的行、把结果落库、把指标算出来
    路由与 DTO                          <- 已有 HTTP 用例

这条缝此前从未跑过：**它要么一次跑通，要么在"真实检索返回的 trace 形状"
与"我以为的形状"之间暴露差异** —— 而那种差异在构造的输入里永远看不到。

⚠️ 它**自己造数据、自己删**：不碰库里已有的 bundle / case。探针数据集在
结尾按 bundle_id 删掉（级联带走 case 与结果行）。
"""

from __future__ import annotations

import asyncio
import uuid
from pathlib import Path

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

__all__ = ["main"]


def _dsn() -> str:
    import yaml

    raw = yaml.safe_load(
        (
            Path(__file__).resolve().parents[2]
            / "config" / "datasource" / "postgre.yaml"
        ).read_text(encoding="utf-8")
    )

    return raw["config"]["dsn"]


async def _pick_probe_chunk(session) -> dict:
    """
    从真语料里挑一块做探针，并**用它的正文拼一个问题**。

    用正文里真实出现的词做查询，是为了让"命中"成为**可达**的 —— 如果
    随手编一句无关的话，跑出来必然全 miss，而那种"跑通了"看不出任何
    东西（它只证明流程没崩，不证明它能把命中的判成命中）。
    """

    row = (
        await session.execute(
            text(
                """
                SELECT c.document_chunk_id, c.document_version_id,
                       c.content, c.content_hash,
                       d.document_type_definition_id, d.document_id,
                       d.title, d.document_key
                FROM document_chunk c
                JOIN document_version v
                  ON v.document_version_id = c.document_version_id
                JOIN document d ON d.document_id = v.document_id
                WHERE c.chunker_key = 'TEXT_TXT_RECURSIVE_CHUNKER'
                  AND length(c.content) > 200
                ORDER BY c.document_chunk_id
                LIMIT 1
                """
            )
        )
    ).one()

    content = row[2]

    return {
        "chunk_id": row[0],
        "document_version_id": row[1],
        "content": content,
        "content_hash": row[3],
        "dtd_id": row[4],
        "document_id": row[5],
        "title": row[6],
        "document_key": row[7],
        "question": _question_from(content),
    }


def _question_from(content: str) -> str:
    """
    从正文里挑一句**像人话**的做查询词。

    ⚠️ 不能直接取正文头 N 个字：实测这个语料的正文头是 markdown 标题与
    日期，取出来是这个样子 ——

        '2012-06-09：实施方案  #### 2011-07-07 ｜ 分红送转'

    那不是一句人话，检索几乎必然 miss。而**一次全 miss 的"跑通"什么都
    证明不了**：它只说明流程没崩，不说明它能把命中的判成命中。

    所以跳掉标题、引用、列表、表格这些行，取第一段连续散文。
    """

    for line in content.split("\n"):

        line = line.strip()

        if len(line) < 20:
            continue

        if line.startswith(("#", ">", "|", "-", "*", "```")):
            continue

        return line[:60]

    # 整块都是结构化行（表格类）。退回取长度合适的一段，总比空着好。
    return content[:60].replace("\n", " ").strip()


async def main() -> int:

    from app.main import app

    print("启动 lifespan ……", flush=True)

    async with app.router.lifespan_context(app):

        from app.common.factories import get_service_factory
        from app.common.models import Base
        from app.services.document.domain.entities import (
            RetrievalEvalBundle,
            RetrievalEvalCase,
            RetrievalEvalCaseGoldChunk,
            RetrievalEvalRun,
        )
        from app.services.document.domain.schemas.enum import (
            EvalBundleStatus,
            EvalGoldValidationMethod,
            EvalJobStatus,
        )
        from app.services.document.domain.tasks import (
            RetrievalEvalEvaluateParams,
        )
        from app.services.document.application.retrieval_eval_service import (
            RetrievalEvalService,
        )

        engine = create_async_engine(_dsn())
        maker = async_sessionmaker(engine, expire_on_commit=False)

        bundle_id = uuid.uuid4()
        run_id = uuid.uuid4()

        try:

            # ---- 造探针数据集（直接写库，不走出题那一步：那要钱） ----
            async with maker() as session:

                probe = await _pick_probe_chunk(session)

                print(f"探针 chunk: {probe['title']}（{probe['document_key']}）", flush=True)
                print(f"查询词    : {probe['question'][:50]!r}", flush=True)

                # 用**表的 Core 接口**建行，而不是 ORM 模型类。
                #
                # 理由：这里要造的是一份"已经 ready、有一道题"的探针数据集，
                # 而走服务层建它就得真去出题（要钱）。直接插行是探针该做的
                # 事 —— 但列名要跟着表走，所以用 `Base.metadata.tables[...]`
                # 而不是手写 INSERT SQL（表改名时手写的那份会静默指到别处）。
                bundle_model = Base.metadata.tables[
                    "retrieval_eval_bundle"
                ]

                await session.execute(
                    bundle_model.insert().values(
                        bundle_id=bundle_id,
                        name="评测执行冒烟（探针）",
                        status=EvalBundleStatus.READY.value,
                        schema_version="1.0",
                        selection_config={
                            "document_type_definition_ids": [
                                str(probe["dtd_id"])
                            ],
                        },
                        generation_config={"questions_per_chunk": 1},
                        source_chunk_count=1,
                        case_count=1,
                        dataset_sha256="0" * 64,
                    )
                )

                case_model = Base.metadata.tables["retrieval_eval_case"]
                case_id = uuid.uuid4()

                await session.execute(
                    case_model.insert().values(
                        eval_case_id=case_id,
                        bundle_id=bundle_id,
                        document_type_definition_id=probe["dtd_id"],
                        document_version_id=probe["document_version_id"],
                        document_id=probe["document_id"],
                        source_document_title=probe["title"],
                        source_document_key=probe["document_key"],
                        source_chunk_id=probe["chunk_id"],
                        source_content_hash=probe["content_hash"] or "",
                        question=probe["question"],
                        answer_original=probe["content"][:30],
                        answer_start_offset=0,
                        answer_end_offset=30,
                        question_order=0,
                        question_sha256=uuid.uuid4().hex * 2,
                        validation_status="validated",
                    )
                )

                gold_model = Base.metadata.tables[
                    "retrieval_eval_case_gold_chunk"
                ]

                await session.execute(
                    gold_model.insert().values(
                        eval_case_gold_chunk_id=uuid.uuid4(),
                        eval_case_id=case_id,
                        chunk_id=probe["chunk_id"],
                        is_source_chunk=True,
                        validation_method=(
                            EvalGoldValidationMethod.SOURCE_CHUNK.value
                        ),
                        source_content_hash=probe["content_hash"],
                        judgment={},
                    )
                )

                await session.execute(
                    Base.metadata.tables["retrieval_eval_run"].insert().values(
                        eval_run_id=run_id,
                        bundle_id=bundle_id,
                        status=EvalJobStatus.QUEUED.value,
                        max_k=10,
                        include_stage_trace=True,
                        total_cases=1,
                    )
                )

                await session.commit()

            print("探针数据已写入，开始评测 ……", flush=True)

            # ---- 跑一轮（直接调 handler 主体，不经过任务队列） ----
            #
            # 不调 `create_eval_run`：那会提交一条真任务，而 worker 也会
            # 捡起来跑一遍 —— 两遍并发写同一批结果行，谁赢不好说。
            # 任务投递那一段由检索评测的装配冒烟与路由用例各自验过。
            factory = get_service_factory()

            async with factory.scope(RetrievalEvalService) as service:

                await service.evaluate(
                    RetrievalEvalEvaluateParams(eval_run_id=run_id),
                    None,
                )

                detail = await service.get_eval_run(run_id)
                stats = await service.get_stage_stats(run_id)
                cases = await service.list_run_cases(run_id)
                trace = await service.get_case_trace(run_id, case_id)

            print()
            print("=== 轮次 ===")
            print(f"  状态      : {detail.status.value}")
            print(f"  跑通/失败 : {detail.completed_cases}/{detail.failed_cases}")
            print(f"  指标      : hit@1={detail.metrics.hit_rate_at_1} "
                  f"hit@10={detail.metrics.hit_rate_at_k} "
                  f"MRR={detail.metrics.mrr_at_k} "
                  f"p50={detail.metrics.p50_latency_ms}ms")

            print()
            print("=== 单条结果 ===")
            for item in cases.items:
                print(f"  gold_hit={item.gold_hit} gold_rank={item.gold_rank} "
                      f"latency={item.latency_ms}ms")
                print(f"  top-5: {[str(c.chunk_id)[:8] for c in item.top_candidates]}")

            print()
            print("=== 宏观丢因（每段进出与金标）===")
            for stage in stats.stages:
                # ⚠️ 字段名取自 `StageStatItem`（**统计**视图），不是
                # `RetrieveStageTrace`（单次痕迹视图）—— 两者都有
                # input/output 的概念，但一个叫 average_input_count（多
                # 条 case 的平均），一个叫 input_count（这一次的）。
                # 用错会 AttributeError，而它只在打印那一行才炸。
                print(
                    f"  {stage.stage_key:24s} src={stage.source_key:14s} "
                    f"avg {stage.average_input_count}->"
                    f"{stage.average_output_count} "
                    f"before={stage.gold_present_before_count} "
                    f"after={stage.gold_present_after_count} "
                    f"drop={stage.gold_drop_count} "
                    f"pass={stage.gold_pass_rate} "
                    f"{stage.drop_reason_counts}"
                )

            print()
            print("=== 逐阶段痕迹 ===")
            for stage in trace.stages:
                rows = [
                    f"{str(c.chunk_id)[:8]}#{c.rank}"
                    f"{'' if c.selected else '✗' + str(c.drop_reason)}"
                    for c in stage.top_candidates[:6]
                ]
                print(f"  {stage.stage_key:24s} {rows}")

        finally:

            # ---- 清场 ----
            #
            # ⚠️ 两件事都比看起来重要：
            #
            # 一、**删除顺序不能反。** 三张表对外都是 `ON DELETE RESTRICT`
            #     （数据集是历史断言，不该被顺手删掉 —— 那是有意的设计）。
            #     直接从 bundle 删会撞 RESTRICT，报的是
            #     `RestrictViolationError ... retrieval_eval_run_bundle_id_fkey`
            #     —— 看起来像"删不掉"，其实是"顺序错了"。
            #     所以：run（级联带走 case_result → 两张阶段表）
            #           -> case（级联带走 gold_chunk）
            #           -> bundle（级联带走 generation_*）
            #
            # 二、**清场失败绝不能盖住真正的异常。** 上一版把这几条直接写在
            #     `finally` 里，于是"评测本身炸了"被"清场也炸了"顶替掉，
            #     真正的 traceback 一条都看不见 —— 白跑一轮。
            #     这里吞掉清场异常并打出来，让原始失败照常往上抛。
            try:

                async with maker() as session:

                    for table in (
                        "retrieval_eval_run",
                        "retrieval_eval_case",
                        "retrieval_eval_bundle",
                    ):
                        model = Base.metadata.tables[table]
                        await session.execute(
                            delete(model).where(
                                model.c.bundle_id == bundle_id
                            )
                        )

                    await session.commit()

                print("探针数据已清除。")

            except Exception as exc:  # noqa: BLE001 —— 见上面第二条
                print(
                    f"⚠️ 探针数据没清干净（bundle={bundle_id}）："
                    f"{type(exc).__name__}: {exc}"
                )

            await engine.dispose()

    print("lifespan 已干净退出。")
    return 0


if __name__ == "__main__":

    # ⚠️ **必须传 `loop_factory`** —— Windows 上不能光用 `asyncio.run(main())`。
    #
    # `asyncio.run` 默认在 Windows 上建 **ProactorEventLoop**，而 psycopg
    # （LangGraph checkpointer 用的那个）跑不了它。app.main 的 lifespan 会
    # 在连库之前显式拦下，报：
    #
    #     RuntimeError: The running event loop is ProactorEventLoop, but
    #     psycopg ... cannot run on it.
    #
    # 所以这里走 `app/serve.py` 那份 `loop_factory` —— 与
    # `python run_server.py` 完全同一条路径。**不传它的话，任何"跑一遍
    # lifespan"的脚本都会在启动那一刻就退出**，而它看起来像"应用起不来"。
    from app.serve import loop_factory

    raise SystemExit(
        asyncio.run(main(), loop_factory=loop_factory())
    )
