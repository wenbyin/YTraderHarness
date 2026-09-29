"""
通用图运行器。

本类刻意不知道任何业务细节：它拿一个 GraphDefinition（已编译的图
+ 节点目录 + context 工厂）和一个 GraphRunRecorder，负责

    1. 建 run 行（早于图开始运行）
    2. 置 running
    3. 挂 callback 兜底
    4. ainvoke / astream
    5. 汇总节点行，推导终态并写回

新增 pipeline 不需要新的 runner —— 这正是把本类放在 infra 而不是
某个业务模块下的原因。

事务边界
--------
runner **绝不在图运行期间持有事务**：

    Step 1  开短事务插 run 行 + 置 processing，提交关闭
    Step 2  ainvoke（此时没有任何业务锁在手上）
    Step 3  另开短事务写终态

因为 Step 1 的行已经提交，**失败路径永远写得进去** ——
这是"部分失败不丢观测"的实现方式。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import UUID

from .callbacks import PersistingCallbackHandler
from .recorder import (
    GraphRunRecorder,
    RunOutcome,
    RunSnapshot,
    RunStatus,
    RunSummary,
)
from .registry import (
    GraphDefinition,
    RunContext,
)

logger = logging.getLogger(__name__)

__all__ = ["GraphRunner", "RunRequest"]


@dataclass(frozen=True, slots=True)
class RunRequest:
    """一次运行的输入。"""

    definition: GraphDefinition

    pipeline_run_id: UUID
    thread_id: str

    #: 初始 state。续跑时传 None，由 checkpoint 恢复。
    input_state: dict[str, Any] | None

    #: 运行记录的其余字段（subject / trigger / config / 拓扑）。
    #: 由业务侧填好，runner 只负责落库。
    run_snapshot: RunSnapshot

    #: True 表示用同一个 thread_id 从 checkpoint 续跑。
    resume: bool = False

    #: 本次运行覆盖 runner 默认的 durability。
    #:
    #:     "sync"  每个 super-step 落一次 checkpoint（可续跑）
    #:     "exit"  只在结束时落一次（写入最少，不可续跑）
    #:
    #: None 表示用 GraphRunner 构造时给的默认值。
    #: 必须在运行**开始前**定 —— run 级启动的运行中途崩溃后
    #: 没有中间 checkpoint 可恢复，事后改也救不回来。
    durability: str | None = None


class GraphRunner:
    """
    跑一个图，并把过程写成可观测数据。

    durability="sync"：
        每个 super-step 落一次 checkpoint，进程被杀后才能真正续跑。
        因为 state 里只有 ArtifactRef 没有正文，这个 I/O 代价是可控的。
    """

    def __init__(
            self,
            *,
            recorder: GraphRunRecorder,
            durability: str = "sync",
    ) -> None:
        self._recorder = recorder
        self._durability = durability

    # ========================================================
    # Entry point
    # ========================================================

    async def run(
            self,
            request: RunRequest,
    ) -> RunOutcome:

        definition = request.definition
        run_id = request.pipeline_run_id

        # ----------------------------------------------------
        # 1. 建 run 行。必须在图开始运行之前提交，
        #    否则图跑挂了就没有任何记录可写。
        # ----------------------------------------------------

        await self._recorder.create_run(
            request.run_snapshot
        )

        started_at = datetime.now(timezone.utc)

        await self._recorder.mark_run_running(
            run_id, started_at=started_at
        )

        # ----------------------------------------------------
        # 2. 组装 context 与 callback
        # ----------------------------------------------------

        run_context = RunContext(
            pipeline_run_id=run_id,
            thread_id=request.thread_id,
            trigger_source=(
                request.run_snapshot.trigger_source
            ),
            recorder=self._recorder,
        )

        context = definition.context_factory(run_context)

        handler = PersistingCallbackHandler(
            pipeline_run_id=run_id,
            recorder=self._recorder,
            handled_node_ids=definition.callback_node_ids,
            node_labels={
                s.node_id: s.label
                for s in definition.node_specs
            },
            node_sequences={
                s.node_id: s.sequence
                for s in definition.node_specs
            },
        )

        config: dict[str, Any] = {
            "configurable": {
                "thread_id": request.thread_id,
            },
            # 让 LangGraph 的根 run_id 等于我们的 pipeline_run_id，
            # 观测行与 LangSmith trace 因此可以直接对上。
            "run_id": str(run_id),
            "callbacks": [handler],
        }

        # ----------------------------------------------------
        # 3. 跑图。这里不持有任何业务事务。
        # ----------------------------------------------------

        error: BaseException | None = None

        try:
            await definition.compiled.ainvoke(
                request.input_state,
                config,
                context=context,
                durability=(
                    request.durability or self._durability
                ),
            )

        except BaseException as exc:  # noqa: BLE001
            # 刻意捕获 BaseException 而不是 Exception：
            # asyncio.CancelledError 也要走终结路径，
            # 否则 run 行会永远停在 running。
            error = exc
            # 用 exception 而不是 warning：带 traceback 才能定位到
            # 具体是哪一行。节点的 error_message 只存了异常文本，
            # 复杂异常光看文本查不动。
            logger.exception(
                "Graph %s run %s failed: %s",
                definition.name,
                run_id,
                type(exc).__name__,
            )

        # ----------------------------------------------------
        # 4. 汇总真实落库的节点结果，推导终态
        # ----------------------------------------------------

        summary = await self._recorder.summarize(run_id)

        finished_at = datetime.now(timezone.utc)

        outcome = RunOutcome(
            status=_derive_status(error, summary),
            finished_at=finished_at,
            duration_ms=int(
                (finished_at - started_at).total_seconds()
                * 1000
            ),
            error_type=(
                None if error is None else type(error).__name__
            ),
            error_node_id=_attribute_node(
                error, summary
            ),
            error_message=(
                None
                if error is None
                else f"{type(error).__name__}: {error}"
            ),
            node_count=summary.node_count,
            failed_node_count=summary.failed_node_count,
        )

        await self._recorder.finish_run(run_id, outcome)

        return outcome


# ============================================================
# Status derivation
# ============================================================


def _derive_status(
        error: BaseException | None,
        summary: RunSummary,
) -> RunStatus:
    """
    从节点行的真实结果推导运行终态。

    为什么不能只看 error：

        扇出 3 路，1 路挂 —— 图**会**抛异常，但那不是 failed，
        而是 **partial**：文档在那 2 路索引里是真实可检索的。
        这个信息必须保留，否则前端只能笼统显示"失败"。

    为什么不能只看 summary：

        图可能在没有任何节点行的情况下就挂了（例如 context 工厂
        抛错），此时 summary 全 0，只能靠 error 判断。
    """

    if summary.node_count == 0:
        # 一个节点行都没有：图没跑起来，或第一个节点前就炸了。
        return RunStatus.FAILED

    if summary.failed_node_count == 0:
        if error is not None:
            # 节点都成功了，但 state 合并 / 收尾阶段抛错。
            return RunStatus.FAILED
        return RunStatus.COMPLETED

    if summary.completed_node_count == 0:
        # 有节点，但一个成功的都没有。
        return RunStatus.FAILED

    return RunStatus.PARTIAL


def _attribute_node(
        error: BaseException | None,
        summary: RunSummary,
) -> str | None:
    """
    把失败归因到某个节点。

    优先用实际落到库里的失败节点；节点行里查不到（例如图还没
    进入任何节点就挂了）时返回 None，由前端展示 run 级错误。
    """

    if error is None:
        return None

    if summary.failed_node_ids:
        return summary.failed_node_ids[0]

    return None
