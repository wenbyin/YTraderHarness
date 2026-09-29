"""
QA 测试数据集生成服务。

**手动触发**：人工调一次、看结果、再决定要不要继续。不挂在任何管线
步骤上，也不会被调度器自动拉起。

为什么这个服务比它看起来"重"
----------------------------
它一个人串起了四件事，每件都有各自的失败模式：

    找 clean 正文      该版本没跑过管线 / cleaning 节点没留 file_id
    切窗口 + 抽样      正文为空 / 窗口数不足
    调云端 LLM         超时 / 限流 / 返回的引用不是逐字原文
    与 chunk 求交      区间不落在任何 chunk 内

所以返回体里带了完整的统计与失败明细，而不是只给一个文件路径 ——
"生成了 3 条"和"生成了 3 条、另外 47 个窗口全失败了"是两回事。
"""

from __future__ import annotations

import logging
import random
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from app.common.exceptions import NotFoundError
from app.common.llm import get_chat_model
from app.common.utils import split_blocks, split_sections

from app.common.service.base import BaseService, ServiceContext

from ..const import QA_OUTPUT_DIR
from ..domain.schemas.qa_dataset import (
    QADatasetFailure,
    QADatasetGenerateRequest,
    QADatasetGenerateResponse,
)
from ..pipeline.observability.artifacts import ArtifactStore
from ..qa.generator import generate_dataset
from ..qa.overlap import ChunkRange
from ..qa.storage import write_dataset
from ..repository.document_chunk import DocumentChunkRepository
from ..repository.document_version import DocumentVersionRepository
from ..repository.pipeline_run import PipelineRunRepository

__all__ = [
    "QADatasetService",
    "QADatasetServiceImpl",
]

logger = logging.getLogger(__name__)

#: 一次取 chunk 的页大小。求交要该版本的**全部** chunk，而仓库接口
#: 是分页的（默认 limit=20）。取 500 是为了让常见文档（几百到一千多条）
#: 一两页就取完 —— 翻几十页纯属浪费往返。
_CHUNK_PAGE = 500

#: 响应里最多回传多少条失败明细。全量回传在"整批都挂了"时会让响应体
#: 膨胀到几百 KB，而调用方真正需要的是**分类计数**（在 drop_reasons 与
#: windows_failed 里），明细只是给排查用的样本。
_MAX_REPORTED_FAILURES = 20


class QADatasetService(BaseService):
    """数据集生成。"""

    @abstractmethod
    async def generate(
            self,
            request: QADatasetGenerateRequest,
    ) -> QADatasetGenerateResponse:
        ...


class QADatasetServiceImpl(QADatasetService):

    def __init__(
            self,
            *,
            version_repository: DocumentVersionRepository,
            chunk_repository: DocumentChunkRepository,
            run_repository: PipelineRunRepository,
            artifact_store: ArtifactStore,
            output_dir: str = QA_OUTPUT_DIR,
            ctx: ServiceContext | None = None,
    ) -> None:

        super().__init__(ctx)

        self._versions = version_repository
        self._chunks = chunk_repository
        self._runs = run_repository
        self._artifacts = artifact_store
        self._output_dir = output_dir

    async def generate(
            self,
            request: QADatasetGenerateRequest,
    ) -> QADatasetGenerateResponse:

        version_id = request.document_version_id

        # ---- 1. 版本存在吗 ----------------------------------
        #
        # 先查一次不是为了拿数据，是为了**早失败**：版本不存在时，
        # 后面每一步都会以各自的方式失败（查不到 run、读不到正文），
        # 而那些报错都指向次要原因，让人误以为问题在管线而不在 id。

        version = await self._versions.get(version_id)

        if version is None:
            raise NotFoundError(
                f"文档版本不存在：{version_id}"
            )

        # ---- 2. 取 clean 正文 --------------------------------

        clean_text = await self._load_clean_text(version_id)

        if not clean_text:
            raise ValueError(
                f"版本 {version_id} 读不到 clean 正文。"
                "该版本可能还没跑过写入管线，或 cleaning 节点没有留下"
                "产物引用 —— 没有正文就无法生成数据集。"
            )

        # ---- 3. 切窗口 + 抽样 --------------------------------

        # ---- 切事件窗口 --------------------------------
        #
        # 一个窗口 = 一条事件（`#### 2026-03-20 ｜ 股东户数` 那种）。
        #
        # 正文最开头那段（文档头）没有标题，**排除在出题池之外** ——
        # 它是文档元信息不是事件，从里面出的题（"这份档案的数据来源是
        # 什么"）对检索评测没有意义。但它会作为**背景**喂给模型，
        # 见下面的 document_context。
        all_sections = split_sections(clean_text, max_chars=None)
        blocks = [x for x in all_sections if x.title]

        if not blocks:
            raise ValueError(
                f"版本 {version_id} 的正文切不出任何事件窗口（正文长度为 "
                f"{len(clean_text)}）。"
            )

        # ---- 文档头：当**上下文**，不参与抽样 ---------------
        #
        # 正文最开头那段（第一个标题之前）是 `# 东方财富（300059.SZ）
        # 大事提醒 · 语义化事件档案` 这类整体信息，含**公司名与证券代码**。
        #
        # 它必须单独喂给模型：事件正文里只有事件本身、不含公司名，
        # 模型无从知道这份档案是谁的，只能写"公司"这种占位词 —— 那样
        # 产出的问题没有主体，检索时会把各家公司的同类事件全捞回来
        # （实测就是这样）。
        #
        # 块模式下它已经包含在第一块里，但**只有第一块有**；其余块
        # 看不到，所以仍然要单独摘出来，作为所有块共用的背景。
        head_sections = split_sections(clean_text, max_chars=None)
        head = (
            head_sections[0]
            if head_sections and not head_sections[0].title
            else None
        )
        document_context = head.text if head is not None else ""

        chosen = _sample(
            blocks, request.max_windows, request.sample_seed,
        )

        # ---- 4. 取该版本全部 chunk 区间 ----------------------

        chunk_ranges = await self._load_chunk_ranges(version_id)

        if not chunk_ranges:
            raise ValueError(
                f"版本 {version_id} 没有任何带字符偏移的 chunk。"
                "没有区间就无法把答案映射到目标 chunk —— 这正是本工具"
                "存在的意义，缺了它生成的数据集没有价值。"
            )

        # ---- 5. 生成 ----------------------------------------

        model = get_chat_model(
            request.provider,
            temperature=request.temperature,
            timeout=request.llm_timeout_seconds,
        )

        logger.info(
            "开始生成数据集：version=%s provider=%s 窗口 %d/%d 并发 %d",
            version_id,
            request.provider or "(default)",
            len(chosen),
            len(blocks),
            request.concurrency,
        )

        cases, stats = await generate_dataset(
            clean_text=clean_text,
            sections=chosen,
            chunks=chunk_ranges,
            model=model,
            document_version_id=version_id,
            document_context=document_context,
            max_questions_per_window=request.max_questions_per_window,
            concurrency=request.concurrency,
        )

        # ---- 5.5 全部失败就别伪装成成功 ----------------------
        #
        # 单块失败隔离是对的（一个 429 不该让整批陪葬），但**全部**失败
        # 是另一回事：那说明问题出在批次级（并发超限、key 失效、模型名
        # 写错），而不是某一块运气不好。
        #
        # 不拦的话端点会返回 200 + "生成了 0 条"，调用方以为跑通了、
        # 只是这批数据恰好没出题 —— 实际原因被埋在一堆 warning 里。
        # 实测踩过：账号并发上限是 1 而默认并发设了 4，每个块都 429。
        if stats.windows_total and stats.windows_ok == 0:
            reasons = "；".join(
                f.reason for f in stats.failures[:3]
            ) or "（无明细）"

            raise ValueError(
                f"全部 {stats.windows_total} 个窗口都失败了，一条样本都没有"
                f"产出。这通常不是单个块的问题，而是批次级的：账号并发/"
                f"速率超限、API Key 失效、或模型名写错。"
                f"前几条失败原因：{reasons}"
            )

        # ---- 6. 落盘 ----------------------------------------

        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        path = write_dataset(
            cases,
            output_dir=self._output_dir,
            document_version_id=str(version_id),
            stamp=stamp,
            meta={
                **stats.to_meta(),
                "provider": request.provider or "(default)",
                "temperature": request.temperature,
                "sample_seed": request.sample_seed,
                "max_windows": request.max_windows,
                "max_questions_per_window": (
                    request.max_questions_per_window
                ),
                "concurrency": request.concurrency,
                # 记录下来才能判断这批数据的覆盖面：只跑了 10/1257 个
                # 窗口的数据集，和跑了全量的，结论强度完全不同。
                "windows_available": len(blocks),
                "clean_text_chars": len(clean_text),
            },
        )

        logger.info(
            "数据集已生成：%s（%d 条样本，%d/%d 窗口成功，耗时 %.1fs）",
            path,
            len(cases),
            stats.windows_ok,
            stats.windows_total,
            stats.elapsed_seconds,
        )

        return QADatasetGenerateResponse(
            dataset_path=str(path),
            case_count=len(cases),
            windows_total=stats.windows_total,
            windows_ok=stats.windows_ok,
            windows_failed=stats.windows_failed,
            questions_generated=stats.questions_generated,
            questions_kept=stats.questions_kept,
            questions_dropped=stats.questions_dropped,
            drop_reasons=stats.drop_reasons,
            failures=[
                QADatasetFailure(
                    section=f.title,
                    span=(f.start, f.end),
                    reason=f.reason,
                )
                for f in stats.failures[:_MAX_REPORTED_FAILURES]
            ],
            elapsed_seconds=round(stats.elapsed_seconds, 1),
        )

    # ========================================================
    # internals
    # ========================================================

    async def _load_clean_text(
            self,
            version_id: UUID,
    ) -> str | None:
        """
        找到该版本**最近一次**运行里 cleaning 节点的产物，读回正文。

        为什么是"最近一次"：一个版本可能被 reprocess 过多次，早期运行
        的 clean 正文对应的是当年的切分结果，与现在库里的 chunk 偏移
        未必对得上。用最近一次才与当前 chunk 同一批。
        """

        runs, _ = await self._runs.list_runs(
            document_version_id=version_id,
            limit=1,
        )

        if not runs:
            logger.warning(
                "版本 %s 没有任何流水线运行记录。", version_id,
            )
            return None

        nodes = await self._run_repository_all_nodes(
            runs[0].pipeline_run_id
        )

        for node in nodes:

            if node.node_id != "cleaning":
                continue

            summary = node.output_summary or {}
            ref = summary.get("ref") or {}
            file_id = ref.get("file_id")

            if not file_id:
                continue

            text = await self._artifacts.get_text(UUID(str(file_id)))

            if text:
                return text

        return None

    async def _run_repository_all_nodes(
            self,
            pipeline_run_id: UUID,
    ) -> list[Any]:
        """
        取一次运行的全部节点记录。

        单独包一层只是为了在**这里**说明为什么不用分页接口：
        `list_node_runs` 是分页的（默认 20），而一个版本的扇出分支可能
        有好几路 × 十几个节点，20 条装不下，会漏掉 cleaning。
        `list_all_node_runs` 才是对的。
        """

        return await self._runs.list_all_node_runs(pipeline_run_id)

    async def _load_chunk_ranges(
            self,
            version_id: UUID,
    ) -> list[ChunkRange]:
        """
        取该版本全部带偏移的 chunk。

        翻页而不是一次要大 limit：`DocumentChunkRepository.list` 的
        `limit` 没有上限约束，但一次拉几千行会在内存里堆出不必要的
        对象；分页还能在取的过程中提前发现有偏移缺失。
        """

        ranges: list[ChunkRange] = []
        offset = 0

        while True:

            page, total = await self._chunks.list(
                document_version_id=version_id,
                offset=offset,
                limit=_CHUNK_PAGE,
            )

            for chunk in page:

                if (
                        chunk.start_index is None
                        or chunk.end_index is None
                ):
                    continue

                ranges.append(
                    ChunkRange(
                        chunk_id=chunk.document_chunk_id,
                        chunk_index=chunk.chunk_index,
                        start=chunk.start_index,
                        end=chunk.end_index,
                    )
                )

            offset += len(page)

            if not page or offset >= total:
                break

        return ranges


def _sample(
        sections: list[Any],
        limit: int,
        seed: int,
) -> list[Any]:
    """
    从全部窗口里随机抽 limit 个，**结果按原顺序返回**。

    为什么随机而不是取前 N 个：文档是按事件类型分节的（前 200 行可能
    全是「机构调研」，中间才是「股东户数」）。取前 N 个会让小批量试验
    永远只看得到开头那几种类型，调出来的参数在别的类型上未必成立。

    为什么抽完还要排序：生成阶段按顺序并发，保持原序让日志里的
    "窗口 X" 与在文档中的位置对应得上，排查时不用来回换算。
    """

    if limit >= len(sections):
        return list(sections)

    picked = random.Random(seed).sample(sections, limit)

    return sorted(picked, key=lambda s: s.start)
