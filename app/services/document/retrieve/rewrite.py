"""
查询改写的两个实现：`common`（不调模型）与 `llm`（调一次模型，**可批量**）。

注册
----
照精排那一套：声明 `rewriter_kinds`，注册表反射收进来
（见 registry.py 的 `_collect_rewriters`）。本文件是 retrieve 包的兄弟
模块，`import_package_modules(recursive=False)` 会自动收进来 ——
**不需要**在注册表里加任何一行。

`common` 刻意只做日期
---------------------
它只做一件事：把 query 里的日期**两种写法都展开**。

    2007年4月12日  ->  追加 "2007-04-12"
    2007-04-12     ->  追加 "2007年4月12日"

为什么两样都要：这份语料的正文散文写中文（`2007年4月12日，公司披露…`），
而 `#### 2007-04-12 ｜ 对外担保` 这种小标题写短横线；两种写法在 BM25
那一路是**不同的词项**，只写一种就有一半的块匹配不上。

⚠️ 短横线形式**必须补零**：它是拿去 `match_phrase` 在**正文**里按位置相邻
找的串，而正文的小标题写 `#### 2007-04-02 ｜ …` —— 不补零就对不上那一条。
中文形式**不补零**（正文的散文就那么写的）。

⚠️ 与索引侧 `content_dates` 的键不是同一个串：那个是**没有分隔符**的
`20070402`（喂字段、当查询键），这里是有分隔符的、给人看也拿去正文里找的
两种写法。形态都由 `services/document/dates.py` 一处定义。

⚠️ 明确**不做**：实体→business key 替换、子查询拆分、模板词加权。
这三条都会改变问题的语义，得单独量过才敢默认开 —— 别"顺手补全"。

`llm` 的失败姿态
----------------
调模型的一律**永不抛**：异常/超时/回包解析不了，都退回原文并标
`degraded`。改写失败不该让一次检索失败（契约见 base.py 的
`QueryRewriter`）。

批量的风险在**串号**
-------------------
`rewrite_many` 一次改一批（省掉每条都重付的固定开销）。批量最危险的失败
不是"出错"，是**把甲的改写安到乙头上** —— 每一条读起来都合理。所以：
提示词要求逐条回编号，这里**逐条校验编号**，对不上就整批不用、回落到
逐条。提示词与校验缺一不可。
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ValidationError

from app.agent.context.processor.messages import (
    message_text as llm_message_text,
)
from app.agent.model import ModelRequest, ModelRole, get_model_registry
from app.agent.prompt import get_prompt_provider

from .. import dates as dates_module
from ..domain.schemas import QueryRewriterConfig
from ..domain.schemas.enum import QueryRewriterKind
from .base import QueryRewriter, RewriteResult

__all__ = [
    "CommonQueryRewriter",
    "LlmQueryRewriter",
]

logger = logging.getLogger(__name__)


# ============================================================
# common
# ============================================================

# 日期的正则、形态与"两种写法"都在 `dates.py` 里（一份词汇表，四个调用方：
# 这里的改写器、ES 侧那个 content_dates 字段、以及索引期的字段计算）。
# 改形态要一起改。


class CommonQueryRewriter(QueryRewriter):
    """
    不调模型的改写：日期两种写法都展开。

    幂等是硬要求（见 `QueryRewriter.rewrite` 的契约）：实现方式是
    **只追加缺的那种写法**，且追加前先看规范形式在不在原文里 ——
    所以把输出再喂一遍不会长出第二条，也不会越滚越长。

    不覆盖 `rewrite_many`：默认的"逐条循环"对它就是最优解（批量省的是
    模型调用的固定开销，而它压根不调模型）。
    """

    rewriter_kinds = (QueryRewriterKind.COMMON,)

    async def rewrite(
            self,
            *,
            query: str,
            config: QueryRewriterConfig,
    ) -> RewriteResult:

        triples, skipped = dates_module.scan(query)

        if not triples:
            return RewriteResult(
                text=query,
                changed=False,
                detail={
                    "dates": [],
                    "skipped": skipped,
                    "boost_phrases": [],
                    "date_keys": [],
                },
            )

        appended: list[str] = []
        rows: list[dict[str, Any]] = []

        for year, month, day in triples:

            dash, chinese = dates_module.phrase_forms(year, month, day)

            # `in query` 而不是正则再扫一遍：这里判的是"那种写法在不在
            # 原文里"，字符串包含就够，而且与追加用的串是同一份拼法。
            missing = [
                form for form in (dash, chinese) if form not in query
            ]

            for form in missing:
                if form not in appended:
                    appended.append(form)

            rows.append({
                "canonical": dash,
                "appended": missing,
                "key": dates_module.canonical_key(year, month, day),
            })

        # 短语加权清单：每个日期的**两种规范写法都给**。
        #
        # 两种都给，而不是只给"原文里缺的那种"：正文散文写中文、小节标题
        # 写短横线，而它们落在**不同的块**上（`#### 2007-04-12 ｜ 对外担保`
        # 那个标题常常被切进上一块）。只加一种，就有一半的块拿不到这份
        # 权重 —— 而"哪一块才是这块的标题"在这里是看不到的。
        phrases = tuple(
            dict.fromkeys(
                form
                for year, month, day in triples
                for form in dates_module.phrase_forms(year, month, day)
            )
        )

        # 与索引侧 `content_dates` 字段**同一种形态**的键（补零的 YYYYMMDD），
        # 供精确加权用 —— 不管查询端写的是中文还是短横线，都能对上
        # （见 dates.py 的模块说明）。
        keys = tuple(
            dates_module.canonical_key(year, month, day)
            for year, month, day in triples
        )

        detail = {
            "dates": rows,
            "skipped": skipped,
            "boost_phrases": list(phrases),
            "date_keys": list(keys),
        }

        if not appended:
            return RewriteResult(
                text=query,
                changed=False,
                boost_phrases=phrases,
                date_keys=keys,
                detail=detail,
            )

        return RewriteResult(
            text=f"{query} {' '.join(appended)}",
            changed=True,
            boost_phrases=phrases,
            date_keys=keys,
            detail=detail,
        )


# ============================================================
# llm
# ============================================================

#: 后台/在线调用用的 agent_type。与 model.yaml 里的 binding 必须一致
#: —— 名字取"服务"而不是图里的 agent（与 retrieval_eval_* 同一口径）。
_REWRITE_AGENT_TYPE = "document_retrieve_rewrite"

#: 从回包里剔 ``` 围栏。提示词里写死了"只输出一个 JSON 对象"
#: （不设 response_format，与 eval/generator.py 同一口径）。
_FENCE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>.*?)\s*```\s*$",
    re.DOTALL,
)


class _RewriteItem(BaseModel):
    """回包里的**一条**改写结果。"""

    index: int
    rewritten_query: str
    changed: bool = False


class _BatchRewritePayload(BaseModel):
    """
    LLM 改写器的回包形状（批量）。

    ⚠️ `index` 是**必填**而不是靠列表顺序推：提示词要求"逐条对应、不得
    串号"，而校验只能靠这个显式编号 —— 没有它就只能相信顺序，而串号
    恰恰是批量最危险、最难发现的失败（每条读起来都合理）。
    """

    items: list[_RewriteItem]


def _load_object(text: str) -> dict[str, Any]:
    """
    模型输出 -> JSON 对象。

    与 `eval/output.py` 的 `_load_object` 是同一条规矩（去 ``` 围栏、
    顶层必须是对象），但**刻意不复用**：那个在 eval 包里而且不公开，
    而改写器是**检索侧**能力（线上也会走），不该反向依赖评测层。
    """

    raw = (text or "").strip()

    if not raw:
        raise ValueError("模型返回了空输出（提示词要求一个 JSON 对象）。")

    fenced = _FENCE.match(raw)

    if fenced is not None:
        raw = fenced.group("body").strip()

    payload = json.loads(raw)

    if not isinstance(payload, dict):
        raise ValueError(
            "模型输出的顶层不是对象（提示词要求一个 JSON 对象）。"
            f"实际类型：{type(payload).__name__}。"
        )

    return payload


class LlmQueryRewriter(QueryRewriter):
    """
    调模型改写 query —— **批量优先**（单条就是批量为 1）。

    runtime **懒取**：注册表是无参构造实现（`rewriter_cls()`），所以这里
    不能有 `__init__` 依赖；顺带的好处是没配这个 role 的环境不会在装配期
    就炸 —— 而是真的用到时才报，并且报错会被接住变成 `degraded`。

    超时 / 重试 / fallback / 用量统计都由 ModelRuntime 负责（见
    `agent/model/runtime.py`）：这里只管"拿到文本、校验条数与编号、
    失败退回"。
    """

    rewriter_kinds = (QueryRewriterKind.LLM,)

    async def rewrite(
            self,
            *,
            query: str,
            config: QueryRewriterConfig,
    ) -> RewriteResult:
        """单条 = 批量为 1 —— 契约只有一份，省掉两套解析。"""

        results = await self.rewrite_many(queries=[query], config=config)

        return results[0]

    async def rewrite_many(
            self,
            *,
            queries: list[str],
            config: QueryRewriterConfig,
    ) -> list[RewriteResult]:

        if not queries:
            return []

        try:

            return await self._rewrite_batch(
                queries=queries, config=config,
            )

        except Exception as exc:  # noqa: BLE001 —— 契约：永不抛

            logger.warning(
                "批量改写不可用（%d 条），回落到逐条：%s: %s",
                len(queries),
                type(exc).__name__,
                exc,
            )

        # 批量这条路走不通时**逐条重试**，而不是整批退化成"不改"。
        #
        # 逐条的成功率更高（没有串号风险、每条的输出也短得多），代价只是
        # 多几次调用；而"整批用原文"会让一次回包抖动抹掉整轮的改写 ——
        # 第 2 轮的意义就是"换个说法再查一次"，全轮没换说法这一轮就白跑了
        # （数字上表现为"第 2 轮没用"，而看不出是改写没生效）。
        return [
            await self._rewrite_one(query=query, config=config)
            for query in queries
        ]

    # --------------------------------------------------------
    # 内部
    # --------------------------------------------------------

    async def _rewrite_batch(
            self,
            *,
            queries: list[str],
            config: QueryRewriterConfig,
    ) -> list[RewriteResult]:
        """
        一次调用改一批，并**逐条校验编号**。

        Raises:
            ValueError / ValidationError: 条数不对、编号对不上、或回包不成形。
                调用方（`rewrite_many`）据此回落到逐条 —— 宁可多花几次
                调用，也不接受"可能串了号"的结果。
        """

        response = await self._invoke(queries=queries, config=config)

        payload = _BatchRewritePayload.model_validate(
            _load_object(llm_message_text(response.message))
        )

        items = {item.index: item for item in payload.items}

        if len(items) != len(payload.items):
            raise ValueError(
                f"回包里有重复编号（共 {len(payload.items)} 条）—— 分不清"
                "哪条对应哪条。"
            )

        if set(items) != set(range(1, len(queries) + 1)):
            raise ValueError(
                f"回包的编号与输入对不上：期望 1..{len(queries)}，"
                f"实际 {sorted(items)}。"
            )

        results: list[RewriteResult] = []

        for index, query in enumerate(queries, start=1):

            text = (items[index].rewritten_query or "").strip() or query

            results.append(
                RewriteResult(
                    text=text,
                    # 以**文本**为准而不是模型自报的 changed：模型说改了但
                    # 吐回一模一样的串时，报"改过"会让结果读起来像换了个
                    # 说法，其实没有。
                    changed=text != query,
                    # **不给短语加权**：模型会顺手补出它"觉得该有"的词
                    # （实测它给一条 2007 年的题补了公司全称与"被担保人"
                    # 这类公告用语）。把模型吐的词当**短语级**权重，等于让它
                    # 决定排序公式里的一项，而那一项没有量过。要用它的输出
                    # 就只用文本（BM25 按普通词项处理，错了也只是多几个词）。
                    boost_phrases=(),
                    detail={
                        "model": config.model,
                        "batch_size": len(queries),
                        "reported_changed": items[index].changed,
                        "boost_phrases": [],
                    },
                )
            )

        return results

    async def _rewrite_one(
            self,
            *,
            query: str,
            config: QueryRewriterConfig,
    ) -> RewriteResult:
        """
        逐条改写（批量失败后的回落路径）。

        **不抛**：任何失败都退回原文 + `degraded=True`（契约见 base.py）。
        """

        try:

            batch = await self._rewrite_batch(
                queries=[query], config=config,
            )

            return batch[0]

        except Exception as exc:  # noqa: BLE001 —— 契约就是永不抛

            logger.warning(
                "改写失败，本条回退原文：%s: %s",
                type(exc).__name__,
                exc,
            )

            return RewriteResult(
                text=query,
                changed=False,
                degraded=True,
                detail={"error": f"{type(exc).__name__}: {exc}"},
            )

    async def _invoke(
            self,
            *,
            queries: list[str],
            config: QueryRewriterConfig,
    ) -> Any:
        """
        调一次模型（不含解析）。批量与逐条共用。

        ⚠️ `disable_thinking=True`：这条调用的**输出形状完全由提示词给定**
        （只回一个 JSON 对象），正是该关思考的情形。不关的代价实测过：
        同端点一条简单得多的抽取，`completion_tokens=51` 里 `reasoning=41`
        （约 92%）—— 思考与正文**共用** `max_tokens`，批量时条数一多，
        正文就会被挤到写不完（`Unterminated string`）。关了之后同一条
        提示词的输出是 9 token。

        它与模型级那个开关（model.yaml 的 `disable_thinking`）**不是一回事**：
        那个是一律关、会连主循环要推给用户的思维链一起关掉；这个只对
        当场这次调用生效。
        """

        runtime = get_model_registry().get_runtime(
            agent_type=_REWRITE_AGENT_TYPE,
            model_role=ModelRole.QUERY_REWRITE,
        )

        request = ModelRequest(
            messages=(
                SystemMessage(
                    content=get_prompt_provider().node_prompt(
                        ModelRole.QUERY_REWRITE
                    )
                ),
                HumanMessage(content=_render_user_message(queries)),
            ),
            disable_thinking=True,
            metadata={
                "purpose": "query.rewrite",
                "agent_type": _REWRITE_AGENT_TYPE,
                "model_role": ModelRole.QUERY_REWRITE,
                "batch_size": len(queries),
                # 模型标识进了 metadata 而不是请求参数：这里只用于观测
                # （谁改的），真正的模型由 model.yaml 的 binding 决定 ——
                # 两处都能选模型会让"到底是哪个模型"说不清。
                "model": config.model,
            },
        )

        # ⚠️ 只在调用方**显式**给了 timeout_ms 时才额外套一层。
        #
        # 此前这里有个 20s 的硬编码默认值，它比该 model_role 的 profile
        # 自己配的超时（30s）更紧 —— 于是"profile 里那个数"形同虚设，
        # 而超时的表现是**静默回退原文**（degraded），看起来只是"这次
        # 改写没生效"。实测：一轮评测里出现 `TimeoutError` 回退，第 2 轮
        # 的数字因此混进了"改写没跑成"。
        #
        # 超时/重试/总时长本来就归 ModelRuntime 的 policy 管（见
        # agent/model/runtime.py），这一层只该是"调用方想更紧"的补充。
        if config.timeout_ms is None:
            return await runtime.ainvoke(request)

        return await asyncio.wait_for(
            runtime.ainvoke(request), timeout=config.timeout_ms / 1000,
        )


def _render_user_message(queries: list[str]) -> str:
    """
    用户消息那半（**一批**，带编号）。

    ⚠️ 这里 import 提示词模块里的渲染函数会给「检索侧反向依赖 agent
    提示词包」—— 那是**刻意的**：提示词按 model_role 住在 agent 侧是仓库
    的既有约定（见 agent/prompt/prompts/__init__.py）。
    """

    from app.agent.prompt.prompts.query_rewrite import (
        render_user_message,
    )

    return render_user_message(queries)
