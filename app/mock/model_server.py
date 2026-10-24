"""
开发用的 OpenAI-compatible Mock 模型服务。

为什么是进程内守护线程
----------------------
Phase 0 的目标是"启动主服务、切前端就能测"，而不是先搭一套独立部署。
守护线程让 mock 与主服务共生命周期：主服务起来它就监听，主服务退出它就
结束；前端不需要额外起进程。

⚠️ 它不适合压测
--------------
线程与主服务共享同一个进程/GIL。高并发下 mock 的 CPU 会污染被测系统；
真正的压测阶段（Phase 1+）应把它换成独立进程或独立主机。这个取舍写在
`design_agent_eval/stress_capacity_plan.md` 里。

它做什么
--------
    POST /v1/chat/completions   OpenAI 兼容；支持流式与非流式
    GET  /v1/models             给 warmup / 调试用
    GET  /health                就绪探针
    GET  /stats                 请求数、活跃流、实际 TTFT、字符数

回复策略
--------
    普通对话       随机文本；TTFT / 字数 / 速率均来自截断正态分布
    分类           识别强制 tool schema（含 behavior + execution_plan），
                   返回合法结构化结果；execution_plan 可由
                   mock_model.yaml 的 execution_plan 强制（stress_eval
                   固定 react），否则随机；capability_needs 从 system
                   prompt 的候选清单里随机选
    带 tools 的调用 按配置概率随机返回 tool_calls；数量与数值参数用正态分布，
                   具体选哪个工具是分类选择，按等概率抽取
    response_format 非分类的结构化输出，按 JSON schema 随机生成

⚠️ 随机性的边界
--------------
"所有随机都符合正态分布"在数学上只对**连续量**成立（延迟、字数、速率、
数值参数、调用数量）。工具名、行为枚举这类**分类量**没有正态分布的定义，
只能按等概率/权重抽取；代码里对这两类分开处理，且注释写清楚。
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from aiohttp import web

from app.common.mock_model_config import (
    MOCK_EXECUTION_PLANS,
    MockModelConfig,
)

__all__ = ["MockModelServer"]

logger = logging.getLogger(__name__)


#: 分类行为 / 执行方案的合法值。刻意不 import app.agent.state：
#: mock 只依赖配置与 aiohttp，不应该把 Agent 图那一整串依赖拖进来。
#: 执行方案暂不含 "plan" —— planner 节点还没接通，返回 plan 会把请求
#: 路由到一个不存在的路径。
_BEHAVIORS: tuple[str, ...] = (
    "chat",
    "qa",
    "analysis",
    "creation",
    "modification",
    "research",
    "execution",
    "planning",
    "decision_support",
    "other",
)

_EXECUTION_PLANS: tuple[str, ...] = MOCK_EXECUTION_PLANS

#: 记忆后台任务的固定合法输出。
#:
#: 这些任务不走 response_format，也没有 tools —— HTTP 层看不出 model_role，
#: 只能靠 system prompt 识别。返回空结果而不是随机文本：随机文本会让
#: `parse_extraction` / `parse_conflicts` 直接抛 JSONDecodeError（任务被
#: 标 FAILED），空数组/空变更则是这些解析器明确接受的正常结果。
_MEMORY_RESPONSES: dict[str, str] = {
    # 情境列表；空 = 这一轮没有值得记的东西（cursor 仍会推进）。
    "memory_extract": "[]",
    # 冲突判定列表；空 = 没有需要合并/覆盖的已有记忆。
    "memory_dedup": "[]",
    # L2 整合：没有文件变更，也不请求更新画像。
    "memory_scene": json.dumps(
        {"operations": [], "persona_update_request": False},
        ensure_ascii=False,
    ),
    # L3 画像：明确声明本轮不改。
    "memory_persona": json.dumps({"no_change": True}, ensure_ascii=False),
}

#: 分类提示词里的候选能力行：``- retrieval：检索 —— ...``
_CANDIDATE_LINE = re.compile(r"^-\s+([^\s：:]+)[：:]", re.MULTILINE)

#: 随机文本语料。只求可读、可渲染，不含任何业务结论。
_CORPUS = (
    "这是 mock 模型生成的一段回复，用于本地联调和前端渲染测试。",
    "它不包含真实业务结论，也不应被当作投资建议。",
    "当前阶段主要验证流式链路、首 token 延迟和并发行为。",
    "如果需要更长的回复，会重复这些句子直到达到配置的字符数。",
)

_CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
}


# ============================================================
# 请求计划
# ============================================================


@dataclass(frozen=True, slots=True)
class _Plan:
    """一次请求的完整回复计划。先生成、再按计划流式/非流式发送。"""

    role: str
    model: str
    response_id: str
    text: str
    tool_calls: tuple[dict[str, Any], ...]
    finish_reason: str
    ttft_ms: float
    chars_per_second: float
    chunk_chars: int
    prompt_chars: int
    completion_chars: int

    @property
    def total_seconds(self) -> float:
        """TTFT + 生成本文/参数的时间。"""

        return self.ttft_ms / 1000.0 + (
            self.completion_chars / max(self.chars_per_second, 1.0)
        )

    @property
    def usage(self) -> dict[str, int]:
        # 粗略的 4 字符 ≈ 1 token。mock 不追求 tokenizer 精度，只需要
        # 让 trace/前端拿到非空 usage。
        prompt = max(1, round(self.prompt_chars / 4))
        completion = max(1, round(self.completion_chars / 4))

        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": prompt + completion,
        }


# ============================================================
# 统计
# ============================================================


class _Stats:
    """单事件循环内读写，因此不需要锁。"""

    def __init__(self) -> None:
        self.started_at = time.time()
        self.requests = 0
        self.streams = 0
        self.non_streams = 0
        self.classifications = 0
        self.memory_responses = 0
        self.tool_responses = 0
        self.errors = 0
        self.disconnects = 0
        self.chars = 0
        self.duration_ms_sum = 0.0
        self.ttft_ms_sum = 0.0
        self.ttft_count = 0

    def snapshot(self) -> dict[str, Any]:
        avg_ttft = (
            self.ttft_ms_sum / self.ttft_count
            if self.ttft_count
            else None
        )
        avg_duration = (
            self.duration_ms_sum / self.ttft_count
            if self.ttft_count
            else None
        )

        return {
            "started_at": self.started_at,
            "uptime_seconds": round(time.time() - self.started_at, 1),
            "requests": self.requests,
            "streams": self.streams,
            "non_streams": self.non_streams,
            "classifications": self.classifications,
            "memory_responses": self.memory_responses,
            "tool_responses": self.tool_responses,
            "errors": self.errors,
            "disconnects": self.disconnects,
            "chars": self.chars,
            "avg_ttft_ms": None if avg_ttft is None else round(avg_ttft, 1),
            "avg_duration_ms": (
                None if avg_duration is None else round(avg_duration, 1)
            ),
        }


# ============================================================
# 服务
# ============================================================


class MockModelServer:
    """一个跑在守护线程里的 aiohttp OpenAI-compatible 服务。"""

    def __init__(self, config: MockModelConfig) -> None:
        self._config = config
        self._rng = random.Random(config.seed)
        self._stats = _Stats()

        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self._started = threading.Event()
        self._error: BaseException | None = None
        self._lock = threading.Lock()
        self._active = 0

        self._app = self._build_app()

    # --------------------------------------------------------
    # 生命周期
    # --------------------------------------------------------

    def start(self) -> None:
        """启动线程并等待端口真正 listen；失败时抛出。"""

        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return

            self._started.clear()
            self._error = None

            self._thread = threading.Thread(
                target=self._run,
                name="mock-model-server",
                daemon=True,
            )
            self._thread.start()

            if not self._started.wait(timeout=10.0):
                raise RuntimeError(
                    "mock model server 启动超时（10s 内没有就绪）。"
                )

            if self._error is not None:
                raise RuntimeError(
                    f"mock model server 启动失败：{self._error!r}"
                ) from self._error

    def stop(self) -> None:
        """停止 aiohttp 服务并等待线程退出。可重复调用。"""

        loop = self._loop

        if loop is not None and loop.is_running():
            loop.call_soon_threadsafe(loop.stop)

        thread = self._thread

        if thread is not None:
            thread.join(timeout=5.0)

        self._thread = None
        self._loop = None
        self._runner = None

    def _run(self) -> None:
        """线程入口：独立事件循环 + aiohttp AppRunner。"""

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        runner: web.AppRunner | None = None

        try:
            runner = web.AppRunner(self._app, access_log=None)
            loop.run_until_complete(runner.setup())

            site = web.TCPSite(
                runner,
                self._config.host,
                self._config.port,
            )
            loop.run_until_complete(site.start())

            self._runner = runner
            logger.info(
                "Mock model server listening on http://%s:%d/v1",
                self._config.host,
                self._config.port,
            )
            self._started.set()
            loop.run_forever()

        except BaseException as exc:  # noqa: BLE001 - 交给 start() 抛出
            self._error = exc
            self._started.set()
            logger.exception("Mock model server 运行失败。")

        finally:
            if runner is not None:
                try:
                    loop.run_until_complete(runner.cleanup())
                except Exception:
                    logger.exception("Mock model server 清理失败。")

            try:
                loop.close()
            except Exception:
                pass

    # --------------------------------------------------------
    # 路由
    # --------------------------------------------------------

    def _build_app(self) -> web.Application:
        app = web.Application(middlewares=[_cors_middleware])
        app.router.add_get("/", self._index)
        app.router.add_get("/health", self._health)
        app.router.add_get("/stats", self._stats_handler)
        app.router.add_get("/v1/models", self._models)
        app.router.add_post(
            "/v1/chat/completions",
            self._chat_completions,
        )

        return app

    async def _index(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "service": "mock-model",
                "openai_compatible": True,
                "endpoints": [
                    "POST /v1/chat/completions",
                    "GET /v1/models",
                    "GET /health",
                    "GET /stats",
                ],
            }
        )

    async def _health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})

    async def _stats_handler(self, request: web.Request) -> web.Response:
        payload = self._stats.snapshot()
        payload["active_requests"] = self._active

        return web.json_response(payload)

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {
                "object": "list",
                "data": [
                    {
                        "id": "mock-chat",
                        "object": "model",
                        "created": int(self._stats.started_at),
                        "owned_by": "mock",
                    }
                ],
            }
        )

    # --------------------------------------------------------
    # chat completions
    # --------------------------------------------------------

    async def _chat_completions(
            self,
            request: web.Request,
    ) -> web.StreamResponse:
        self._stats.requests += 1
        self._active += 1
        started = time.perf_counter()

        try:
            try:
                payload = await request.json()

            except Exception as exc:
                self._stats.errors += 1
                return web.json_response(
                    {"error": {"message": str(exc), "type": "invalid_json"}},
                    status=400,
                )

            if not isinstance(payload, Mapping):
                self._stats.errors += 1
                return web.json_response(
                    {"error": {"message": "body 必须是 JSON 对象"}},
                    status=400,
                )

            plan = self._plan(payload)

            if self._config.log_requests:
                logger.info(
                    "mock model role=%s model=%s stream=%s "
                    "ttft=%.0fms chars=%d tools=%d id=%s",
                    plan.role,
                    plan.model,
                    bool(payload.get("stream")),
                    plan.ttft_ms,
                    plan.completion_chars,
                    len(plan.tool_calls),
                    plan.response_id,
                )

            if payload.get("stream"):
                return await self._stream(request, plan, started)

            await asyncio.sleep(plan.total_seconds)

            self._stats.non_streams += 1
            self._record(
                plan,
                actual_ttft_ms=plan.ttft_ms,
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

            return web.json_response(self._completion_payload(plan))

        except Exception as exc:  # noqa: BLE001 - mock 不把异常抛给 aiohttp
            self._stats.errors += 1
            logger.exception("mock model 请求失败。")

            return web.json_response(
                {"error": {"message": str(exc), "type": "mock_error"}},
                status=500,
            )

        finally:
            self._active -= 1

    # --------------------------------------------------------
    # 计划生成
    # --------------------------------------------------------

    def _plan(self, payload: Mapping[str, Any]) -> _Plan:
        messages = list(payload.get("messages") or ())
        system = _system_text(messages)
        tools = list(payload.get("tools") or ())
        tool_choice = payload.get("tool_choice")
        response_format = payload.get("response_format")

        model = str(payload.get("model") or "mock-chat")

        role = self._detect_role(
            model=model,
            system=system,
            tools=tools,
            tool_choice=tool_choice,
            response_format=response_format,
        )

        mode = (
            self._config.classification
            if role in {
                "classification",
                "structured",
                "memory_extract",
                "memory_dedup",
                "memory_scene",
                "memory_persona",
            }
            else self._config.stream
        )

        prompt_chars = sum(
            len(_message_text(message)) for message in messages
        )

        response_id = f"chatcmpl-mock-{uuid.uuid4().hex[:24]}"

        text = ""
        tool_calls: tuple[dict[str, Any], ...] = ()
        finish_reason = "stop"

        if role == "classification":
            forced = _forced_tool(tools, tool_choice)
            schema = (
                _function_schema(forced)
                if forced is not None
                else _structured_schema(None, response_format)
            )
            args = self._classification_args(system, schema)

            if forced is not None:
                name = _tool_name(forced) or "classification"
                tool_calls = (_tool_call(name, args),)
                finish_reason = "tool_calls"
            else:
                # response_format=json_schema 的客户端期待正文是 JSON，
                # 不是 tool_calls。
                text = json.dumps(args, ensure_ascii=False)
                finish_reason = "stop"

        elif role == "structured":
            forced = _forced_tool(tools, tool_choice)
            schema = _structured_schema(forced, response_format)

            if forced is not None:
                args = self._sample_schema(schema, required_only=True)
                tool_calls = (_tool_call(_tool_name(forced), args),)
                finish_reason = "tool_calls"

            else:
                args = self._sample_schema(schema, required_only=True)
                text = json.dumps(args, ensure_ascii=False)
                finish_reason = "stop"

        elif role in _MEMORY_RESPONSES:
            text = _MEMORY_RESPONSES[role]

        elif role == "tool_use":
            tool_calls = self._maybe_tool_calls(tools)

            if tool_calls:
                finish_reason = "tool_calls"
            else:
                text = _random_text(
                    mode.response_chars.sample_int(self._rng),
                    self._rng,
                )

        else:
            # text / summarization / planner / 未知角色
            text = _random_text(
                mode.response_chars.sample_int(self._rng),
                self._rng,
            )

        completion_chars = len(text) or sum(
            len(call["function"]["arguments"]) for call in tool_calls
        )

        return _Plan(
            role=role,
            model=model,
            response_id=response_id,
            text=text,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            ttft_ms=mode.ttft_ms.sample(self._rng),
            chars_per_second=mode.chars_per_second.sample(self._rng),
            chunk_chars=self._config.chunk_chars,
            prompt_chars=prompt_chars,
            completion_chars=max(1, completion_chars),
        )

    def _detect_role(
            self,
            *,
            model: str,
            system: str,
            tools: Sequence[Mapping[str, Any]],
            tool_choice: Any,
            response_format: Any,
    ) -> str:
        # 优先按模型名判断：--mock 启动时每个 model_role 有独立的
        # mock-role-* 模型。这是准确判据，不走任何提示词启发式。
        from_model = _role_from_model(model, has_tools=bool(tools))

        if from_model is not None:
            return from_model

        forced = _forced_tool(tools, tool_choice)

        if forced is not None and _is_classification_schema(forced):
            return "classification"

        if _is_classification_response_format(response_format):
            return "classification"

        # 记忆后台任务：没有 tools / response_format，只能靠 system prompt
        # 识别。放在 classifier 之前是因为这些任务的提示词里有
        # "memories"/"operations" 一类词，先按更具体的角色收口。
        if _looks_like_memory_extract(system):
            return "memory_extract"

        if _looks_like_memory_dedup(system):
            return "memory_dedup"

        if _looks_like_memory_scene(system):
            return "memory_scene"

        if _looks_like_memory_persona(system):
            return "memory_persona"

        if response_format is not None:
            return "structured"

        if forced is not None:
            return "structured"

        if _looks_like_classifier(system):
            return "classification"

        if tools:
            return "tool_use"

        if _looks_like_planner(system):
            # planner 节点还没接通。这里不伪造 plan 结构，只按普通文本回，
            # 等它真正有调用方时再补专用分支。
            return "planner"

        if _looks_like_summarizer(system):
            return "summarization"

        return "text"

    # --------------------------------------------------------
    # 分类结果
    # --------------------------------------------------------

    def _classification_args(
            self,
            system: str,
            schema: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        schema = schema or {}
        properties = (
            schema.get("properties", {})
            if isinstance(schema, Mapping)
            else {}
        )

        behavior_values = tuple(properties.get("behavior", {}).get("enum") or _BEHAVIORS)
        plan_values = tuple(
            properties.get("execution_plan", {}).get("enum") or _EXECUTION_PLANS
        )

        # stress_eval 把 mock_model.yaml 的 execution_plan 钉成 react：
        # 压测只走主链路，不随机分叉到 other。None = 维持随机。
        # 配置值不在本次 schema 的枚举里时回退到随机，避免造出
        # 分类解析器不认的输出。
        forced_plan = self._config.execution_plan

        execution_plan = (
            forced_plan
            if forced_plan is not None and forced_plan in plan_values
            else self._rng.choice(plan_values)
        )

        candidates = list(self._config.capability_pool) or _candidate_codes(system)
        needs: list[str] = []

        if candidates:
            count = self._rng.randint(0, min(2, len(candidates)))
            needs = self._rng.sample(candidates, count)

        args: dict[str, Any] = {
            "behavior": self._rng.choice(behavior_values),
            "execution_plan": execution_plan,
            "capability_needs": needs,
            "capability_config": {},
            # mock 分类默认打开 L0/L1 预召回：与真实模型一样走
            # context_policy.memory_retrieve，才能验证"召回 -> 注入"链路。
            # 要压测主链路时把 YTRADER_MEMORY_CONTEXT 设为 0。
            "retrieve_l0": "true",
            "retrieve_l1": "true",
        }

        # schema 里还有别的必填字段（例如旧版的 modules）时用通用采样补齐，
        # 避免结构化解析因为缺字段整次降级。
        required = schema.get("required") or ()
        for name in required:
            if name not in args and name in properties:
                args[name] = self._sample_schema(properties[name])

        return args

    # --------------------------------------------------------
    # 工具调用
    # --------------------------------------------------------

    def _maybe_tool_calls(
            self,
            tools: Sequence[Mapping[str, Any]],
    ) -> tuple[dict[str, Any], ...]:
        if not tools:
            return ()

        if self._rng.random() > self._config.tool_call.probability:
            return ()

        count = self._config.tool_call.count.sample_int(self._rng)

        if count <= 0:
            return ()

        count = min(count, len(tools), 4)
        calls: list[dict[str, Any]] = []

        for _ in range(count):
            tool = self._rng.choice(list(tools))
            function = tool.get("function", {}) or {}
            name = str(function.get("name") or "mock_tool")
            arguments = self._sample_schema(
                function.get("parameters") or {},
                required_only=True,
            )
            calls.append(_tool_call(name, arguments))

        return tuple(calls)

    def _sample_schema(
            self,
            schema: Any,
            *,
            required_only: bool = False,
    ) -> Any:
        """
        按 JSON Schema 采样一个值。

        数值型用正态分布并夹到 schema 的 minimum/maximum；字符串枚举按
        等概率抽取；数组长度取最小长度（避免生成无意义的超长参数）。
        """

        if not isinstance(schema, Mapping):
            return "mock"

        if "const" in schema:
            return schema["const"]

        enum = schema.get("enum")

        if isinstance(enum, list) and enum:
            return self._rng.choice(enum)

        for key in ("anyOf", "oneOf", "allOf"):
            options = schema.get(key)

            if isinstance(options, list) and options:
                return self._sample_schema(
                    self._rng.choice(options),
                    required_only=required_only,
                )

        schema_type = schema.get("type")

        if isinstance(schema_type, list):
            schema_type = next(
                (item for item in schema_type if item != "null"),
                schema_type[0] if schema_type else None,
            )

        if schema_type == "object" or "properties" in schema:
            properties = schema.get("properties") or {}
            required = set(schema.get("required") or ())
            result: dict[str, Any] = {}

            for name, child in properties.items():
                if name in required or (
                    not required_only and self._rng.random() < 0.35
                ):
                    result[name] = self._sample_schema(child)

            return result

        if schema_type == "array":
            items = schema.get("items") or {}
            minimum = int(schema.get("minItems") or 0)
            maximum = schema.get("maxItems")

            length = max(1, minimum)

            if maximum is not None:
                length = min(length, int(maximum))

            return [
                self._sample_schema(items) for _ in range(length)
            ]

        if schema_type == "string":
            return "mock"

        if schema_type in {"integer", "number"}:
            minimum = schema.get("minimum")
            maximum = schema.get("maximum")

            if minimum is not None and maximum is not None:
                mean = (float(minimum) + float(maximum)) / 2.0
                std = max((float(maximum) - float(minimum)) / 6.0, 0.0)
            else:
                mean = 10.0
                std = 3.0

            value = self._rng.gauss(mean, std)

            if minimum is not None:
                value = max(value, float(minimum))

            if maximum is not None:
                value = min(value, float(maximum))

            return int(round(value)) if schema_type == "integer" else round(value, 2)

        if schema_type == "boolean":
            return self._rng.choice([True, False])

        if schema_type == "null":
            return None

        return "mock"

    # --------------------------------------------------------
    # 流式输出
    # --------------------------------------------------------

    async def _stream(
            self,
            request: web.Request,
            plan: _Plan,
            started: float,
    ) -> web.StreamResponse:
        headers = {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            **_CORS_HEADERS,
        }

        response = web.StreamResponse(status=200, headers=headers)

        await response.prepare(request)

        first_chunk_at: float | None = None
        finished = False

        try:
            await asyncio.sleep(plan.ttft_ms / 1000.0)
            first_chunk_at = time.perf_counter()

            await self._sse(
                response,
                self._chunk(plan, delta={"role": "assistant"}),
            )

            if plan.tool_calls:
                for index, call in enumerate(plan.tool_calls):
                    await self._sse(
                        response,
                        self._chunk(
                            plan,
                            delta={
                                "tool_calls": [
                                    {"index": index, **call}
                                ]
                            },
                        ),
                    )
                    await asyncio.sleep(
                        len(call["function"]["arguments"])
                        / max(plan.chars_per_second, 1.0)
                    )

            else:
                for piece in _split_text(plan.text, plan.chunk_chars):
                    await self._sse(
                        response,
                        self._chunk(plan, delta={"content": piece}),
                    )
                    await asyncio.sleep(
                        len(piece) / max(plan.chars_per_second, 1.0)
                    )

            await self._sse(
                response,
                self._chunk(
                    plan,
                    delta={},
                    finish_reason=plan.finish_reason,
                    usage=plan.usage,
                ),
            )
            await response.write(b"data: [DONE]\n\n")
            finished = True

            self._stats.streams += 1
            self._record(
                plan,
                actual_ttft_ms=(
                    (first_chunk_at - started) * 1000.0
                    if first_chunk_at is not None
                    else plan.ttft_ms
                ),
                duration_ms=(time.perf_counter() - started) * 1000.0,
            )

        except (ConnectionResetError, asyncio.CancelledError):
            self._stats.disconnects += 1
            logger.info("mock model 客户端断开：%s", plan.response_id)

        except Exception:
            self._stats.errors += 1
            logger.exception("mock model 流式输出失败：%s", plan.response_id)

        finally:
            if not finished:
                try:
                    await response.write_eof()
                except Exception:
                    pass

        return response

    @staticmethod
    def _chunk(
            plan: _Plan,
            *,
            delta: Mapping[str, Any],
            finish_reason: str | None = None,
            usage: Mapping[str, int] | None = None,
    ) -> dict[str, Any]:
        chunk: dict[str, Any] = {
            "id": plan.response_id,
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": plan.model,
            "choices": [
                {
                    "index": 0,
                    "delta": dict(delta),
                    "finish_reason": finish_reason,
                }
            ],
        }

        if usage is not None:
            chunk["usage"] = dict(usage)

        return chunk

    @staticmethod
    async def _sse(
            response: web.StreamResponse,
            payload: Mapping[str, Any],
    ) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        await response.write(b"data: " + data + b"\n\n")

    # --------------------------------------------------------
    # 非流式响应与统计
    # --------------------------------------------------------

    def _completion_payload(self, plan: _Plan) -> dict[str, Any]:
        message: dict[str, Any] = {
            "role": "assistant",
            "content": plan.text or None,
        }

        if plan.tool_calls:
            message["tool_calls"] = list(plan.tool_calls)

        return {
            "id": plan.response_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": plan.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": plan.finish_reason,
                }
            ],
            "usage": plan.usage,
        }

    def _record(
            self,
            plan: _Plan,
            *,
            actual_ttft_ms: float,
            duration_ms: float,
    ) -> None:
        self._stats.chars += plan.completion_chars
        self._stats.ttft_ms_sum += max(0.0, actual_ttft_ms)
        self._stats.ttft_count += 1
        self._stats.duration_ms_sum += max(0.0, duration_ms)

        if plan.tool_calls:
            self._stats.tool_responses += 1

        if plan.role == "classification":
            self._stats.classifications += 1

        if plan.role.startswith("memory_"):
            self._stats.memory_responses += 1


# ============================================================
# 中间件与工具函数
# ============================================================


@web.middleware
async def _cors_middleware(
        request: web.Request,
        handler: Any,
) -> web.StreamResponse:
    if request.method == "OPTIONS":
        return web.Response(status=204, headers=_CORS_HEADERS)

    response = await handler(request)

    for key, value in _CORS_HEADERS.items():
        response.headers.setdefault(key, value)

    return response


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text")

                if isinstance(text, str):
                    parts.append(text)

        return "\n".join(parts)

    return ""


def _system_text(messages: Sequence[Mapping[str, Any]]) -> str:
    return "\n".join(
        _message_text(message)
        for message in messages
        if message.get("role") == "system"
    )


def _forced_tool(
        tools: Sequence[Mapping[str, Any]],
        tool_choice: Any,
) -> Mapping[str, Any] | None:
    if not tools:
        return None

    if isinstance(tool_choice, Mapping):
        function = tool_choice.get("function") or {}
        name = function.get("name")

        if name:
            for tool in tools:
                if _tool_name(tool) == name:
                    return tool

            return {
                "type": "function",
                "function": {"name": name, "parameters": {}},
            }

        return None

    if tool_choice == "required" and len(tools) == 1:
        return tools[0]

    return None


def _tool_name(tool: Mapping[str, Any] | None) -> str:
    if not tool:
        return ""

    function = tool.get("function") or {}

    return str(function.get("name") or "")


def _function_schema(
        tool: Mapping[str, Any] | None,
) -> Mapping[str, Any]:
    if not tool:
        return {}

    function = tool.get("function") or {}
    schema = function.get("parameters") or {}

    return schema if isinstance(schema, Mapping) else {}


def _is_classification_schema(tool: Mapping[str, Any] | None) -> bool:
    properties = _function_schema(tool).get("properties") or {}

    return "behavior" in properties and "execution_plan" in properties


def _is_classification_response_format(response_format: Any) -> bool:
    if not isinstance(response_format, Mapping):
        return False

    schema = _structured_schema(None, response_format)
    properties = schema.get("properties") or {}

    return "behavior" in properties and "execution_plan" in properties


def _structured_schema(
        forced: Mapping[str, Any] | None,
        response_format: Any,
) -> Mapping[str, Any]:
    if forced is not None:
        return _function_schema(forced)

    if isinstance(response_format, Mapping):
        if response_format.get("type") == "json_object":
            return {"type": "object", "properties": {}}

        json_schema = response_format.get("json_schema") or {}

        if isinstance(json_schema, Mapping):
            schema = json_schema.get("schema") or {}

            if isinstance(schema, Mapping):
                return schema

    return {"type": "object", "properties": {}}


def _role_from_model(model: str, *, has_tools: bool) -> str | None:
    """
    `mock-role-<key>` -> 明确的 mock 角色。

    这是 --mock 主路径的判据：model name 由 with_mock_bindings 按
    `model_role` 生成，因此不会像 system prompt 那样被上下文里的画像 /
    记忆标题误触发。
    """

    prefix = "mock-role-"

    if not model.startswith(prefix):
        return None

    key = model[len(prefix):]
    mapping = {
        "reasoning": "tool_use" if has_tools else "text",
        "classification": "classification",
        "summarization": "summarization",
        "memory-atom-extract": "memory_extract",
        "memory-atom-dedup": "memory_dedup",
        "memory-scene-consolidate": "memory_scene",
        "memory-persona-extract": "memory_persona",
        # 以下角色暂时按文本处理；它们各自的解析器还没接专用输出。
        "query-rewrite": "text",
        "retrieve-rerank": "text",
        "eval-qa-generation": "text",
        "eval-cross-chunk-judge": "text",
    }

    return mapping.get(key)


def _looks_like_classifier(system: str) -> bool:
    # 只用**分类提示词独有**的短语。不要拿 execution_plan /
    # capability_needs 这类字段名当判据：它们也可能出现在别的 system
    # prompt 或工具描述里，误判会让主循环收到一个分类 tool_call。
    markers = (
        "Agent Builder",
        "任务分诊",
        "分类调用",
    )

    return any(marker in system for marker in markers)


def _looks_like_memory_extract(system: str) -> bool:
    markers = (
        "情境切分与记忆提取专家",
        "待提取的新消息",
        "情境切分",
    )

    return any(marker in system for marker in markers)


def _looks_like_memory_dedup(system: str) -> bool:
    markers = (
        "冲突检测",
        "record_id",
        "store / skip / update / merge",
    )

    return any(marker in system for marker in markers)


def _looks_like_memory_scene(system: str) -> bool:
    markers = (
        "记忆整合架构师",
        "现有场景文件",
        "persona_update_request",
    )

    return any(marker in system for marker in markers)


def _looks_like_memory_persona(system: str) -> bool:
    markers = (
        "User Narrative Profile",
        "Team Operating Doctrine",
        "no_change",
    )

    return any(marker in system for marker in markers)


def _looks_like_planner(system: str) -> bool:
    markers = ("planner", "plan-and-execute", "计划节点")

    return any(marker in system for marker in markers)


def _looks_like_summarizer(system: str) -> bool:
    markers = ("摘要", "summarization", "上下文压缩")

    return any(marker in system for marker in markers)


def _candidate_codes(system: str) -> list[str]:
    """从分类 prompt 的 ``- code：名称`` 行里抽出能力 code。"""

    seen: set[str] = set()
    result: list[str] = []

    for match in _CANDIDATE_LINE.finditer(system):
        code = match.group(1).strip()

        if code and code not in seen:
            seen.add(code)
            result.append(code)

    return result


def _tool_call(
        name: str,
        arguments: Any,
) -> dict[str, Any]:
    return {
        "id": f"call_mock_{uuid.uuid4().hex[:12]}",
        "type": "function",
        "function": {
            "name": name or "mock_tool",
            "arguments": json.dumps(arguments, ensure_ascii=False),
        },
    }


def _random_text(length: int, rng: random.Random) -> str:
    """生成指定字符数的可读文本；长度不足时至少返回一个句子。"""

    if length <= 0:
        return ""

    parts: list[str] = []
    total = 0

    while total < length:
        sentence = rng.choice(_CORPUS)
        parts.append(sentence)
        total += len(sentence)

    return "".join(parts)[:length]


def _split_text(text: str, size: int) -> list[str]:
    if not text:
        return [""]

    return [
        text[index:index + size]
        for index in range(0, len(text), size)
    ]
