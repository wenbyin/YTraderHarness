"""
Mock model server 的冒烟测试。

跑法：

    .venv/Scripts/python.exe -m app.mock.mock_model_smoke

覆盖四条路径：

    1. 非流式文本
    2. 流式文本（SSE + [DONE]）
    3. 分类的结构化输出（强制 tool_choice）
    4. 带 tools 的自动工具调用

不依赖主应用、数据库、Redis，也不访问真实模型。
"""

from __future__ import annotations

import asyncio
import json
import sys
from typing import Any

import aiohttp

from app.common.mock_model_config import (
    MockModelConfig,
    MockToolCallConfig,
    NormalSpec,
)
from app.mock.model_server import MockModelServer

HOST = "127.0.0.1"
PORT = 18081
BASE = f"http://{HOST}:{PORT}"


def check(label: str, condition: bool) -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")

    if not condition:
        raise AssertionError(label)


async def _post(
        session: aiohttp.ClientSession,
        payload: dict[str, Any],
        *,
        stream: bool = False,
) -> Any:
    async with session.post(
        f"{BASE}/v1/chat/completions",
        json=payload,
    ) as response:
        if response.status != 200:
            text = await response.text()
            raise AssertionError(
                f"HTTP {response.status}: {text}"
            )

        if stream:
            return await response.text()

        return await response.json()


async def _run() -> None:
    async with aiohttp.ClientSession() as session:

        # 1. 非流式文本
        body = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {"role": "user", "content": "你好，介绍一下你自己"}
                ],
                "stream": False,
            },
        )
        text = body["choices"][0]["message"]["content"]

        check("非流式返回非空正文", bool(text))
        check("非流式 finish_reason=stop",
              body["choices"][0]["finish_reason"] == "stop")
        check("非流式带 usage",
              body["usage"]["completion_tokens"] > 0)

        # 2. 流式文本
        raw = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {"role": "user", "content": "流式回复测试"}
                ],
                "stream": True,
            },
            stream=True,
        )
        frames = [
            line[len("data: "):]
            for line in raw.splitlines()
            if line.startswith("data: ")
        ]
        chunks = [
            json.loads(frame)
            for frame in frames
            if frame != "[DONE]"
        ]
        content = "".join(
            choice["delta"].get("content") or ""
            for chunk in chunks
            for choice in chunk.get("choices", [])
        )

        check("流式收到 [DONE]", frames[-1] == "[DONE]")
        check("流式至少有 2 个 chunk", len(chunks) >= 2)
        check("流式正文非空", bool(content))
        check("流式带 usage", "usage" in chunks[-1])

        # 3. 分类：强制 tool_choice
        classification_schema = {
            "type": "object",
            "properties": {
                "behavior": {"type": "string"},
                "execution_plan": {"type": "string"},
                "capability_needs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "capability_config": {"type": "object"},
                "retrieve_l0": {"type": "string"},
                "retrieve_l1": {"type": "string"},
            },
            "required": [
                "behavior",
                "execution_plan",
                "capability_needs",
                "capability_config",
                "retrieve_l0",
                "retrieve_l1",
            ],
        }

        body = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Agent Builder 的分类调用：判行为 / 执行方案 / "
                            "能力需求。可选能力清单：\n"
                            "- retrieval：检索\n"
                            "- memory：记忆\n"
                        ),
                    },
                    {"role": "user", "content": "帮我查一下昨天的行情"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "_ClassificationOutput",
                            "parameters": classification_schema,
                        },
                    }
                ],
                "tool_choice": {
                    "type": "function",
                    "function": {"name": "_ClassificationOutput"},
                },
                "stream": False,
            },
        )
        calls = body["choices"][0]["message"]["tool_calls"]
        arguments = json.loads(calls[0]["function"]["arguments"])

        check("分类返回 tool_calls",
              body["choices"][0]["finish_reason"] == "tool_calls")
        check("分类 tool 名正确",
              calls[0]["function"]["name"] == "_ClassificationOutput")
        check("分类 behavior 合法",
              arguments["behavior"] in {
                  "chat", "qa", "analysis", "creation", "modification",
                  "research", "execution", "planning",
                  "decision_support", "other",
              })
        check("分类 execution_plan 固定 react",
              arguments["execution_plan"] == "react")
        check("分类 capability_needs 是列表",
              isinstance(arguments["capability_needs"], list))
        check("分类 retrieve_l0/l1 是字符串",
              isinstance(arguments["retrieve_l0"], str)
              and isinstance(arguments["retrieve_l1"], str))

        # 4. 自动工具调用
        body = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {"role": "user", "content": "查一下行情"}
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "query": {"type": "string"},
                                    "limit": {
                                        "type": "integer",
                                        "minimum": 1,
                                        "maximum": 10,
                                    },
                                },
                                "required": ["query", "limit"],
                            },
                        },
                    }
                ],
                "tool_choice": "auto",
                "stream": False,
            },
        )
        calls = body["choices"][0]["message"].get("tool_calls") or []

        check("自动工具调用命中", len(calls) == 1)

        if calls:
            tool_args = json.loads(calls[0]["function"]["arguments"])
            check("工具名正确", calls[0]["function"]["name"] == "lookup")
            check("工具必填参数齐全",
                  "query" in tool_args and "limit" in tool_args)
            check("工具整数参数被夹到范围",
                  1 <= tool_args["limit"] <= 10)

        # 5. 记忆后台任务：返回解析器能接受的空结果
        memory_extract = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是专业的\"情境切分与记忆提取专家\"。",
                    },
                    {"role": "user", "content": "待提取的新消息"},
                ],
                "stream": False,
            },
        )
        extract_text = memory_extract["choices"][0]["message"]["content"]

        check("memory_extract 返回空数组",
              json.loads(extract_text) == [])

        memory_dedup = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "冲突检测：输出 record_id 与 store / skip / update / merge。",
                    },
                    {"role": "user", "content": "判定"},
                ],
                "stream": False,
            },
        )
        dedup_text = memory_dedup["choices"][0]["message"]["content"]

        check("memory_dedup 返回空数组",
              json.loads(dedup_text) == [])

        memory_scene = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "你是记忆整合架构师。现有场景文件见下。",
                    },
                    {"role": "user", "content": "整合"},
                ],
                "stream": False,
            },
        )
        scene = json.loads(
            memory_scene["choices"][0]["message"]["content"]
        )

        check("memory_scene 返回空 operations",
              scene["operations"] == []
              and scene["persona_update_request"] is False)

        memory_persona = await _post(
            session,
            {
                "model": "mock-chat",
                "messages": [
                    {
                        "role": "system",
                        "content": "User Narrative Profile；返回 no_change。",
                    },
                    {"role": "user", "content": "生成画像"},
                ],
                "stream": False,
            },
        )
        persona = json.loads(
            memory_persona["choices"][0]["message"]["content"]
        )

        check("memory_persona 返回 no_change=true",
              persona["no_change"] is True)

        # 5.1 模型名优先于 system prompt：主 agent 的上下文里可能带着
        #     "# User Narrative Profile" / "no_change" 之类的画像文本，
        #     但它们不能被误判成 memory_persona。
        reasoning = await _post(
            session,
            {
                "model": "mock-role-reasoning",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "# User Narrative Profile\n"
                            "no_change: true\n"
                            "这是主 agent 上下文里注入的画像。"
                        ),
                    },
                    {"role": "user", "content": "今天行情怎么样"},
                ],
                "stream": False,
            },
        )
        reasoning_text = reasoning["choices"][0]["message"]["content"]

        check("reasoning 不会被画像文本误判成 memory_persona",
              reasoning_text != '{"no_change": true}'
              and bool(reasoning_text))

        persona_by_model = await _post(
            session,
            {
                "model": "mock-role-memory-persona-extract",
                "messages": [
                    {
                        "role": "system",
                        "content": "任意 system prompt，不需要靠它猜角色。",
                    },
                    {"role": "user", "content": "生成画像"},
                ],
                "stream": False,
            },
        )
        persona_model_payload = json.loads(
            persona_by_model["choices"][0]["message"]["content"]
        )

        check("model 名明确指定 memory_persona",
              persona_model_payload["no_change"] is True)

        # 6. 统计
        async with session.get(f"{BASE}/stats") as response:
            stats = await response.json()

        check("stats 有请求计数", stats["requests"] >= 10)
        check("stats 有实际 TTFT", stats["avg_ttft_ms"] is not None)


def main() -> int:
    # capability -> tool 的 mock 解析：忽略 code，固定池里随机选 N 个。
    from app.mock.capability import select_mock_tool_keys
    from app.services.tool.registry.registry import get_tool_registry

    selected = select_mock_tool_keys(["retrieval", "memory"])
    pool = {
        "mock.echo",
        "mock.lookup",
        "mock.calc",
        "mock.current_time",
        "mock.ping",
    }

    check("mock capability 选中 1~5 个",
          1 <= len(selected) <= 5)
    check("mock capability 只从固定池里选",
          set(selected) <= pool)

    specs = get_tool_registry().scan(packages=("app.mock",))
    registered = {
        spec.tool_key
        for spec in specs
        if spec.tool_key.startswith("mock.")
    }

    check("五个 mock 工具已登记", registered == pool)

    config = MockModelConfig(
        enabled=True,
        host=HOST,
        port=PORT,
        seed=7,
        log_requests=False,
        execution_plan="react",
        tool_call=MockToolCallConfig(
            probability=1.0,
            count=NormalSpec(mean=1, std=0, min=1, max=1),
        ),
        capability_pool=["retrieval", "memory"],
    )

    server = MockModelServer(config)
    server.start()

    try:
        asyncio.run(_run())

    finally:
        server.stop()

    print("\nMock model smoke: ALL OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
