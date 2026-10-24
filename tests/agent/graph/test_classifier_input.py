"""
Agent Builder 交给分类 ContextBuilder 的状态快照。

分类需要历史摘要作为背景，但**不能**在分类阶段生成新摘要：

    include_summary=True     注入 state 里已有的那份
    allow_compaction=False   钉死生成分支

主 agent 的 context_policy 可能带着 allow_compaction=True；如果这里
不显式覆盖，分类调用会先多调一次摘要模型 —— 延迟与成本都翻倍，
而观测里看起来只是"分类慢了一点"。
"""

from __future__ import annotations

from app.agent.graph.react import agent_builder_node as node


class _FakePromptProvider:
    def node_prompt(self, model_role, /, **variables) -> str:
        return "分类提示词"


def test_graph_state_enables_readonly_summary(
        monkeypatch,
) -> None:
    monkeypatch.setattr(
        node, "get_prompt_provider", lambda: _FakePromptProvider()
    )

    state = {
        "conversation_id": "c1",
        "user_id": "u1",
        "summary": {
            "text": "此前聊过 AAPL。",
            "revision": 1,
            "summarized_until_message_id": "m1",
        },
        # 主 agent 的策略：压缩是开着的。
        "context_policy": {"allow_compaction": True},
    }

    snapshot = node._graph_state(state, [])

    assert snapshot["summary"]["text"] == "此前聊过 AAPL。"

    policy = snapshot["context_policy"]

    assert policy["include_summary"] is True
    assert policy["allow_compaction"] is False
    assert policy["include_history"] is True
    assert policy["min_recent_turns"] == 1
    assert policy["max_recent_turns"] == 1
