"""DeepSeek chat 客户端：把 assistant 的 `reasoning_content` 带回请求。

为什么需要这个子类
------------------
DeepSeek 的思考模式要求：多轮对话里，assistant 消息的
`reasoning_content` 必须原样回传；只传 content 会 400：

    The `reasoning_content` in the thinking mode must be passed back to the API.

langchain 的 OpenAI 兼容转换 `_convert_message_to_dict` **不读**
`additional_kwargs["reasoning_content"]`，所以历史里的思维链会被丢掉。
`ChatDeepSeek` 自己也没有覆盖这一点（它只在解析响应时写入）。

⚠️ 只在思考模式未显式关闭时注入
-----------------------------
调用方通过 `extra_body={"thinking": {"type": "disabled"}}` 关思考时
（结构化输出 / 带工具 / 显式 disable_thinking），请求体里已经带了
`extra_body.thinking`，这时不回传 reasoning_content：
非思考模式的请求没有这个字段，回传它没有意义，也不该赌 provider 一定忽略。

实现方式
--------
覆盖 `_get_request_payload`：先让父类生成 payload（它已经完成
`AIMessage -> OpenAI dict` 的转换），再按消息顺序把
`additional_kwargs["reasoning_content"]` 补回对应的 assistant 项。
payload 里的 messages 与源 messages 一一对应，顺序不变。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.messages import AIMessage
from langchain_deepseek import ChatDeepSeek

__all__ = ["ReasoningChatDeepSeek"]


class ReasoningChatDeepSeek(ChatDeepSeek):
    """DeepSeek 客户端：思考模式下保留 assistant 的 reasoning_content。"""

    def _get_request_payload(
            self,
            input_: Any,
            *,
            stop: list[str] | None = None,
            **kwargs: Any,
    ) -> dict[str, Any]:

        payload = super()._get_request_payload(
            input_, stop=stop, **kwargs
        )

        if _thinking_disabled(payload):
            return payload

        try:
            messages = self._convert_input(input_).to_messages()
        except Exception:  # noqa: BLE001 - 转换失败就交给父类的 payload
            return payload

        payload_messages = payload.get("messages")

        if not isinstance(payload_messages, list):
            return payload

        for source, target in zip(messages, payload_messages):

            if not isinstance(source, AIMessage):
                continue

            if not isinstance(target, dict):
                continue

            if target.get("role") != "assistant":
                continue

            target["reasoning_content"] = _reasoning_content(source)

        return payload


def _reasoning_content(source: AIMessage) -> str:
    """
    取 assistant 的思维链；没有就返回空串。

    两种存放形状都要认（与 agent/context/processor/messages.py 的
    `message_reasoning` 一致）：

        additional_kwargs["reasoning_content"]     非流式 / 旧形状
        content 里 type=reasoning 块的 reasoning    流式 v3 形状

    ⚠️ **没有 reasoning 信息也必须回传空串，不能省略字段**
    ------------------------------------------------------
    DeepSeek 在思考模式下要求**每一条** assistant 消息都带
    `reasoning_content`。实测：传空串可以，缺字段 400。

    这条不只是为了「有思维链」的消息：分类器合成的
    `enter_plan_mode` AIMessage、非思考轮留下的回答、别的 provider
    产出的历史消息，都没有 reasoning 信息。只要它们还在 messages 里，
    下一轮思考模式的请求就必须把它们补成空串 —— 否则整次调用 400：

        The `reasoning_content` in the thinking mode must be passed back.
    """

    extra = getattr(source, "additional_kwargs", None)

    if isinstance(extra, Mapping) and "reasoning_content" in extra:
        value = extra.get("reasoning_content")

        if isinstance(value, str):
            return value

    content = getattr(source, "content", None)

    if not isinstance(content, list):
        return ""

    chunks: list[str] = []

    for block in content:

        if not isinstance(block, Mapping):
            continue

        if block.get("type") != "reasoning":
            continue

        text = block.get("reasoning")

        if not isinstance(text, str):
            text = block.get("text")

        if isinstance(text, str):
            chunks.append(text)

    return "".join(chunks)


def _thinking_disabled(payload: Mapping[str, Any]) -> bool:
    """请求体里是否显式带 `thinking: {type: disabled}`。"""

    extra = payload.get("extra_body")

    if not isinstance(extra, Mapping):
        return False

    thinking = extra.get("thinking")

    return (
        isinstance(thinking, Mapping)
        and str(thinking.get("type") or "") == "disabled"
    )
