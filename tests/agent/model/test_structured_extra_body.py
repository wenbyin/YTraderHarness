"""
结构化输出的"关思考"适配（思考模式拒绝强制 tool_choice）。

背景：有些端点默认跑在思考模式下，而思考模式拒绝带强制 tool_choice 的请求 ——
那正是 `with_structured_output(method="function_calling")` 发出的东西。实测：

    400 Thinking mode does not support this tool_choice

于是"要结构化输出"的**那一次调用**要额外并进一段 extra_body 把思考关掉。

三块：

    组装    纯函数 thinking_off_extra_body：两个判据缺一不可
    绑定    _bind 真的把它传给了 with_structured_output
    报文    传进去的东西真的会进请求体（**不联网**）

⚠️ 第三块为什么值得单独写
----------------------
"kwargs 被静默丢掉"是这条链路上唯一没有报错的失效方式：`_bind` 传了、
`with_structured_output` 收了、然后什么都没发生。所以这里不看代码怎么写，
而是把 langchain 真正会发的 payload 取出来断言（`_get_request_payload`
的产物就是交给 SDK 的那一份，SDK 再把它浅并进 JSON body）。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from langchain_core.messages import HumanMessage

from app.agent.model import (
    LangChainProviderAdapter,
    ModelDefinition,
    ModelRequest,
)
from app.agent.model.adapters import thinking_off_extra_body
from app.agent.model.provider import ModelClient

#: deepseek 那条实测出来的值（见 model.yaml 的 structured_output_extra_body）。
_THINKING_OFF = {"thinking": {"type": "disabled"}}


def _definition(**overrides) -> ModelDefinition:

    return ModelDefinition.make(
        "deepseek",
        "deepseek-v4-flash",
        **{"structured_output_extra_body": _THINKING_OFF, **overrides},
    )


def _request(*, response_format=None) -> ModelRequest:

    return ModelRequest(
        messages=(HumanMessage("你好"),),
        response_format=response_format,
    )


class _FakeChatModel:
    """只实现适配器会用到的那几个方法。"""

    def __init__(self) -> None:
        self.bound_tools = None
        self.output_format = None
        self.structured_kwargs: dict = {}
        #: 纯文本那条路走 `bind(**extra)` —— 单独记，免得与上面那个混在一起
        #: （`_bind` 的三条分支互斥，但测试要能指出是哪一条生效了）。
        self.bound_kwargs: dict = {}

    def bind_tools(self, tools):
        self.bound_tools = list(tools)
        return self

    def with_structured_output(self, fmt, **kwargs):
        self.output_format = fmt
        self.structured_kwargs = dict(kwargs)
        return self

    def bind(self, **kwargs):
        self.bound_kwargs = dict(kwargs)
        return self

    async def ainvoke(self, messages, config=None):
        return SimpleNamespace(content="ok", usage_metadata=None)


# ============================================================
# 组装：纯函数
# ============================================================


def test_declared_and_needed_yields_extra_body() -> None:
    """模型声明了 + 这次要结构化输出 -> 带上。"""

    body = thinking_off_extra_body(
        _definition(), _request(response_format=SimpleNamespace)
    )

    assert body == {"extra_body": _THINKING_OFF}


def test_not_needed_yields_nothing() -> None:
    """这次不要结构化输出 -> 什么都不加（推理 / 摘要照旧走思考模式）。"""

    body = thinking_off_extra_body(_definition(), _request())

    assert body == {}


def test_not_declared_yields_nothing() -> None:
    """模型没声明 -> 什么都不加（qwen / kimi 这些不冲突的端点）。"""

    plain = ModelDefinition.make("qwen", "qwen3.7-flash")

    body = thinking_off_extra_body(
        plain, _request(response_format=SimpleNamespace)
    )

    assert body == {}


def test_returned_mapping_is_a_copy() -> None:
    """
    返回的是**拷贝**：调用方（或 langchain）就地改它不该污染模型定义 ——
    定义进了客户端缓存，改了会让"缓存过的客户端"与"定义"悄悄不一致。
    """

    definition = _definition()

    body = thinking_off_extra_body(
        definition, _request(response_format=SimpleNamespace)
    )

    body["extra_body"]["thinking"]["type"] = "enabled"

    assert definition.structured_output_extra_body == _THINKING_OFF


# ============================================================
# 一律关：按模型声明，不看调用形状
# ============================================================


def test_always_off_covers_plain_text_calls() -> None:
    """
    声明了 `disable_thinking` 的模型，**纯文本调用**也关思考。

    这条钉的是 2026-10-13 那次泄漏的**形状**：既不带 tools、也不带
    response_format、调用方也没要求 —— 三条形状判据一条都不占，于是它照旧
    跑在思考模式下，把工具调用写成了正文（run 51b31758，614 字全是标记）。
    """

    body = thinking_off_extra_body(
        _definition(disable_thinking=True), _request()
    )

    assert body == {"extra_body": _THINKING_OFF}


def test_always_off_without_the_extra_body_is_a_config_error() -> None:
    """
    置了开关却没有那段 extra_body -> **装配期就抛**，不留到运行期。

    那样写的话它是"静默不生效"：请求照发、思考照跑、什么都不报 —— 而这里
    正是为了防"看起来装好了其实没拦住"。
    """

    with pytest.raises(ValueError) as error:

        ModelDefinition.make("deepseek", "x", disable_thinking=True)

    assert "structured_output_extra_body" in str(error.value)


# ============================================================
# 绑定：_bind 真的传下去了
# ============================================================


async def test_bind_passes_extra_body_only_for_structured_calls() -> None:
    """
    要结构化输出时传；不要时不传 —— 后者才是重点：不传的话这个模型的
    推理 / 摘要调用照旧走思考模式，而"每次都带"会把它们一起关掉。

    两个用例各用一个全新的假客户端：它是**有状态**的（记录上一次收到
    什么），复用会让第二条断言看到第一条留下的值。
    """

    adapter = LangChainProviderAdapter()

    async def _invoke(*, response_format) -> _FakeChatModel:
        model = _FakeChatModel()
        client = ModelClient(
            definition=_definition(),
            client=model,
            adapter=adapter,
            timeout=5.0,
        )
        await adapter.ainvoke(
            client, _request(response_format=response_format), timeout=5.0
        )
        return model

    structured = await _invoke(response_format=SimpleNamespace)
    assert structured.structured_kwargs["extra_body"] == _THINKING_OFF

    plain = await _invoke(response_format=None)
    assert plain.structured_kwargs == {}
    assert plain.output_format is None


async def test_bind_passes_extra_body_for_plain_text_calls() -> None:
    """
    **纯文本调用**也要带上（`disable_thinking` 的全部意义就在这里）。

    ⚠️ 这条钉的是一个真实漏过的分支
    -----------------------------
    `_bind` 原来只有两条分支（`bind_tools` / `with_structured_output`），
    纯文本调用两条都不占 —— 于是 `thinking_off_extra_body` 算出来的
    `extra` 被丢掉，模型照旧跑在思考模式下。症状：配置里写着"一律关"、
    读起来像装好了，实际**这一类调用一条都没拦住**（2026-10-13 的实测：
    带工具的那轮没推理，纯文本那轮有 —— 差别在这里）。
    """

    adapter = LangChainProviderAdapter()
    model = _FakeChatModel()

    client = ModelClient(
        definition=_definition(disable_thinking=True),
        client=model,
        adapter=adapter,
        timeout=5.0,
    )

    await adapter.ainvoke(client, _request(), timeout=5.0)

    assert model.bound_kwargs == {"extra_body": _THINKING_OFF}
    # 另外两条分支一个都不该被动到。
    assert model.bound_tools is None
    assert model.output_format is None


async def test_bind_does_not_touch_plain_calls_without_the_switch() -> None:
    """
    没声明开关的模型，纯文本调用一个多余参数都不加。

    这条是上一条的对照组：`extra` 为空时**不能**去 bind —— 空 bind 无害，
    但"纯文本也想加点什么"的下一步就是把不该加的加上去，而多一个
    extra_body 在某些端点上就是一条 400。
    """

    adapter = LangChainProviderAdapter()
    model = _FakeChatModel()

    client = ModelClient(
        definition=_definition(),
        client=model,
        adapter=adapter,
        timeout=5.0,
    )

    await adapter.ainvoke(client, _request(), timeout=5.0)

    assert model.bound_kwargs == {}


async def test_bind_leaves_other_models_alone() -> None:
    """没声明的模型照旧 —— 一次多余的 extra_body 就是一条 400。"""

    adapter = LangChainProviderAdapter()
    model = _FakeChatModel()

    client = ModelClient(
        definition=ModelDefinition.make("qwen", "qwen3.7-flash"),
        client=model,
        adapter=adapter,
        timeout=5.0,
    )

    await adapter.ainvoke(
        client,
        _request(response_format=SimpleNamespace),
        timeout=5.0,
    )

    assert model.structured_kwargs == {}


# ============================================================
# 报文：真的会进请求体
# ============================================================


def test_extra_body_reaches_the_request_payload() -> None:
    """
    真 langchain 客户端：传进去的 extra_body 出现在 payload 里。

    这条钉住的是最后一段 —— `with_structured_output` 的 kwargs 会经
    `bind_tools(**bind_kwargs)` 落到第 0 步的绑定上，再由
    `_get_request_payload` 并进 payload（`{**default_params, **kwargs}`）。
    少了它，"传了但被丢掉"在别的用例里看不出来。

    断言的是**交给 SDK 的那一份**，不是网线上的字节：SDK 之后还会
    `_merge_mappings(json_data, extra_json)` 把它并进 JSON body ——
    那一层不该由 langchain 的用例来验。
    """

    from langchain_deepseek import ChatDeepSeek

    chat = ChatDeepSeek(
        model="deepseek-v4-flash",
        api_key="test-not-a-real-key",
    )

    adapter = LangChainProviderAdapter()

    client = ModelClient(
        definition=_definition(),
        client=chat,
        adapter=adapter,
        timeout=5.0,
    )

    runnable = adapter._bind(
        client, _request(response_format=SimpleNamespace)
    )

    # 结构化输出返回的是 RunnableSequence，第 0 步是绑好写法的模型。
    # 形状变了就该在这里炸 —— 正是我们要知道的。
    bound = getattr(runnable, "steps", [None])[0]
    assert bound is not None, (
        f"with_structured_output 的返回形状变了：{type(runnable).__name__}"
    )

    payload = chat._get_request_payload(
        [HumanMessage("你好")], **bound.kwargs
    )

    assert payload["extra_body"] == _THINKING_OFF

    # 顺带确认我们**没有**把强制 tool_choice 弄丢 —— 走的就是这条路，
    # 所以必须靠关思考来解决，而不是换 method。
    assert "tool_choice" in payload


def test_plain_call_extra_body_reaches_the_request_payload() -> None:
    """
    纯文本那条路传进去的东西**真的会进请求体**（不联网）。

    与上面那条同一条理由：`bind(**extra)` 也可能是"传了但被丢掉"，而那条路
    上没有任何报错。这里断言的是交给 SDK 的那一份 payload。
    """

    from langchain_deepseek import ChatDeepSeek

    chat = ChatDeepSeek(
        model="deepseek-v4-flash",
        api_key="test-not-a-real-key",
    )

    adapter = LangChainProviderAdapter()

    client = ModelClient(
        definition=_definition(disable_thinking=True),
        client=chat,
        adapter=adapter,
        timeout=5.0,
    )

    runnable = adapter._bind(client, _request())

    # 纯文本这条路返回的是 RunnableBinding，kwargs 挂在它上面。
    assert getattr(runnable, "kwargs", None) == {"extra_body": _THINKING_OFF}

    payload = chat._get_request_payload(
        [HumanMessage("你好")], **runnable.kwargs
    )

    assert payload["extra_body"] == _THINKING_OFF
    # 纯文本调用没有工具也没有 tool_choice —— 这条分支不该顺手带上别的。
    assert "tool_choice" not in payload
    assert not payload.get("tools")


def test_real_model_definition_carries_the_declaration() -> None:
    """
    model.yaml 里那条声明真的装配进了定义。

    没有这条，上面全部用例都能通过而生产环境依旧 400：那三个用例喂的是
    自己造的定义。
    """

    from app.agent.model import get_model_registry

    definition = get_model_registry().definition(
        "deepseek:deepseek-v4-flash"
    )

    assert definition.structured_output_extra_body == _THINKING_OFF, (
        "model.yaml 里 deepseek-v4-flash 的 structured_output_extra_body "
        "没装配上 —— 分类降级会回来。"
    )

    # ⚠️ `disable_thinking` **没有**声明，这是刻意的（2026-10-13 撤掉）
    # -----------------------------------------------------------------
    # 它曾经是 True：纯文本那一类调用一条形状判据都不占，于是思考照开、
    # 模型把工具调用写成正文。撤掉是因为产品要**给用户看思维链** —— 一律关
    # 就等于 reasoning_content 那条链路永远没有数据，而这个模型是主循环
    # （reasoning_default）唯一的 primary。
    #
    # 代价（纯文本轮的 DSML 泄漏）由出口的闸兜底，见 model.yaml 那段取舍。
    # 读到"那次泄漏"的说明时别顺势以为这个开关还开着。
    assert definition.disable_thinking is False, (
        "deepseek-v4-flash 又声明了 disable_thinking —— 纯文本调用会不再产出"
        "思维链，reasoning_content / message.reasoning.* 在聊天里会一直是空的。"
        "要确认这是有意的（拿看不到推理换掉那一类泄漏的风险），见 model.yaml。"
    )
