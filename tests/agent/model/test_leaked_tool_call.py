"""
"模型把工具调用写成正文"时的出口过滤。

⚠️ 样本是**拼出来**的，不是抄进来的
--------------------------------
第一版把标记当字面量抄了进来，抄成了**半角**竖线，而库里存的是**全角**
（U+FF5C）。两者在等宽字体里几乎一样，代码看着完全正常 —— 直到拿真实数据
比对，才发现一处都匹配不上（hits=0）。所以下面一律从 `LEAK_MARKER_TOKEN`
拼，换个宽度就能造出另一种形态，而"两种都认"本身也成了被测行为。

数据来源：`agent_run:51b31758` —— 614 字里全是标记，149 条 delta 与
completed 的全文都是它，而 finish_reason 是 stop、运行表记 completed。
"""

from __future__ import annotations

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from app.agent.model import (
    LangChainProviderAdapter,
    ModelDefinition,
    ModelRequest,
)
from app.agent.model.errors import ModelProviderError
from app.agent.model.leaked_tool_call import (
    LEAK_MARKER_TOKEN,
    LeakGate,
    strip_leaked_tool_calls,
)
from app.agent.model.provider import ModelClient

#: 存储里那一种（全角竖线）。
_FULLWIDTH = "｜"


def _marker_sample(*, width: str = "|") -> str:
    """那次泄漏的真实形状。"""

    token = LEAK_MARKER_TOKEN.replace("|", width)
    opened = "<" + token
    closed = "</" + token

    return (
        opened + " calls>\n"
        + opened + ' invoke name="memory_atom_search">\n'
        + opened + ' parameter name="query" string="true">爬山 大蜀山</'
        + token
        + " parameter>\n"
        + closed + " invoke>\n"
        # ⚠️ 真实那次是**两个** invoke 块（模型想同时查两件事）。少一个的话
        # "区段之间那点空白会不会漏出去"这件事就测不到。
        + opened + ' invoke name="memory_atom_search">\n'
        + opened + ' parameter name="query" string="true">合肥 户外</'
        + token
        + " parameter>\n"
        + closed + " invoke>\n"
        + closed + " calls>"
    )


# ============================================================
# 整段文本
# ============================================================


@pytest.mark.parametrize("width", ["|", _FULLWIDTH])
def test_a_pure_leak_leaves_nothing(width: str) -> None:
    """整条回复只有标记（真实形状）-> 一个字都不剩。两种竖线宽度都要认。"""

    cleaned, hits = strip_leaked_tool_calls(_marker_sample(width=width))

    assert hits == 3
    assert cleaned.strip() == ""


def test_prose_around_a_leak_survives() -> None:
    """正文 + 泄漏 + 正文：两头都要留下来（中间允许剩点空白）。"""

    text = "先说结论：没查到。\n" + _marker_sample() + "\n换个说法再查吗？"

    cleaned, hits = strip_leaked_tool_calls(text)

    assert hits == 3
    assert cleaned.startswith("先说结论：没查到。")
    assert cleaned.endswith("换个说法再查吗？")
    # 一个标记字符都不许剩 —— 别只断言"没有 DSML"：竖线与尖括号才是
    # 最容易漏出去的那几个（实测漏过 6 个）。
    assert "<" not in cleaned and "|" not in cleaned


@pytest.mark.parametrize(
    "text",
    [
        "紫蓬山海拔 284 米。",
        "公式 a|b 是位或运算。",
        "结尾正好一个竖线 |",
        "结尾正好一个左尖括号 <",
        "两个竖线 || 也不是标记。",
    ],
)
def test_normal_text_is_untouched(text: str) -> None:
    """
    正常正文（含孤立的竖线与尖括号）一字符不动。

    ⚠️ 这条比它看起来重要：那两行全角/半角的差别就是靠这类样本钉住的 ——
    一个"什么都不匹配"的过滤器对正常文本是**完美透明**的。
    """

    assert strip_leaked_tool_calls(text) == (text, 0)


# ============================================================
# 流式闸
# ============================================================


def test_char_by_char_feeding_matches_the_whole_text_result() -> None:
    """
    一个字一个字喂（真实分片就是这么碎的）与整段过滤结果**完全一致**。

    这是"必须是状态机"那句话的实测：那次泄漏的分片是 1~10 个字，标记跨在
    分片之间。
    """

    text = "先说结论：没查到。\n" + _marker_sample() + "\n换个说法再查吗？"

    gate = LeakGate()
    streamed = "".join(gate.feed(ch) for ch in text) + gate.flush()

    assert streamed == strip_leaked_tool_calls(text)[0]
    assert gate.hits == 3


def test_nothing_marker_like_is_released_while_a_leak_is_open() -> None:
    """
    逐字喂整条泄漏时，放出去的只有**空白** —— 一个标记字符都不许出去。

    （不是"一个字都没有"：两个 invoke 块之间那个换行是在区段**关闭之后**
    放的，它照常出去。所以要断言的是"出去的里面没有标记字符"。）
    """

    gate = LeakGate()

    released = "".join(gate.feed(ch) for ch in _marker_sample()) + gate.flush()

    assert released.strip() == ""
    assert "<" not in released and "|" not in released


def test_a_trailing_open_angle_is_held_then_released() -> None:
    """
    正常正文以 `<` 结尾时**扣一下、最后原样放出去**，不吞掉。

    这是 `flush` 存在的理由：闸必须分得清"扣住的可能是前缀"与"真的是泄漏"。
    代价就是那一个字符要等到流结束才发出去 —— 所以这里两个都要断言：过程中
    被扣住（那不是 bug），结束时一个字不少。
    """

    gate = LeakGate()

    assert gate.feed("答案是 3 ") == "答案是 3 "
    assert gate.feed("<") == ""          # 扣住：它可能是标记的开头
    assert gate.flush() == "<"           # 流结束：确认不是标记，放出去

    assert gate.hits == 0


# ============================================================
# 适配器出口（落到 state / DB 的那一份）
# ============================================================


class _FakeChatModel:
    """只实现适配器会用到的那几个方法。"""

    def __init__(self, text: str) -> None:
        self._message = AIMessage(content=text)

    def bind_tools(self, tools):
        return self

    with_structured_output = None

    async def ainvoke(self, messages, config=None):
        return self._message


def _client(text: str) -> ModelClient:

    return ModelClient(
        definition=ModelDefinition.make("deepseek", "deepseek-v4-flash"),
        client=_FakeChatModel(text),
        adapter=LangChainProviderAdapter(),
    )


async def test_adapter_raises_when_the_whole_answer_is_a_leak() -> None:
    """
    整条回复都是标记 -> **抛**，不是一个空正文的"成功"。

    与 `_astream` 里"流里一块都没有就抛"同一个立场：空消息会以 completed
    收尾，而模型其实什么都没说 —— 那种成功最难查。
    """

    adapter = LangChainProviderAdapter()

    with pytest.raises(ModelProviderError) as error:

        await adapter.ainvoke(
            _client(_marker_sample()),
            ModelRequest(messages=(HumanMessage("紫蓬山怎么样"),)),
            timeout=0,
        )

    assert "工具调用写成了正文" in error.value.message


async def test_adapter_keeps_the_prose_and_drops_the_leak() -> None:
    """
    部分泄漏：正文留下、标记去掉，而且**这条不算失败**（没有抛）。

    只断言到"正文还在、标记字符一个不剩"，不断言具体留了几个换行 ——
    那些是分片边界的产物，钉死它只会让将来调整规则时到处红。
    """

    adapter = LangChainProviderAdapter()

    result = await adapter.ainvoke(
        _client("先说结论：没查到。\n" + _marker_sample()),
        ModelRequest(messages=(HumanMessage("紫蓬山怎么样"),)),
        timeout=0,
    )

    content = result.message.content

    assert content.startswith("先说结论：没查到。")
    assert "<" not in content and "|" not in content
