"""
前置验证：切 v3 会不会改掉**模型调用**路径。

为什么必须先跑这个
------------------
v3 的 mux 固定注册 `MessagesTransformer`（`required_stream_modes =
("messages",)`，见 langgraph/stream/transformers.py:198），而
`_collect_stream_modes` 取所有 transformer 的并集来决定底层 `astream`
请求哪些模式（pregel/main.py:398-414）。`transformers=` 参数**只能加不能减**
（main.py:3589-3600 是写死的 factories 列表）。所以 `version="v3"` 下
`"messages"` **必然**在 stream_modes 里，没有开关能关掉它。

而 "messages" 一旦在 stream_modes 里，pregel 就挂上 `StreamMessagesHandlerV2`
（main.py:2810-2827），它继承 `_V2StreamingCallbackHandler`。于是
`BaseChatModel` 的每一次 invoke/ainvoke 都会命中
`_should_use_protocol_streaming`（langchain_core/.../chat_models.py:587-620）——
它**优先于** `_should_stream`（同文件 :1943 的注释："preferred over v1 when
any attached handler opts in"），把调用改道到内容块事件发生器，从事件里聚合出
`ChatResult`；聚合不出消息时直接抛：

    RuntimeError("v2 stream finished without producing a message")
    （chat_models.py:1961-1963）

对照本仓库
----------
`app/agent/model/adapters.py` 的 `_can_stream`（:241-266）**刻意**让
`response_format` 非空的调用退回 `runnable.ainvoke`，理由写在函数注释里：
"with_structured_output 的流式各家实现不一致，有些根本不支持"。改道会把这条
保护整个绕过 —— 结构化输出的调用也会走流式聚合。

判定
----
三格全绿 -> 可以进下一步（换驱动）。
structured output 那一格红了 -> **停下来讨论**，不要换驱动。

⚠️ 本脚本不从"失败即中断"的 check 起步
--------------------------------------
仓库其它 smoke 的 `check` 一失败就 `raise`。这里刻意改成**收齐全部格再判**：
要看的是一张 3×2 的矩阵（三条路径 × 两个驱动），中断在第一个红格上就看不到
其余格是不是也坏了 —— 而"坏的范围"正是这次要问的问题。

⚠️ 需要真模型
--------------
本脚本用**真** registry + **真**模型（`model.yaml` 的 reasoning_default /
classification_default），因此需要 `app/config/providers.yaml` 的
`llm` 段与凭据可解析。
连不上就直接失败并打原始异常 —— 不跳过：这个验证的结论是"能不能换驱动"，
跳过等于没有结论。

跑法（必须用模块方式，直接跑脚本会让 `app` 包不在 sys.path 上）：
    .venv/Scripts/python.exe -m app.agent.model.v3_reroute_smoke
"""

from __future__ import annotations

import asyncio
import sys
import traceback
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from app.agent.model import ModelRequest, get_model_registry

# ============================================================
# 阳性对照：改道到底有没有发生
# ============================================================
#
# ⚠️ 没有这一层，下面的全绿是**没有意义的**。
#
# "结构化输出没坏"有两种可能的解释：
#   (a) 改道发生了，但它是良性的；
#   (b) 改道压根没触发 —— 那这次实验什么都没证明。
#
# 光看结果分不出这两者。所以要直接盯住决定改道的那个方法
# （`_should_use_protocol_streaming`，chat_models.py:587），把它包一层记下
# 每次调用的返回值。v3 那几格必须**至少命中一次且返回 True**，
# 否则整轮结论作废。

_reroute_calls: list[bool] = []

_original_should_use_protocol_streaming = (
    BaseChatModel._should_use_protocol_streaming
)


def _spy_should_use_protocol_streaming(self: Any, **kwargs: Any) -> bool:
    decision = _original_should_use_protocol_streaming(self, **kwargs)
    _reroute_calls.append(decision)
    return decision


BaseChatModel._should_use_protocol_streaming = _spy_should_use_protocol_streaming

# ============================================================
# 断言工具（收齐再判，理由见模块 docstring）
# ============================================================

#: 本轮的判定结果。空 -> 全绿。
_failures: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        _failures.append(label)
        if detail:
            print(f"         {detail}")


def section(title: str) -> None:
    print(f"\n{title}")


# ============================================================
# 探针图
# ============================================================


class _ProbeState(TypedDict, total=False):
    """节点产出的最小记录。两个驱动要比的就是这几个字段。"""

    text: str
    tool_names: list[str]
    structured_ok: bool


@tool
def get_weather(city: str) -> str:
    """查某个城市的天气。"""

    return f"{city}：晴，24℃"


class _Sentiment(BaseModel):
    """结构化输出的目标类型。刻意小 —— 只验"能不能拿到解析后的对象"。"""

    label: str = Field(description="情感标签：positive / negative / neutral")
    confidence: float = Field(description="置信度，0 到 1")


def _build_app(
        *,
        agent_type: str,
        model_role: str,
        tools: tuple[Any, ...] = (),
        response_format: Any = None,
        prompt: str = "只回复两个字：收到",
) -> Any:
    """
    单节点的极小图：唯一一件事就是调一次 `runtime.ainvoke`。

    刻意**不**复用 `graph/react` —— 那里有 planner / agent_builder / tools
    一堆节点，任何一个出问题都会污染这次实验的结论。这里只问一个问题：
    "同一个模型调用，在 v3 下还正常吗"。
    """

    registry = get_model_registry()
    runtime = registry.get_runtime(agent_type=agent_type, model_role=model_role)

    async def _probe(state: _ProbeState) -> _ProbeState:
        response = await runtime.ainvoke(
            ModelRequest(
                messages=(HumanMessage(prompt),),
                tools=tuple(tools),
                response_format=response_format,
            )
        )

        message = response.message

        return {
            "text": (getattr(message, "text", "") or ""),
            # ⚠️ 只取名字，不取参数 —— 参数在不同调用间天然会有措辞差异，
            # 拿它做对比会把"模型自由发挥"误判成"改道坏了"。
            "tool_names": [
                call.get("name")
                for call in (getattr(message, "tool_calls", None) or [])
            ],
            "structured_ok": response_format is None
                             or isinstance(message, BaseModel),
        }

    graph = StateGraph(_ProbeState)
    graph.add_node("probe", _probe)
    graph.add_edge(START, "probe")
    graph.add_edge("probe", END)

    return graph.compile(checkpointer=InMemorySaver())


# ============================================================
# 两个驱动
# ============================================================


async def _drive_ainvoke(app: Any) -> dict[str, Any]:
    """基线：今天的生产路径。"""

    return await app.ainvoke(
        {},
        {"configurable": {"thread_id": str(uuid4())}},
    )


async def _drive_v3(app: Any) -> dict[str, Any]:
    """
    受验：v3 驱动。

    ⚠️ `await app.astream_events(...)` **本身不启动图**。图由
    `await run.output()` 驱动（它内部循环 `adrive_until_done`，见
    langgraph/stream/run_stream.py:30-33）。
    """

    run = await app.astream_events(
        {},
        {"configurable": {"thread_id": str(uuid4())}},
        version="v3",
    )

    async with run:
        return await run.output()


async def _run_case(
        title: str,
        *,
        agent_type: str,
        model_role: str,
        tools: tuple[Any, ...] = (),
        response_format: Any = None,
        prompt: str = "只回复两个字：收到",
        expect_tools: list[str] | None = None,
) -> None:
    """
    跑一格：同一条路径分别过两个驱动，各自记下结果或异常。

    ⚠️ 两个驱动用**不同**的 thread_id：共用会让第二次运行读到第一次的
    checkpoint，把"改道坏没坏"和"历史串了"混在一起。
    """

    section(title)

    results: dict[str, dict[str, Any] | None] = {}
    errors: dict[str, str] = {}
    reroute: dict[str, list[bool]] = {}

    for label, drive in (("ainvoke", _drive_ainvoke), ("v3", _drive_v3)):
        app = _build_app(
            agent_type=agent_type,
            model_role=model_role,
            tools=tools,
            response_format=response_format,
            prompt=prompt,
        )

        _reroute_calls.clear()

        try:
            results[label] = await drive(app)
            errors[label] = ""
        except BaseException as exc:  # noqa: BLE001
            # 刻意捕获 BaseException：CancelledError 也要落进矩阵，
            # 否则它会以"脚本崩了"的形式出现，看不出是哪一格。
            results[label] = None
            errors[label] = f"{type(exc).__name__}: {exc}"
            print(f"\n  --- {label} 原始异常 ---")
            traceback.print_exc()
        finally:
            # 即便抛了也要把阳性对照记下来 —— "改道触发了但随后炸了"
            # 与"改道没触发所以没炸"是完全不同的两件事。
            reroute[label] = list(_reroute_calls)

    for label in ("ainvoke", "v3"):
        check(f"{label} 跑通", results[label] is not None, errors[label])

    # ---- 阳性对照：v3 必须真的走了改道那条分支 --------------------------------

    v3_calls = reroute["v3"]
    check(
        "v3 上改道分支确实被命中且返回 True（阳性对照）",
        any(v3_calls),
        f"决策序列={v3_calls}（空 = 没命中，本轮结论作废）",
    )

    baseline_calls = reroute["ainvoke"]
    check(
        "ainvoke 基线**没有**走改道（对照组的阴性结果）",
        not any(baseline_calls),
        f"决策序列={baseline_calls}（出现 True = 基线本身就被改了，"
        "两边不可比）",
    )

    # ---- 逐条对比 -----------------------------------------------------------------

    both_ok = results["ainvoke"] is not None and results["v3"] is not None

    if not expect_tools and response_format is None:
        # ⚠️ 另两格**不能**断言正文非空，两个理由各不相同：
        #   - 带工具：模型返回 tool_calls 时 content 本就是空的
        #     （实测 ainvoke 与 v3 都是 ''）。
        #   - 结构化输出：`resp.message` 是**解析后的 BaseModel**，不是
        #     AIMessage，压根没有 .text（实测两边都是 ''）。
        # 拿这两种正确的空当失败，会把"行为符合预期"误判成"改道坏了"。
        check(
            "两个驱动都产出了正文",
            both_ok
            and bool((results["ainvoke"] or {}).get("text"))
            and bool((results["v3"] or {}).get("text")),
            f"ainvoke={(results['ainvoke'] or {}).get('text')!r} "
            f"v3={(results['v3'] or {}).get('text')!r}",
        )

    if expect_tools:
        for label in ("ainvoke", "v3"):
            got = (results[label] or {}).get("tool_names") or []
            check(
                f"{label} 保留了 tool_calls",
                got == expect_tools,
                f"期望 {expect_tools}，实际 {got}",
            )

    if response_format is not None:
        # 这一格是本脚本的重点：改道若坏，症状通常是 v3 抛
        # RuntimeError("v2 stream finished without producing a message")
        # 或拿不到解析后的 BaseModel。
        check(
            "v3 也拿到了解析后的结构化对象",
            bool((results["v3"] or {}).get("structured_ok")),
            f"structured_ok={(results['v3'] or {}).get('structured_ok')}",
        )


# ============================================================
# 主流程
# ============================================================


async def main() -> int:
    print("=" * 64)
    print("v3 模型调用改道 —— 前置验证（真模型）")
    print("=" * 64)

    # ---- 1. 普通调用：_can_stream 为真，adapter 走自己的 astream 循环 ----------
    await _run_case(
        "一、普通调用（reasoning_default / deepseek-v4-flash）",
        agent_type="react",
        model_role="reasoning",
    )

    # ---- 2. 带工具：验证 tool_calls 没在聚合里被丢掉 --------------------------
    #
    # adapters.py 在累积 chunk 那段专门写过"用 + 拼 tool_call_chunks 拼错就等于
    # 把工具调用丢掉"。改道后这次聚合换到了 langchain 的 ChatModelStream 里做，
    # 所以这条必须单独看一眼。
    await _run_case(
        "二、带工具调用（tool_calls 是否被聚合逻辑丢掉）",
        agent_type="react",
        model_role="reasoning",
        tools=(get_weather,),
        prompt="用 get_weather 工具查一下北京的天气，城市参数填「北京」。",
        expect_tools=["get_weather"],
    )

    # ---- 3. 结构化输出：_can_stream 刻意拦下的那条路径 ------------------------
    await _run_case(
        "三、结构化输出（classification_default / qwen-flash）",
        agent_type="react",
        model_role="classification",
        response_format=_Sentiment,
        prompt="判断这句话的情感：今天心情不错。",
    )

    print()
    print("=" * 64)

    if _failures:
        print(f"有 {len(_failures)} 格没过：")
        for name in _failures:
            print(f"  - {name}")
        print()
        print("⚠️ 若「结构化输出」那一格红了：不要换驱动，先讨论。")
        return 1

    print("全部通过。可以进下一步：换驱动 + 双通道打印。")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
