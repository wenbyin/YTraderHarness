"""
langgraph v3 协议事件**探针** —— 把拿到的事件打到控制台，用来定投影格式。

这是一段**临时**代码：格式定下来、投影规则写进 transformer 之后它就该被删掉。
所以它刻意长得不像生产代码 —— 直接 `print`，不做分级，不做采样。

为什么要有两个入口
==================
本来只想注册一个 `StreamTransformer` 把所有事件打出来。做不到，原因是库里的
一条硬约束：

    `StreamMux._forward`（langgraph/stream/_mux.py:496-505）原文：
    "Forwarded events bypass the transformer pipeline to avoid infinite
     recursion (a transformer that pushes to a channel during `process()`
     would re-trigger itself). These events are visible in this mux's main
     event log but are not passed through transformers' `process()` methods."

也就是**具名 channel 的转发事件绕过 `process()`**。内置里至少 `lifecycle` 是具名
channel（`LifecycleTransformer`，transformers.py:629），任何 `custom:<name>` 同理。
所以单个 transformer 看到的**不是全集**。

于是这里给两个入口，前缀区分：

    ProbeTransformer     进 mux 管道 —— 打**经过管道**的事件
    print_main_log_event 迭代主日志 —— 打**落进日志**的事件（带 `seq`）

⚠️ 两个通道是**互补的，不是包含关系** —— 谁都不是谁的超集
--------------------------------------------------------
这一点实测过，别按直觉猜（直觉会说"主日志是全量"，**错**）：

**方向一：transformer 看得到、主日志没有。**
`_TasksLifecycleBase.process`（`LifecycleTransformer` 与 `SubgraphTransformer`
的基类，**两者都默认注册**）对 `tasks` 事件无条件 `return False`
（transformers.py:458-471，原文："Tasks events are folded into the synthesized
projections; suppress from the main event log so iterators don't double-see the
same information in two shapes"）。而 `push` 的语义是"**任一** transformer 返回
False 就抑制"（_mux.py:289-292）。

实测（单节点图）：transformer 看到 `tasks×2 + values×1`，主日志只送出
`values` 且 `seq=1` —— 赋号发生在抑制判断**之后**，所以被抑制的事件连 `seq`
都不占。注意**每个 transformer 都会拿到事件**，抑制只影响主日志，不影响
`process` 的调用 —— 所以探针照样能看到 `tasks`。

**方向二：主日志有、transformer 没有。**
`lifecycle` 与任何 `custom:<name>` 走 `_forward`，绕过管道（见上）。

**对下一步的意义**：`tasks` 是**节点边界**的来源，而它只走 transformer 通道 ——
这反而是 transformer 路线的一个真实优势（主日志里永远看不到 `tasks`，
因为两个默认 transformer 无条件抑制它，而 `transformers=` 只能加不能减，
没法把内置的摘掉）。真正的分工是：

    transformer 通道 -> values / messages / tasks / custom（经管道的）
    主日志通道       -> lifecycle / custom:<name> 转发

两者打出来的**差集**就是这条分界线的实测答案。

⚠️ transformer 里任何异常都会把**成功的运行变成 FAILED**
-------------------------------------------------------
`StreamMux.push` / `apush` 是内联调用 `process` 的（_mux.py:289-296、371-378），
异常一路上抛到 `AsyncGraphRunStream._apump_next` 的 `except`（run_stream.py:475-478）
→ `afail(e)` → `await run.output()` 里 `raise err` → 落到
`agent_run_dispatch.py` 的 `except BaseException` → 终态记成 FAILED。

一个纯打印的观测层**不该有把成功运行变失败的能力**。所以本模块的
`process` / `finalize` / `fail` **全部包 try/except 且绝不外抛** —— 打印失败就
是少一行日志，不是一次运行失败。

⚠️ `process()` 里读不到 `seq`
------------------------------
`push()` 是先跑完 transformer 管道、**然后**才在事件上赋 `seq`（_mux.py:277-283、
293-296）。所以 transformer 侧的行 `seq` 恒为 null，只有主日志侧有值。子 mux 更是
从不赋值（`_assign_seq=False`）。要全局有序，用**主日志**那一侧。

⚠️ 打印发生在**锁内**
--------------------
这段跑在 `agent_run_dispatch.py` 的线程锁里，慢打印会直接拉长"同一 thread 下一个
请求 409"的窗口。所以建议把 stdout 重定向到文件，别留在终端：

    python run_server.py > v3_events.log 2>&1

跑一次真实会话后把日志拿来定格式。

⚠️ 本模块**当前没有接在任何地方**
=================================
运行事件已经由 app/agent/event/run_event_projector.py 正式投影，所以这里
的探针从生产路径上**摘掉了** —— 全仓只有它自己引用自己。

要把它接回去（排查协议形状时），只需要在 agent_run_dispatch.py 的
`transformers=[...]` 里加一项：

    *probe_transformers(),          # AGENT_EVENT_TRACE=1 时才展开

⚠️ 但它会**改变被测系统**：本类的 `required_stream_modes` 声明了
`custom` / `updates`，而底层 `stream_mode` 是所有已注册 transformer 声明
模式的**并集** —— 接上探针等于额外请求两个模式。调试开关扰动了被观测的
对象，所以它不该常驻在生产路径上，这也是它被摘下来的原因之一。

开关
====
`AGENT_EVENT_TRACE`（默认 **关**）。开着时 `probe_transformers()` 返回
`[ProbeTransformer]`，关着时返回 `[]`（连注册都不注册，不给泵白加一趟
`process`）。注意：摘掉接线之后这个开关**不再有任何作用** —— 保留它是为
了接回去时行为一致。
"""

from __future__ import annotations

import json
import os
from collections import Counter
from typing import Any, ClassVar, Final

from langgraph.stream import StreamTransformer

__all__ = [
    "EXTRA_STREAM_MODES",
    "MAIN_LOG_PREFIX",
    "TRANSFORMER_PREFIX",
    "TRACE_ENABLED",
    "ProbeSummary",
    "ProbeTransformer",
    "print_main_log_event",
    "print_summary",
    "probe_transformers",
]


# ============================================================
# 开关（沿用 app/services/document/pipeline/settings.py:35-39 的形状）
# ============================================================

_ENV_PREFIX: Final[str] = "AGENT_EVENT_"


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(_ENV_PREFIX + name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


#: 默认**关**。
#:
#: 采集阶段默认开着（那时它是唯一的数据来源）。现在运行事件由
#: app/agent/event/run_event_projector.py 正式投影，要看事件直接从
#: Redis / SSE 那边看即可 —— 探针退化成"协议形状不明时的一次性排查工具"，
#: 需要时 `AGENT_EVENT_TRACE=1` 打开。
TRACE_ENABLED: Final[bool] = _env_flag("TRACE", False)

#: 两个前缀是给 grep / 分割日志用的，别改着玩 —— 定格式时要靠它们区分来源。
TRANSFORMER_PREFIX: Final[str] = "[V3-XFORMER]"
MAIN_LOG_PREFIX: Final[str] = "[V3-MAINLOG]"


#: 探针额外要请求的 stream mode。
#:
#: ⚠️ 不声明就**根本不存在**，不是"声明了才打印"
#: -------------------------------------------------
#: 底层 `astream(stream_mode=...)` 的取值是"所有已注册 transformer 的
#: `required_stream_modes` 并集"（`_collect_stream_modes`，pregel/main.py:398-414）。
#: 默认可用的只有 `values` / `messages` / `tasks`（来自三个内置 transformer）。
#: 其余模式没有任何人请求，图就不会产生它们的事件 —— 探针一行都收不到，
#: 而症状与"这个 mode 没有事件"完全一样。
#:
#: 默认只加这三条，各有明确目的：
#:     tools    工具调用的协议事件 —— `tool.called/completed` 的来源
#:     custom   `get_stream_writer()` 的输出 —— 将来业务事件不依赖 publisher 的通道
#:     updates  节点产出的 state 增量
#:
#: 刻意**不**默认加 `debug` 与 `checkpoints`：前者与 `tasks` 高度重叠（都是节点
#: 边界），后者是整个 state 的快照，两个都会把日志淹掉，而我们没有用到它们的
#: 需求。要看就设 `AGENT_EVENT_MODES=debug,checkpoints`（逗号分隔，会**替换**
#: 默认值而不是追加）。
_DEFAULT_EXTRA_MODES: Final[tuple[str, ...]] = ("tools", "custom", "updates")


def _env_modes() -> tuple[str, ...]:
    raw = os.environ.get(_ENV_PREFIX + "MODES")
    if raw is None:
        return _DEFAULT_EXTRA_MODES
    return tuple(part.strip() for part in raw.split(",") if part.strip())


EXTRA_STREAM_MODES: Final[tuple[str, ...]] = _env_modes()


# ============================================================
# 打印
# ============================================================


def _jsonable(value: Any) -> str:
    """
    尽量序列化；不能序列化的**降级成 repr**，而不是让整行消失。

    事件里混着 langchain 消息对象、pydantic 实例、异常……`default=str`
    兜住大多数，兜不住的用 repr 再兜一层。宁可这行难看，也不要这行不存在 ——
    漏一条事件的症状是"格式里少一块"，事后完全看不出来。
    """

    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:  # noqa: BLE001
        return repr(value)


def _format(
        prefix: str,
        *,
        scope: tuple[str, ...] | list[str],
        event: Any,
        seq: Any = None,
) -> str:
    """
    一行：前缀 + 单行 JSON。

    单行是刻意的 —— 多行 JSON 重定向之后没法 grep，也没法按 `method` 切片。
    """

    if not isinstance(event, dict):
        return f"{prefix} {_jsonable({'raw': event})}"

    params = event.get("params")

    payload: dict[str, Any] = {
        "scope": list(scope or ()),
        "method": event.get("method"),
        "event_id": event.get("event_id"),
        # transformer 侧恒为 None（见模块 docstring），主日志侧才有值。
        # 不用 event["seq"] 兜底：那正是读不到的那个字段，写出来会让人
        # 以为"transformer 也能拿到 seq，只是这次是 null"。
        "seq": seq,
        "namespace": (params or {}).get("namespace")
        if isinstance(params, dict)
        else None,
        "params": params,
    }

    return f"{prefix} {_jsonable(payload)}"


def _emit(line: str) -> None:
    """
    唯一一处真正写 stdout 的地方。

    ⚠️ `flush=True` 不是可选的：stdout 重定向到文件时是**块缓冲**的，
    进程被 kill（或跑太久被手动中断）时缓冲区里没刷出去的那部分直接丢 ——
    而"采集到一半的日志"恰恰是这类实验最常见的产出。逐行 flush 换来的是
    "拿到的都是完整的"。

    整体再包一层 try：打印失败不该有任何后果。
    """

    try:
        print(line, flush=True)
    except Exception:  # noqa: BLE001
        pass


# ============================================================
# 汇总表
# ============================================================


class ProbeSummary:
    """
    `method -> (条数, 首条样例)`。

    为什么要计数而不是只打：采集的目的是定格式，而**"某个 method 一条都没
    出现"与"它出现了但被漏掉"在逐条日志里长得一模一样**。汇总表把这件事
    变成可判定的 —— 表里没有的 method，就是真的没有（或者没被请求）。

    ⚠️ 由**每次运行**自己持有一个，不要做成模块级单例
    ---------------------------------------------------
    `InProcessGraphDispatcher` 会并发跑多个 handler（不同会话/线程）。模块级
    汇总会让几条同时进行、本不相干的运行把计数混进同一张表里 —— 而这张表
    正是用来判断"某个 method 到底有没有出现过"的，混了就失去意义。
    """

    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.samples: dict[str, str] = {}

    def record(self, method: Any, line: str) -> None:
        key = str(method)
        self.counts[key] += 1
        self.samples.setdefault(key, line)

    def lines(self, title: str) -> list[str]:
        if not self.counts:
            return [f"{title} 汇总：**一条事件都没有**"]

        out = [f"{title} 汇总（{sum(self.counts.values())} 条，按 method）:"]
        for method, count in sorted(
                self.counts.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            out.append(f"    {method:<24} {count:>6} 条")
        return out


def print_main_log_event(
        event: Any,
        *,
        summary: ProbeSummary | None = None,
) -> None:
    """
    打一条**主日志**事件（带 `seq`）。

    `summary` 给了就顺手记一笔 —— 调用方（一次运行）自己持有它，
    这样并发运行之间不会互相污染（理由见 `ProbeSummary` 的说明）。不给就只打。

    ⚠️ 这一侧**不是全量**（差集见模块 docstring）：`tasks` 被两个默认 transformer
    无条件抑制，永远到不了这里。

    ⚠️ 与 transformer 侧的区别不只是前缀：主日志里 `seq` 在 push 时已经赋好了
    （_mux.py:293-296），所以这一侧是**可排序**的。
    """

    if not TRACE_ENABLED:
        return

    try:
        # 主日志事件里 namespace 已经嵌在 params 里，scope 就是它自己 ——
        # 不像 transformer 有构造时的 scope 可写。
        params = event.get("params") if isinstance(event, dict) else None
        namespace = (
            (params or {}).get("namespace") if isinstance(params, dict) else None
        )

        line = _format(
            MAIN_LOG_PREFIX,
            scope=namespace or (),
            event=event,
            seq=event.get("seq") if isinstance(event, dict) else None,
        )

        if summary is not None:
            summary.record(
                event.get("method") if isinstance(event, dict) else "?",
                line,
            )

        _emit(line)

    except Exception as exc:  # noqa: BLE001
        _emit(f"{MAIN_LOG_PREFIX} !! 打印失败: {exc!r}")


def print_summary(summary: ProbeSummary, title: str) -> None:
    """打一张汇总表。由调用方在一次运行结束后显式调。"""

    if not TRACE_ENABLED:
        return

    for line in summary.lines(title):
        _emit(line)


# ============================================================
# transformer
# ============================================================


class ProbeTransformer(StreamTransformer):
    """
    把经过 mux 管道的协议事件打出来。**它看到的不是全集**（见模块 docstring）。

    ⚠️ 形状约束（违反会在注册期炸，不是运行期）：
      1. 必须是**类或 `factory(scope)` 形状**，不能传实例 ——
         `_normalize_stream_transformer_factories` 对实例直接 `TypeError`
         （main.py:430-436）。
      2. 构造必须收 `scope` —— mux 按 `factory(scope)` 调（_mux.py:134-138）。
      3. `init()` 是**抽象方法**且必须返回 dict（_mux.py:243-247）。
         这里返回 `{}`：探针不贡献投影键，于是不进 `run.extensions`、不参与
         channel 自动转发、也不跟内置 transformer 抢键名。

    ⚠️ 它按 scope **逐个子图**实例化（`_mux.py:193-225`）。所以每个实例只看到
    自己那个命名空间的事件，汇总也是按作用域分别打的 —— 这是有意的：
    "根图看到一个 method、子图看到另一个"本身就是定格式时要的信息。
    """

    #: 探针要额外请求的模式（见 `EXTRA_STREAM_MODES` 的说明）。不加这一行，
    #: `tools` / `custom` / `updates` 的事件根本不会被产生。
    required_stream_modes: ClassVar[tuple[str, ...]] = EXTRA_STREAM_MODES

    def __init__(self, scope: tuple[str, ...] = ()) -> None:
        super().__init__(scope)
        self._summary = ProbeSummary()

    def init(self) -> dict[str, Any]:
        return {}

    def process(self, event: Any) -> bool:
        """
        看一眼，打一行，**始终放行**。

        ⚠️ 必须 `return True`。返回 `False` 会把事件从主日志里**抑制掉**
        （_mux.py:290-292），于是主日志那一侧就看不到这条 —— 探针会把
        自己要对比的对象弄丢。

        ⚠️ 只读不改：转发事件的对象是共享的（子 mux 不复制，见
        `_mux.py:277-283`），改它会污染别的消费者。
        """

        try:
            if TRACE_ENABLED:
                line = _format(
                    TRANSFORMER_PREFIX,
                    scope=self.scope,
                    event=event,
                )
                self._summary.record(
                    event.get("method") if isinstance(event, dict) else "?",
                    line,
                )
                _emit(line)
        except Exception as exc:  # noqa: BLE001
            _emit(f"{TRANSFORMER_PREFIX} !! 打印失败: {exc!r}")

        return True

    def finalize(self) -> None:
        """
        mux 关闭时打本作用域的汇总。

        ⚠️ 绝不外抛：`StreamMux.close` 会把第一个 finalize 异常在清理完成后
        重新抛出（_mux.py:300-310），而那会被记成一次**运行失败**。
        """

        try:
            print_summary(
                self._summary,
                f"{TRANSFORMER_PREFIX} scope={list(self.scope)}",
            )

        except Exception:  # noqa: BLE001
            pass

    def fail(self, err: BaseException) -> None:
        """运行失败时留一行。同样绝不外抛（理由同 `finalize`）。"""

        try:
            if TRACE_ENABLED:
                _emit(
                    f"{TRANSFORMER_PREFIX} scope={list(self.scope)} "
                    f"!! 运行失败: {err!r}"
                )
        except Exception:  # noqa: BLE001
            pass


def probe_transformers() -> list[type[StreamTransformer]]:
    """
    给 `astream_events(..., transformers=...)` 用的列表。

    关闭时返回**空列表**而不是仍然注册一个不打印的 transformer ——
    后者依然会吃掉每一条事件的 `process` 调用，白白给泵加一趟开销。
    """

    return [ProbeTransformer] if TRACE_ENABLED else []
