"""
ReAct 图的 state。

只有一个字段：对话本身。

⚠️ 这里**刻意反着** document 流水线的规矩，且必须写清楚
------------------------------------------------------
`app/services/document/pipeline/state.py` 开头有一大段"state 里绝不
放正文"：那里只放 `ArtifactRef`（一个指向 File 模块的指针），正文
走 artifact_store，`assert_checkpoint_safe` 还会在每次节点后断言
state 体积不超过 64KB。

**本图的 messages 就是正文，而且是故意如此。** 理由：

    对话的"本体"就是这些消息 —— 没有第二个地方可以指向它。
    document 那边正文有 File 模块这个外部归属，所以能只存引用；
    一次对话没有（把消息存进某个 store 再传 id，等于把
    LLM 调用要用的东西藏到一个模型看不见的地方）。

因此：

    ❌ **不要**对这张图施加 `assert_checkpoint_safe`。
       它会（正确地）报"state 太大"，而这里的"太大"是正常的。
       runner 那边是否开启由参数控制，本图不参与那项断言。

    边界由 compaction 负责，不是靠把消息挪走。
       `ContextBuilderType.COMPACTION` 存在的理由就是这件事：
       上下文长到超出预算时，由它摘要压缩，而不是让 state 变空。

没有第三件事可做：既不能塞进外部存储，也不能裁到某个上限以下 ——
两者都会让模型看不到它该看到的东西。所以这里的答案是"接受它，
并在别处解决增长"。

⚠️ 未声明的键会被 LangGraph **静默丢弃**
----------------------------------------
state 的 channel 由这个 TypedDict 的字段声明。往返回值里塞一个
没声明的键，LangGraph 不报错也不写入 —— 写了、没生效、没有任何
迹象。要加通道就得先在这里加字段。

因此下面 ContextBuilder 需要的那一组通道**必须**在这里声明（哪怕
调用方暂时不填）：它们是 Context Processor 的绑定根（`when` 与
`inputs` 都从这些路径解析，见 app/agent/context/processor/state.py）。

为什么用 TypedDict 而不是 pydantic
----------------------------------
与 document 流水线同一条理由：节点之间传的是 dict，pydantic 会在
每次读写时做一遍校验，而它挡不住的东西（比如 Send payload 绕过
校验）才是真正会出错的地方 —— 于是它只贡献了开销与"看起来很安全"
的错觉。

⚠️ ContextBuilder 相关的通道由谁写
-------------------------------
    conversation_id / user_id / context_policy
                    -> **调用方 / Agent Builder**（它们才知道这次是谁、
                       用什么策略）。ContextBuilder 只读不写。
    user_context / retrieved_context / exposed_tools
                    -> 前置节点或外部服务（画像 / 检索 / 能力选择）。
                       ContextBuilder 从这些通道读，并通过 `context_policy`
                       上的开关决定注不注入。
    summary / memory / context_result
                    -> **图本身**（agent 节点把 ContextBuilder 的产出写回）：
                       摘要的新版本、L0/L1 的召回命中、以及"这一轮用了哪些
                       上下文"的快照。

（`classification` / `duty_key` 不在上面这组里 —— 它们不是 ContextBuilder
的通道，见各自字段的说明。）
"""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from app.agent.state import AgentClassification

__all__ = ["ReactState", "ToolBranchPayload"]


class ReactState(TypedDict):
    """
    一次 ReAct 运行的全部状态。

    messages 用 `add_messages` 归约，不是 `operator.add` ——
    差别在三处，三处都会静默地错：

        按 id 去重 / 覆盖   同一条消息被重新发出时是**替换**而不是
                           追加（重试、编辑历史都依赖这一点）。
                           `operator.add` 会得到两份。
        RemoveMessage 墓碑  压缩上下文时要能删掉旧消息。
                           `operator.add` 只会把墓碑也追加进去。
        身份稳定的合并      节点返回的部分列表会与已有列表按 id 合并。

    这三件事都是 ReAct + compaction 真的会走到的路径，因此不能图省事
    写成 `operator.add`。

    ⚠️ 其余通道都是"最后写入者赢"
    ---------------------------
    它们没有 reducer。因此：

        写入方只该是**决定它的人**（见下面每个字段的说明）
        节点**不要**每轮都重复写同一个值（那会让 channel 看起来一直在
        变，而依赖它的 Processor gate 就说不清"到底谁改的"）

    唯一例外是 `context_result`：agent 节点每轮都会覆盖它 —— 那正是它
    的语义（"最近一次上下文构建长什么样"）。

    ⚠️ ContextBuilder 需要的通道**必须在这里声明**
    -------------------------------------------
    未声明的键会被 LangGraph 静默丢弃。下面那一组是 Context Processor
    的绑定根（`when` 与 `inputs` 都从这些路径解析，见
    app/agent/context/processor/state.py）—— 因此它们不是"可选的便利
    字段"，而是通道契约的一部分。

    谁写谁读（这条分工决定了每个通道的写入方）：

        conversation_id / user_id / context_policy
            调用方 / Agent Builder 写（它们才知道这次是谁、用什么策略）；
            ContextBuilder 只读。
        user_context / retrieved_context / exposed_tools
            前置节点或外部服务写；ContextBuilder 读，并按策略决定注不注入。
        summary / memory / context_result
            **图写**：agent 节点把 ContextBuilder 的产出回写（摘要的新版本、
            L0/L1 的召回命中、以及"这一轮用了哪些上下文"的快照）。
    """

    messages: Annotated[list[AnyMessage], add_messages]

    # ========================================================
    # 会话与主体（ContextBuilder 的绑定根）
    # ========================================================

    #: 会话标识。图里没有它时，ContextBuilder 用 thread_id 顶上。
    conversation_id: str

    #: 主体（租户 / 用户）。画像与记忆都挂在它上面。
    #: 由派发时从 `agent_run.user_id` 填入（见 agent_run_dispatch 的
    #: _initial_state）；那一列为 NULL 时（加这列之前建的运行）留空串 ——
    #: 空串是约定的「没有用户」，画像处理器会静默跳过（不是错误）。
    user_id: str

    # ========================================================
    # 上下文输入
    # ========================================================

    #: 本次上下文构建的**策略覆盖项**（基线策略 ⊕ 这里的键）。
    #:
    #: 例：`{"include_memory": False, "max_recent_turns": 4}`
    #:
    #: ⚠️ 值必须**类型正确**：`include_memory: "yes"` 会让 ContextBuilder
    #: 以 `ContextStateError` 失败（规格 §九：gate 必须是 bool，不做隐式
    #: 转换）。这是刻意的 —— 把写错的状态悄悄当成 True 最难查。
    #:
    #: ⚠️ 它同时是**运行期数据**的载体，不只是策略开关
    #: ----------------------------------------------
    #: Agent Builder 是这一通道的唯一写者（见 _merged_policy），它把
    #: 本轮**查库得来**的两个键也写在这里：
    #:
    #:     tool_keys        本轮暴露给模型的工具（能力 code -> key）
    #:     tool_overrides   那些工具的人工说明覆盖
    #:
    #: 之所以借用策略通道而不是新开一条：键列表与它的说明必须同源同轮，
    #: 而 capability 处理器早就绑定了 `context_policy.tool_keys`。因此
    #: 这一通道里既有开关也有数据，**值必须 JSON 可序列化** —— 它会被
    #: checkpoint 落库（UUID / 实体对象一类塞进来会炸在写检查点那一步，
    #: 而不是在这里）。
    context_policy: dict[str, Any]

    #: 用户画像（`{"profile": {...}}`，供 `user_context.profile` 绑定）。
    #: 也可以什么都不放，由 ProfileSource 去取。
    user_context: dict[str, Any]

    #: L0/L1 的召回命中（记忆处理器在 INITIAL 召回一次，之后每轮迭代复用）。
    #:
    #: 形状：`{"sections": [{"name", "label", "text", "source_ids",
    #: "token_estimate"}], "degraded": [{"layer", "reason"}]}`
    #:
    #: ⚠️ 装的是**渲染好的分节**（而不是原始命中 DTO）：`text` 是给模型看的
    #: 正文，读路径因此不依赖渲染函数与 DTO 形状，且文本逐字节稳定 ——
    #: `context_hash` 不会因为重渲染而漂。
    #:
    #: ⚠️ 它**不是**"每轮都带的记忆"：L2/L3 由记忆处理器每次构建现读（纯
    #: SQL，便宜）。这个通道存在的唯一理由是 **L0/L1 贵**（embedder +
    #: 向量检索），所以一个运行只召回一次，结果在这里过夜。
    #:
    #: 写入方是 **agent 节点**（经 `ModelRequest.metadata` 的
    #: `context_state_updates` 回写，白名单见 agent_node 的
    #: `_STATE_UPDATE_KEYS`）；ContextBuilder 只读。
    memory: dict[str, Any]

    #: 知识库检索结果（`{"items": [{"id": ..., "text": ...}]}`）。
    #: 检索本身属于 Document 模块，这里只放结果。
    retrieved_context: dict[str, Any]

    #: 本次可暴露给模型的工具对象（BaseTool 那一类）。
    #: 权限判定与执行不在这里（ContextBuilder 只负责"给模型看哪些"）。
    exposed_tools: list[Any]

    # ========================================================
    # 摘要（ContextBuilder 读，图写）
    # ========================================================

    #: 对话摘要：`{"text": ..., "revision": ..., "summarized_until_message_id": ...}`。
    #:
    #: agent 节点在 summary 阶段真的生成过一版之后把它写回来（见
    #: `ModelRequest.metadata["context_state_updates"]`），因此它是一个
    #: **跨轮存活**的事实：下一轮的历史处理器按
    #: `summarized_until_message_id` 排除已被覆盖的消息。
    summary: dict[str, Any]

    # ========================================================
    # 观测（ContextBuilder 产出，图写）
    # ========================================================

    #: 最近一次上下文构建的快照（规格 §二十七）。
    #:
    #: 它回答"这一轮模型到底看到了什么"：选中的历史 id、注入的记忆 /
    #: 资料 id、预算、模型窗口、上下文哈希、每个 Processor 的执行状态。
    #: **没有正文**（正文在 messages 与 checkpoint 里）。
    context_result: dict[str, Any]

    # ========================================================
    # 输入分类（Agent Builder 写，路由读）
    # ========================================================

    #: Agent Builder 对本次输入的初次理解（见
    #: app/agent/state/agent_state.py 的 AgentClassification）。
    #:
    #: builder 节点**每轮覆盖**它 —— 与 `context_result` 同理，语义是
    #: "这一轮输入的判断"，不是跨轮累积的事实。
    #:
    #: ⚠️ builder 节点那次分类调用**不写 messages**：它是内部步骤，把
    #: 那段问答塞进对话历史会让主模型和下一轮的历史选择器都看见它。
    #: 因此"这一轮发生了什么分类"只从这里读得到。
    classification: AgentClassification

    # ========================================================
    # 主 agent 的职责提示词（派发写，builder 节点读）
    # ========================================================

    #: 这一轮用哪份**职责提示词**（duty_key）—— 如 `stock_research`。
    #:
    #: 由 `_initial_state` 从 `agent_run.duty_key` 填入（建运行那一刻定下的，
    #: 见 agent_run_service_impl.create_run），builder 节点拿它向
    #: PromptProvider 取文本，并把文本写进 `context_policy.system_prompt`。
    #: 空串 = "没指定"，builder 节点据此回落默认职责。
    #:
    #: ⚠️ 它**不是** ContextBuilder 的通道（不在 agent_node 的
    #: `_CONTEXT_STATE_CHANNELS` 里）：ContextProcessor 不读它，读它的是
    #: builder 节点自己 —— 提示词的选择发生在**节点**上，不在上下文流水线
    #: 的某个阶段里（值进流水线是下一步的事，那时它是 policy.system_prompt）。
    duty_key: str


class ToolBranchPayload(TypedDict):
    """
    `tools` 扇出分支的 Send payload —— 分支节点能看到的**全部** state。

    ⚠️ Send 的 payload **取代** state（已实测）：分支节点看不到父状态
    的任何字段。所以"这个分支需要什么"必须在这里列全 —— 漏一个就是
    KeyError，而那个报错不会告诉你是因为 Send 的语义。

    反过来，**不该带的也别带**：payload 随 pending task 一起进
    checkpoint，多一个字段就多一份被恢复的东西。下面三个字段都是
    "分支确实需要、且父状态里拿不到"的。

    为什么本图不需要 document 那种"路由屏障节点"
    ------------------------------------------
    那条约定（`app/services/document/pipeline/state.py` 的 BranchInput
    里有完整说明）是：分支节点的**下游边**看到的是**父状态**，不是
    分支状态。document 有 4 个扇出阶段串行，所以每两个之间必须夹一个
    屏障来"等所有分支跑完再发下一批"。

    本图只有**一个**扇出阶段：`tools` 的下游是 `agent`，而分支写回的
    `messages` 经 `add_messages` 汇进父状态，agent 因此在所有分支跑完
    之后只执行一次、看得到全部结果。屏障在这里是多余的。
    """

    #: 这个分支负责的**那一次**工具调用（`AIMessage.tool_calls` 里的一项：
    #: `{"name": ..., "args": {...}, "id": ...}`）。
    call: dict[str, Any]

    #: 本次请求暴露的工具 key **原文**。
    #:
    #: ⚠️ 携带的是输入而不是"允许集"这个结论：允许集由分支用
    #: `selection.select_specs` 现算 —— 与 capability 装配（给模型看
    #: 哪些）是同一个纯函数、同一个输入，两个集合恒等是算出来的。
    #: 路由只搬运原文，不参与判定。
    tool_keys: list[str]

    #: 这个分支的稳定身份，用来算观测的 `branch_key`
    #: （见 `infra/graph/ids.py` 的 `branch_scope_id`）。
    #:
    #: ⚠️ 为什么放在 payload 里而不是节点里现推
    #: 它会被 checkpoint **原样恢复**，所以 resume 时同一分支拿到的还是
    #: 同一个值。这比"节点里调一个纯函数"更强：不依赖任何执行期状态。
    #:
    #: ⚠️ 为什么不能只用 tool_call_id
    #: 很多 provider 每轮都重发 `"call_1"` 这种 id。观测行的唯一键是
    #: `(pipeline_run_id, node_id, branch_key, attempt)` —— 同一 run 的
    #: 两轮用同一个 key，两个分支的行会**合并成一行**。路由因此拼上
    #: AIMessage 的 id（每轮不同）。
    call_scope: str
