"""
Model 模块的领域类型。

它们只描述"模型是什么"与"一次模型调用长什么样"，不含任何 provider、
SDK、LangGraph 或 Agent 业务概念 —— 这一层是**纯形状**。

为什么是 dataclass 而不是 pydantic
----------------------------------
与 `app/agent/context/builder/contracts.py` 同一条理由：这些对象在**每次
模型调用**的路径上被创建与比较，而 pydantic 的校验挡不住真正会出错的
东西（provider 返回的怪东西），只贡献开销与"看起来很安全"的错觉。

另一个理由更实际：`ModelRequest.messages` 的类型是 langchain 的
`BaseMessage`，pydantic 会在**类创建时**求值注解，于是 import 本模块
就要拉起 langchain_core。这里的字段一律用 `Any` + 模块 docstring 说明，
运行时不碰 langchain。

冻结与"不要就地改"
------------------
所有类型都是 frozen。但 `metadata` / `config` 这类 Mapping 字段是
**防御性拷贝**出来的普通 dict（不是 MappingProxyType），因为：

    ModelResponse.metadata 会进节点观测并被 JSON 序列化写库
    （见 infra/graph 的 recorder），MappingProxyType 序列化不了。

代价是"拿到 dict 的人可以就地改"。约定是：**不要就地改**，它们是
共享的。要改就构造新的（`dataclasses.replace`）。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, ClassVar
from uuid import uuid4

from .redaction import redact_mapping

__all__ = [
    "ModelAttempt",
    "ModelCallSummary",
    "ModelCapabilities",
    "ModelDefinition",
    "ModelPricing",
    "ModelProfile",
    "ModelRequest",
    "ModelResolveContext",
    "ModelResponse",
    "ModelRole",
    "ModelRuntimePolicy",
    "ModelUsage",
    "ProviderResult",
    "new_invocation_id",
    "required_capabilities_for",
]


# ============================================================
# 模型角色
# ============================================================


class ModelRole:
    """
    **模型侧**的角色词汇表：这一次模型调用是干什么用的。

    ⚠️ 它不是 `agent_type`
    ----------------------
    两者是 Resolver 的两个入参，职责完全不同：

        agent_type   **Agent Builder / JEV** 决定的：用哪张图 / 哪种
                     Agent。它的词汇表属于 Agent 侧，本模块**不定义**
                     它，只把它当作字符串键。（规格 §6 明写：不要让
                     Resolver 再去理解用户输入或做 Agent 选择。）
        model_role   **本模块**定义的：同一个 Agent 里不同的模型用途。
                     react 图的推理调用与它的摘要调用，是同一个
                     agent_type、两个 model_role。

    刻意**不是** Enum：角色是会长的（今天五个，明天可能八个），而
    强制枚举会让"加一个角色"变成"改 Model 模块的公共类型"。这里的
    常量只是给调用方少打字，Resolver 接受任意字符串。
    """

    REASONING = "reasoning"
    SUMMARIZATION = "summarization"
    EVALUATION = "evaluation"
    CLASSIFICATION = "classification"
    VISION = "vision"
    FAST = "fast"

    #: L1 记忆抽取与去重（memory_atom）。**都不在任何图里** —— 它们是
    #: `memory.l1.extract` 那条后台任务里的一前一后两次调用。语义上仍归
    #: "节点功能性"（这次调模型干什么），与 VISION / FAST 同类。
    #:
    #: ⚠️ 两个角色，两个 agent_type
    #: --------------------------
    #: `memory_atom_extract` / `memory_atom_dedup` 各查各的绑定 ——
    #: 它们的取舍不同（抽取要"读得多、写得全"，去重要"判得准"），
    #: 挤一个 agent_type 就没法只调其中一个的模型。
    MEMORY_ATOM_EXTRACT = "memory.atom.extract"
    MEMORY_ATOM_DEDUP = "memory.atom.dedup"

    #: L2 场景整合（memory_scene）。同样是**后台任务里的一次调用**，
    #: 不在任何图里。
    #:
    #: ⚠️ 值取 `memory.scene.consolidate` 而不是 `memory.l2.extract`
    #: ------------------------------------------------------
    #: 与上面那对同一口径：**动作**名，不是 handler 名（handler 仍然叫
    #: `memory.l2.extract`，见 domain/tasks.py，那里有一段专门说这件事）。
    #: 抽一个角色名是为了让"这次调模型干什么"在 model.yaml 里一眼可读 ——
    #: `memory.l2.extract` 会让人以为这个角色管整条链。
    MEMORY_SCENE_CONSOLIDATE = "memory.scene.consolidate"

    #: L3 画像生成（memory_persona）。同样是**后台任务里的一次调用**，
    #: 不在任何图里。
    #:
    #: ⚠️ 两份系统提示词（个人 / 团队）共用**这一个** role
    #: --------------------------------------------
    #: 它们由同一份绑定发起、同一个模型、同一条任务（`memory.l3.extract`），
    #: 只是产物形态按 scope 有没有 org 分叉。拆成两个 role 就等于把"选哪份
    #: 提示词"这件事抄到服务侧的第二个地方（绑定与 role 各选一次），而它们
    #: 一旦漂了，症状是"团队画像用了个人模板"—— 输出仍然是一份像样的
    #: markdown，不报错。选哪一份的唯一落点见
    #: `prompts/memory_persona_extract.py` 的 `_render`。
    #:
    #: ⚠️ 值取 `memory.persona.extract`（动作）而不是 handler 名
    #: `memory.l3.extract` —— 与上面两条同一口径。
    MEMORY_PERSONA_EXTRACT = "memory.persona.extract"

    #: 记忆上下文的**注入**（画像 + 场景索引进主模型的上下文）。
    #:
    #: ⚠️ 与其他四个 memory role **不是一类**：上面四条各是一次后台模型
    #: 调用（都有自己的 agent_type 与绑定），而这一条**不发起任何调用** ——
    #: 它是"主模型每轮看到的记忆那一段长什么样"。所以：
    #:
    #:     消费侧   上下文装配（把这段拼进 system）—— 还没落地，
    #:              见 services/memory/__init__.py 的"L2/L3 的消费侧"
    #:     管理端   管理端预览（`POST /memory/retrieval`）按它渲染 prompt
    #:              预览，见 prompts/memory_retrieval.py
    #:
    #: 现在**没有** model.yaml 绑定指向它，这是对的：一旦有绑定，就说明
    #: 有人拿它去调模型了，而它的定位是"拼进别人的上下文"，不是"自己调一次"。
    #:
    #: ⚠️ 文本当前是**占位**（空）—— 提示词的正文还没写。空串在管理端
    #: 预览里会呈现成 `prompt.system: null`（契约里就是这么写的：
    #: "尚未实现时为 null"）。见那一份文件的说明。
    MEMORY_RETRIEVAL = "memory.retrieval"

    # ---- 检索评测（document eval）：同样是后台任务里的一次调用 ----
    #
    # 与上面几条 memory role 是同一类：**不在任何图里**，由
    # `retrieval_eval.generate` 那条后台任务发起。走 Model Runtime 而不是
    # 直接拿 provider 客户端，是为了白拿它那套 timeout / retry / fallback
    # 与异常归一化 —— 造题是几百次调用的一批活，逐次自己写重试必然漏掉
    # 某些分支，而漏掉的表现是"这批数据集少了一部分题"，不报错。

    #: 出题：从一段 chunk 正文里产出「问题 + 逐字答案」。
    #:
    #: ⚠️ 值取动作名 `eval.qa_generation` 而不是任务名（任务叫
    #: `retrieval_eval.generate`）—— 与 memory 那几条同一口径：任务里可能
    #: 还有别的模型调用，role 描述的是**这一次调用干什么**。
    EVAL_QA_GENERATION = "eval.qa_generation"

    #: 跨 chunk 判金标：这道题别的 chunk 是不是也能独立回答。
    #:
    #: 与上一条分两个 role，因为两者取舍不同：出题要"读得细、问得准"，
    #: 裁定要"判得保守"（宁漏勿滥 —— 错误地多标一个金标会让召回率虚高，
    #: 且虚高的部分无法察觉）。挤一个 role 就没法只调其中一个的模型。
    EVAL_CROSS_CHUNK_JUDGE = "eval.cross_chunk.judge"

    # ---- 检索侧：召回之前改写 query ----
    #
    # ⚠️ 值**不带 `eval.` 前缀**，与上面两条不一样：改写在**线上**检索
    # 路径上也能用（`DocumentRetrieveRequest.query_rewriter`），不是评测
    # 专属。评测只是它的第一个调用方（阶梯的逐轮参数）。
    #
    # 与 memory 那几条同类：一次调用、不在任何图里；走 Model Runtime
    # 白拿 timeout / retry / fallback。

    #: 查询改写：把用户那句话改写成更适合检索的形式。
    #:
    #: 改不动的必须**原样返回**并标 `changed=false` —— 硬改会改变问题的
    #: 语义，而语义一变，"这条 case 为什么没命中"就再也归因不到检索上了。
    QUERY_REWRITE = "query.rewrite"

    #: 精排：给一批候选正文打相关性分（`RerankerType.LLM`）。
    #:
    #: 与改写同属检索侧（线上也会走，所以不带 `eval.` 前缀），但**每次
    #: 调用的输入大得多**：它要读 N 条候选正文，所以窗口（送几条进来）
    #: 由实现自己封顶 —— 那是成本的主要变量。
    #:
    #: 与 cross-encoder 那条本地精排是**并列的两条实现**，不是替代关系：
    #: 前者便宜、稳定；这条能带上领域约定（"这批语料里什么算相关"）。
    RERANK = "retrieve.rerank"


# ============================================================
# 能力
# ============================================================


@dataclass(frozen=True, slots=True)
class ModelCapabilities:
    """
    一个模型**能做什么**。

    它参与两件事，两件都是硬的：

        模型选择      Resolver 之外，Runtime 在每次调用前按"这次需要
                      什么"过滤候选链（见 runtime.py）
        fallback 相容 一个需要 tool_calling 的调用，**不允许**降级到
                      不支持 tool_calling 的模型上（规格 §8.3）

    默认全 False 而不是全 True：能力是**声明**出来的，默认值代表
    "没声明"，而"没声明"必须表现为保守（不能拿它降级），不是乐观。
    """

    tool_calling: bool = False
    structured_output: bool = False
    streaming: bool = False
    vision: bool = False
    reasoning: bool = False

    #: 五个字段名，顺序即展示顺序。写成类属性而不是到处 `fields()`：
    #: 展示与比较两处都要"稳定的顺序"，而从 dataclass 反射出来的顺序
    #: 会随字段声明的位置变。
    FIELDS: ClassVar[tuple[str, ...]] = (
        "tool_calling",
        "structured_output",
        "streaming",
        "vision",
        "reasoning",
    )

    # --------------------------------------------------------

    def missing(
            self,
            required: "ModelCapabilities | None",
    ) -> tuple[str, ...]:
        """
        `required` 里有哪些是**本模型没有**的。

        返回名字而不是 bool，是为了让报错能直接说出"缺哪个"——
        `ModelCapabilityError` 的信息质量全靠它。
        """

        if required is None:
            return ()

        return tuple(
            name
            for name in self.FIELDS
            if getattr(required, name) and not getattr(self, name)
        )

    def satisfies(
            self,
            required: "ModelCapabilities | None",
    ) -> bool:
        """本模型是否满足 `required`。"""

        return not self.missing(required)

    def merged(
            self,
            other: "ModelCapabilities | None",
    ) -> "ModelCapabilities":
        """
        按位或合并两个**要求**（不是两个模型的能力）。

        用在"节点声明的要求 ∪ 请求内容推导出的要求"这一步：两处都
        只写自己知道的那几位，合起来才是这次调用完整的要求。
        """

        if other is None:
            return self

        return ModelCapabilities(
            **{
                name: getattr(self, name) or getattr(other, name)
                for name in self.FIELDS
            }
        )

    def as_dict(self) -> dict[str, bool]:

        return {name: getattr(self, name) for name in self.FIELDS}

    def describe(self) -> str:
        """`tool_calling, structured_output`；一个都没有时是 `-`。"""

        enabled = [name for name in self.FIELDS if getattr(self, name)]

        return ", ".join(enabled) if enabled else "-"

    @classmethod
    def from_dict(
            cls,
            data: Mapping[str, Any] | None,
    ) -> "ModelCapabilities":
        """
        从配置里读一份能力声明。

        ⚠️ 不认识的键**报错**而不是忽略：能力名写错一个字母
        （`tool_callingg: true`）时静默忽略的后果是"这个模型看起来
        支持工具，运行时才发现不支持"，而报错指向的是行号明确的配置。
        """

        if not data:
            return cls()

        unknown = sorted(set(data) - set(cls.FIELDS))

        if unknown:
            raise ValueError(
                f"不认识的能力声明：{unknown}。"
                f"可用的是：{list(cls.FIELDS)}。"
            )

        return cls(**{name: bool(data[name]) for name in cls.FIELDS if name in data})

    @classmethod
    def none(cls) -> "ModelCapabilities":
        return cls()

    @classmethod
    def full(cls) -> "ModelCapabilities":
        return cls(**{name: True for name in cls.FIELDS})


def required_capabilities_for(request: "ModelRequest") -> ModelCapabilities:
    """
    从**请求内容**推导这次需要的能力。

    只推导两件能确定的事：

        有 tools            -> 需要 tool_calling
        有 response_format  -> 需要 structured_output

    ⚠️ 刻意**不**猜 vision / reasoning
    --------------------------------
    "messages 里有图片所以需要 vision"是一句看起来聪明、实际会错的
    推断：图片可能以 URL 字符串、markdown、多模态 block 三种形态出现，
    猜错的代价是"本来能跑的调用被判成能力不符"。这两项必须由**知道
    自己在干什么的人**显式声明（节点的 `ModelResolveContext`，或
    `ModelRequest.required_capabilities`）。

    宁可让一个真需要 vision 的调用在 provider 那里报错，也不要让
    一堆不需要的调用因为猜错而被拦下。
    """

    return ModelCapabilities(
        tool_calling=bool(request.tools),
        structured_output=request.response_format is not None,
    )


# ============================================================
# 价格与用量
# ============================================================


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """
    单价（**每 1000 token**）。

    刻意不做"内置价目表"：模型价格随时在变，而模块里的一份硬编码
    价目表只会变成一份过期数据，且没有任何机制提醒它过期。价格写在
    配置里（model.yaml 的 `pricing:`），谁改配置谁负责。
    """

    input_per_1k: float | None = None
    output_per_1k: float | None = None
    currency: str = "USD"

    def estimate(self, usage: "ModelUsage | None") -> float | None:
        """
        估算这次调用的成本。

        只按**报出来的** token 算；provider 没报用量（不少流式 / 兼容
        端点不报）时返回 None —— 编一个 0 出来会让"这家没报"与
        "这次真的免费"分不清，而成本报表最怕这个。
        """

        if usage is None:
            return None

        total = 0.0
        priced = False

        if usage.input_tokens is not None and self.input_per_1k is not None:
            total += usage.input_tokens / 1000.0 * self.input_per_1k
            priced = True

        if usage.output_tokens is not None and self.output_per_1k is not None:
            total += usage.output_tokens / 1000.0 * self.output_per_1k
            priced = True

        if not priced:
            return None

        return round(total, 6)


@dataclass(frozen=True, slots=True)
class ModelUsage:
    """
    一次调用的 token 用量。

    三个字段都可空：可空表示"provider 没报"，与 0 是两件事。
    """

    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None

    @classmethod
    def from_metadata(
            cls,
            data: Mapping[str, Any] | None,
    ) -> "ModelUsage | None":
        """
        从各家五花八门的用量结构里读一份出来。

        认两种命名（覆盖 langchain / openai / anthropic 的常见形态）：

            input_tokens / output_tokens / total_tokens   langchain 归一化后
            prompt_tokens / completion_tokens             OpenAI 原始

        一个都认不出来就返回 None（= 没报），不是全 0 的 ModelUsage。
        """

        if not data:
            return None

        input_tokens = _first_int(
            data, "input_tokens", "prompt_tokens", "input_token_count"
        )
        output_tokens = _first_int(
            data, "output_tokens", "completion_tokens", "output_token_count"
        )
        total_tokens = _first_int(data, "total_tokens", "total_token_count")

        if total_tokens is None and (
                input_tokens is not None or output_tokens is not None
        ):
            # 各家不一定都报 total；两个分项都有时自己加，只有一个时
            # **不**加（会把缺的那半当成 0，得到一个偏小的总数）。
            if input_tokens is not None and output_tokens is not None:
                total_tokens = input_tokens + output_tokens

        if input_tokens is None and output_tokens is None and total_tokens is None:
            return None

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

    def as_dict(self) -> dict[str, int | None]:

        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
        }


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """
    **Provider Adapter 的返回值** —— 一次成功的 provider 调用。

    它是 Runtime 与适配器之间的内部契约：适配器只负责"把请求发出去、
    把响应用量拿回来并归一化异常"，retry / fallback / 观测全在 Runtime。
    """

    message: Any

    usage: ModelUsage | None = None

    provider: str = ""
    model: str = ""

    #: provider 侧附带的元信息（finish_reason、response id…）。
    #: 会经过脱敏后进观测。
    metadata: Mapping[str, Any] = field(default_factory=dict)


# ============================================================
# 模型定义
# ============================================================


@dataclass(frozen=True, slots=True)
class ModelDefinition:
    """
    一个**具体**模型（provider + model name + 能力 + 凭据 + 参数）。

    它是 Factory 的输入、Resolver 的输出的组成单位，也是观测里"这次
    到底是谁答的"的答案。
    """

    #: 稳定标识，约定是 `provider:model`（如 `deepseek:deepseek-v4-flash`）。
    #:
    #: ⚠️ 它是**契约键**：profile 的 primary / fallbacks、别名表、观测行
    #: 里都用它。改它等于换一个模型（历史观测会指向一个不存在的定义）。
    id: str

    provider: str

    model: str

    capabilities: ModelCapabilities = field(
        default_factory=ModelCapabilities
    )

    #: 这个模型还能被哪些名字引用（`sonnet` -> `anthropic:claude-sonnet`）。
    #:
    #: ⚠️ 别名进**统一命名空间**（与模型 id、profile id 同域），冲突会
    #: 在注册时报错 —— 见 registry.py。
    aliases: tuple[str, ...] = ()

    #: 凭据引用（名字，不是秘密）。None = 用 provider 自己配置里的凭据。
    credential_ref: str | None = None

    #: 传给 provider 客户端的构造参数（temperature、max_tokens…）。
    #:
    #: 这是**provider 参数**，不是 Agent 参数：Agent 侧不该在这里塞
    #: "这次对话要干什么"。它进 Factory 的缓存键，因此必须是可比较的
    #: 简单值。
    config: Mapping[str, Any] = field(default_factory=dict)

    #: **需要结构化输出的那次调用**要额外并进请求体的那一段。
    #:
    #: 为什么需要它：有些端点默认跑在"思考模式"下，而思考模式拒绝带
    #: 强制 tool_choice 的请求 —— 那正是
    #: `with_structured_output(method="function_calling")` 发出的东西。
    #: 实测 deepseek 直接 400：
    #:
    #:     400 Thinking mode does not support this tool_choice
    #:
    #: 于是"要结构化输出"的调用必须先关掉思考。关法是请求体里的一个键
    #: （deepseek 是 `{"thinking": {"type": "disabled"}}`），所以这里存的是
    #: **要合并的那一段**而不是一个布尔开关 —— 换一家供应商只改配置。
    #:
    #: ⚠️ 它**只在 `request.response_format` 非空时**才用，不是每次调用都带：
    #: 推理、摘要这些调用照旧走思考模式。写成 `config` 里的构造参数做不到
    #: 这一点 —— 那是**构造期**参数，会作用于这个客户端的每一次请求，
    #: 而"结构化输出"是**调用期**才决定的事。
    structured_output_extra_body: Mapping[str, Any] = field(
        default_factory=dict
    )

    #: 这个模型的思考模式**一律关掉**，不看这次调用长什么样。
    #:
    #: 与 `ModelRequest.disable_thinking` 同名同义，作用域不同：那个是
    #: "这一次调用请关"，这个是"这个模型一律关"。
    #:
    #: 为什么要它（2026-10-13 实测）
    #: ---------------------------
    #: `structured_output_extra_body` 原来的用法是"这次调用与思考模式冲突时
    #: 才关"，判据看的是**调用形状**（带工具 / 带结构化输出 / 显式开关）。
    #: 而第二次 DSML 泄漏的形状恰好是"既不带 tools、也不带 response_format"
    #: 的**纯文本**调用 —— 三条判据一条都不占，于是它照旧跑在思考模式下，
    #: 把工具调用写成正文吐给了用户（149 条 delta + 全文，run 51b31758）。
    #: 按调用形状猜，永远会有下一种漏法。
    #:
    #: 判据是"这个模型的思考模式**没有实证收益**，而它有已知副作用"。
    #: ⚠️ 不是 `capabilities.reasoning` —— 那个说的是"它是不是推理档"
    #: （model.yaml 里明写了这两者不是一回事：flash 声明 reasoning: false，
    #: 可它默认就跑在思考模式下）。
    #:
    #: ⚠️ 它是**可选**的，今天一个模型都没开
    #: ------------------------------------
    #: flash 2026-10-13 撤掉了它：思维链要留给用户看（`agent_message.
    #: reasoning_content` + `message.reasoning.*`），一律关就等于那条链路
    #: 永远没有数据。撤掉之后那个纯文本形状的泄漏由**出口的闸**兜底
    #: （见 leaked_tool_call）。读到那次泄漏的说明时别顺势以为"所以 flash
    #: 现在开着这个开关" —— 它没开。
    #:
    #: ⚠️ 置 True 却没有 `structured_output_extra_body` 是**配置错**
    #: ------------------------------------------------------------
    #: 关思考的写法逐 provider 不同（deepseek 是 `thinking: {type: disabled}`，
    #: qwen 是 `enable_thinking: false`），少了那一段这个开关会静默不生效。
    #: 所以 `__post_init__` 直接抛，不留到"运行期看起来正常但没拦住"。
    disable_thinking: bool = False

    pricing: ModelPricing | None = None

    display_name: str = ""

    #: 上下文窗口（token）。
    #:
    #: ⚠️ 它是**预算**的依据，不是"能不能跑"的依据：Context Builder 用它算
    #: history_budget（规格 §二十一：必须优先用 Registry，而不是硬编码
    #: 模型名或窗口）。不填就是 None —— 那时 Context Builder 回落到策略里
    #: 的兜底窗口，并在快照里注明"用的是兜底值"（可见，而不是静默）。
    context_window: int | None = None

    def __post_init__(self) -> None:

        if not self.id.strip():
            raise ValueError("ModelDefinition.id 不能为空。")

        if not self.provider.strip():
            raise ValueError(
                f"模型 {self.id!r} 没有声明 provider —— "
                "Factory 靠它找 provider adapter。"
            )

        if not self.model.strip():
            raise ValueError(
                f"模型 {self.id!r} 没有声明 model（provider 侧的模型名）。"
            )

        if self.context_window is not None and self.context_window <= 0:
            raise ValueError(
                f"模型 {self.id!r} 的 context_window 必须为正，"
                f"得到 {self.context_window!r}。"
            )

        if self.disable_thinking and not self.structured_output_extra_body:
            raise ValueError(
                f"模型 {self.id!r} 声明了 disable_thinking，却没有 "
                "structured_output_extra_body —— 关思考是**逐 provider** 的"
                "请求体参数，少了那一段这个开关会静默不生效（请求照发、思考"
                "照跑、什么都不报）。\n"
                "在 model.yaml 里补上那一段：deepseek 是 "
                "`thinking: {type: disabled}`，qwen 是 "
                "`enable_thinking: false`。"
            )

        object.__setattr__(
            self, "aliases", tuple(dict.fromkeys(self.aliases))
        )

        # 只读视图：config 会被 Factory 当成构造参数读，也会进日志。
        # 就地改它会让"缓存过的客户端"与"定义"悄悄不一致。
        object.__setattr__(
            self, "config", MappingProxyType(dict(self.config))
        )

        # 同上。它是**请求体**的一部分，改它的后果比改 config 更隐蔽：
        # 下一次调用的报文就变了，而没有任何东西会报错。
        object.__setattr__(
            self,
            "structured_output_extra_body",
            MappingProxyType(dict(self.structured_output_extra_body)),
        )

    # --------------------------------------------------------

    @classmethod
    def make(
            cls,
            provider: str,
            model: str,
            **kwargs: Any,
    ) -> "ModelDefinition":
        """按 `provider:model` 约定拼出 id。"""

        return cls(
            id=f"{provider}:{model}",
            provider=provider,
            model=model,
            **kwargs,
        )

    @property
    def target(self) -> str:
        """`provider:model` 形态的目标串（客户端构造时用）。"""

        return f"{self.provider}:{self.model}"

    def with_config(
            self,
            **overrides: Any,
    ) -> "ModelDefinition":
        """
        派生一份改了 config 的定义。

        用途是 profile 级别的参数覆盖（同一次调用链里给 fallback 配
        不同的 max_tokens），因此返回**新对象**而不是就地改。
        """

        merged = dict(self.config)
        merged.update(overrides)

        return replace(self, config=merged)

    def describe(self) -> dict[str, Any]:
        """进日志 / 目录接口的一份摘要。**已脱敏**。"""

        return {
            "id": self.id,
            "provider": self.provider,
            "model": self.model,
            "capabilities": self.capabilities.as_dict(),
            "aliases": list(self.aliases),
            "credential_ref": self.credential_ref,
            "config": redact_mapping(dict(self.config)),
            "config_keys": sorted(self.config),
            "context_window": self.context_window,
        }


# ============================================================
# 运行策略与 Profile
# ============================================================


@dataclass(frozen=True, slots=True)
class ModelRuntimePolicy:
    """
    一次模型调用的**可靠性策略** —— Runtime 的输入。

    它只谈模型调用本身（超时、重试、降级），不碰任何 Agent 语义：
    这里没有"要不要人工确认""工具该不该执行"的位置（见规格 §10）。
    """

    #: **单次 provider attempt** 的超时（秒）。
    #:
    #: 是 attempt 级而不是"整次调用"级：fallback 链上有 N 家，每家都
    #: 该有自己的预算，否则第一家的超时会把后面几家的时间吃光。
    timeout: float = 60.0

    #: 同一个候选模型最多再试几次（总尝试次数 = max_retries + 1）。
    #:
    #: 默认 2，与既有 `get_chat_model(max_retries=2)` 的体感一致 ——
    #: 区别是现在这个数字由 Runtime 统一执行，而**客户端内部重试被关掉**
    #: （见 factory.py），因此不会有"3 × 3 = 9 次可计费调用"那类叠加。
    max_retries: int = 2

    #: 哪些错误码值得重试（见 errors.DEFAULT_RETRYABLE_ERRORS）。
    #:
    #: 只有**落在表里的**错误码会被重试；表外那些即使类的 disposition
    #: 是 RETRY，也会退化成"换下一个候选"。这让"这次不要重试"成为
    #: 一个可以按 profile 收窄的开关，而不是必须改代码。
    retryable_errors: tuple[str, ...] = (
        "timeout",
        "rate_limit",
        "unavailable",
        "provider_error",
    )

    #: 是否允许降级到 fallback。
    fallback_enabled: bool = True

    #: 整条候选链的时间上限（秒）；None = 不限。
    #:
    #: ⚠️ 必须存在一个默认值：max_retries=2 × 3 个候选 × timeout=60
    #: 最坏是 9 分钟，而 LLM Node 的上游（HTTP 请求 / 图运行）几乎一定
    #: 等不了那么久。这里给一个"比最坏情况小得多、又比正常调用宽裕"的数。
    total_timeout: float | None = 180.0

    #: 重试前的基础退避（秒），按 attempt 指数增长；0 = 不退避。
    #:
    #: 默认 0 是刻意的：退避是**为了绕开限流**，而单机开发路径上
    #: "等一下再试"只会把失败拖长。要跑生产就把 profile 里这一项设上。
    retry_backoff: float = 0.0

    #: 退避的上限（秒）。
    max_retry_backoff: float = 8.0

    def __post_init__(self) -> None:

        if self.timeout <= 0:
            raise ValueError(
                f"ModelRuntimePolicy.timeout 必须为正，得到 {self.timeout!r}。"
            )

        if self.max_retries < 0:
            raise ValueError(
                "ModelRuntimePolicy.max_retries 不能为负 "
                f"（得到 {self.max_retries!r}）—— 它是“额外再试几次”，"
                "不要用它表达“不重试”，那用 retryable_errors=() 或 "
                "max_retries=0。"
            )

        if self.total_timeout is not None and self.total_timeout <= 0:
            raise ValueError(
                "ModelRuntimePolicy.total_timeout 必须为正或 None，"
                f"得到 {self.total_timeout!r}。"
            )

        if self.retry_backoff < 0:
            raise ValueError("ModelRuntimePolicy.retry_backoff 不能为负。")

        object.__setattr__(
            self, "retryable_errors", tuple(self.retryable_errors)
        )

    def with_(self, **changes: Any) -> "ModelRuntimePolicy":
        """派生一份改了若干项的策略。"""

        return replace(self, **changes)

    @property
    def max_attempts(self) -> int:
        """单个候选模型的最多尝试次数。"""

        return self.max_retries + 1

    def allows_retry(self, code: str) -> bool:
        """
        这个错误码是否值得重试。

        表为空 = **什么都不重试**（这是"关掉重试"的表达方式，比
        max_retries=0 更明确：它连"可重试的错误"这个概念都关掉了）。
        """

        return code in self.retryable_errors

    def describe(self) -> dict[str, Any]:

        return {
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retryable_errors": list(self.retryable_errors),
            "fallback_enabled": self.fallback_enabled,
            "total_timeout": self.total_timeout,
            "retry_backoff": self.retry_backoff,
            "max_retry_backoff": self.max_retry_backoff,
        }


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """
    `Agent Type + Model Role` 解析出来的**模型策略**。

    它是"用哪个模型、失败了退到哪、按什么策略调"的完整答案 ——
    规格 §5 里 Resolver 的返回值。

    注意 primary / fallbacks 里放的是**模型的引用**（模型 id 或别名），
    不是 ModelDefinition 本身：定义归 Registry 管，profile 只引用它。
    这样一份 profile 可以在不改动的情况下被重新指向另一个具体模型。
    """

    id: str

    #: 主模型：模型 id 或模型别名。
    primary: str

    #: 降级顺序：从前往后试。空 = 不降级。
    fallbacks: tuple[str, ...] = ()

    policy: ModelRuntimePolicy = field(
        default_factory=ModelRuntimePolicy
    )

    display_name: str = ""
    description: str = ""

    def __post_init__(self) -> None:

        if not self.id.strip():
            raise ValueError("ModelProfile.id 不能为空。")

        if not self.primary.strip():
            raise ValueError(f"ModelProfile {self.id!r} 没有 primary。")

        object.__setattr__(self, "fallbacks", tuple(self.fallbacks))

        candidates = self.candidates

        duplicates = _duplicates(candidates)

        if duplicates:
            # 重复候选的后果不是报错而是**多花一次钱**：同一个模型被
            # 当成两个候选各试一遍，日志里看起来"两家都挂了"。
            raise ValueError(
                f"ModelProfile {self.id!r} 的候选里有重复：{duplicates}。"
                "同一个模型在一次调用里只该出现一次。"
            )

    @property
    def candidates(self) -> tuple[str, ...]:
        """primary + fallbacks，按尝试顺序。"""

        return (self.primary, *self.fallbacks)

    def describe(self) -> dict[str, Any]:

        return {
            "id": self.id,
            "primary": self.primary,
            "fallbacks": list(self.fallbacks),
            "policy": self.policy.describe(),
            "display_name": self.display_name,
            "description": self.description,
        }


# ============================================================
# 请求 / 响应
# ============================================================


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """
    一次模型调用的**输入**。

    与 `app/agent/context/builder/contracts.py::ModelRequest` 的关系
    ---------------------------------------------------------------
    两者字段一一对应，但**刻意是两个类**，转换由 LLM Node 显式做
    （见 graph/react/agent_node.py）。理由：

        builder 那个是**上下文构建规格**的 final_output（对外契约，
        context_builder.yaml 里的 type == model_request）；这一个属于
        Model 模块。让任一方 import 另一方，都会把"上下文构建"与
        "模型调用"这两件事绑成一件 —— 它们是相邻的两层，不是一层。

    冻结 + 防御性拷贝：它会跨越 Runtime 的多次 attempt 被反复使用
    （retry / fallback 用的是**同一个 request 对象**，见规格 §9），
    因此不允许被任何一次 attempt 修改。
    """

    messages: tuple[Any, ...]

    #: 本次可用的工具（langchain BaseTool，或任何 provider 客户端
    #: 的 bind_tools 认的东西）。
    tools: tuple[Any, ...] = ()

    #: 结构化输出要求（pydantic 模型 / "json_object" / schema dict）。
    response_format: Any = None

    #: 调用级元信息（追踪 id、标签）。**不是**给模型看的正文。
    #: 发出前会脱敏（见 runtime.py）。
    metadata: Mapping[str, Any] = field(default_factory=dict)

    #: 这次调用**显式要求**的能力。None = 只按请求内容推导
    #: （见 required_capabilities_for）。
    required_capabilities: ModelCapabilities | None = None

    #: 这次调用**不需要推理链**，请把思考模式关掉。
    #:
    #: ⚠️ 它解决的是一个**预算**问题，不只是速度问题
    #: ----------------------------------------
    #: 有些端点默认跑在思考模式下，而 `max_tokens` 是**思考与答案共用**的。
    #: 一次"按给定格式把内容填进去"的调用（抽取 / 分类 / 摘要），推理会吃掉
    #: 一大截预算，答案写到一半就被截断 —— 表现是 JSON **在字符串中间断掉**
    #: （`Unterminated string`），看起来像模型乱写，其实是没写完。
    #:
    #: 什么时候该置 True：**输出形状由我们给定**（schema / 提示词里写死的
    #: JSON 结构），模型不需要"想"该怎么答。什么时候**别**置：推理题、需要
    #: 权衡的写作、以及要不要调工具这种判断 —— 那些正是思考有价值的地方。
    #:
    #: 关法由**模型定义**声明（model.yaml 的 `structured_output_extra_body`），
    #: 因为"关思考"这个私有参数各 provider 写法不同（deepseek 是
    #: `thinking: {type: disabled}`，qwen 是 `enable_thinking: false`）。
    #: 没有声明的模型这里置了也没用（静默不生效，不报错）。
    disable_thinking: bool = False

    #: 这次调用**显式要求保留思考模式** —— 即使带 tools 也不关。
    #:
    #: 默认 False：沿用"带 tools -> 关思考"的形状判据。ReAct 主循环
    #: （react）置 True，让模型在工具调用之间保留推理链。
    #:
    #: ⚠️ 优先级
    #: ----------
    #:     response_format 非空        仍然关 —— 结构化输出必须发强制
    #:                                 tool_choice，思考模式会 400 / 截断
    #:     definition.disable_thinking 仍然关 —— 模型级硬开关
    #:     request.disable_thinking    仍然关 —— 调用方显式要求关
    #:     enable_thinking            覆盖"带 tools"这一条
    #:
    #: 代价：deepseek-v4-flash 在思考模式下有把工具调用写成 DSML 正文的
    #: 已知风险，由 leaked_tool_call 的两道闸兜底（可见的重试 / 报错）。
    enable_thinking: bool = False

    def __post_init__(self) -> None:

        # list -> tuple：调用方（节点）手上常常是 list，而这里必须
        # 不可变 —— 否则一次 fallback 里的适配器改了它，另一次看到的
        # 就是改过的请求。
        object.__setattr__(self, "messages", tuple(self.messages))
        object.__setattr__(self, "tools", tuple(self.tools))
        object.__setattr__(self, "metadata", dict(self.metadata))

    def describe(self) -> dict[str, Any]:
        """
        进观测的摘要。**只有计数与键名，没有正文**。

        正文进观测是另一类问题（体积、隐私），且 span 里本来就有
        `input_summary` 只放计数这条约定（见 infra/graph/instrumentation.py）。
        """

        return {
            "message_count": len(self.messages),
            "tool_count": len(self.tools),
            "has_response_format": self.response_format is not None,
            "disable_thinking": self.disable_thinking,
            "enable_thinking": self.enable_thinking,
            "metadata_keys": sorted(str(key) for key in self.metadata),
        }


@dataclass(frozen=True, slots=True)
class ModelAttempt:
    """
    **一次 provider attempt** 的结果（成功或失败）。

    它是"一次 runtime.ainvoke 内部发生了什么"的最小记录单位。规格 §9
    要求把"一次调用"与"多次 attempt"分开，这个类型就是那个区分的载体：

        一次 LLM Node execution
            -> 一次 ModelRuntime.ainvoke
                -> N 个 ModelAttempt
    """

    provider: str
    model: str

    #: 在候选链里的位置：0 = primary，1.. = fallback 的顺序。
    candidate_index: int = 0

    #: 这个候选模型的第几次尝试（1 起）。
    attempt: int = 1

    latency_ms: int = 0

    success: bool = False

    usage: ModelUsage | None = None

    error_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    estimated_cost: float | None = None

    #: 这一次 attempt 实际拿到的超时预算（可能被 total_timeout 压缩）。
    timeout: float | None = None

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"

    @property
    def fallback(self) -> bool:
        return self.candidate_index > 0

    def as_metadata(self) -> dict[str, Any]:
        """
        进观测的一份快照。

        ⚠️ error_message 已在 `ModelError` 构造时脱敏；这里再过一遍
        redaction 是**廉价的双保险** —— 观测行是持久化的，一次泄漏
        比一次多余的正则匹配贵得多。
        """

        return {
            "provider": self.provider,
            "model": self.model,
            "model_id": self.model_id,
            "candidate_index": self.candidate_index,
            "fallback": self.fallback,
            "attempt": self.attempt,
            "latency_ms": self.latency_ms,
            "success": self.success,
            "timeout": self.timeout,
            "error_type": self.error_type,
            "error_code": self.error_code,
            "error_message": redact_mapping(self.error_message),
            "estimated_cost": self.estimated_cost,
            **(
                self.usage.as_dict()
                if self.usage is not None
                else {
                    "input_tokens": None,
                    "output_tokens": None,
                    "total_tokens": None,
                }
            ),
        }


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """
    一次 `runtime.ainvoke` 的**结果**。

    调用方（LLM Node）只看这个对象：它已经包含了"谁答的、用了多少、
    降级过没有"的全部信息，节点不需要再去问任何 provider 细节。
    """

    #: provider 返回的消息（langchain 侧就是 AIMessage）。
    message: Any

    provider: str
    model: str

    usage: ModelUsage | None = None

    #: 本次调用从开始到返回的墙钟耗时（含重试与退避）。
    latency_ms: int = 0

    #: 一共发生了多少次 provider attempt（>= 1）。
    attempts: int = 1

    #: 是否**降级**过：重试过、或用过 fallback、或剔除过不相容候选。
    #:
    #: 单列成必填字段而不是让调用方自己去数 metadata：它是"这次答案
    #: 的可靠性打折了"的唯一信号，值得在类型上就看得见。
    degraded: bool = False

    #: 逐次 attempt 的明细 + profile / 策略等。**已脱敏**，且可 JSON 序列化
    #: （它会经节点观测写库）。
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:

        object.__setattr__(self, "metadata", dict(self.metadata))

    @property
    def model_id(self) -> str:
        return f"{self.provider}:{self.model}"


@dataclass(frozen=True, slots=True)
class ModelCallSummary:
    """
    一次 `runtime.ainvoke` 的**整体**观测记录（给 observer）。

    与 ModelResponse 的区别：它是"这次调用好不好"的判定，成功失败都有
    —— 因此观察者只装这一个入口，不用从异常里再拼一次。
    """

    invocation_id: str

    agent_type: str
    model_role: str
    profile_id: str

    started_at: float
    latency_ms: int

    success: bool
    degraded: bool

    provider: str | None = None
    model: str | None = None

    attempts: tuple[ModelAttempt, ...] = ()

    usage: ModelUsage | None = None
    estimated_cost: float | None = None

    #: 因**能力不相容**被剔除的候选（模型引用），按 profile 里的顺序。
    incompatible_candidates: tuple[str, ...] = ()

    error_type: str | None = None
    error_code: str | None = None
    error_message: str | None = None

    #: 调用方给的 metadata 的**脱敏副本**（追踪 id 之类），不含正文。
    request_metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def provider_attempts(self) -> int:
        return len(self.attempts)

    def describe(self) -> dict[str, Any]:
        """
        日志 / 观测用的一份扁平描述。

        字段名与规格 §8.6 列的那一组对齐（provider / model / attempt /
        latency / tokens / estimated_cost / success / error_type），
        并补上"降级"与"被剔除的候选"两项 —— 少了它们，"这次为什么慢"
        与"为什么没用上 fallback"都查不出来。
        """

        return {
            "invocation_id": self.invocation_id,
            "agent_type": self.agent_type,
            "model_role": self.model_role,
            "profile_id": self.profile_id,
            "provider": self.provider,
            "model": self.model,
            "success": self.success,
            "degraded": self.degraded,
            "provider_attempts": self.provider_attempts,
            "latency_ms": self.latency_ms,
            "input_tokens": self.usage.input_tokens if self.usage else None,
            "output_tokens": self.usage.output_tokens if self.usage else None,
            "total_tokens": self.usage.total_tokens if self.usage else None,
            "estimated_cost": self.estimated_cost,
            "incompatible_candidates": list(self.incompatible_candidates),
            "error_type": self.error_type,
            "error_code": self.error_code,
        }

    def as_metadata(self) -> dict[str, Any]:
        """给观测行的完整快照（含逐次 attempt）。**已脱敏**。"""

        return {
            **self.describe(),
            "error_message": redact_mapping(self.error_message),
            "attempts": [attempt.as_metadata() for attempt in self.attempts],
            "request_metadata": redact_mapping(dict(self.request_metadata)),
        }


@dataclass(frozen=True, slots=True)
class ModelResolveContext:
    """
    `get_runtime(...)` 的**可选**上下文。

    它是"解析时需要知道、但由调用方（LLM Node）提供"的那点信息。刻意
    只有三项，且三项都必须是**模型维度**的：

        required_capabilities  这次调用需要哪些能力（模型维度的硬约束）
        credential_ref         覆盖这次要用的凭据（多账号 / 多租户）
        attributes             解析路由用的键值（租户、区域、成本档…）

    ⚠️ 它**不允许**出现 Agent / Graph 业务信息
    -------------------------------------------------
    不放 thread_id、不放用户消息、不放 graph state、不放"是否高风险工具"。
    那些是 LLM Node / Graph 的事（规格 §11）；一旦进来了，Resolver 就会
    慢慢长成第二个 Agent Builder —— 而那正是规格 §6 禁止的事。

    attributes 里放什么由部署决定（Resolver 只做等值匹配），但它是
    **路由依据**而不是业务上下文：`{"locale": "zh"}` 可以，
    `{"user_question": "..."}` 不行。
    """

    required_capabilities: ModelCapabilities | None = None

    credential_ref: str | None = None

    attributes: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:

        object.__setattr__(
            self, "attributes", dict(self.attributes or {})
        )

    def cache_key(self) -> tuple[Any, ...]:
        """
        用于 Registry 缓存 runtime 的键。

        只取**影响解析与策略**的字段：能力要求、凭据覆盖、属性。
        顺序稳定（属性按键排序），否则同一个 context 两次会算出两个键，
        缓存形同虚设 —— 而那会让每次模型调用都重建一次 runtime。
        """

        capabilities = self.required_capabilities

        return (
            tuple(
                (name, getattr(capabilities, name))
                for name in ModelCapabilities.FIELDS
            )
            if capabilities is not None
            else None,
            self.credential_ref,
            tuple(sorted((str(k), str(v)) for k, v in self.attributes.items())),
        )


# ============================================================
# 小工具
# ============================================================


def new_invocation_id() -> str:
    """
    一次 `runtime.ainvoke` 的关联 id。

    它出现在每一次 attempt 的观测里，用来把"一次调用内部的多次尝试"
    与别的并发调用分开 —— 没有它，同一个 profile 的并发调用在日志里
    是糊在一起的。
    """

    return uuid4().hex


def _first_int(
        data: Mapping[str, Any],
        *keys: str,
) -> int | None:
    """按顺序取第一个"看起来是 token 数"的值。"""

    for key in keys:

        value = data.get(key)

        if isinstance(value, bool):
            continue

        if isinstance(value, int) and value >= 0:
            return value

        if isinstance(value, float) and value >= 0:
            return int(value)

    return None


def _duplicates(values: Sequence[str]) -> list[str]:
    """出现超过一次的取值（保持首次出现的顺序）。"""

    seen: set[str] = set()
    duplicates: list[str] = []

    for value in values:

        if value in seen and value not in duplicates:
            duplicates.append(value)

        seen.add(value)

    return duplicates
