"""
云端 LLM 的配置与客户端。

用 init_chat_model 统一构造，但有两家不走它
-------------------------------------------
主路径是 langchain 的 `init_chat_model(model, model_provider=...)` ——
它按 provider 分派到对应的客户端类，调用方不必知道底下是 `ChatOpenAI`
还是 `ChatDeepSeek`。

**但 deepseek 与 kimi 的处境不一样**（实测 langchain 1.4.2）：

    deepseek   在 init_chat_model 的 27 个 provider 里 -> 直接可用
    kimi       **不在**，即使装了 langchain-moonshot -> 报 Unsupported

`langchain-moonshot` 本身是好的，它提供 `ChatMoonshot`（基类
`BaseChatOpenAI`），只是没被注册进 `init_chat_model` 的分派表。所以这一家
绕开它、直接构造 —— 见 `_DIRECT_PROVIDERS`。

两张表都只在 **app/common/llm.py** 里，加第三家时按它的归属选一张改。

（对比：本地 Ollama 那条路走的是 `langchain_ollama`，与这里**互不相干**。
本地那套服务的是 embedding / reranker 这类必须在本机跑的小模型；这里的
是生成式大模型，本机跑不动也没必要跑。）

配置住在 app/config/，但不在被扫描的那个子目录里
------------------------------------------------
`llm.yaml` 与 casdoor、模型目录、事件流并排放在 `app/config/` 下，
与读它的本模块**不同目录**。

它不会被数据源加载器扫到，因为加载器只 glob 交给它的那**一个**目录，
而 `app/main.py` 给它的是 `app/config/datasource/`（见
infra/datasource/loader.py）。所以"不含 `type` 的 yaml 混在配置目录里"
这件事在这里不成立 —— 它在上一层。

（历史：这份配置原先住在 app/common/ 下，为的是躲开加载器的整目录
扫描 —— 那时判据还是"没有 `type` 就 raise"，一个空文件都能让应用起不来。
那条判据后来放宽成"跳过 + 打 warning"，躲的理由就没了，文件搬回了
app/config/。见 tests/infra/test_datasource_loader.py。）

密钥只在用的时候才校验
----------------------
配置加载**不**校验密钥是否为空 —— 只跑检索、不跑 LLM 的场景不该被一条
缺失的 key 拦在门外。校验推迟到 `get_chat_model()`，那时报错才指向真正
的问题，并给出该设哪个环境变量。
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from time import perf_counter
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.common.utils import expand_env

__all__ = [
    "ChatModelWarmup",
    "LLMConfig",
    "LLMProviderConfig",
    "get_chat_model",
    "load_llm_config",
    "warmup_chat_models",
]


#: 走 `init_chat_model` 的 provider —— 值是传给它的 `model_provider`。
#:
#: 大多数家都在 `init_chat_model` 的支持列表里，这里列出**本项目实际用到**
#: 的那几个，而不是把 27 个全抄一遍：多抄一个就多一处会过期的信息。
#:
#: ⚠️ 值可以是**别名**：本项目自己的 provider 名（llm.yaml 的键）与
#: `init_chat_model` 认的名字不必相同。qwen 就是这一种 —— 它的端点是
#: OpenAI 兼容的（`/compatible-mode/v1`），`init_chat_model` 也不认得
#: "qwen" 这个名字，因此映射到它认得的 "openai"，由 base_url 指向
#: 阿里那台（base_url 由 get_chat_model 从 provider 配置透传，见下）。
_INIT_PROVIDERS: dict[str, str] = {
    "deepseek": "deepseek",
    "qwen": "openai",
    # 开发用的进程内 mock：端点也是 OpenAI 兼容，复用 openai 适配器。
    "mock": "openai",
}


#: `init_chat_model` **不认得**、但 langchain 生态里有独立包的 provider。
#:
#: ⚠️ 实测 langchain 1.4.2：`init_chat_model` 支持 27 个 provider，
#: **其中没有 moonshot** —— 哪怕 `langchain-moonshot` 已经装好。直接传
#: `model_provider="moonshot"` 会得到
#: `ValueError: Unsupported provider='moonshot'`。
#:
#: `langchain-moonshot` 本身是好的，它提供了 `ChatMoonshot`（基类是
#: `BaseChatOpenAI`），所以这一家绕开 `init_chat_model` 直接构造即可。
#:
#: 值 = (模块名, 类名)。写成字符串是为了**惰性 import** —— 见 get_chat_model。
_DIRECT_PROVIDERS: dict[str, tuple[str, str]] = {
    "kimi": ("langchain_moonshot", "ChatMoonshot"),
    "moonshot": ("langchain_moonshot", "ChatMoonshot"),
    # DeepSeek 的思考模式要求多轮对话回传 assistant 的
    # reasoning_content；langchain 的 OpenAI 兼容转换会把它丢掉。
    # 用子类在 _get_request_payload 里补回去（见该模块 docstring）。
    "deepseek": ("app.common.deepseek_chat", "ReasoningChatDeepSeek"),
}

#: 默认配置文件位置 —— app/config/llm.yaml。
#:
#: 不再用 `Path(__file__).with_name(...)`：配置已经不住在本模块旁边了。
#: 从 __file__ 往上推而不是写相对路径，理由和 tests 里那条一样 ——
#: 相对路径只在 CWD 恰好是仓库根时成立，而"读配置"这件事在 IDE 里点运行、
#: 从别处跑脚本时都会发生。
DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "config" / "llm.yaml"
)


class LLMProviderConfig(BaseModel):
    """一个提供方的连接信息。"""

    model_config = ConfigDict(extra="forbid")

    label: str = Field(description="展示用名称。")

    base_url: str = Field(
        description="OpenAI 兼容端点，**必须带 /v1**。",
    )

    model: str = Field(description="模型名。")

    api_key: str = Field(
        default="",
        description=(
            "密钥。**不要写死在这里** —— 用 `${VAR}` 从环境变量取。"
            "空串表示没配，调用时才会报错。"
        ),
    )

    temperature: float | None = Field(
        default=None,
        description=(
            "该 provider 的默认采样温度。**None = 根本不传这个参数**，"
            "由服务端用它自己的默认值。"
            "\n\n"
            "为什么默认是不传：temperature 的**合法范围由模型决定**，"
            "不是由我们决定。实测 kimi-k3 只接受 1 —— 传 0 会得到"
            " `400 invalid temperature: only 1 is allowed for this model`，"
            "而且整个窗口的生成直接失败。不传就没有这个风险，换模型"
            "也不用跟着改。"
            "\n\n"
            "只有确实需要「同一份输入每次出一模一样结果」时才在这里"
            "显式写一个该模型接受的值。"
        ),
    )


class LLMConfig(BaseModel):
    """整份 llm.yaml。"""

    model_config = ConfigDict(extra="forbid")

    default_provider: str = Field(
        description="不指定 provider 时用哪个。",
    )

    providers: dict[str, LLMProviderConfig] = Field(
        description="provider 名 -> 连接信息。",
    )

    def resolve(
            self,
            provider: str | None,
    ) -> tuple[str, LLMProviderConfig]:
        """
        定出这次用哪个提供方。

        返回 `(名字, 配置)` —— 名字要带回去，日志里说得出"这次是谁答的"。
        """

        name = provider or self.default_provider

        config = self.providers.get(name)

        if config is None:
            known = "、".join(sorted(self.providers)) or "（无）"
            raise ValueError(
                f"未知的 LLM provider: {name!r}。"
                f"已配置的有：{known}。"
                "请检查 app/config/llm.yaml。"
            )

        return name, config


@lru_cache(maxsize=4)
def load_llm_config(
        path: str | None = None,
) -> LLMConfig:
    """
    读配置并展开环境变量占位符。

    按路径缓存 —— 同一份配置在一次进程里只解析一次。改了 yaml 要重启
    才生效，与项目里其它配置（数据源、模型路径）的立场一致。
    """

    file_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not file_path.exists():
        raise ValueError(
            f"LLM 配置文件不存在：{file_path}。"
            "该文件随代码一起提供，缺失说明部署不完整。"
        )

    raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}

    return LLMConfig.model_validate(expand_env(raw))


def get_chat_model(
        provider: str | None = None,
        *,
        model: str | None = None,
        api_key: str | None = None,
        temperature: float | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        config: LLMConfig | None = None,
        **model_kwargs: Any,
) -> Any:
    """
    取一个配置好的 chat 客户端。

    类型标成 `Any` 是因为各 provider 返回的类不同（`ChatDeepSeek` /
    `ChatMoonshot` / ...），而且本模块对它们**惰性 import**，模块级拿不到
    那些类型。调用方按 langchain 的统一接口用即可（`ainvoke` /
    `with_structured_output`）。

    `temperature` 三级优先：

        get_chat_model(temperature=X)     本次调用显式指定
          ↓ 没有则
        llm.yaml 里该 provider 的 temperature
          ↓ 没有则
        **不传这个参数**，用服务端的默认值

    最后一档是默认行为，也是最重要的一档：temperature 的合法范围由模型
    决定。实测 `kimi-k3` 只接受 1，传 0 直接 400 并让整个窗口失败。不传
    就没有这个风险。

    `model` / `api_key` 是给 **Model 模块的 Factory** 留的覆盖口
    ----------------------------------------------------------
    本文件描述的是"一个 provider 怎么连"，而模型目录
    （`app/agent/model/model.yaml`）描述的是"有哪些模型、用哪份凭据"。
    同一个 provider 底下常常有多个模型（`deepseek-chat` /
    `deepseek-reasoner`），也可能有多把 key（生产 / 沙箱 / 多租户），
    所以这两个值必须能被调用方覆盖：

        model      覆盖 llm.yaml 里该 provider 的 model
                   （缺省 None = 不覆盖，保持既有行为）
        api_key    覆盖 llm.yaml 里该 provider 的 api_key
                   （缺省 None = 用配置里那份；给了就不再校验配置里
                   那份是否为空 —— 覆盖的意义就是不用它）

    这两个参数**不引入第二套配置**：连接信息、provider 分派、密钥展开
    仍然只有本文件这一份。Model 模块只是把"用哪个模型 / 哪份凭据"
    这两件事参数化。

    `**model_kwargs` 是**provider 参数的透传口**
    ------------------------------------------
    `max_tokens` / `top_p` / `frequency_penalty` / `extra_body` 这类
    参数**各家不同、也随模型变**，因此本文件不替它们定签名，原样交给
    客户端构造。模型目录（model.yaml 的 `config:`）里的键就走这条路。

    为什么值得留这个口子：不留下它，"某个模型要多一个参数"就只能改
    本文件 —— 而那意味着**每接一个模型都要动共享内核**。有了它，
    provider 特有的东西留在配置里（谁用谁负责），本文件只管"怎么连"。

    已具名的参数（provider / model / api_key / temperature / timeout /
    max_retries / config）不在其中：它们是**每个 provider 都成立**的
    共同部分，因此有明确的默认与文档。

    惰性 import 的理由与 embedder/rerank 那边一致：这些客户端会拖起
    `openai` + `httpx` 一整串依赖，而本模块被 `app.common` 下的其他代码
    间接引用 —— 在模块顶层 import 它们会让"只想读个配置"的场景也付这份
    启动成本。
    """

    resolved = config or load_llm_config()

    name, provider_config = resolved.resolve(provider)

    # 覆盖优先。`api_key` 给了就不再要求配置里那份非空 —— 显式覆盖
    # 的语义就是"不用那一份"。
    effective_key = (
        api_key if api_key is not None else provider_config.api_key
    )

    effective_model = model or provider_config.model

    if not effective_key.strip():
        raise ValueError(
            f"LLM provider {name!r} 没有配置 API Key。"
            f"请设置环境变量后重启服务"
            f"（见 app/config/llm.yaml 里 {name} 那段的 api_key）。"
        )

    # 三级优先：调用方显式指定 > provider 配置 > **不传**。
    #
    # 最后一档是"不传"而不是某个具体值，理由见函数 docstring ——
    # 给一个自认为安全的默认值（比如 0）会在只接受 1 的模型上直接
    # 400，而那是**整个窗口**的失败，不是一条样本的失败。
    effective_temperature = (
        temperature
        if temperature is not None
        else provider_config.temperature
    )

    common: dict[str, Any] = {
        "api_key": effective_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }

    # provider 特有参数原样透传。
    #
    # 不可能覆盖上面那几个：provider / model / api_key / temperature /
    # timeout / max_retries / config 都是**具名参数**，Python 不会把它们
    # 同时收进 **model_kwargs。
    common.update(model_kwargs)

    if effective_temperature is not None:
        common["temperature"] = effective_temperature

    # base_url 交给 provider 的默认值，只有在配置里**显式写了**才覆盖。
    # 写死传进去会让"用 provider 自己的默认地址"这个选项消失，
    # 而那正是 init_chat_model 存在的意义之一。
    if provider_config.base_url:
        common["base_url"] = provider_config.base_url

    # ---- 不走 init_chat_model 的那几家 --------------------
    direct = _DIRECT_PROVIDERS.get(name)

    if direct is not None:

        module_name, class_name = direct

        try:
            client_class = getattr(
                import_module(module_name), class_name,
            )

        except ImportError as exc:
            raise RuntimeError(
                f"provider {name!r} 需要 {module_name}，但它没有安装。"
                f"请 pip install {module_name.replace('_', '-')}"
                "（见 requirements.txt）。"
            ) from exc

        return client_class(model=effective_model, **common)

    # ---- 其余走 init_chat_model --------------------------

    try:
        from langchain.chat_models import init_chat_model

    except ImportError as exc:
        raise RuntimeError(
            "调用云端 LLM 需要 langchain，但它没有安装。"
            "请 pip install langchain（见 requirements.txt）。"
        ) from exc

    try:
        return init_chat_model(
            effective_model,
            model_provider=_INIT_PROVIDERS.get(name, name),
            **common,
        )

    except ValueError as exc:
        # init_chat_model 对不认识的 provider 会抛
        # "Unsupported provider=..."。这个报错本身不说明"该怎么修"，
        # 所以补上两条可行路径。
        raise ValueError(
            f"{exc}\n"
            f"provider {name!r} 不在 init_chat_model 的支持列表里。"
            "两条路：在 app/common/llm.py 的 _INIT_PROVIDERS 里给出它"
            "认得的别名，或在 _DIRECT_PROVIDERS 里登记对应的"
            "（模块名, 类名）直接构造。"
        ) from exc


# ============================================================
# 启动预热
# ============================================================


@dataclass(frozen=True, slots=True)
class ChatModelWarmup:
    """
    一次 provider 预热的账。

    `warmed` 与 `skipped` **都要**记：只报一句"预热完成"会让"某一家
    其实没装上 / 没配 key"这件事从启动日志里消失，而那正是之后某个
    fallback 候选第一次被用到时才会炸的东西。
    """

    warmed: tuple[str, ...] = ()

    #: `(provider 名, 原因)`。原因分两种，见 `_NotWorthWarming`。
    skipped: tuple[tuple[str, str], ...] = ()

    elapsed: float = 0.0

    def describe(self) -> str:
        """一行账，给启动日志用。"""

        parts = [f"已预热 {len(self.warmed)} 家"]

        if self.warmed:
            parts.append("（" + "、".join(self.warmed) + "）")

        if self.skipped:
            parts.append(
                "；跳过 "
                + "、".join(
                    f"{name}（{reason}）" for name, reason in self.skipped
                )
            )

        parts.append(f"；耗时 {self.elapsed:.2f}s")

        return "".join(parts)


class _NotWorthWarming(Exception):
    """
    **有理由地**跳过 —— 与"热失败了"是两件事。

    分开的理由：日志里"kimi（没配 API Key）"和
    "kimi（ImportError: No module named 'langchain_moonshot'）"
    该被读出不同的结论（一个是配置没开，一个是装少了包），
    而合成一句"预热跳过 2 家"两种都看不出来。
    """


def warmup_chat_models(
        config: LLMConfig | None = None,
) -> ChatModelWarmup:
    """
    把 provider SDK 的 **import 成本**挪到启动阶段。

    为什么需要
    ----------
    实测（loop_watchdog 抓到的现场）：进程起来后的**第一次**模型调用
    会卡在 `get_chat_model` 里的一串惰性 import 上 ——

        langchain.chat_models -> langchain_openai -> openai.types...

    （挂在 PyCharm 调试器下这一串实测 7 秒）。而它是**同步**的、跑在
    事件循环线程上，于是它按住循环的这段时间里，SSE 那条
    `XREAD ... BLOCK 1000` 的读超时定时器在循环恢复的瞬间先到期，
    把"回复早就到了"谎报成：

        TimeoutError: Timeout reading from localhost:6379

    也就是说：**那条 Redis 报错的根因在这里，与 Redis 无关**。完整
    链路见 app/common/loop_watchdog.py 与 app/agent/event/reader.py
    的模块 docstring。

    预热把这份成本从"第一个请求"挪到"服务就绪之前"——总额不变，
    但没人在这段时间里等。**它不是并发优化，是延迟搬家。**

    为什么只热 import、不建客户端
    -----------------------------
    建客户端需要凭据、需要 definition（每个模型的 timeout / 参数都
    不同），那是 `app/agent/model/` 的目录知识，不是本文件知道的。
    本函数只用它能回答的那件事：**这个 provider 的包该不该 import**。
    （要连客户端一起预热是另一件事，属于组合根 —— 见
    app/agent/model/factory.py 里 ModelClientFactory 的说明。）

    失败**不抛**
    ------------
    预热是优化，不是前置条件。装少了一个包不该让服务起不来（那家
    可能压根没被 model.yaml 引用）。每一家的结果都记在返回的
    `ChatModelWarmup` 里，由调用方决定怎么打日志。
    """

    resolved = config or load_llm_config()

    warmed: list[str] = []
    skipped: list[tuple[str, str]] = []

    started = perf_counter()

    for name in sorted(resolved.providers):

        try:
            _warm_one(name, resolved.providers[name])

        except _NotWorthWarming as exc:
            skipped.append((name, str(exc)))

        except Exception as exc:
            # 连类名一起记：ImportError / ValueError / AttributeError
            # 在这里的含义完全不同，丢掉类名就等于丢掉线索。
            skipped.append((name, f"{type(exc).__name__}: {exc}"))

        else:
            warmed.append(name)

    return ChatModelWarmup(
        warmed=tuple(warmed),
        skipped=tuple(skipped),
        elapsed=perf_counter() - started,
    )


def _warm_one(name: str, provider_config: LLMProviderConfig) -> None:
    """
    热一个 provider 会用到的那串 import。热不了就抛。

    **不建客户端、不联网** —— 只把模块拉进 `sys.modules`。

    这里热的是**两段**互相独立的惰性 import：客户端**构造**那段（本函数
    的主体）与**结构化输出**那段（`_warm_structured_output_path`）。
    少了后者的后果见它的说明 —— 那不是理论风险，是实测抓到过的现场。
    """

    if not provider_config.api_key.strip():
        raise _NotWorthWarming(
            "没配 API Key —— 它在 get_chat_model 里必然报错，"
            "为它付 import 成本不合理（那家的包很可能根本没装）"
        )

    direct = _DIRECT_PROVIDERS.get(name)

    if direct is not None:

        # 不走 init_chat_model 的那几家：模块名与类名就写在表里，
        # 类名正好是下面判断 mro 要用的东西。
        module_name, class_name = direct

        _warm_structured_output_path(
            getattr(import_module(module_name), class_name)
        )

        return

    # ⚠️ 与 get_chat_model 用的是**同一个表达式**。这里多写一份映射，
    # 就多一处会过期的信息（而且过期的方式很隐蔽：请求路径 import
    # 的是 A 包，预热热的是 B 包，表现为"预热了但没生效"）。
    key = _INIT_PROVIDERS.get(name, name)

    # 借 langchain 自己的分派表把 provider 键落到模块名上。
    #
    # 用**私有**函数（`_get_chat_model_creator`）是有意的取舍：它正是
    # `init_chat_model` 内部调用的那一个，而且带 lru_cache —— 这里热过
    # 之后，请求路径直接命中缓存。自己维护一份"provider -> 包名"的表
    # 则是抄 langchain 那 27 家，抄错一家的症状是"某个 provider 永远
    # 首调卡顿"，且不会被任何测试发现。
    #
    # 代价是绑了 langchain 的内部结构：它改名时这里会显式报错（下面
    # 那条 RuntimeError），而不是静默失效 —— 这正是可接受的降级。
    base = import_module("langchain.chat_models.base")

    creator = getattr(base, "_get_chat_model_creator", None)

    if creator is None:
        raise RuntimeError(
            "langchain 的 _get_chat_model_creator 不在了（内部结构变了）—— "
            "预热要跟着改，见 app/common/llm.py 的 _warm_one。"
        )

    # langchain 返回的是 `functools.partial(creator_func, cls=类)`，因此
    # 顺带就能拿到"将来会构造哪个客户端类"—— 下面判断 mro 要用。
    created = creator(key)

    _warm_structured_output_path(
        getattr(created, "keywords", {}).get("cls")
    )


def _warm_structured_output_path(client_class: Any) -> None:
    """
    热**结构化输出**那条分支上的惰性 import。

    为什么它是独立的一段
    --------------------
    预热 `langchain_openai` **碰不到它**。`BaseChatOpenAI._astream` 在
    payload 带 `response_format` 时走的是另一条分支：

        # langchain_openai/chat_models/base.py:2103
        response_stream = self.root_async_client.beta.chat.completions.stream(...)

    而 openai SDK 的 `client.beta` 是 **cached_property**，首次访问才

        # openai/_client.py:1212
        from .resources.beta import AsyncBeta

    —— 那一串会拉进 `resources/beta/realtime/**` 与
    `types/beta/realtime/**` 整棵生成树。构造客户端不碰 `.beta`，因此
    只热 `langchain_openai` 是**不生效**的：要等真发一次带
    `response_format` 的流式请求才付这份钱。

    本地实测这正是一条 7.4 秒停顿（loop_watchdog 的现场）：

        .../langchain_openai/.../base.py:2103 in _astream
          -> openai/_client.py:1212 in beta
             -> openai/resources/beta/beta.py:40
                -> .realtime.realtime -> types.beta.realtime.*

    而 `response_format` 正是本项目分类/工具调用（`with_structured_output`）
    每次都要带的东西 —— 所以它必然发生在**第一个请求**上。

    判据为什么看 mro 而不是 provider 名
    ----------------------------------
    `deepseek` 与 `kimi` 都不叫 "openai"，但它们的客户端
    （`ChatDeepSeek` / `ChatMoonshot`）都继承 `BaseChatOpenAI`，
    因此**同样**走这条分支。按 provider 名判断会漏掉它们。

    `client_class` 为 None（拿不到类）时**照样热**：漏热 = 那个 7.4 秒
    的停顿原样回来，多热 = 启动多花零点几秒。两者不对等，所以不确定
    时选多热。
    """

    mro = getattr(client_class, "__mro__", None)

    if mro is not None and not any(
            klass.__module__ == "langchain_openai.chat_models.base"
            for klass in mro
    ):
        # 不是 BaseChatOpenAI 的后代 —— 它有自己的 _astream，不走 .beta。
        return

    # 用 openai 自己的惰性钩子取符号（正是 _client.py 那一行做的事），
    # 而不是直接 import 子模块：它内部挪了位置也仍然解析得到。
    from openai.resources.beta import AsyncBeta  # noqa: F401
