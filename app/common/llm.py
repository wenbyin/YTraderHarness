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

为什么配置读取放在 app/common 而不是 app/config
-----------------------------------------------
`app/config/` 被数据源加载器整目录扫描，多一个不含 `type` 的 yaml 会
**直接让启动失败**。理由与取舍写在 `llm.yaml` 的开头。

密钥只在用的时候才校验
----------------------
配置加载**不**校验密钥是否为空 —— 只跑检索、不跑 LLM 的场景不该被一条
缺失的 key 拦在门外。校验推迟到 `get_chat_model()`，那时报错才指向真正
的问题，并给出该设哪个环境变量。
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "LLMConfig",
    "LLMProviderConfig",
    "get_chat_model",
    "load_llm_config",
]


#: 走 `init_chat_model` 的 provider —— 值是传给它的 `model_provider`。
#:
#: 大多数家都在 `init_chat_model` 的支持列表里，这里列出**本项目实际用到**
#: 的那几个，而不是把 27 个全抄一遍：多抄一个就多一处会过期的信息。
_INIT_PROVIDERS: dict[str, str] = {
    "deepseek": "deepseek",
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
}

#: 默认配置文件位置 —— 与读它的本模块同目录。
DEFAULT_CONFIG_PATH = Path(__file__).with_name("llm.yaml")


#: `${NAME}` / `${NAME:-default}`。
#:
#: 与 infra/datasource/loader.py 的 `_expand_env` 是**同一套语法**，
#: 但那边是那个类的私有方法、且绑着数据源的语义（必填项缺失就跳过整个
#: 数据源），复用它反而要迁就那套语义。这里按同样的规则重写一份，
#: 语法一致、行为独立。
_ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
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
                "请检查 app/common/llm.yaml。"
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

    return LLMConfig.model_validate(_expand_env(raw))


def get_chat_model(
        provider: str | None = None,
        *,
        temperature: float | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        config: LLMConfig | None = None,
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

    惰性 import 的理由与 embedder/rerank 那边一致：这些客户端会拖起
    `openai` + `httpx` 一整串依赖，而本模块被 `app.common` 下的其他代码
    间接引用 —— 在模块顶层 import 它们会让"只想读个配置"的场景也付这份
    启动成本。
    """

    resolved = config or load_llm_config()

    name, provider_config = resolved.resolve(provider)

    if not provider_config.api_key.strip():
        raise ValueError(
            f"LLM provider {name!r} 没有配置 API Key。"
            f"请设置环境变量后重启服务"
            f"（见 app/common/llm.yaml 里 {name} 那段的 api_key）。"
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
        "api_key": provider_config.api_key,
        "timeout": timeout,
        "max_retries": max_retries,
    }

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

        return client_class(model=provider_config.model, **common)

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
            provider_config.model,
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
# internals
# ============================================================


def _expand_env(value: Any) -> Any:
    """
    递归展开配置里的 `${NAME}` / `${NAME:-default}`。

    与数据源加载器同一套语法（含 `:-` 的 POSIX 语义：未设置**和**设置为
    空串都算缺省）。区别只有一个：这里**没有"必填"的概念** —— 缺省值
    省略时展开成空串，而不是抛异常。

    理由：数据源缺 host 就没法工作，早失败是对的；而 LLM 的 key 缺了，
    只影响真正要调 LLM 的那条路径。让"没配 key"在启动时就炸，会把
    只用检索的场景一起拦下。
    """

    if isinstance(value, dict):
        return {key: _expand_env(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_expand_env(item) for item in value]

    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:
        name = match.group(1)
        default = match.group(2)

        env_value = os.getenv(name)

        if env_value:
            return env_value

        return default if default is not None else ""

    return _ENV_PATTERN.sub(replace, value)
