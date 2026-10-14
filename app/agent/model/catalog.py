"""
模型目录（model.yaml）的加载与装配。

这一层只做两件事
---------------
    读一份 YAML -> ModelCatalog（已校验的声明）
    ModelCatalog -> DefaultModelRegistry（装配好的注册表）

它**不是**第二套配置系统
----------------------
本项目的配置约定一直是一份 YAML + `yaml.safe_load` + 启动期校验
（`app/config/datasource/*.yaml` 由数据源加载器读，`app/config/llm.yaml` 由
`app/common/llm.py` 读）。这里照同一套来，没有引入新的配置框架、
没有新的环境变量语法、没有第二份 DI 容器。

⚠️ 与 `app/config/llm.yaml` 的分工（重要）
-----------------------------------------
    llm.yaml        **连接**：base_url / api_key / provider 分派 / 客户端类
    model.yaml      **声明**：有哪些模型、它们什么能力、谁引用谁、什么策略

因此新增一个 provider 的完整步骤是两处各一行：

    llm.yaml    加一段连接信息（含 `${ENV}`）
    model.yaml  providers: 里加一条（`adapter: langchain`）+ 模型/绑定

两者都不含明文密钥：llm.yaml 用环境变量占位，model.yaml 只写
`credential_ref` 的**名字**。

配置在 `app/config/` 下的摆放
---------------------------
    app/config/llm.yaml          连接：base_url / api_key / 客户端类
    app/config/model.yaml        声明：本文件读的那份
    app/config/event_stream.yaml 事件流参数
    app/config/casdoor.yaml      鉴权
    app/config/datasource/*.yaml 数据源 —— **只有这个子目录**被
                                 infra/datasource/loader.py 扫描

本文件不在 datasource/ 里，因此不会被数据源加载器看到。这是刻意的：
加载器只 glob 交给它的那一个目录（`app/main.py` 给的是 datasource/），
所以"配置目录里混进不含 `type` 的 yaml"这件事从结构上就不会发生，
而不是靠"大家记得别放进去"。
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .adapters import LangChainProviderAdapter
from .credentials import CredentialResolver, EnvVarCredentialResolver
from .errors import ModelConfigurationError
from .observer import ModelCallObserver
from .provider import ProviderAdapter
from .registry import DefaultModelRegistry
from .resolver import ModelRoleBinding
from .types import (
    ModelCapabilities,
    ModelDefinition,
    ModelPricing,
    ModelProfile,
    ModelRuntimePolicy,
)

__all__ = [
    "DEFAULT_CATALOG_PATH",
    "ModelCatalog",
    "ProviderDeclaration",
    "build_model_registry",
    "get_model_registry",
    "load_model_catalog",
    "reset_model_registry",
    "set_model_registry",
]

logger = logging.getLogger(__name__)


#: 默认目录文件位置 —— app/config/model.yaml。
#:
#: 不再用 `Path(__file__).with_name(...)`：配置已经不住在本模块旁边了
#: （与 app/common/llm.py 的 DEFAULT_CONFIG_PATH 同形）。
DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "model.yaml"
)


#: 顶层允许的键。多一个就报错（拼错一个键的后果是**整段静默失效**，
#: 而"我明明配了"是最常见的排查起点）。
_TOP_LEVEL_KEYS = frozenset({
    "providers",
    "credentials",
    "models",
    "profiles",
    "aliases",
    "bindings",
})


@dataclass(frozen=True, slots=True)
class ProviderDeclaration:
    """`providers:` 里的一条 —— 只声明"用哪个适配器"，连接信息在 llm.yaml。"""

    name: str

    #: 适配器标识（`langchain`）。决定用哪个 ProviderAdapter 实现。
    adapter: str = "langchain"

    label: str = ""
    description: str = ""


@dataclass(frozen=True, slots=True)
class ModelCatalog:
    """
    一份**已校验**的模型目录。

    它是"配置读出来的样子"，还不认识 Runtime / Factory —— 装配是
    `build_model_registry` 的事。分开的好处是：目录可以被测试直接构造
    （不必有文件系统），也可以被将来的 `/models` 目录接口读取。
    """

    providers: tuple[ProviderDeclaration, ...] = ()

    models: tuple[ModelDefinition, ...] = ()

    profiles: tuple[ModelProfile, ...] = ()

    #: 顶层 `aliases:`：逻辑名 -> 模型 id 或 profile id。
    #:
    #: （模型自带的 `aliases:` 在 ModelDefinition 里，注册时自动生效。）
    aliases: Mapping[str, str] = field(default_factory=dict)

    bindings: tuple[ModelRoleBinding, ...] = ()

    #: `credential_ref -> 环境变量名`。**只有名字，没有秘密。**
    credentials: Mapping[str, str] = field(default_factory=dict)

    #: 这份目录从哪来（文件路径）。报错与日志里要说得出来。
    source: str = ""

    def __post_init__(self) -> None:

        object.__setattr__(self, "aliases", dict(self.aliases))
        object.__setattr__(self, "credentials", dict(self.credentials))

    # --------------------------------------------------------

    def model_ids(self) -> tuple[str, ...]:

        return tuple(item.id for item in self.models)

    def describe(self) -> dict[str, Any]:

        return {
            "source": self.source,
            "providers": [item.name for item in self.providers],
            "models": list(self.model_ids()),
            "profiles": [item.id for item in self.profiles],
            "aliases": dict(self.aliases),
            "bindings": [item.describe() for item in self.bindings],
            "credentials": sorted(self.credentials),
        }


# ============================================================
# 加载
# ============================================================


def load_model_catalog(
        path: str | Path | None = None,
) -> ModelCatalog:
    """
    读并校验一份模型目录。

    **不做环境变量展开**：这份文件里不该有秘密（只有 credential_ref
    的名字与 provider 参数），因此不需要那套 `${VAR}` 语法 —— 少了它
    就少一处"两套展开规则"的可能。

    校验分两轮：
        形状      每个字段的类型 / 必填 / 未知键（就地报错，带行号级的键名）
        引用      模型 -> provider、profile -> 模型、binding -> profile
                  （第二轮的报错必须说清"谁引用了不存在的东西"）
    """

    file_path = Path(path) if path else DEFAULT_CATALOG_PATH

    if not file_path.exists():
        raise ModelConfigurationError(
            f"模型目录文件不存在：{file_path}。"
            "该文件随代码一起提供，缺失说明部署不完整。"
        )

    try:
        raw = yaml.safe_load(file_path.read_text(encoding="utf-8")) or {}

    except yaml.YAMLError as exc:

        raise ModelConfigurationError(
            f"模型目录 {file_path} 不是合法的 YAML：{exc}"
        ) from exc

    if not isinstance(raw, Mapping):
        raise ModelConfigurationError(
            f"模型目录 {file_path} 的顶层必须是一个映射，"
            f"得到 {type(raw).__name__}。"
        )

    unknown = sorted(set(raw) - _TOP_LEVEL_KEYS)

    if unknown:
        raise ModelConfigurationError(
            f"模型目录 {file_path} 里有不认识的顶层键：{unknown}。"
            f"可用的是：{sorted(_TOP_LEVEL_KEYS)}。"
        )

    source = str(file_path)

    try:
        providers = _parse_providers(raw.get("providers"), source)
        credentials = _parse_credentials(raw.get("credentials"), source)
        models = _parse_models(raw.get("models"), providers, credentials, source)
        profiles = _parse_profiles(raw.get("profiles"), models, source)
        aliases = _parse_aliases(raw.get("aliases"), models, profiles, source)
        bindings = _parse_bindings(raw.get("bindings"), profiles, aliases, source)

    except ModelConfigurationError:
        # 已经是"带文件路径的配置错"（解析器自己抛的那种），原样向上。
        raise

    except ValueError as exc:

        # 领域类型构造时的校验（比如 profile 的 fallbacks 里有重复、
        # 模型缺 provider）。统一包成 ModelConfigurationError 并补上
        # **是哪个文件**：裸 ValueError 里没有文件名，而排查的第一步
        # 永远是"这是哪份配置的问题"。
        raise ModelConfigurationError(
            f"模型目录 {source} 里有不合法的声明：{exc}"
        ) from exc

    return ModelCatalog(
        providers=providers,
        models=models,
        profiles=profiles,
        aliases=aliases,
        bindings=bindings,
        credentials=credentials,
        source=source,
    )


# ============================================================
# 装配
# ============================================================


def build_model_registry(
        catalog: ModelCatalog | None = None,
        *,
        adapters: Mapping[str, ProviderAdapter] | None = None,
        observer: ModelCallObserver | None = None,
        credentials: CredentialResolver | None = None,
        path: str | Path | None = None,
        default_max_retries: int = 0,
) -> DefaultModelRegistry:
    """
    把一份目录装配成可用的注册表。

    装配顺序是**硬约束**（providers -> models -> profiles -> aliases ->
    bindings）：每一步都要求它引用的东西已经登记，因此顺序错了会在
    启动期报错，而不是运行期某个模型悄悄取不到。

    Parameters
    ----------
    adapters
        `adapter 名 -> ProviderAdapter 实例`。默认只有 `langchain`。
        接入自研网关时在这里加一项，**不用改**目录格式。
    credentials
        凭据解析器。默认按目录里的 `credentials:` 建一个环境变量解析器
        （缺变量时回落到 llm.yaml 里 provider 自己的凭据）。
    """

    resolved = catalog or load_model_catalog(path)

    impls: dict[str, ProviderAdapter] = {
        "langchain": LangChainProviderAdapter(),
    }

    for name, adapter in (adapters or {}).items():
        impls[name] = adapter

    registry = DefaultModelRegistry(
        observer=observer,
        credentials=(
            credentials
            if credentials is not None
            else EnvVarCredentialResolver(resolved.credentials)
        ),
        default_max_retries=default_max_retries,
    )

    for declaration in resolved.providers:

        adapter = impls.get(declaration.adapter)

        if adapter is None:
            raise ModelConfigurationError(
                f"provider {declaration.name!r} 声明的适配器 "
                f"{declaration.adapter!r} 没有实现。"
                f"可用的适配器：{sorted(impls)}。\n"
                "接入新的 provider 类型时，给 build_model_registry 传 "
                "adapters={...}，不要往目录里写一个不存在的名字。"
            )

        registry.register_provider(
            declaration.name,
            adapter,
            label=declaration.label,
            description=declaration.description,
        )

    for definition in resolved.models:
        registry.register_model(definition)

    for profile in resolved.profiles:
        registry.register_profile(profile)

    for alias, target in resolved.aliases.items():
        registry.register_alias(alias, target)

    for binding in resolved.bindings:
        registry.register_binding(binding)

    logger.info(
        "模型注册表已装配：%d 个 provider、%d 个模型、%d 个 profile、"
        "%d 条绑定（来源 %s）。",
        len(resolved.providers),
        len(resolved.models),
        len(resolved.profiles),
        len(resolved.bindings),
        resolved.source or "（内存目录）",
    )

    return registry


# ============================================================
# 进程级单例
# ============================================================
#
# 与 `app/services/tool/registry/processor_registry.py` 的
# `get_processor_registry()` 同一形状：模型注册表是**代码 + 配置**层面的
# 事实，一个进程一份。组合根（main.py 的 lifespan）调 `get_model_registry()`
# 拿到它并注入 register_react_graph；测试用 `set_model_registry()` 换一份。

_registry: DefaultModelRegistry | None = None


def get_model_registry() -> DefaultModelRegistry:
    """
    取进程级注册表，首次调用时按 `model.yaml` 装配。

    ⚠️ 首次调用会读文件并构造注册表，因此**应当在启动期**调（组合根），
    而不是在第一次真实对话里 —— 放在启动期，"目录里少一个模型"这类
    配置错会让应用起不来，而不是等某个用户踩到。
    """

    global _registry

    if _registry is None:
        _registry = build_model_registry()

    return _registry


def set_model_registry(
        registry: DefaultModelRegistry | None,
) -> None:
    """
    更换进程级注册表。

    组合根用它注入一份**定制装配**（额外的 provider adapter、观测者、
    凭据解析器）；测试用它注入一份内存目录。
    """

    global _registry

    _registry = registry


def reset_model_registry() -> None:
    """
    丢弃进程级注册表（**只给测试用**）。

    与 `reset_factories()` 同一条立场：生产上丢弃它没有正当理由 ——
    它缓存着 Runtime 与 provider 客户端，可能正被并发调用引用。
    """

    global _registry

    _registry = None


# ============================================================
# 解析（internals）
# ============================================================


def _parse_providers(
        raw: Any,
        source: str,
) -> tuple[ProviderDeclaration, ...]:

    if raw is None:
        return ()

    if not isinstance(raw, Mapping):
        raise _bad(source, "providers", "映射（provider 名 -> 声明）", raw)

    declarations: list[ProviderDeclaration] = []

    for name, body in raw.items():

        item = _mapping(source, f"providers.{name}", body)

        _reject_unknown(
            source,
            f"providers.{name}",
            item,
            {"adapter", "label", "description"},
        )

        declarations.append(
            ProviderDeclaration(
                name=str(name),
                adapter=str(item.get("adapter", "langchain")),
                label=str(item.get("label", "") or ""),
                description=str(item.get("description", "") or ""),
            )
        )

    return tuple(declarations)


def _parse_credentials(
        raw: Any,
        source: str,
) -> dict[str, str]:

    if raw is None:
        return {}

    if not isinstance(raw, Mapping):
        raise _bad(
            source,
            "credentials",
            "映射（credential_ref -> 环境变量名）",
            raw,
        )

    result: dict[str, str] = {}

    for ref, env_name in raw.items():

        if not isinstance(env_name, str) or not env_name.strip():
            raise _bad(
                source,
                f"credentials.{ref}",
                "非空字符串（**环境变量名**，不是密钥本身）",
                env_name,
            )

        result[str(ref)] = env_name.strip()

    return result


def _parse_models(
        raw: Any,
        providers: Sequence[ProviderDeclaration],
        credentials: Mapping[str, str],
        source: str,
) -> tuple[ModelDefinition, ...]:

    if raw is None:
        return ()

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise _bad(source, "models", "列表（每个模型一项）", raw)

    known_providers = {item.name for item in providers}
    definitions: list[ModelDefinition] = []

    for index, body in enumerate(raw):

        item = _mapping(source, f"models[{index}]", body)

        _reject_unknown(
            source,
            f"models[{index}]",
            item,
            {
                "id",
                "provider",
                "model",
                "display_name",
                "capabilities",
                "aliases",
                "credential_ref",
                "config",
                "pricing",
                "context_window",
                "structured_output_extra_body",
                "disable_thinking",
            },
        )

        provider = str(item.get("provider", "") or "")
        model = str(item.get("model", "") or "")

        if not provider or not model:
            raise _bad(
                source,
                f"models[{index}]",
                "至少要有 provider 与 model",
                item,
            )

        if provider not in known_providers:
            raise ModelConfigurationError(
                f"{source} 的 models[{index}] 用了未声明的 provider "
                f"{provider!r}。已声明：{sorted(known_providers) or '（无）'}。\n"
                "provider 必须在 providers: 里登记 —— 适配器靠它解析。"
            )

        model_id = str(item.get("id") or f"{provider}:{model}")

        credential_ref = item.get("credential_ref")

        if credential_ref is not None:
            credential_ref = str(credential_ref)
            if credential_ref not in credentials:
                # 只在**声明了 credentials 段**时才要求登记：不写
                # credentials 段的部署走"ref 直接当环境变量名"那条约定
                # （见 credentials.py），不该被这里拦下。
                if credentials:
                    raise ModelConfigurationError(
                        f"{source} 的 models[{index}] 引用了未登记的 "
                        f"credential_ref {credential_ref!r}。"
                        f"已登记：{sorted(credentials)}。"
                    )

        definitions.append(
            ModelDefinition(
                id=model_id,
                provider=provider,
                model=model,
                capabilities=_parse_capabilities(
                    source, f"models[{index}].capabilities", item.get("capabilities")
                ),
                aliases=tuple(
                    str(alias)
                    for alias in _list(
                        source, f"models[{index}].aliases", item.get("aliases")
                    )
                ),
                credential_ref=credential_ref,
                config=_mapping(
                    source, f"models[{index}].config", item.get("config")
                ),
                # 只对"要结构化输出"的调用生效，不是构造期参数 ——
                # 见 ModelDefinition.structured_output_extra_body 的说明。
                structured_output_extra_body=_mapping(
                    source,
                    f"models[{index}].structured_output_extra_body",
                    item.get("structured_output_extra_body"),
                ),
                # 上面那段 extra_body 的第二个用法：**一律**关（不看调用
                # 形状）。见 ModelDefinition.disable_thinking 的说明 ——
                # 它置 True 而上面那段为空时，ModelDefinition 会直接抛。
                disable_thinking=bool(item.get("disable_thinking", False)),
                pricing=_parse_pricing(
                    source, f"models[{index}].pricing", item.get("pricing")
                ),
                display_name=str(item.get("display_name", "") or ""),
                context_window=_optional_positive_int(
                    source,
                    f"models[{index}].context_window",
                    item.get("context_window"),
                ),
            )
        )

    return tuple(definitions)


def _parse_capabilities(
        source: str,
        where: str,
        raw: Any,
) -> ModelCapabilities:

    if raw is None:
        return ModelCapabilities()

    if not isinstance(raw, Mapping):
        raise _bad(source, where, "映射（能力名 -> bool）", raw)

    unknown = sorted(set(raw) - set(ModelCapabilities.FIELDS))

    if unknown:
        raise ModelConfigurationError(
            f"{source} 的 {where} 里有不认识的能力：{unknown}。"
            f"可用的是：{list(ModelCapabilities.FIELDS)}。\n"
            "能力名写错一个字母，后果是“这个模型看起来支持，运行时才发现"
            "不支持” —— 所以这里报错而不是忽略。"
        )

    return ModelCapabilities.from_dict(raw)


def _parse_pricing(
        source: str,
        where: str,
        raw: Any,
) -> ModelPricing | None:

    if raw is None:
        return None

    item = _mapping(source, where, raw)

    _reject_unknown(
        source, where, item, {"input_per_1k", "output_per_1k", "currency"}
    )

    def price(key: str) -> float | None:

        value = item.get(key)

        if value is None:
            return None

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _bad(source, f"{where}.{key}", "数字（每 1000 token）", value)

        return float(value)

    return ModelPricing(
        input_per_1k=price("input_per_1k"),
        output_per_1k=price("output_per_1k"),
        currency=str(item.get("currency", "USD") or "USD"),
    )


def _parse_profiles(
        raw: Any,
        models: Sequence[ModelDefinition],
        source: str,
) -> tuple[ModelProfile, ...]:
    """
    解析 profiles，并校验 primary / fallbacks **引用得到模型**。

    引用可以是模型 id，也可以是模型自带的别名 —— 后者是常用写法
    （"primary: flash" 比 "primary: deepseek:deepseek-v4-flash" 好读，
    而且换具体模型时只改一处）。

    校验放在这里（启动期）而不是运行期：`(agent_type, model_role)` 解析
    成功却引用了不存在的模型，是一种"看起来配好了"的坏状态。
    """

    if raw is None:
        return ()

    if not isinstance(raw, Mapping):
        raise _bad(source, "profiles", "映射（profile id -> 声明）", raw)

    references = {item.id for item in models}

    for item in models:
        references.update(item.aliases)

    profiles: list[ModelProfile] = []

    for profile_id, body in raw.items():

        where = f"profiles.{profile_id}"
        item = _mapping(source, where, body)

        _reject_unknown(
            source,
            where,
            item,
            {"primary", "fallbacks", "policy", "display_name", "description"},
        )

        primary = str(item.get("primary", "") or "")

        if not primary:
            raise _bad(source, f"{where}.primary", "模型 id 或模型别名", item)

        fallbacks = tuple(
            str(reference)
            for reference in _list(source, f"{where}.fallbacks", item.get("fallbacks"))
        )

        for reference in (primary, *fallbacks):

            if reference not in references:
                raise ModelConfigurationError(
                    f"{source} 的 {where} 引用了未登记的模型 {reference!r}。"
                    f"已登记：{sorted(references) or '（无）'}。"
                )

        profiles.append(
            ModelProfile(
                id=str(profile_id),
                primary=primary,
                fallbacks=fallbacks,
                policy=_parse_policy(
                    source, f"{where}.policy", item.get("policy")
                ),
                display_name=str(item.get("display_name", "") or ""),
                description=str(item.get("description", "") or ""),
            )
        )

    return tuple(profiles)


def _parse_policy(
        source: str,
        where: str,
        raw: Any,
) -> ModelRuntimePolicy:

    if raw is None:
        return ModelRuntimePolicy()

    item = _mapping(source, where, raw)

    _reject_unknown(
        source,
        where,
        item,
        {
            "timeout",
            "max_retries",
            "retryable_errors",
            "fallback_enabled",
            "total_timeout",
            "retry_backoff",
            "max_retry_backoff",
        },
    )

    def number(key: str, default: Any) -> Any:

        value = item.get(key)

        if value is None:
            return default

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise _bad(source, f"{where}.{key}", "数字", value)

        return float(value)

    total_timeout = _optional_number(
        source, f"{where}.total_timeout", item
    )

    retryable = item.get("retryable_errors")

    if retryable is None:
        retryable_errors = ModelRuntimePolicy().retryable_errors
    else:
        retryable_errors = tuple(
            str(code)
            for code in _list(source, f"{where}.retryable_errors", retryable)
        )

    max_retries = item.get("max_retries", ModelRuntimePolicy().max_retries)

    if isinstance(max_retries, bool) or not isinstance(max_retries, int):
        raise _bad(source, f"{where}.max_retries", "整数", max_retries)

    return ModelRuntimePolicy(
        timeout=number("timeout", ModelRuntimePolicy().timeout),
        max_retries=max_retries,
        retryable_errors=retryable_errors,
        fallback_enabled=bool(
            item.get("fallback_enabled", True)
        ),
        total_timeout=total_timeout,
        retry_backoff=number("retry_backoff", 0.0),
        max_retry_backoff=number("max_retry_backoff", 8.0),
    )


def _parse_aliases(
        raw: Any,
        models: Sequence[ModelDefinition],
        profiles: Sequence[ModelProfile],
        source: str,
) -> dict[str, str]:
    """
    顶层别名：逻辑名 -> 模型 id 或 profile id。

    ⚠️ 目标必须存在，且**在读取时就被摊平成一跳**
    --------------------------------------------
    注册表只支持一跳（别名不能指向别名），因此这里把"指向模型别名的
    别名"（`vision_default: kimi-code`）就地解析成模型 id
    （`vision_default: kimi:kimi-k2.7-code`）。摊平发生在**启动期**，
    于是运行期只需要一次字典查找。

    目标必须在同一份文件里已经存在：校验在这里做，是为了让"别名指向
    一个拼错的名字"在启动期暴露 —— 否则它会在某次模型调用时才以
    "某个角色解析不出来"的形式出现。
    """

    if raw is None:
        return {}

    if not isinstance(raw, Mapping):
        raise _bad(source, "aliases", "映射（别名 -> 目标）", raw)

    model_ids = {item.id for item in models}
    profile_ids = {item.id for item in profiles}

    #: 模型别名 -> 模型 id（用于把两跳摊成一跳）。
    model_alias_map = {
        alias: item.id
        for item in models
        for alias in item.aliases
    }

    targets = model_ids | profile_ids | set(model_alias_map)

    aliases: dict[str, str] = {}

    for alias, target in raw.items():

        name = str(alias)
        value = str(target)

        if value not in targets:
            raise ModelConfigurationError(
                f"{source} 的 aliases.{name} 指向的 {value!r} 既不是模型，"
                f"也不是 profile。可指向的目标：{sorted(targets) or '（无）'}。"
            )

        aliases[name] = model_alias_map.get(value, value)

    return aliases


def _parse_bindings(
        raw: Any,
        profiles: Sequence[ModelProfile],
        aliases: Mapping[str, str],
        source: str,
) -> tuple[ModelRoleBinding, ...]:

    if raw is None:
        return ()

    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise _bad(source, "bindings", "列表（每条一项）", raw)

    referenceable = {item.id for item in profiles}
    referenceable.update(aliases)

    bindings: list[ModelRoleBinding] = []

    for index, body in enumerate(raw):

        where = f"bindings[{index}]"
        item = _mapping(source, where, body)

        _reject_unknown(
            source,
            where,
            item,
            {"agent_type", "model_role", "profile", "attributes", "description"},
        )

        profile = str(item.get("profile", "") or "")

        if profile not in referenceable:
            raise ModelConfigurationError(
                f"{source} 的 {where} 引用了未知的 profile {profile!r}。"
                f"可引用：{sorted(referenceable) or '（无）'}。"
            )

        bindings.append(
            ModelRoleBinding(
                agent_type=str(item.get("agent_type", "") or ""),
                model_role=str(item.get("model_role", "") or ""),
                profile=profile,
                attributes={
                    str(key): str(value)
                    for key, value in _mapping(
                        source, f"{where}.attributes", item.get("attributes")
                    ).items()
                },
                description=str(item.get("description", "") or ""),
            )
        )

    return tuple(bindings)


# ============================================================
# 小工具
# ============================================================


def _bad(
        source: str,
        where: str,
        expected: str,
        actual: Any,
) -> ModelConfigurationError:
    """一条能直接照着修的配置错。"""

    return ModelConfigurationError(
        f"{source} 的 {where} 需要{expected}，"
        f"得到 {type(actual).__name__}: {actual!r}。"
    )


def _mapping(
        source: str,
        where: str,
        raw: Any,
) -> dict[str, Any]:

    if raw is None:
        return {}

    if not isinstance(raw, Mapping):
        raise _bad(source, where, "映射", raw)

    return {str(key): value for key, value in raw.items()}


def _list(
        source: str,
        where: str,
        raw: Any,
) -> list[Any]:

    if raw is None:
        return []

    if isinstance(raw, str) or not isinstance(raw, Sequence):
        raise _bad(source, where, "列表", raw)

    return list(raw)


def _optional_number(
        source: str,
        where: str,
        item: Mapping[str, Any],
) -> float | None:
    """
    读一个"可空数字"。

    三种写法是三件不同的事，必须分得开：

        没写这个键          -> 用默认值（见 ModelRuntimePolicy）
        写了 null           -> **显式不限**（total_timeout 的 None）
        写了数字            -> 用这个数
    """

    key = where.rsplit(".", 1)[-1]

    if key not in item:
        # 用默认策略里那一个 —— 不在这里再写一遍默认值，否则两处漂移。
        return getattr(ModelRuntimePolicy(), key)

    value = item[key]

    if value is None:
        return None

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _bad(source, where, "数字或 null", value)

    return float(value)


def _optional_positive_int(
        source: str,
        where: str,
        raw: Any,
) -> int | None:
    """
    读一个"可空的正整数"（`context_window` 那种）。

    写了就必须是正数：0 或负数会让预算分配直接算错，而那种错误的表现是
    "上下文被裁到只剩一条消息"，离配置很远。
    """

    if raw is None:
        return None

    if isinstance(raw, bool) or not isinstance(raw, int) or raw <= 0:
        raise _bad(source, where, "正整数或 null", raw)

    return raw


def _reject_unknown(
        source: str,
        where: str,
        item: Mapping[str, Any],
        allowed: set[str] | frozenset[str],
) -> None:

    unknown = sorted(set(item) - set(allowed))

    if unknown:
        raise ModelConfigurationError(
            f"{source} 的 {where} 里有不认识的键：{unknown}。"
            f"可用的是：{sorted(allowed)}。"
        )
