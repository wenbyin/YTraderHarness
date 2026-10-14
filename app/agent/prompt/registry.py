"""
节点提示词目录：反射发现 `prompts/` 下的模块级 `PROMPT`。

与其它注册表同一套路（见 app/agent/context/processor/registry.py）
----------------------------------------------------------------
    import_package_modules(...)  先把模块 import 一遍（否则看不到任何东西）
    遍历 / 收集                   这里扫的是**模块级变量**而不是子类，
                                  因此用 iter_package_modules 直接拿模块对象
    @lru_cache(maxsize=1)        进程一份；提示词是常量，没有失效问题
    MappingProxyType             返回只读视图，防止调用方就地改
    冲突显式报错                  不静默取其一

⚠️ 发现不到任何提示词也要报错
---------------------------
`pkgutil.walk_packages` **不走进没有 `__init__.py` 的目录**（见
app/common/utils/reflection.py 的三条坑）。所以"目录被删 / `__init__.py`
被删"的表现是**空表**而不是异常 —— 症状是每个 model_role 都
PromptNotFoundError，而原因离得很远。这里把空表直接当成装配错误报出来。

⚠️ 模块里没有 PROMPT 也报错
-------------------------
`prompts/` 下每一个文件都该声明一个 `PROMPT`。多一个文件却什么都没声明，
说明有人复制了模板没改完 —— 静默跳过它，等于"我加了提示词但没生效"。

**不**在这里 catch ImportError：包里某个模块 import 失败（写错了 import）
与"包不存在"是两回事，混在一个 try 里会把后者误报成前者。让原始异常带着
真实文件名往上抛（reflection.py 的 docstring 专门说过这一点）。
"""

from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from types import MappingProxyType

from app.common.utils.reflection import iter_package_modules

from .contracts import PromptDefinition, PromptRegistryError

__all__ = [
    "PROMPT_PACKAGES",
    "PROMPT_VARIABLE",
    "found_prompts",
    "reset_prompt_registry",
]


#: 提示词实现所在的包。加一个新包（比如把某类提示词拆出去）改这里。
PROMPT_PACKAGES: tuple[str, ...] = ("app.agent.prompt.prompts",)


#: 模块级声明用的变量名 —— 一个模块一份提示词。
PROMPT_VARIABLE = "PROMPT"


@lru_cache(maxsize=1)
def found_prompts() -> Mapping[str, PromptDefinition]:
    """
    全部节点提示词，键是 `model_role`。

    第一次调用时才 import 那些模块（`lru_cache` 之后不再走这条路），
    因此 import 错误发生在**第一次取提示词**时，而不是 import 本模块时。
    组合根在启动期预热一次，就是为了把它提前到启动日志，而不是第一个请求
    （见 app/main.py 的 5.2）。
    """

    found: dict[str, PromptDefinition] = {}

    #: key -> 声明它的模块名。只为了报错时说清"谁和谁撞了"。
    sources: dict[str, str] = {}

    for module in iter_package_modules(*PROMPT_PACKAGES):

        name = module.__name__
        definition = getattr(module, PROMPT_VARIABLE, None)

        if definition is None:
            raise PromptRegistryError(
                f"{name} 里没有 {PROMPT_VARIABLE}。\n"
                f"{PROMPT_PACKAGES[0]} 下的每个文件都必须声明一个"
                f"模块级 {PROMPT_VARIABLE}（PromptDefinition），"
                "否则它不会被任何地方取到 —— 而这个「没生效」是静默的。"
            )

        if not isinstance(definition, PromptDefinition):
            raise PromptRegistryError(
                f"{name} 的 {PROMPT_VARIABLE} 不是 PromptDefinition，"
                f"而是 {type(definition).__name__}。"
            )

        existing = sources.get(definition.key)

        if existing is not None:
            # 两个模块抢同一个 model_role：取其一等于"哪个生效取决于
            # import 顺序"，与 ToolRegistry 撞 key 同一套取舍。
            raise PromptRegistryError(
                f"model_role {definition.key!r} 的提示词被两个模块占用："
                f"{existing} 与 {name}。请删掉其中一个。"
            )

        sources[definition.key] = name
        found[definition.key] = definition

    if not found:
        raise PromptRegistryError(
            f"{PROMPT_PACKAGES[0]} 下没有发现任何提示词。\n"
            "最常见的原因：那个目录（或它的 __init__.py）被删了 —— "
            "pkgutil.walk_packages 不会走进没有 __init__.py 的目录，"
            "于是失败是**空表**而不是异常。"
        )

    return MappingProxyType(found)


def reset_prompt_registry() -> None:
    """
    丢弃发现结果（**只给测试用**）。

    发现结果只跟代码有关，不跟运行期状态有关 —— 生产路径上不需要它。
    """

    found_prompts.cache_clear()
