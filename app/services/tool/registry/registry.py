"""
Tool 注册表：进程级的工具清单。

它怎么被填满
------------
**不是靠谁调用 register()**，而是 @tool 描述符在
`__set_name__` 里自己注册进来的（见 decorator.py）——
类体执行的那一刻就完成了登记，没有任何中央清单要维护。

那还要 scan() 干什么
--------------------
`__set_name__` 只在**模块被 import 时**触发。如果一个 service
模块恰好没被任何路径 import 到，它的工具就不在表里 ——
而"这个进程 import 过什么"是运行期状态，不该影响
"这个代码库里有哪些工具"这个事实。

所以同步之前先 scan()：把配置的包走一遍，缺的模块补上，
保证结果与进程的 import 状态无关。这与 document / retrieve /
tool-processor / context-builder 那四个注册表是同一个理由，走的是
同一个实现（app/common/utils/reflection.py）。

进程级单例
----------
与 app/common/model_factory.py 的 _factory + get_model_factory()
同一形状：装饰器在 import 期就要能拿到注册表，没有注入的入口。
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from types import ModuleType

from app.common.utils import iter_module_tree

from .spec import SOURCE_REF_VARIABLE, ToolSpec, derive_source_ref

__all__ = [
    "TOOL_SOURCE_PACKAGES",
    "ToolRegistry",
    "get_tool_registry",
    "iter_package_modules",
]

logger = logging.getLogger(__name__)


#: @tool 可能出现的包 —— 扫描的根。
#:
#: 新增一个带工具的模块：把它的 application 包加进来即可。
#: 不用改注册表、不用改发现器、更不用改同步服务。
TOOL_SOURCE_PACKAGES: tuple[str, ...] = (
    # @tool 声明所在：各业务模块的服务实现。
    "app.services.document.application",
    "app.services.file.application",
    # 记忆检索的两个工具（`memory.search` L0 原文 / `memory_atom.search`
    # L1 提炼记忆）都声明在这里的两个服务实现上。
    #
    # ⚠️ 漏了它 = 那两个 @tool **永远不会被登记**，而症状是"模型说它没有
    # 这个工具" —— 主 agent 的职责提示词里正写着"需要回忆时用 memory.search"，
    # 于是模型会去调一个不存在的工具。加这一行之前就是这个状态（2026-10-10
    # 发现并补上）。
    "app.services.memory.application",
    "app.services.tool.application",
    # langchain @tool 声明所在：跨多个 service 的组合工具。
    "app.agent.tools",
)


class ToolRegistry:
    """按 tool_key 登记与获取工具。"""

    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}

    # ========================================================
    # 登记
    # ========================================================

    def register(self, spec: ToolSpec) -> None:
        """
        登记一个工具。由 @tool 描述符调用，不给人手调。

        重复登记的判据是 **entrypoint**（模块 + 类 + 方法名），
        不是函数对象本身：

            同 key + 同 entrypoint  -> 允许，覆盖
                （模块被 reload、或以两个名字 import 到同一个文件时
                 会走到这条；函数对象是新的，但工具还是同一个）

            同 key + 不同 entrypoint -> 报错
                （两个地方想用同一个 key 声明不同的工具，
                 静默取其一比报错危险得多 —— 与 ModelFactory
                 以及两个工厂的 register 是同一套取舍）
        """

        existing = self._specs.get(spec.tool_key)

        if existing is not None:
            if existing.entrypoint != spec.entrypoint:
                raise ValueError(
                    f"tool_key {spec.tool_key!r} 已被另一个方法占用："
                    f"{existing.entrypoint} != {spec.entrypoint}。"
                    "同一个 key 必须指向同一个工具。"
                )

            self._specs[spec.tool_key] = spec
            return

        self._specs[spec.tool_key] = spec

    # ========================================================
    # 读取
    # ========================================================

    def get(self, tool_key: str) -> ToolSpec | None:
        return self._specs.get(tool_key)

    def all(self) -> list[ToolSpec]:
        return sorted(
            self._specs.values(), key=lambda s: s.tool_key
        )

    def keys(self) -> list[str]:
        return sorted(self._specs)

    def clear(self) -> None:
        """清空。**只给测试用** —— 生产上清空注册表没有正当理由。"""

        self._specs.clear()

    def __contains__(self, tool_key: object) -> bool:
        return tool_key in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    # ========================================================
    # 扫描
    # ========================================================

    def scan(
            self,
            packages: tuple[str, ...] | None = None,
    ) -> list[ToolSpec]:
        """
        把配置的包走一遍，然后返回全量工具。

        **两种工具，两种收法**
        --------------------
        `@tool` 服务方法        走一遍包**只是触发 import** —— 真正的登记
                                发生在各模块的类体执行时（`__set_name__`）。
                                这里不解析模块内容，因此新增一个 .py
                                文件就自动生效，不用回来改本文件。

        模块级 langchain 工具   它**没有描述符协议**（是模块命名空间里的
                                一个 BaseTool 实例，不是类的方法），
                                import 不会登记任何东西 —— 必须遍历模块
                                属性去找。见 `_register_langchain_tools`。

        ⚠️ 第二种曾经整条缺失
        -------------------
        后果不是"少一个工具"，而是**同步与运行时对不上**：同步那条路
        （`LangchainToolScanner` → `tool_definition`）看得到它，而
        `select_specs` 只读本注册表 —— 于是库里明明有这条工具，模型却
        永远拿不到，每一轮只留一条容易被忽略的 WARNING。
        """

        target_packages = (
                packages
                if packages is not None
                else TOOL_SOURCE_PACKAGES
        )

        for package_name in target_packages:
            self._import_package(package_name)

        self._register_langchain_tools(target_packages)

        return self.all()

    def _register_langchain_tools(
            self,
            packages: tuple[str, ...],
    ) -> None:
        """
        登记模块级的 langchain 工具。

        ⚠️ 惰性 import：`scanner.py` 在**模块级** import 本模块
        （它要用 TOOL_SOURCE_PACKAGES 与 iter_package_modules），
        反向 import 只有放进函数体才不会成环。

        发现逻辑复用 `iter_langchain_tools` —— 与同步那条路是**同一个**
        遍历。各写一份的话，两处对"哪些算模块级工具"的判定早晚分叉，
        而那正是这个方法出现之前那个 bug 的形状。
        """

        from .scanner import iter_langchain_tools

        for value, module, attr_name in iter_langchain_tools(packages):

            declared = getattr(module, SOURCE_REF_VARIABLE, None)

            source_ref = (
                declared
                if isinstance(declared, str) and declared
                else derive_source_ref(module.__name__)
            )

            name = getattr(value, "name", None) or attr_name

            self.register(
                ToolSpec(
                    tool_key=f"{source_ref}.{name}",
                    name=name,
                    # 没有 owner 类 —— 执行时直接调 `tool`
                    # （见 infra/agent/tool_executor.py 的分支）。
                    tool=value,
                    description=getattr(value, "description", None),
                    source_ref=source_ref,
                )
            )

    def _import_package(self, package_name: str) -> None:
        """
        导入一个包下的全部子模块。

        走一遍的目的只是**触发 import** —— @tool 的登记发生在
        各模块的类体执行时（__set_name__），不是这里解析出来的。
        """

        for _ in iter_package_modules(package_name):
            pass


def iter_package_modules(
        package_name: str,
) -> Iterator[ModuleType]:
    """
    导入一个包下的全部子模块，并逐个交出来。

    两个调用点：`ToolRegistry.scan()`（触发 @tool 自注册）与
    `LangchainToolScanner`（找模块级的 BaseTool 实例）。

    遍历本身在 `app/common/utils/reflection.py`（本项目四个注册表共用
    同一套发现机制）。**本函数保留**，因为它是工具侧的公开出口
    （`__all__` 与 registry/__init__.py 都转发它），而且它带一条
    工具侧独有的策略：

        包不存在时只警告、不抛 —— TOOL_SOURCE_PACKAGES 里可能有还
        没落地的模块（例如 file 目前就没有带工具的服务），那不该让
        整次同步失败。

    ⚠️ 这条策略的作用域是**包自己的 import**，不含包内某个模块。
    "包不存在"与"包在、但里面某个 .py import 崩了"是两回事：前者该
    跳过，后者是真实的代码错误，必须让它炸出来。所以这里自己
    `import_module`，只把**遍历**交给 `iter_module_tree` —— 直接
    `yield from iter_package_modules(...)` 会让那个 try 连子模块的
    ImportError 一起吞掉，并把它们误报成"包不存在"。
    """

    try:
        package = importlib.import_module(package_name)

    except ImportError:
        logger.warning(
            "工具扫描：包 %s 不存在，跳过。", package_name,
        )
        return

    yield from iter_module_tree(package)


#: 进程级单例。装饰器在 import 期就要拿到它，没有注入的入口。
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """取进程级注册表。"""

    return _registry
