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
保证结果与进程的 import 状态无关。这与
document/pipeline/registry.py 的 _import_capability_modules
是同一个理由、同一套写法。

进程级单例
----------
与 app/common/model_factory.py 的 _factory + get_model_factory()
同一形状：装饰器在 import 期就要能拿到注册表，没有注入的入口。
"""

from __future__ import annotations

import importlib
import logging
import pkgutil
from collections.abc import Iterator
from types import ModuleType

from .spec import ToolSpec

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

        走一遍的目的只是**触发 import** —— 真正的登记发生在
        各模块的类体执行时（__set_name__）。这里不解析模块内容，
        因此新增一个 .py 文件就自动生效，不用回来改本文件。
        """

        for package_name in (
                packages
                if packages is not None
                else TOOL_SOURCE_PACKAGES
        ):
            self._import_package(package_name)

        return self.all()

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

    两处要用，所以抽成公共函数：
        ToolRegistry.scan()       触发 @tool 自注册
        LangchainToolScanner      找模块级的 BaseTool 实例

    包不存在时只警告、不抛：TOOL_SOURCE_PACKAGES 里可能有还没落地
    的模块（例如 file 目前就没有带工具的服务），那不该让整次同步
    失败。

    不依赖各包的 __init__.py 逐个 re-export：新增一个 .py 文件就
    自动生效，不用回来改任何清单。
    """

    try:
        package = importlib.import_module(package_name)

    except ImportError:
        logger.warning(
            "工具扫描：包 %s 不存在，跳过。", package_name,
        )
        return

    if not hasattr(package, "__path__"):
        return

    for module_info in pkgutil.walk_packages(
            package.__path__,
            package.__name__ + ".",
    ):
        yield importlib.import_module(module_info.name)


#: 进程级单例。装饰器在 import 期就要拿到它，没有注入的入口。
_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """取进程级注册表。"""

    return _registry
