"""
处理器注册表：把 `processor_key` 解析成实现。

与 PipelineRegistry 是同一套做法
--------------------------------
`app/services/document/pipeline/registry.py` 的 `found_capabilities()`
先 `_import_capability_modules()` 遍历包触发 import，再
`_leaf_subclasses(基类)` 收叶子类。这里逐字照抄那两个函数 ——
理由完全相同：`__subclasses__()` 只认识**已经被 import 的类**，
所以反射之前必须先把模块走一遍。

为什么注册表住在服务侧、实现住在 agent 侧
------------------------------------------
    app/agent/tools/processor/          实现（重试 / 超时 / 审批）
    app/services/tool/registry/         注册表（消费方）

与 document 那边完全对称：DocumentProcessor 不知道 PipelineRegistry
存在。注册表是"管理面要读的东西"，让服务层反过来 import agent 包
会把依赖方向拧反。

与 @tool 那套注册表的区别
-------------------------
`@tool` 是**声明即注册**（描述符在 `__set_name__` 里自己登记），
因为那个描述符活在 service 类体里、import 时必然执行。
处理器没有这样一个天然的挂载点（它们不被谁 import），因此走
反射发现 —— 与 document 的能力目录一致。
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from app.agent.tools.processor.base import BaseToolProcessor
from app.common.exceptions import NotFoundError

__all__ = [
    "PROCESSOR_PACKAGES",
    "ToolProcessorRegistry",
    "get_processor_registry",
]


#: 处理器实现所在的包 —— 反射发现的根。
#:
#: 新增一个实现：丢进这些包下的某个 .py 里，继承三个 type 基类之一，
#: 声明 name / supported_hooks / supported_execution_modes 即可。
#: 不用改注册表、不用改发现器、更不用改服务。
PROCESSOR_PACKAGES: tuple[str, ...] = (
    "app.agent.tools.processor",
)


def _import_processor_modules() -> None:
    """
    导入处理器包下的所有模块。

    `__subclasses__()` 只认识已经被导入的类，所以反射之前必须先把
    模块走一遍。不依赖各包的 `__init__.py` 逐个 re-export：
    新增一个 .py 文件就自动生效，不用回来改任何清单。

    ⚠️ `walk_packages` 不会走进没有 `__init__.py` 的目录 ——
    因此 processor/ 下的每个子包都必须留着那个空文件。
    """

    for package_name in PROCESSOR_PACKAGES:

        package = importlib.import_module(package_name)

        if not hasattr(package, "__path__"):
            continue

        for module_info in pkgutil.walk_packages(
                package.__path__,
                package.__name__ + ".",
        ):
            importlib.import_module(module_info.name)


def _leaf_subclasses(cls: type) -> list[type]:
    """
    深度优先收集 cls 下所有"最小子类"，即自己不再有子类的那些。

    与 pipeline/registry.py 的同名函数一致：中间基类（只做共性、
    这里就是三个 type 基类）不进结果 —— 它们对注册表没有意义。
    """

    result: list[type] = []

    for subclass in cls.__subclasses__():

        children = _leaf_subclasses(subclass)

        if children:
            result.extend(children)
        else:
            result.append(subclass)

    return result


class ToolProcessorRegistry:
    """
    按 `processor_key` 登记与获取处理器**实现**。

    构造后即冻结使用 —— 实现是代码侧事实，运行期不会变。
    """

    def __init__(
            self,
            processors: dict[str, BaseToolProcessor],
    ) -> None:
        self._processors = processors

    # ========================================================
    # 发现
    # ========================================================

    @classmethod
    def default(cls) -> "ToolProcessorRegistry":
        """按 PROCESSOR_PACKAGES 反射发现全部实现。"""

        return cls(cls.found_processors())

    @staticmethod
    def found_processors() -> dict[str, BaseToolProcessor]:
        """
        反射发现 BaseToolProcessor 的所有具体实现。

        三条排除规则，各自对应一种"这不是一个可挂载的实现"：

            BaseToolProcessor 的直接子类
                三个 type 基类本身。不能靠 `inspect.isabstract()`
                排除 —— 本模块的基类**故意没有抽象方法**（见
                base.py 的说明），因此它们是"可实例化"的。
                按结构排除：type 基类必须是直接子类，实现必须挂在
                它们之下。

            processor_type is None
                直接继承了 BaseToolProcessor 却忘了挂到某个 type
                基类下，因此类别未知。与 document 那边
                `role is None` 的能力不进目录是同一条规则。

            抽象类
                将来可能出现的中间基类（比如"所有需要审批通道的
                拦截器"）。与 document 一样靠 `isabstract()` 跳过。

        ⚠️ 键冲突与空 name 都**显式报错**，不用字典推导。
        推导会让后一个实现静默顶掉前一个 —— 症状是"某个处理器在
        能力目录里凭空消失"，没有任何报错，极难查。这与
        PipelineRegistry 的能力键冲突检查是同一套立场。
        """

        _import_processor_modules()

        result: dict[str, BaseToolProcessor] = {}

        for processor_cls in _leaf_subclasses(BaseToolProcessor):

            if BaseToolProcessor in processor_cls.__bases__:
                continue

            if processor_cls.processor_type is None:
                continue

            if inspect.isabstract(processor_cls):
                continue

            instance = processor_cls()

            key = instance.get_key()

            if not key:
                raise ValueError(
                    f"{processor_cls.__name__} 没有声明 name。"
                    "每个具体实现都要有自己的注册键 —— "
                    "空键会让它挂不上任何链，且在目录里无法区分。"
                )

            existing = result.get(key)

            if existing is not None:
                raise ValueError(
                    f"处理器键冲突：{key!r} 同时被 "
                    f"{type(existing).__name__} 与 "
                    f"{processor_cls.__name__} 占用。"
                    "每个实现都要声明自己的 name，不能沿用别人的 —— "
                    "否则先注册的那个会被静默顶掉。"
                )

            result[key] = instance

        return result

    # ========================================================
    # 读取
    # ========================================================

    def get(
            self,
            processor_key: str,
    ) -> BaseToolProcessor | None:
        """
        **读路径**：查不到返回 None。

        链里保留着"代码里已经删掉的处理器"的行，读链时不该因此
        报错 —— 返回 None 让组装层把 `processor` 字段整个省掉
        （见 application/_assembly.py）。
        """

        return self._processors.get(processor_key)

    def require(self, processor_key: str) -> BaseToolProcessor:
        """
        **写路径**：查不到抛 NotFoundError（→ 404）。

        与 `GET /processors/{processor_key}` 对未注册 key 的响应
        一致：那个定义本身就是一个资源，不存在就是 404。
        两类调用方共用一条消息，前端拿到的解释也就只有一种。
        """

        processor = self._processors.get(processor_key)

        if processor is not None:
            return processor

        raise NotFoundError(
            f"处理器实现不存在：{processor_key!r}。"
            f"当前注册的有：{self.keys()}。"
        )

    def contains(self, processor_key: str) -> bool:
        return processor_key in self._processors

    def keys(self) -> list[str]:
        return sorted(self._processors)

    def all(self) -> list[BaseToolProcessor]:
        return [
            self._processors[key] for key in self.keys()
        ]

    def __contains__(self, processor_key: object) -> bool:
        return processor_key in self._processors

    def __len__(self) -> int:
        return len(self._processors)


#: 进程级单例。
#:
#: 为什么不每次现建：`found_processors()` 要走一遍包 import +
#: 类实例化，而它会被**每个请求**的服务构造调到。document 那边
#: 的 PipelineRegistry 只在 lifespan 建一次，我们付不起这个成本。
#:
#: 与 app/services/tool/registry/registry.py 的 get_tool_registry()
#: 同一形状 —— 进程级事实就该只有一个。
_registry: ToolProcessorRegistry | None = None


def get_processor_registry() -> ToolProcessorRegistry:
    """取进程级注册表，首次调用时构建。"""

    global _registry

    if _registry is None:
        _registry = ToolProcessorRegistry.default()

    return _registry
