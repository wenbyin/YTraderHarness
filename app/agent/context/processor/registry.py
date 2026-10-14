"""
Context Processor 的反射发现（第 5 个注册表）。

与另外四个注册表同一套做法
------------------------
    app/services/document/pipeline/registry.py        写入链能力
    app/services/document/retrieve/registry.py        查询期阶段
    app/services/tool/registry/processor_registry.py  工具处理器
    app/agent/context/registry.py                     Context Builder
    **本模块**                                        Context Processor

先遍历包触发 import（`__subclasses__()` 只认识**已经被 import 的类**），
再收叶子类。两个函数都在 `app/common/utils/reflection.py`（那份 docstring
把三个坑一次说清了：没有 `__init__.py` 的目录、先 import 后反射、
抽象类要在结果上过滤而不能在递归里剪枝）。

为什么是反射而不是一张清单
------------------------
新增一个 Processor = 在 `processors/` 下加一个 .py 文件 + 声明 spec +
实现 process，**不用回来改任何清单**。清单一定会漂移，而漂移的表现是
"我加的那个处理器没跑"（没有任何报错）。

冲突与非法声明都**显式报错**
------------------------
    没有 spec（不是 ContextProcessorSpec）  -> base.__init_subclass__ 已经拦
    name 为空 / 重复                        -> 这里报错
    order 重复                              -> pipeline 构造时报错
    gate 与 required 矛盾                   -> pipeline 构造时报错

零参构造是硬要求（与另外四个注册表一致）：依赖不走构造注入，而走
工作集上的依赖束（见 dependencies.py）。
"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from app.common.utils import import_package_modules, leaf_subclasses

if TYPE_CHECKING:  # pragma: no cover - 只为类型
    from .base import ContextProcessor
    from .pipeline import ContextProcessorPipeline

__all__ = [
    "PROCESSOR_PACKAGES",
    "create_pipeline",
    "found_processors",
]


#: Processor 实现所在的包 —— 反射发现的根。
#:
#: 新增一个实现：丢进这些包下的某个 .py 里，继承 ContextProcessor，
#: 声明 spec 并实现 process 即可。不用改注册表、不用改流水线、
#: 更不用改 Builder。
PROCESSOR_PACKAGES: tuple[str, ...] = (
    "app.agent.context.processor.processors",
)


@lru_cache(maxsize=1)
def found_processors() -> Mapping[str, "ContextProcessor"]:
    """
    反射发现 `ContextProcessor` 的全部实现，`name` -> 实例。

    排除规则只有一条（比另外四个注册表简单，因为 `process` 是
    `@abstractmethod`）：

        `inspect.isabstract()` 为真的类不是实现
            —— `ContextProcessor` 基类自己（它留着抽象的 process）、
               以及将来可能出现的抽象中间基类都在此列

    ⚠️ 这里**没有**"直接子类不是实现"那条规则
    --------------------------------------
    另外四个注册表要靠它排除 type 基类（`BaseContextBuilder` 的直接
    子类 = 六个 type 基类）。这里不需要：`ContextProcessor` 有抽象
    方法，`isabstract()` 已经把它自己排除掉了；而加上那条规则反而会
    把**所有**实现都排除掉（它们全都是直接子类 —— 处理器没有 type
    分类这一层）。这个坑实测踩过一次：发现结果为空，且不报错。

    ⚠️ 并且只收**住在 PROCESSOR_PACKAGES 里**的实现
    -------------------------------------------
    `__subclasses__()` 是**进程级**的：任何被 import 过的模块里定义的
    子类都在里面（测试里的桩、脚本里临时写的变体、将来某处"顺手"
    继承的一个类……）。不加这条过滤，"这个进程 import 过什么"就会决定
    "生产流水线里有哪些 Processor"。

    那种事故的两种长相都很糟：

        启动期 order 冲突（一个测试桩的 order 恰好与生产的一致）
        更糟：一个本不该上线的处理器**真的参与了每一次上下文构建**

    与"实现必须丢进这些包"这条约定一致：发现只认那些包。
    """

    import_package_modules(*PROCESSOR_PACKAGES)

    from .base import ContextProcessor

    prefixes = tuple(f"{name}." for name in PROCESSOR_PACKAGES)

    result: dict[str, ContextProcessor] = {}

    for processor_cls in leaf_subclasses(
            ContextProcessor, skip_abstract=True
    ):

        if not str(processor_cls.__module__).startswith(prefixes):
            continue

        instance = processor_cls()

        key = instance.get_key()

        if not key:
            raise ValueError(
                f"{processor_cls.__name__} 的 spec.name 为空。"
                "每个 Processor 都必须有自己的注册键 —— 空键会让它"
                "在观测里无法区分。"
            )

        existing = result.get(key)

        if existing is not None:
            raise ValueError(
                f"Context Processor 名冲突：{key!r} 同时被 "
                f"{type(existing).__name__} 与 "
                f"{processor_cls.__name__} 占用。"
                "name 是注册键也是观测名，重复会让"
                "「这一步是谁跑的」说不清。"
            )

        result[key] = instance

    return MappingProxyType(result)


def processors() -> tuple["ContextProcessor", ...]:
    """全部 Processor，按 order（注册表的便捷出口）。"""

    return tuple(
        sorted(
            found_processors().values(),
            key=lambda item: item.spec_of().order,
        )
    )


def create_pipeline(
        *,
        required_stages: object = (),
        processors_override: object = None,
) -> "ContextProcessorPipeline":
    """
    造一条流水线（Builder 在类构造期调一次）。

    Parameters
    ----------
    required_stages
        该 Builder 声明为 required 的阶段（见 builder/agent_invoke.py）。
        流水线用它校验"gate 与 required 不矛盾"。
    processors_override
        显式给一组 Processor（测试 / 定制装配）。默认用反射发现的结果。
    """

    from .pipeline import ContextProcessorPipeline

    items = (
        tuple(processors_override)  # type: ignore[arg-type]
        if processors_override is not None
        else processors()
    )

    return ContextProcessorPipeline(
        processors=items,
        required_stages=required_stages,  # type: ignore[arg-type]
    )
