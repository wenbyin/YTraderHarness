"""
Context Builder 的发现与解析。

只有两个函数，不是一个注册表类
------------------------------
早先这里是一个 `ContextBuilderRegistry`（get / require / contains /
keys / all / __len__ / …），理由是"与另外四个注册表形状一致"。但那个
形状是被 document 流水线**逼出来的**：那边的能力由运行时数据选择
（`pipeline_config` 是数据库里的字符串），所以 `DocumentPipelineContext`
必须持注册表、每个节点都要查。

Context Builder 没有那个前提：本图用哪个 `builder_type` 是**编译期
常量**（`catalog.CONTEXT_BUILDER_TYPE`）。于是：

    解析在**启动时**做一次（组合根调 create_context_builder）
    实例直接进 ReactContext，执行路径上不再查表

一个 9 方法的类服务 1 个调用方，不值。剩下两个函数各自对应一件真实
的事：**发现全部**（启动期校验 + 将来的 `/context-builders` 目录）
与**按类型取一个**（组合根接线）。

实例是**共享的**，每个类型只构造一次
------------------------------------
`found_builders()` 缓存了结果，因此每个 `builder_type` 只有一个实例，
进程内所有并发的模型调用共用它。

这之所以安全，靠的是 `BaseContextBuilder.__setattr__` 封死了实例属性
—— 实例上**不可能**存在"这次调用"的状态。那不是一句约定：约定挡不住
`self.foo = ...`，而违反的症状是"偶发地拼错了上下文"，几乎不可能从
日志里看出来。机器保证让违约在写下的那一刻就炸。

为什么与实现同在 agent 侧（与 tool 模块不同）
--------------------------------------------
    发现函数要 import 基类才能反射子类。

放 `app/common` 会形成 `app/common -> app/agent`，而 app/agent 已经
import app/common（见 app/agent/tools/catalog_tool.py）—— 包级成环。

（枚举是另一回事：纯词汇、零依赖，所以放 app/common 名副其实。见
app/common/context/__init__.py。）

反射三件事由 app/common/utils/reflection.py 承担
------------------------------------------------
`walk_packages` 不认没有 `__init__.py` 的目录、必须先 import 后
`__subclasses__`、抽象类要在结果上过滤而不能在递归里剪枝 —— 那三条
坑的说明只写在共享内核那一份里。
"""

from __future__ import annotations

from functools import lru_cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from app.common.context import ContextBuilderType
from app.common.exceptions import NotFoundError
from app.common.utils import (
    import_package_modules,
    leaf_subclasses,
)

if TYPE_CHECKING:
    from .builder.base import BaseContextBuilder

__all__ = [
    "CONTEXT_BUILDER_PACKAGES",
    "create_context_builder",
    "found_builders",
]


#: 实现所在的包 —— 反射发现的根。
#:
#: 新增一个实现：丢进这些包下的某个 .py 里，挂到某个 type 基类之下，
#: 声明 enabled_stages 并实现对应阶段方法即可。不用改本模块、不用改
#: 发现逻辑、更不用改任何清单。
CONTEXT_BUILDER_PACKAGES: tuple[str, ...] = (
    "app.agent.context.builder",
)


@lru_cache(maxsize=1)
def found_builders() -> Mapping[ContextBuilderType, "BaseContextBuilder"]:
    """
    反射发现 `BaseContextBuilder` 的全部实现，`builder_type` -> 实例。

    四条排除规则，各自对应一种"这不是一个可用的实现"：

        BaseContextBuilder 的直接子类
            六个 type 基类本身。**不能靠 `inspect.isabstract()` 排除**
            —— 本模块的基类故意没有抽象方法（见 base.py 的说明），
            因此它们是"可实例化"的。按结构排除。

        builder_type is None
            直接继承了 BaseContextBuilder 却忘了挂到某个 type 基类下
            （`__init_subclass__` 已经会拦，这里是第二道）。

        抽象类
            将来可能出现的中间基类（比如"所有需要知识库检索的
            builder"）。

        空 enabled_stages
            声明不参与任何阶段的实现什么都构建不出来（`build()` 会
            直接抛）。它不是"合法的退化实现"，是没写完。

    ⚠️ 键冲突**显式报错**，不用字典推导。推导会让后一个实现静默顶掉
    前一个 —— 症状是"某个 builder 凭空消失"，没有任何报错。

    ⚠️ `status=DISABLED` 的实现**照常返回**：目录要能按
    `include_disabled` 列出它（规格里 disabled 只是元数据，不是删除）。
    在这里过滤它会让目录与实际能取到的东西对不上。

    结果被 `lru_cache` 缓存并包成只读视图 —— 每个类型只构造一次实例
    （共享的前提），而调用方改不坏缓存。测试要重来一遍就调
    `found_builders.cache_clear()`。

    ⚠️ 只发现**住在本包**的实现（见下面 prefixes 的说明）
    -------------------------------------------------
    这条过滤同时挡住"测试里的桩被当成生产实现"与"某个模块顺手继承
    BaseContextBuilder 就进了注册表"两种事故。
    """

    import_package_modules(*CONTEXT_BUILDER_PACKAGES)

    from .builder.base import BaseContextBuilder

    #: 只收**住在 CONTEXT_BUILDER_PACKAGES 里**的实现。
    #:
    #: `__subclasses__()` 是**进程级**的：任何被 import 过的模块里定义的
    #: 子类都在里面（测试里的桩、脚本里的变体、将来某个模块"顺手"继承
    #: 的一个类……）。不加这条过滤，"这个进程 import 过什么"就会决定
    #: "生产上用的是哪个 Context Builder" —— 而那种事故的两种长相都很糟：
    #:
    #:     键冲突（两个实现抢同一个 builder_type）-> 应用**起不来**
    #:     静默顶掉（顺序恰好反过来）-> 生产用了测试桩，且毫无迹象
    #:
    #: 与"实现必须丢进这些包"这条约定一致：发现只认那些包。
    prefixes = tuple(f"{name}." for name in CONTEXT_BUILDER_PACKAGES)

    result: dict[ContextBuilderType, "BaseContextBuilder"] = {}

    for builder_cls in leaf_subclasses(
            BaseContextBuilder, skip_abstract=True
    ):

        if not str(builder_cls.__module__).startswith(prefixes):
            continue

        if BaseContextBuilder in builder_cls.__bases__:
            continue

        if builder_cls.builder_type is None:
            continue

        if not builder_cls.enabled_stages:
            continue

        # 零参构造：与 tool / document 两个注册表一样，实现必须能
        # 无参建出来。依赖走应用级工厂（get_service_factory），
        # 不走构造注入 —— 见 base.py 关于"实例上不能有状态"的说明。
        instance = builder_cls()

        key = instance.get_key()

        existing = result.get(key)

        if existing is not None:
            raise ValueError(
                f"Context Builder 键冲突：{key.value!r} 同时被 "
                f"{type(existing).__name__} 与 "
                f"{builder_cls.__name__} 占用。"
                "一个 builder_type 只能有一个实现 —— 否则先注册的"
                "那个会被静默顶掉，而症状是「某个 builder 没生效」。"
            )

        result[key] = instance

    return MappingProxyType(result)


def create_context_builder(
        builder_type: ContextBuilderType,
) -> "BaseContextBuilder":
    """
    按类型取一个 builder 实例。**组合根在启动时调。**

    没有这个类型的实现就抛 —— 用 `NotFoundError` 与
    `ToolProcessorRegistry.require` 一致（将来 `/context-builders/{type}`
    对未注册类型也正是 404）。同一条消息服务两处是划算的：安静降级
    会让模型收到一个没有 system prompt、没有历史、也没有工具的请求，
    而日志里一切正常。

    ⚠️ 调用时机是**启动期**，不是每次模型调用 —— 见模块 docstring。
    放在启动期的好处是"没有任何 agent_invoke 实现"会让应用起不来，
    而不是等第一次真实对话才失败。
    """

    builder = found_builders().get(builder_type)

    if builder is not None:
        return builder

    raise NotFoundError(
        f"Context Builder 实现不存在：{builder_type.value!r}。"
        f"当前注册的有："
        f"{[k.value for k in sorted(found_builders(), key=lambda k: k.value)] or '（无）'}。"
    )
