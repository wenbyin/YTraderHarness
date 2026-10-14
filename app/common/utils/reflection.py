"""
反射发现：遍历包、收集叶子子类。

本项目有**四个注册表**用同一套发现机制，它们各自抄了一份：

    app/services/document/pipeline/registry.py   写入链能力
    app/services/document/retrieve/registry.py   查询期阶段
    app/services/tool/registry/processor_registry.py   工具处理器
    app/agent/context/registry.py                Context Builder

四份 `_leaf_subclasses` 里三份逐字相同、一份行为不同（见下），
五份"遍历包并 import"的帮手只是名字不同。重复的代价不是行数，是
**同一条坑要踩四次** —— 那些坑（`__init__.py`、先 import 后
`__subclasses__`）每一份都得重新写一遍注释。因此统一到这里。

为什么能放在共享内核
--------------------
两个函数都是**纯标准库**：接收一个基类 / 一个包名，不 import
app.services / app.agent / app.infra。因此放在 app/common 不违反
"共享内核不反向依赖"这条规则（对比 Context Builder 的**注册表**：
它要 import 自己的基类才能反射，所以只能与基类同侧）。

⚠️ 坑一：`pkgutil.walk_packages` 不会走进没有 `__init__.py` 的目录
------------------------------------------------------------------
删掉某个子包的 `__init__.py`，`walk_packages` 会**静默跳过**它 ——
于是那个包里的实现全部消失，注册表返回空字典而不是报错。本项目多个
`__init__.py` 里都为此留了警告。

⚠️ 坑二：反射之前必须先把模块 import 一遍
-----------------------------------------
`__subclasses__()` 只认识**已经被 import 的类**。"代码里有哪些实现"
与"这个进程 import 过什么"是两件事，后者不该决定前者。所以先
`import_package_modules(...)` 再 `leaf_subclasses(...)`，顺序不能反。

⚠️ 坑三：抽象类要在**结果**上过滤，不能在递归里剪枝
--------------------------------------------------
这是本项目真踩过的坑：曾经有一份实现写成

    for subclass in cls.__subclasses__():
        if inspect.isabstract(subclass):
            continue          # ← 错：整棵子树都没了
        ...

它想表达的是"中间的抽象基类不注册"，但 `continue` 跳过的是**整个
分支**：一个挂在抽象中间基类下的**具体**实现永远不会被发现。而那种
写法恰恰是为"将来出现 BaseXxx 这类中间基类"准备的 —— 等真的出现了，
它自己声明的场景反而失效，且失败方式是"某个实现凭空消失"。

正确做法是先收**叶子**（自己没有子类的那些），再对结果过滤抽象 ——
中间基类本来就不是叶子，不会进结果。`skip_abstract=True` 就是这么做的。
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from collections.abc import Iterator
from types import ModuleType

__all__ = [
    "import_package_modules",
    "iter_module_tree",
    "iter_package_modules",
    "leaf_subclasses",
]


def iter_module_tree(
        package: ModuleType,
        *,
        recursive: bool = True,
        skip_private: bool = False,
) -> Iterator[ModuleType]:
    """
    遍历一个**已经导入**的包，import 并逐个交出它的子模块。

    与 `iter_package_modules` 的分工：那个负责"按名字找到包"，
    这个负责"走一遍包"。分开是因为**两者的错误语义不同** ——
    "包不存在"与"包里某个模块 import 失败"是完全两回事，混在一个
    try 里会把后者误报成前者（见 app/services/tool/registry/
    registry.py，它正是靠这个区分来决定是警告还是让它炸）。

    Parameters
    ----------
    recursive
        True（默认）走 `walk_packages`，**递归进子包**。
        实现按类型拆进子包时（比如
        app/agent/tools/processor/interceptor/）必须用这个。
        False 走 `iter_modules`，只看当前这一层。
    skip_private
        跳过 `_` 开头的模块。只在 `recursive=False` 时有意义 ——
        `walk_packages` 本来就会跳过下划线开头的模块名。

    传进来的不是一个包（没有 `__path__`，比如普通模块）时安静返回空 ——
    调用方拿到的就是"没有子模块"，不必自己判。
    """

    if not hasattr(package, "__path__"):
        return

    if recursive:

        for module_info in pkgutil.walk_packages(
                package.__path__,
                package.__name__ + ".",
        ):
            yield importlib.import_module(module_info.name)

        return

    for module_info in pkgutil.iter_modules(package.__path__):

        if skip_private and module_info.name.startswith("_"):
            continue

        yield importlib.import_module(
            f"{package.__name__}.{module_info.name}"
        )


def iter_package_modules(
        package_name: str,
        *,
        recursive: bool = True,
        skip_private: bool = False,
) -> Iterator[ModuleType]:
    """
    按**包名**导入并遍历它的模块。

    ⚠️ 这是个**生成器**：`importlib.import_module(package_name)` 在
    第一次 `next()` 时才执行。所以"包不存在"的 ImportError 不会在
    调用处抛，而要包住**迭代**才行：

        try:
            for module in iter_package_modules(name):
                ...
        except ImportError:
            ...

    要区分"包不存在"与"包里某个模块 import 失败"时，别用这个 ——
    自己先 `importlib.import_module` 再交给 `iter_module_tree`。
    """

    yield from iter_module_tree(
        importlib.import_module(package_name),
        recursive=recursive,
        skip_private=skip_private,
    )


def import_package_modules(
        *package_names: str,
        recursive: bool = True,
        skip_private: bool = False,
) -> None:
    """
    把若干包下的模块**全部 import 一遍**，不要返回值。

    用途只有一个：在 `leaf_subclasses()` 之前触发子类定义（坑二）。

    与 `iter_package_modules` 的关系就是"消费掉"——需要模块对象本身
    时用那个生成器（例如扫描模块级变量，见
    app/services/tool/registry/scanner.py）。
    """

    for package_name in package_names:

        for _ in iter_package_modules(
                package_name,
                recursive=recursive,
                skip_private=skip_private,
        ):
            pass


def leaf_subclasses(
        cls: type,
        *,
        skip_abstract: bool = False,
) -> list[type]:
    """
    深度优先收集 `cls` 下所有"叶子子类"，即自己不再有子类的那些。

    中间基类（只做共性：`TextDocumentParser`、三个 ToolProcessor
    type 基类、六个 ContextBuilder type 基类……）**不进结果** ——
    它们对注册表没有意义。

    Parameters
    ----------
    skip_abstract
        True 时把结果里的抽象类滤掉。注意滤的是**叶子**：中间基类
        本来就不是叶子，压根不会出现在结果里 —— 因此这个开关不会
        误伤"抽象中间基类下面的具体实现"（坑三）。

    ⚠️ 调用前必须先 import 相关模块（坑二），否则拿到空列表且不报错。

    ⚠️ `__subclasses__()` 是**弱引用**的，但"类定义期校验失败"的类
    会短暂留在里面：`__init_subclass__` 抛异常时，类对象已经被创建
    并登记过了。正常情况下这不构成问题 —— 那样异常会让模块 import
    整个失败，进程根本走不到注册。只有在你**捕获**了那个异常继续跑
    时（测试里常见），才会看到幽灵子类；此时注册表报的键冲突是**对的
    行为**，不是误报。
    """

    result: list[type] = []

    for subclass in cls.__subclasses__():

        children = leaf_subclasses(
            subclass, skip_abstract=skip_abstract
        )

        if children:
            result.extend(children)
            continue

        if skip_abstract and inspect.isabstract(subclass):
            continue

        result.append(subclass)

    return result
