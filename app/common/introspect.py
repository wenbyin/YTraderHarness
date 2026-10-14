"""
契约 -> 实现的发现，以及构造参数的类型解析。

为什么要有这个模块
------------------
两个工厂都要回答同一个问题："这个契约对应哪个具体实现？"，以及
"这个构造参数该喂什么？"。

两件事都靠反射完成 —— **没有任何登记表**。因此：

    新增一个实现   不需要在任何地方 register()
    新增一个服务   不需要改任何中央文件

这也让 app.common 得以保持纯叶子：工厂不认识任何具体实现，
只在被调用时按调用方递进来的契约类型去找。
"""

from __future__ import annotations

import importlib
import inspect
import typing

__all__ = [
    "concrete_subclasses",
    "is_broad_type",
    "resolve_impl",
    "unwrap_annotation",
]


def concrete_subclasses(cls: type) -> list[type]:
    """
    递归找出 cls 之下所有**可实例化**的子类。

    抽象中间类要跳过 —— 本项目的能力端口就是多层继承
    （DocumentProcessor -> TextDocumentProcessor -> DocumentParser
    -> TextDocumentParser -> CommonTxtDocumentParser），
    中间那几层带 ABC 且不完整，实例化必然失败。

    与 app/common/utils/reflection.py 的 leaf_subclasses 同一套思路，
    但这里不要求"叶子"（没有子类），只要求"非抽象"。
    """

    found: list[type] = []

    for sub in cls.__subclasses__():
        if not inspect.isabstract(sub):
            found.append(sub)

        found.extend(concrete_subclasses(sub))

    return found


def resolve_impl(contract: type) -> type:
    """
    找出 contract 的具体实现。

    两条路，按顺序：

        1. 已经在内存里的具体子类
           同模块的 Impl（如 qa_dataset_service.py 里 ABC 与 Impl 同文件）
           必然已经 import 过，直接就能看到。

        2. 约定：契约所在模块 X.py 的实现是同级的 X_impl.py
           绝大多数契约与实现分处两个文件，此时 import 那个兄弟模块
           即可让 __subclasses__() 看到它。

    第 2 条是**约定优于配置**：不必登记，也不必在契约旁边维护一张表。
    代价是文件名必须守约定 —— 但整个项目本来就是这个命名。

    多个实现时抛错，而不是挑第一个：静默取其一比报错危险得多
    （与 ModelFactory.register 的取舍一致）。确实需要多实现时，
    调用方用 impl= 显式指定。
    """

    found = concrete_subclasses(contract)

    if not found:
        sibling = f"{contract.__module__}_impl"

        try:
            importlib.import_module(sibling)
        except ImportError:
            pass
        else:
            found = concrete_subclasses(contract)

    if not found:
        raise LookupError(
            f"{contract.__name__} 找不到具体实现。"
            f"已查过它的子类与约定模块 {contract.__module__}_impl 。"
            "要么实现类没写，要么它与契约不在同一个包下 —— "
            "后者用 create(..., impl=实现类) 显式指定。"
        )

    if len(found) > 1:
        names = ", ".join(sorted(c.__name__ for c in found))
        raise LookupError(
            f"{contract.__name__} 有多个实现（{names}），无法自动选择。"
            "请用 create(..., impl=实现类) 显式指定。"
        )

    return found[0]


def unwrap_annotation(annotation: typing.Any) -> typing.Any:
    """
    把 Optional[X] / X | None 拆成 X。

    构造参数里大量出现 `GraphDispatcher | None = None` 这种写法
    （既有依赖是可选的，也是"不传就自己造"的入口），
    匹配池子时必须看里面的那个类型。
    """

    if annotation is None or annotation is inspect.Parameter.empty:
        return None

    origin = typing.get_origin(annotation)

    # X | None 与 Optional[X] 都是 Union[X, None]
    if origin is typing.Union or str(origin) == "<class 'types.UnionType'>":
        args = [
            a for a in typing.get_args(annotation)
            if a is not type(None)
        ]

        if len(args) == 1:
            return args[0]

    return annotation


#: 匹配池子时要跳过的类型。
#:
#: 它们太宽：池子里随便一个字符串都会"是 str"，于是没有任何参数的
#: 默认值能生效。这类参数只按**名字**匹配池子，名字对不上就交给
#: 实现自己的默认值。
_BROAD_TYPES = frozenset({
    str, int, float, bool, bytes, bytearray,
    list, dict, set, frozenset, tuple,
    object, typing.Any,
})


def is_broad_type(tp: typing.Any) -> bool:
    """判断类型是否宽到不适合做"池子里有没有这种值"的匹配。"""

    if tp is None:
        return True

    if tp in _BROAD_TYPES:
        return True

    # list[X] / dict[str, X] 这类泛型别名：看它的原始类型。
    origin = typing.get_origin(tp)

    if origin is not None:
        return is_broad_type(origin)

    return not isinstance(tp, type)
