"""
tool_execution 的两个纯函数。

放在应用层而不是仓储或 domain：它们**不定形状**（形状是
models/sqlalchemy 与 domain/schemas 的事），只是把两个"怎么算"
定在一处 —— 而这两件事今天都各只有一个正确算法。

为什么单独一个模块而不是塞进 _assembly.py
---------------------------------------
_assembly.py 的职责是"实体 -> DTO"的组装（见它的模块 docstring），
和这里的判重键、错误规范化没有关系。混在一起会让两个不相干的
变化互相牵连。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from app.common.utils import canonical_json, sha256_text

__all__ = [
    "execution_error",
    "param_hash_of",
]


def param_hash_of(
        arguments: Mapping[str, Any],
) -> str:
    """
    调用参数 -> 判重键的那一段。

    与 cache 拦截器同一套（`canonical_json` + `sha256_text`，见
    processor/interceptor/cache.py 的 `_cache_key`）：规范化 JSON 保证
    "键序不同但内容相同"算出同一个摘要，sha256 让键定长。

    ⚠️ 算的是**模型给的原始参数**，不是处理器改写之后的
    --------------------------------------------------
    `_wrap_tool_call` 在驱动处理器链**之前**算这个值，因此
    privacy_transformer 的抹除、BEFORE 拦截器补的默认值都不影响它。

    这是刻意的：判重的语义是"**模型**要求做同一件事两次"，而处理器
    的改写是**我们**对同一次要求做的加工 —— 把它算进去会让"同一个
    要求"因为一次脱敏配置的变更而换了键，历史记录就对不上了。

    ⚠️ 代价要说清楚：两次**加工结果不同**的调用会被判成同一次。
    例如某个 BEFORE 处理器按当前时间补了一个参数，模型两次发出同样
    的调用，第二次会被拒。这种处理器与幂等记录天然冲突，配的时候
    要意识到（与 cache 拦截器面对的是同一个问题）。

    参数含不可 JSON 序列化的对象时退化成 `str()` —— 与 cache 那条
    一样的兜底（见 digest.canonical_json 的 default_str）。模型给的
    参数本来就是 JSON 形态，走不到那条路；真走到了说明调用方不是
    工具节点，那时"两个不同对象算出同一个键"的风险由调用方负责。
    """

    return sha256_text(
        canonical_json(dict(arguments))
    )


def execution_error(
        error: BaseException,
) -> dict[str, str]:
    """
    异常 -> `error` 列的 `{"type": ..., "message": ...}`。

    只存类型名与那段文案，**不存堆栈** —— 堆栈进日志。与 checkauth
    的"给模型的文案刻意含糊、细节留服务端日志"是同一条口径
    （processor/interceptor/auth.py 的 `_deny`）：这段东西会经
    ToolMessage 回到模型、进而可能被转述给用户，而堆栈里有路径、
    有内部结构。

    异常类型名留着是因为它对排查有用，且不泄露部署结构。
    """

    return {
        "type": type(error).__name__,
        "message": str(error),
    }
