"""
Mock 能力解析器：忽略真实 capability code，从固定工具池里随机选 N 个。

为什么需要它
----------
分类器返回的 `capability_needs` 是"数据库里有哪些能力"的语义结果；而
Phase 0 不一定有那套种子数据。没有映射时，主模型一个工具都看不到，
于是"随机 tool_call"那条链路永远不会被触发 —— 看起来像 mock 工具没生效。

这里的契约刻意**不依赖 code**：无论分类器说什么，都从
`mock_model.yaml` 的 `capability.tool_keys` 里选 N 个。这样前端联调能
稳定覆盖：

    capability_needs -> tool_keys -> bind_tools -> tool_calls -> 工具执行

真实环境仍然走 ToolQueryService 的数据库查询；本模块只在 --mock 时被
`agent_builder_node._tool_keys_for_capabilities` 调用。
"""

from __future__ import annotations

import random
import threading
from collections.abc import Sequence

from app.mock.settings import load_mock_config

__all__ = ["select_mock_tool_keys"]

_rng: random.Random | None = None
_lock = threading.Lock()


def _rng_instance() -> random.Random:
    """进程级 RNG；固定 seed 时便于复现同一组工具选择。"""

    global _rng

    if _rng is None:
        config = load_mock_config()
        _rng = random.Random(config.seed)

    return _rng


def select_mock_tool_keys(
        codes: Sequence[str] | None = None,
) -> list[str]:
    """
    从固定工具池里随机选 N 个 tool_key。

    Parameters
    ----------
    codes
        真实分类器给出的 capability code。这里**刻意不使用**，保留参数
        只是为了让调用点读起来是"解析这些 capabilities"，而不是一个
        看不懂的零参函数。
    """

    config = load_mock_config().capability
    pool = [str(key) for key in config.tool_keys if str(key).strip()]

    if not pool:
        return []

    with _lock:
        count = config.count.sample_int(_rng_instance())

    count = max(1, min(count, len(pool)))

    with _lock:
        return _rng_instance().sample(pool, count)
