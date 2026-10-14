"""
配置里的 `${NAME}` / `${NAME:-default}` 占位符展开。

同一套语法在仓库里出现多处，这里给出**共用的一份**。数据源加载器
（infra/datasource/loader.py）**刻意不动**：那是那个类的私有方法，且
绑着数据源的语义（必填项缺失就跳过整个数据源），复用它反而要迁就那套
语义。本模块服务的是"不是数据源的那些配置"（llm.yaml、stream.yaml…）。

语义与 POSIX 的 `:-` 一致：变量**未设置**与设置为**空串**都算缺省。
缺省值省略时展开成**空串**，而不是抛异常 —— 这里没有"必填"的概念。
理由（沿用 app/common/llm.py 的立场）：数据源缺 host 就没法工作，早
失败是对的；而密钥 / 前缀这类配置缺了，只影响真正要走到的那条路径，
让它在启动时就炸会把不相干的场景一起拦下。
"""

from __future__ import annotations

import os
import re
from typing import Any

__all__ = ["ENV_PATTERN", "expand_env"]


#: 匹配 `${NAME}` 与 `${NAME:-default}`。
ENV_PATTERN = re.compile(
    r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}"
)


def expand_env(value: Any) -> Any:
    """
    递归展开配置里的占位符。

    dict / list 会**逐层**展开 —— 配置是嵌套的（providers 下面是各自的
    连接信息），只在顶层做替换等于大部分字段都没被展开。
    """

    if isinstance(value, dict):
        return {key: expand_env(item) for key, item in value.items()}

    if isinstance(value, list):
        return [expand_env(item) for item in value]

    if not isinstance(value, str):
        return value

    def replace(match: re.Match) -> str:

        name = match.group(1)
        default = match.group(2)

        env_value = os.getenv(name)

        # 用真值判断而不是 `is not None`：导出成空串的变量按"没配"处理，
        # 与 POSIX `:-` 一致。
        if env_value:
            return env_value

        return default if default is not None else ""

    return ENV_PATTERN.sub(replace, value)
