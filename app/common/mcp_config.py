"""
MCP provider 的配置。

⚠️ 文件名是 mcp_config.py 而不是 mcp.py
--------------------------------------
`mcp` 是第三方 SDK 的**顶层包名**。本文件住在 `app/common/`，一旦
有启动方式把 `app/common` 放进 sys.path（PyCharm 的 source root、
某些 pytest 配置、直接 `python` 某脚本），`import mcp` 就会解析到
这里，而不是 site-packages —— 症状是本项目自己的 MCP 客户端报

    cannot import name 'ClientSession' from 'mcp'

而 SDK 明明装着。改名是唯一稳妥的修法：配置模块没有理由占用一个
与依赖同名的顶层模块名。

为什么 provider 是**配置**而不是注解
------------------------------------
`@tool` 能自注册，前提是「工具是代码里的编译期对象」：装饰器有函数
或类可以挂，`__set_name__` 在 import 时完成登记。MCP 的工具在 import
时根本不存在 —— 它们是 server 的 `tools/list` 返回的**运行时事实**，
代码里没有任何对象可以挂注解。

provider 的连接参数（可执行文件路径、参数、环境变量、超时）是**部署
数据**：不同机器、不同环境各不相同，写进代码等于把路径与密钥钉死在
仓库里。所以这一层走配置，`tools/list` 走协议发现，两者不要混。

为什么默认 allowlist
--------------------
外部 server 升级可能**多出**工具（例如 `execute_trade`）。默认全放行
等于把新增能力自动交给模型；`allowlist` 让「多出来」表现为看不见，
而不是可执行。这是 MCP 最现实的风险：server 的发布节奏不由我们决定。

配置形状见 `app/config/mcp.yaml.template`。真文件已 gitignore ——
它装着本机路径与密钥，跟 providers.yaml 同一待遇。
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field

from app.common.utils.env import expand_env

__all__ = [
    "MCPProviderSettings",
    "MCPSettings",
    "MCPToolFilter",
    "get_mcp_settings",
    "mcp_config_path",
]

logger = logging.getLogger(__name__)

#: 真文件（gitignore）。模板入库。
_CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "mcp.yaml"
)


def mcp_config_path() -> Path:
    """配置文件的绝对路径。给测试与错误提示用。"""

    return _CONFIG_PATH


class MCPToolFilter(BaseModel):
    """
    一个 provider 暴露哪些工具。

    `all`        全放行（信任该 server 的发布节奏）
    `allowlist`  只放行 `allow` 里的（默认；空 = 一个都不放行）
    `denylist`   放行除 `deny` 之外的全部

    `deny` 在任何 mode 下都先判 —— 放一个「已知危险」的名单比
    在两个 mode 里各写一遍排除逻辑更难写错。
    """

    mode: Literal["all", "allowlist", "denylist"] = "allowlist"

    allow: list[str] = Field(default_factory=list)

    deny: list[str] = Field(default_factory=list)

    def admits(self, tool_name: str) -> bool:
        """这个名字是否可以暴露给我们自己的工具注册表。"""

        if tool_name in self.deny:
            return False

        if self.mode == "all":
            return True

        if self.mode == "allowlist":
            return tool_name in self.allow

        # denylist：上面已经排除 deny，其余全放行。
        return True


class MCPProviderSettings(BaseModel):
    """一个 MCP server 的连接与策略。"""

    enabled: bool = True

    #: 目前只实现 stdio。将来加 SSE / streamable HTTP 时，
    #: 这里加一个取值，管理器按它分支 —— 契约不提前编。
    transport: Literal["stdio"] = "stdio"

    #: stdio 的启动命令（可执行文件或命令名）。支持 ${ENV} 展开。
    command: str = ""

    args: list[str] = Field(default_factory=list)

    #: 追加/覆盖的环境变量。MCP SDK 会自动拼上安全的默认继承变量，
    #: 这里只需要写 server 额外需要的那些。
    env: dict[str, str] = Field(default_factory=dict)

    cwd: str | None = None

    #: 启动 + initialize + 首次 list_tools 的总预算。
    startup_timeout_ms: int = Field(
        default=15_000, ge=1_000, le=300_000,
    )

    #: 单次 tool 调用的墙钟上限（毫秒）。
    #:
    #: 这里是**传输层**的兜底；工具级超时由处理器链上的
    #: timeout_interceptor 负责。两者都要有：处理器链没挂的工具
    #: 也不该因为 server 卡死而永远挂着。
    call_timeout_ms: int = Field(
        default=120_000, ge=1_000, le=600_000,
    )

    tools: MCPToolFilter = Field(default_factory=MCPToolFilter)

    def usable(self) -> bool:
        """
        这个 provider 有没有足够的信息去连接。

        缺 command 的 stdio provider 只打 warning、不抛 —— 一份模板
        拷过来还没填路径时，不该让整个应用起不来。
        """

        return (
            self.enabled
            and self.transport == "stdio"
            and bool(self.command.strip())
        )

    @property
    def startup_timeout_seconds(self) -> float:
        return self.startup_timeout_ms / 1000

    @property
    def call_timeout_seconds(self) -> float:
        return self.call_timeout_ms / 1000


class MCPSettings(BaseModel):
    """`mcp.yaml` 的 `mcp:` 段。"""

    enabled: bool = True

    providers: dict[str, MCPProviderSettings] = Field(
        default_factory=dict,
    )

    def active_providers(
            self,
    ) -> dict[str, MCPProviderSettings]:
        """
        可以尝试连接的 provider。

        缺 command 的会被排除并留一条 warning —— 「配了但没填完」
        与「没配」是两件事，静默跳过会让排查从「为什么没有工具」
        变成「配置到底读没读」。
        """

        if not self.enabled:
            return {}

        ready: dict[str, MCPProviderSettings] = {}

        for name, provider in self.providers.items():

            if not provider.enabled:
                continue

            if not provider.usable():

                logger.warning(
                    "MCP provider %s 已启用但不可用（command 为空或"
                    " transport 不支持），本次跳过。", name,
                )
                continue

            ready[name] = provider

        return ready


@lru_cache(maxsize=1)
def get_mcp_settings() -> MCPSettings:
    """
    读 `app/config/mcp.yaml`。

    文件不存在 = MCP 整体关闭（返回 enabled=False 的空设置）。
    这是**默认状态**：没配 provider 的环境不该因为 import 了这个
    模块就多出一个扫描阶段。

    进程内缓存；测试改配置时调 `get_mcp_settings.cache_clear()`。
    """

    if not _CONFIG_PATH.exists():
        return MCPSettings(enabled=False)

    raw: Any = yaml.safe_load(
        _CONFIG_PATH.read_text(encoding="utf-8")
    ) or {}

    section = expand_env(raw.get("mcp") or {})

    return MCPSettings.model_validate(section)
