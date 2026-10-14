"""
Prompt 模块的应用服务。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现（同包同名的
X_impl.py）。

本模块的服务**不依赖任何仓储**（提示词还在 catalog.py 里硬编码），因此
`ServiceFactory.needs_session()` 会判成 False —— 组合根/调用方用
`create()` 构造它，不会为了一段文本去开数据库会话。
"""

from .prompt_service import PromptService
from .prompt_service_impl import PromptServiceImpl

__all__ = [
    "PromptService",
    "PromptServiceImpl",
]
