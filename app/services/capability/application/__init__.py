"""
Capability 模块的应用服务。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现
（同包同名的 X_impl.py）。服务的 __init__ 参数用仓储契约注解，
由 ServiceFactory 自动注入。

_assembly.py 不是服务，是实体 -> 契约的纯函数组装逻辑。放在这里
是因为 tool 模块读列表页时也要用它嵌一份能力进 ToolSummary ——
"能力长什么样"只有一份实现，才不会两处显示不一致。
"""

from .capability_service import CapabilityService
from .capability_service_impl import CapabilityServiceImpl

__all__ = [
    "CapabilityService",
    "CapabilityServiceImpl",
]
