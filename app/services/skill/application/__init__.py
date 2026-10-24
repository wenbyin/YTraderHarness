"""
Skill 模块的应用服务。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现
（同包同名的 X_impl.py）。服务的 __init__ 参数用仓储契约注解，
由 ServiceFactory 自动注入。

_assembly.py 不是服务，是实体 -> 契约的纯函数组装逻辑。
"""

from .skill_service import SkillService
from .skill_service_impl import SkillServiceImpl

__all__ = [
    "SkillService",
    "SkillServiceImpl",
]
