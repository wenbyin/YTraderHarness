"""
Skill 模块的 SQLAlchemy 模型。

import 本包 = skill / skill_capability 两张表注册完毕。

⚠️ skill_capability 有一条**跨表**外键指向 capability（capability
模块），但没有 relationship —— 因此不需要两个模块的模型同时
import 才能配置 mapper；create_all 只需 capability 表已在
metadata 里。initializer 仍会把两个包都 import 一遍，保持与
tool / capability 模块相同的启动顺序。
"""

from .skill import Skill
from .skill_capability import SkillCapability

__all__ = [
    "Skill",
    "SkillCapability",
]
