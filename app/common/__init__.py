"""
共享内核。

各子包通过自己的 __init__.py 做统一出口，调用方从**包**导入，
不钻到叶子模块：

    from app.common.models  import Base        ✓
    from app.common.models.base import Base    ✗

这样叶子模块可以自由拆分 / 改名，调用点不受影响 ——
utils 已经按这个拆成了 file / http / string 三个模块，
调用方一句都不用改。
"""

__all__ = [
    "APIModel",
    "Base",
]

from .models import Base
from .schemas import APIModel
