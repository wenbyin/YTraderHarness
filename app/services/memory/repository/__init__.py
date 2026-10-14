"""
Memory 模块的持久化。

契约 -> 实现的对应**没有任何登记表**：工厂靠 common/introspect.py 的
resolve_impl 反射发现（契约的子类，或同包同名的 X_impl.py）。与 capability /
tool / session 那几个模块同一套约定。

四个契约，按链分层：

    MemoryIndexRepository      L0 —— 挂在 agent_message 上的两张索引表
    MemoryAtomRepository       L1 —— memory_atom + 两张挂载表 + 会话级水位
    MemorySceneRepository      L2 —— memory_scene + 会话级水位（记忆写入时间，
                               不是消息序号）
    MemoryPersonaRepository    L3 —— memory_persona + **scope** 级水位

⚠️ 跨层读一律走**表的主人**那个契约
---------------------------
    L2 读 L1 的增量    MemoryAtomRepository.list_session_delta
    L3 读 L2 的增量    MemorySceneRepository.list_scene_delta
    L3 读自己的画像    MemoryPersonaRepository（本层）

自己写 SQL 去够别人的表，会让同一列的口径在两处各有一份答案 —— 而这类
分叉的症状是"读出来的那批和写进去的那批对不上"，不报错。
"""

from .atom import MemoryAtomRepository
from .atom_impl import MemoryAtomRepositoryImpl
from .memory import MemoryIndexRepository
from .memory_impl import MemoryIndexRepositoryImpl
from .persona import MemoryPersonaRepository
from .persona_impl import MemoryPersonaRepositoryImpl
from .scene import MemorySceneRepository
from .scene_impl import MemorySceneRepositoryImpl

__all__ = [
    "MemoryAtomRepository",
    "MemoryAtomRepositoryImpl",
    "MemoryIndexRepository",
    "MemoryIndexRepositoryImpl",
    "MemoryPersonaRepository",
    "MemoryPersonaRepositoryImpl",
    "MemorySceneRepository",
    "MemorySceneRepositoryImpl",
]
