"""
Memory 模块的 SQLAlchemy 模型。

import 本包 = L0 的两张挂载表 + L1 的三张表与水位表 + L2/L3 的两张画像表
与 L2 水位表注册完毕。

    L0（挂 agent_message）   agent_message_fts / agent_message_vec
    L1（挂 memory_atom）      memory_atom 本体 + memory_atom_fts / memory_atom_vec + memory_atom_cursor
    L2                       memory_scene 本体（无挂载表）+ memory_scene_cursor 水位
    L3                       memory_persona 本体（无挂载表）+ memory_persona_cursor 水位

⚠️ L2/L3 **没有虚表**，这是刻意的
-----------------------------
L0/L1 都挂了关键词 + 向量两张派生表，因为它们的正文是要被**检索**的。
画像不是：它是整段拼进提示词的（场景清单 + 全文），当前没有任何召回入口。
真要做再加挂载表 —— 那是派生数据，源就是这两张本体表。

⚠️ 这里**没有** relationship，也不该有
---------------------------------
这些表都只通过 id 列指向别的表，而 `agent_message` 自己就刻意没有外键
（见它类 docstring 的取舍）。给挂载表加上外键会造出一条单向的强依赖：
会话被删（将来的端点）就删不动这个消息，或者级联把索引删掉而消息还在
—— 两种都不是这里想要的语义。

存亡由**写入方**负责：L0 的索引按 run_id 重写，L1 的记录跟着 memory_atom
的原地更新走（水位行只在 memory.l1.extract 成功时推进），L2 的画像行由
memory.l2.extract 按主键 upsert（水位行只在它成功时推进）。唯一的例外
说明见 memory_atom_cursor.py 与 memory_scene_cursor.py 的模块 docstring（它们
严格说是权威状态，不是派生索引）。

⚠️ memory_atom_vec 与 agent_message_vec 需要 pgvector 扩展
-----------------------------------------------
扩展没装时，本包的 `create_all` 会报 `type "vector" does not exist`。
因此 import 本包只应该在扩展就位之后 —— 顺序见
models/sqlalchemy/agent_message_vec.py 的模块 docstring。
（L2/L3 那三张表与 pgvector 无关，但它们同包，所以照样受这条顺序约束。）
"""

from .agent_message_fts import AgentMessageFts
from .agent_message_vec import AgentMessageVec
from .memory_atom_cursor import MemoryAtomCursor
from .memory_atom_fts import MemoryAtomFts
from .memory_atom import MemoryAtom
from .memory_atom_vec import MemoryAtomVec
from .memory_scene_cursor import MemorySceneCursor
from .memory_persona import MemoryPersona
from .memory_persona_cursor import MemoryPersonaCursor
from .memory_scene import MemoryScene

__all__ = [
    "AgentMessageFts",
    "AgentMessageVec",
    "MemoryAtomCursor",
    "MemoryAtomFts",
    "MemoryAtom",
    "MemoryAtomVec",
    "MemorySceneCursor",
    "MemoryPersona",
    "MemoryPersonaCursor",
    "MemoryScene",
]
