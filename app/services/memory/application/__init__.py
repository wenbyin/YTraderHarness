"""
Memory 模块的应用层。

契约 -> 实现没有登记表：工厂靠 resolve_impl 反射发现（同包同名的
X_impl.py）。服务的 __init__ 参数用仓储契约注解，由 ServiceFactory 自动
注入；`pipeline_registry` 命中池子里的同名条目（见 app/main.py 的
build_factories），因此与 document 模块**共用**同一个 embedder 实例。

汇编六块（每层一个服务，外加一个只读的管理端查询服务）：

    memory_service.py       契约（L0：检索 + 索引两个入口）
    memory_service_impl.py  实现
    memory_atom_service.py      契约（L1：抽取 handler + 检索）
    memory_atom_service_impl.py 实现
    memory_scene_service.py      契约（L2：整合 handler，**没有检索**）
    memory_scene_service_impl.py 实现
    memory_persona_service.py      契约（L3：生成 handler，**没有检索**）
    memory_persona_service_impl.py 实现
    memory_query_service.py        契约（管理端：提取水位 + 跨层溯源，只读）
    memory_query_service_impl.py   实现
    trigger.py              "什么时候排哪条任务" —— L0 按运行、L1 按轮次、
                            L2 按 L1 完成后再延迟一会儿、L3 只要 L2 完成就排
                            （L3 的判定在执行侧，见那里）

memory_query_service 与那四层的关系值得单独说一句：它**不写任何东西**，
而且刻意跨层（水位按会话的归 L1/L2、按 scope 的归 L3），所以它既不是
"第五层"、也不该并进任何一个分层服务里。

⚠️ L2 / L3 的服务**不注入 embedder**
---------------------------
它们都没有向量路（画像没有挂载表，见 models/sqlalchemy/_scope.py 的说明），
所以与 document 的模型栈无关 —— 这也是它们能在冒烟里不起 embedder 就
跑起来的原因。`MemoryQueryService` 同理（纯 SQL，不调模型）。
"""

from .memory_atom_service import MemoryAtomService
from .memory_atom_service_impl import MemoryAtomServiceImpl
from .memory_persona_service import MemoryPersonaService
from .memory_persona_service_impl import MemoryPersonaServiceImpl
from .memory_query_service import MemoryQueryService
from .memory_query_service_impl import MemoryQueryServiceImpl
from .memory_scene_service import MemorySceneService
from .memory_scene_service_impl import MemorySceneServiceImpl
from .memory_service import MemoryService
from .memory_service_impl import MemoryServiceImpl
from .trigger import (
    idempotency_key,
    atom_task_key,
    persona_task_key,
    scene_task_key,
    submit_extract,
    submit_atom_extract,
    submit_persona_extract,
    submit_scene_extract,
    task_id_of,
)

__all__ = [
    "MemoryAtomService",
    "MemoryAtomServiceImpl",
    "MemoryPersonaService",
    "MemoryPersonaServiceImpl",
    "MemoryQueryService",
    "MemoryQueryServiceImpl",
    "MemorySceneService",
    "MemorySceneServiceImpl",
    "MemoryService",
    "MemoryServiceImpl",
    "idempotency_key",
    "atom_task_key",
    "persona_task_key",
    "scene_task_key",
    "submit_extract",
    "submit_atom_extract",
    "submit_persona_extract",
    "submit_scene_extract",
    "task_id_of",
]
