"""
/api/v1 的两棵路由树：受保护的业务接口 + 少数公开接口。

为什么要拆成两个聚合路由
------------------------
前置校验要覆盖**所有**业务接口，但总有那么几个接口在校验之前就要能
用 —— 登录、回调、刷新（理由见 users.py 的模块 docstring）。做法是把
它们放进另一棵树，而不是在校验依赖里维护一张"豁免路径表"：

    豁免表   多一个接口就多一次"要不要加进去"的判断，加错了没人看得出来
    两棵树   接口注册在哪棵树上就是什么待遇，**照着 include 那行读**即可

谁在这棵树上，谁就受保护 —— 这个对应关系是**从这里**读出来的，不需要
跳到依赖实现里看条件分支。

为什么 /docs 与 /openapi.json 不在保护范围内
-------------------------------------------
它们不挂在 /api/v1 下，因此保持开放 —— 这是一个**有意识的决定**：
它意味着接口清单对匿名访问者可见。要收起来就在 main.py 里把
docs_url 按环境关掉，那是另一件事（本文件管不到）。

⚠️ 新增业务路由时，注册到 **protected_router** 上
-----------------------------------------------
没有机制强制这一点（靠的是"照着邻居写"）。guard 见
tests/api/test_auth_enforcement.py：它遍历 protected_router 里
注册的每个路径，断言不带 token 一律 401。
"""

from fastapi import APIRouter, Depends

from app.common.auth import current_user

from .capabilities import router as capabilities_router
from .document_chunks import router as document_chunks_router
from .document_definitions import (
    router as document_definitions_router,
)
from .document_retrieve import router as document_retrieve_router
from .document_versions import router as document_versions_router
from .documents import router as documents_router
from .files import router as files_router
from .memory import router as memory_router
from .memory_atom import router as memory_atom_router
from .mcp_providers import router as mcp_providers_router
from .memory_query import router as memory_query_router
from .pipeline_runs import router as pipeline_runs_router
from .processor_capabilities import (
    router as processor_capabilities_router,
)
from .processors import router as processors_router
from .qa_datasets import router as qa_datasets_router
from .retrieval_eval import router as retrieval_eval_router
from .sessions import router as sessions_router
from .skills import router as skills_router
from .stress_runs import router as stress_runs_router
from .tasks import router as tasks_router
from .tool_processors import router as tool_processors_router
from .tools import router as tools_router
from .users import admin_router as user_admin_router
from .users import protected_router as user_protected_router
from .users import public_router as user_public_router
from .users import require_admin

__all__ = [
    "protected_router",
    "public_router",
]


# ============================================================
# 公开：登录流程本身
# ============================================================
#
# 这几个接口存在的意义就是"在拿到 token 之前"能调 —— 给它们加校验
# 是自相矛盾。它们各自在 users.py 里注明了自己为什么公开。

public_router = APIRouter(
    prefix="/api/v1",
)

public_router.include_router(
    user_public_router,
)


# ============================================================
# 受保护：其余全部业务接口
# ============================================================
#
# `dependencies=[Depends(current_user)]` 是**整个鉴权方案的挂载点**。
# 参照点：它同时让下面每一个接口都能用 `Depends(current_user)` 取到
# 当前用户，而 FastAPI 的依赖缓存按可调用对象去重，**不会验两遍**
# —— 前提是 current_user 始终是同一个模块级函数对象，
# 见 app/common/auth/dependencies.py 的模块 docstring。

protected_router = APIRouter(
    prefix="/api/v1",
    dependencies=[Depends(current_user)],
)


protected_router.include_router(
    files_router,
)

protected_router.include_router(
    document_definitions_router,
)

# 单条 chunk（GET /document-chunks/{document_chunk_id}）。
#
# 与下面任何一个前缀都不重叠（/document-chunks 不是 /document-versions
# 或 /documents 的前缀），段数也固定 —— 所以**注册顺序在这里不承载
# 语义**，放在 document_definitions 旁边只是按名字归类，读起来顺。
protected_router.include_router(
    document_chunks_router,
)

# 能力目录必须排在 document_definitions 之后：
# /document-processors 与 /document-processor-capabilities 前缀相近，
# 先注册更具体的字面路径，避免被宽匹配抢先。
protected_router.include_router(
    processor_capabilities_router,
)

# 必须在 documents_router **之前**：
# /documents/retrieve 与 documents 里的 /documents/{document_id} 前缀
# 相同，而后者把路径参数声明为 UUID —— 先注册的字面量路径必须先
# 命中，否则 "retrieve" 会被当成一个非法 UUID 拿去解析。
protected_router.include_router(
    document_retrieve_router,
)

protected_router.include_router(
    documents_router,
)

# 版本下的产物视图（tab 列表 + 每路的 chunk）。
# 路径前缀是 /document-versions/，与 /documents/{id}/... 不冲突，
# 但排在 documents 之后更稳 —— documents 里有宽匹配的
# /documents/{document_id}。
protected_router.include_router(
    document_versions_router,
)

# 必须放在 documents_router 之后：
# documents 里有 /documents/{document_id} 这样的宽匹配路径，
# 先注册更具体的路由可以避免被它抢先匹配。
protected_router.include_router(
    pipeline_runs_router,
)

# QA 数据集生成。**手动触发**，不在任何管线步骤里。
# 路径是 /qa-datasets:generate，与上面的前缀都不重叠，
# 放在最后注册即可。
protected_router.include_router(
    qa_datasets_router,
)

# 第一层检索评测（数据集侧可用，执行侧是显式 501）。
#
# 前缀 /retrieval-eval/ 与上面任何一个都不重叠，而且它下面的三段
# （bundles / generation-runs / runs）全是**字面量**路径 —— 没有
# /documents/{document_id} 那种"宽匹配的 UUID 参数会吃掉字面量"的问题，
# 所以注册顺序在这里不承载语义，排在哪儿都行。
protected_router.include_router(
    retrieval_eval_router,
)

# 工具定义管理。
#
# tools_router 内部已经把 /tools/discovery/sync 排在 /tools/{tool_id}
# 之前（见 tools.py 的模块 docstring），这里不必再操心顺序。
protected_router.include_router(
    tools_router,
)

# MCP provider（外部工具来源）。
#
# 前缀 /mcp/ 与其余模块都不重叠。排在 tools 之后：前端通常是先拿
# provider 列表渲染筛选项，再调 /tools?source_ref=... 取工具 ——
# 与读代码的顺序一致。
protected_router.include_router(
    mcp_providers_router,
)

# 挂在 Tool 上的处理器链。
#
# tool_processors_router 与 tools_router 共用 /tools/ 前缀，
# 但路径形状不同（/tools/{id}/processors[/...] vs /tools/{id}），
# 不存在宽匹配抢先的问题 —— 排在 tools_router 之后只是为了让
# 读代码的人先看到工具本身，再看挂在它上面的东西。
#
# processors_router 用 /processors 前缀，与 /document-processors
# 不重叠（后者在 processor_capabilities_router 里，前缀不同）。
protected_router.include_router(
    tool_processors_router,
)

protected_router.include_router(
    processors_router,
)

# 能力树（taxonomy）。
#
# capability 的契约、服务、仓储、模型都在 app/services/capability/，
# 路由与其余模块一样留在本目录（capabilities.py）。
#
# 前缀是 /capabilities，与 /tools 不重叠，放在最后注册即可。
protected_router.include_router(
    capabilities_router,
)

# Skill 管理。
#
# 前缀 /skills 与上面任何一个都不重叠；/skills/{id}/capabilities
# 与 /skills/{id} 段数不同，没有宽匹配抢先的问题。
#
# 它只建模 Skill 与能力绑定，不含鉴权 —— 挂在本树上即沿用平台登录。
protected_router.include_router(
    skills_router,
)

# Agent 运行时的会话与运行。
#
# 前缀 /sessions 与 /runs 都不与上面任何一个重叠 —— 注意 /runs 是**顶层**
# 路径（不是 /sessions/{id}/runs 那种嵌套），规格如此，因为两处主语不同：
# 建运行的主语是会话，而查事件 / 恢复 / 取消的主语是运行本身。
protected_router.include_router(
    sessions_router,
)

# 统一异步任务基础设施的查询与控制。
#
# 前缀是 /tasks，与上面任何一个都不重叠（注意没有 /tasks/{id}/runs 这类
# 嵌套 —— 执行链走的是 /tasks/{task_id}/lineage，主语始终是任务）。
#
# ⚠️ 这里**没有**提交任务的路由，是刻意的：任务由业务代码在服务端
# submit，不对外开一个"任意已注册 Handler 都能被触发"的入口。
# 理由见 app/api/v1/tasks.py 的模块 docstring。
#
# /tasks/{task_id}/invoke 不违反这一点：它作用的是一条**已经存在**的
# 任务（把 execute_at 提前到现在），既不创建任务，也不接受 handler 名。
protected_router.include_router(
    tasks_router,
)

# 用户模块里受保护的那部分（目前只有 /auth/me）。
#
# ⚠️ 与上面那行 `public_router.include_router(user_public_router)`
# 是**同一个模块的三棵树**（见 users.py）。/auth/me 之所以不跟
# login/callback/refresh 一起放在公开树上，是因为它受保护这件事
# 应该由 include 的那一行决定，而不是靠端点签名里一行"为了取值"
# 的 Depends 顺带兜着 —— 那行被删掉时不会有人觉得删错了。
protected_router.include_router(
    user_protected_router,
)

# 用户目录检索（/users/search）—— 受保护 **且** 要求 admin_role。
#
# 它注册在 protected_router 上（于是同样要带 token、同样过
# required_groups 那道登录门槛），admin_router 自己再叠一道角色门槛
# （见 users.py 的 require_admin）。
#
# 前缀是 /users，与上面任何一个都不重叠，放在最后注册即可。
protected_router.include_router(
    user_admin_router,
)

# L0 记忆检索（agent_message 的向量 + 关键词召回）。
#
# 前缀是 /memory/search，单条路径、没有 /memory/{id} 这样的宽匹配兄弟，
# 因此**不需要**像 /documents/retrieve 那样靠注册顺序抢字面量 —— 排
# 在哪里都不会被抢先匹配。
#
# 它同时是暴露给主模型的工具之一（`memory.search`，见
# app/services/tool/registry/registry.py 的 TOOL_SOURCE_PACKAGES）——
# 两条入口共用同一个服务方法，身份各自由 HTTP token / 工具运行时给出。
protected_router.include_router(
    memory_router,
)

# L1 记忆检索（memory_atom 的向量 + 关键词召回）。
#
# 与上面那条同形：单条路径 /memory-atom/search，没有宽匹配兄弟，排在哪里
# 都不会被抢先匹配。它同样是暴露给主模型的工具之一（`memory_atom.search`）
# —— 两条入口共用 services/memory/application/memory_atom_service.py 里的
# 同一个方法，身份各自由 HTTP token / 工具运行时给出。
#
# ⚠️ 与 L0 的差别：L1 的记忆是**跨会话**的用户事实，所以调用方多数时候该
# 传 scope=user（见 memory_atom.yaml 与 memory_atom.py 的说明）。
protected_router.include_router(
    memory_atom_router,
)

# 管理端记忆查询（提取进度水位 + 跨层溯源）。
#
# ⚠️ 这一组比上面两条**多一道角色门槛**：Y-Trader-Admin。加在这一行、
# 而不是端点签名里 —— 理由见本模块 docstring 那条"接口注册在哪棵树上
# 就是什么待遇"（与 users.py 里 /users/search 走 admin_router 是同一件事，
# 只是那边的门槛在模块里自己成了一棵树）。
#
# 前缀是 /memory/extraction/* 与 /memory/query/*，与 /memory/search、
# /memory-atom/search 都不重叠，也没有 /memory/{id} 这样的宽匹配兄弟，
# 排在哪里都不会被抢先匹配。
protected_router.include_router(
    memory_query_router,
    dependencies=[Depends(require_admin)],
)

# Mock Model Stress Eval（压测运行与聚合指标）。
#
# 前缀 /stress-runs 与上面任何一个都不重叠，也没有宽匹配兄弟，注册顺序
# 在这里不承载语义。契约权威是 design_agent_eval/openapi.yaml，实现入口
# 在 app/services/stress/。
#
# ⚠️ 当前没有真实 Runner：运行由本地模拟器推进（不发出任何网络请求），
# 指标里的 aggregation_algorithm 为 simulated-v1。详见
# app/services/stress/application/simulated_runner.py。
protected_router.include_router(
    stress_runs_router,
)
