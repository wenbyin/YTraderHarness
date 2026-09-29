from fastapi import APIRouter

from .classifications import router as classifications_router
from .document_definitions import (
    router as document_definitions_router,
)
from .document_retrieve import router as document_retrieve_router
from .document_versions import router as document_versions_router
from .documents import router as documents_router
from .files import router as files_router
from .pipeline_runs import router as pipeline_runs_router
from .processor_capabilities import (
    router as processor_capabilities_router,
)
from .processors import router as processors_router
from .qa_datasets import router as qa_datasets_router
from .tool_processors import router as tool_processors_router
from .tools import router as tools_router

router = APIRouter(
    prefix="/api/v1",
)

router.include_router(
    files_router,
)

router.include_router(
    document_definitions_router,
)

# 能力目录必须排在 document_definitions 之后：
# /document-processors 与 /document-processor-capabilities 前缀相近，
# 先注册更具体的字面路径，避免被宽匹配抢先。
router.include_router(
    processor_capabilities_router,
)

# 必须在 documents_router **之前**：
# /documents/retrieve 与 documents 里的 /documents/{document_id} 前缀
# 相同，而后者把路径参数声明为 UUID —— 先注册的字面量路径必须先
# 命中，否则 "retrieve" 会被当成一个非法 UUID 拿去解析。
router.include_router(
    document_retrieve_router,
)

router.include_router(
    documents_router,
)

# 版本下的产物视图（tab 列表 + 每路的 chunk）。
# 路径前缀是 /document-versions/，与 /documents/{id}/... 不冲突，
# 但排在 documents 之后更稳 —— documents 里有宽匹配的
# /documents/{document_id}。
router.include_router(
    document_versions_router,
)

# 必须放在 documents_router 之后：
# documents 里有 /documents/{document_id} 这样的宽匹配路径，
# 先注册更具体的路由可以避免被它抢先匹配。
router.include_router(
    pipeline_runs_router,
)

# QA 数据集生成。**手动触发**，不在任何管线步骤里。
# 路径是 /qa-datasets:generate，与上面的前缀都不重叠，
# 放在最后注册即可。
router.include_router(
    qa_datasets_router,
)

# 工具定义管理与分类。
#
# tools_router 内部已经把 /tools/discovery/sync 排在 /tools/{tool_id}
# 之前（见 tools.py 的模块 docstring），这里不必再操心顺序。
#
# classifications_router 用 /classifications 前缀，与 /tools 不重叠。
router.include_router(
    tools_router,
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
router.include_router(
    tool_processors_router,
)

router.include_router(
    processors_router,
)

router.include_router(
    classifications_router,
)
