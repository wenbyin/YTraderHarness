from .base import DocumentEmbedder

# 具体实现（bge_small_zh_local / sparse / graph）不需要在这里导出 ——
# 能力注册表按 CAPABILITY_PACKAGES 反射扫描整个包目录，新增 .py 文件
# 即自动收录（见 pipeline/registry.py::found_capabilities）。
#
# 这里只导出 role 基类供类型标注使用。
__all__ = [
    "DocumentEmbedder",
]
