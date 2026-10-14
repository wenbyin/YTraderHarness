"""
Capability 模块的 SQLAlchemy 模型。

import 本包 = capability 表注册完毕。

⚠️ 与 tool 模块的 tool_definition_classification 是**跨模块**的
relationship（Capability.definition_links ↔ 关联行.classification），
两边都用字符串形式的目标名，SQLAlchemy 到 mapper 配置时才去注册表
里解析。因此那个关联模型必须也已经被 import ——
app/infra/datasource/initializer.py 的 import_all_models() 会把两个
模块的模型包都 import 一遍，正常启动路径由此保证。
"""

from .capability import Capability

__all__ = [
    "Capability",
]
