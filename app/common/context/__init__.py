"""
Context Builder 的**共用词汇**（枚举）。

这是全仓库唯一一份 Context Builder 枚举定义 —— 契约层、调度层、
注册表、将来的 `/context-builders` 接口全部从这里取。与
`app/common/models/` 的立场一致：共享内核里的东西，只有一处定义。

⚠️ 为什么在 app/common 而不是 app/services/context 或 app/agent
----------------------------------------------------------------
本包的**全部内容都是纯枚举**，零 import、零副作用。放在共享内核
是名副其实的（见 app/common/__init__.py：这里是"共享内核"）。

反过来说，注册表**不能**放这里：它要
`from app.agent.context.builder.base import BaseContextBuilder`
才能反射出子类，而 app/agent 已经 import app/common（见
app/agent/tools/catalog_tool.py）—— 包级会成环。所以注册表与它的
基类同侧，留在 app/agent/context/registry.py。

一句话：**无依赖的词汇进 common，要反向 import 的机制留在原地。**

导出方式
--------
照 app/common/__init__.py 的规矩：调用方从**包**导入，不钻叶子模块。

    from app.common.context import StageId, ContextBuilderType   ✓
    from app.common.context.enum import StageId                 ✗

这样 enum.py 将来想拆成多个文件时，调用点一个都不用改。

⚠️ 隔壁的 app/common/enums.py 与本包无关
-----------------------------------------
那个文件目前**整段是注释掉的草稿**（一套给前端用的 `__enum_key__`
枚举注册表），没有被任何代码引用，也与 Context Builder 不是一回事。
本包不动它、也不接管它的角色。
"""

from .enum import (
    BuilderExecutionMode,
    ContextBuilderStatus,
    ContextBuilderType,
    StageId,
    StageKind,
)

__all__ = [
    "BuilderExecutionMode",
    "ContextBuilderStatus",
    "ContextBuilderType",
    "StageId",
    "StageKind",
]
