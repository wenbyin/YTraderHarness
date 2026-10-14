"""
工具处理器实现。

这里放的是**实现**，不是契约 —— 契约在 base.py：
    BaseToolProcessor          共性
      ├── ToolInterceptor      改控制流（重试 / 超时 / 审批）
      ├── ToolTransformer      改数据（脱敏 / 补默认值）
      └── ToolObserver         只读旁路（指标 / 审计）

发现方式：**反射遍历本包**
----------------------------
注册表（app/services/tool/registry/processor_registry.py）把本包名
写进 PROCESSOR_PACKAGES，然后 `walk_packages` 逐个 import、
`leaf_subclasses(BaseToolProcessor)` 收叶子类。

因此新增一个处理器 = 在本包下加一个 .py 文件 + 继承某个 type 基类
+ 声明 name / supported_hooks / supported_execution_modes，
**不用回来改本文件，也不用改任何清单**。

⚠️ 子包必须存在 `__init__.py`
------------------------------
`pkgutil.walk_packages` 不会走进没有 `__init__.py` 的目录，
因此 observer/ 与 transformer/ 下即便暂时没有实现，也留着空文件 ——
否则将来往里放第一个实现时会毫无征兆地扫不到。
"""
