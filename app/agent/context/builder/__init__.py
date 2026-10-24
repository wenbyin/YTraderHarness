"""
Context Builder 的契约与实现。

契约
----
    base.py         BaseContextBuilder + 六个 type 基类
    contracts.py    执行期数据形状（ModelRequest 复用 Model 模块的那一个）
    agent_invoke.py 生产实现：把 13 个阶段委托给 Processor 流水线
                    （`app/agent/context/processor/`）

发现方式：**反射遍历本包**
----------------------------
发现逻辑（app/agent/context/registry.py）把本包名写进
CONTEXT_BUILDER_PACKAGES，然后 `walk_packages` 逐个 import、
`leaf_subclasses(BaseContextBuilder)` 收叶子类。

因此新增一个实现 = 在本包下加一个 .py 文件 + 继承某个 type 基类
+ 声明 enabled_stages 并覆写对应阶段方法，**不用回来改本文件，
也不用改任何清单**。

⚠️ 只有**住在本包**的类会被发现
------------------------------
`__subclasses__()` 是进程级的：任何被 import 过的模块里定义的子类
都在里面（测试里的桩、脚本里的变体……）。因此注册表按模块名过滤，
只收住在 CONTEXT_BUILDER_PACKAGES 里的实现 —— 否则"这个进程 import
过什么"会决定"生产上用的是哪个 Builder"（键冲突让应用起不来，
或者更糟：生产静默用了测试桩）。

⚠️ 子包必须存在 __init__.py
------------------------------
`pkgutil.walk_packages` 不会走进没有 `__init__.py` 的目录。将来
按类型拆子包（agent_invoke/、compaction/ …）时，每个子包都要留
那个文件，否则里面的实现会**无声地**扫不到 —— 与
app/agent/tools/processor/__init__.py 的警告是同一条。

为什么分类靠**继承**而不是靠声明
--------------------------------
六个 type 基类各自定死 `builder_type`（见 base.py）。实现不能自己
写这个字段 —— 与 tool 模块把 `processor_type` 定死在 type 基类、
document 模块把 `role` 定死在 role 基类是同一个做法：
声明与实际能力不符时，没有第二处可以撒谎。
"""
