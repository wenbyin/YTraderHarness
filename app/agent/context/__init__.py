"""
Context Builder：访问 LLM 之前的上下文构建。

职责
----
**任何一次 LLM 调用之前，都要经过这里的 ContextBuilder。** 主 Agent、
Summary、决策门…… 区别只在"流水线的哪些阶段参与、参数是什么"，
不在有没有这一层。

    BaseContextBuilder     契约 + 固定 11 步的调度（builder/base.py）
    contracts.py           执行期数据形状（ModelRequest 就是 Model 模块那一个）
    registry.py            按 ContextBuilderType 反射发现实现
    updater.py             调用**之后**的上下文回写（本次仅保留接口）
    processor/             Processor 流水线：自描述依赖 + 统一 State 解析
    builder/agent_invoke.py 生产实现：把 11 个阶段委托给流水线

分工（谁决定什么）
----------------
    Agent Builder / 图        决定 agent_type、Context Policy、注入哪些数据
    processor/ 里的 Processor 决定"自己这一阶段做什么、需要哪些输入"
    BaseContextBuilder        决定阶段参与与固定顺序，产出 ModelRequest
    ModelRegistry             决定用哪个模型、怎么调、失败怎么办

分类轴是**构建策略**，不是业务能力
----------------------------------
六种 builder_type（见 app/common/context/enum.py）说的是"上下文怎么
长出来的"，不是"这个 Agent 会干什么"。业务能力走 skills 注入
system_prompt —— 那是另一个轴，且本次不做。

因此这里**不需要**、也不应该出现任何行业 / 场景 / 部门的名字。

⚠️ 本包必须有 __init__.py
-------------------------
发现逻辑用 `pkgutil.walk_packages` 反射遍历本包找实现，而它**不会
走进没有 __init__.py 的目录**。删掉这个文件会让所有实现无声消失
（发现结果为空，而不是报错）。app/agent/tools/processor/
__init__.py 已经为同一件事留过警告。
"""
