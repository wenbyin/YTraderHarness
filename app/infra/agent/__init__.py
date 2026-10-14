"""
Agent 与业务能力之间的**适配层**。

它是"声明"与"执行"之间缺的那一层：

    app/services/tool/registry/    声明 —— @tool 产出 ToolDescriptor
    app/agent/tools/processor/     契约 —— 拦截器 / 转换器 / 观察者

上面两半都只描述"是什么"与"该怎么做"，没有一个地方真的**把工具跑
起来**、也说不清"这次该暴露哪些工具给模型"。本包就是那件事的落点：

    tool_exposure.py    按 tool_keys 取出可 bind 的工具对象
    chain.py            按 sequence 编译一条处理器链（纯函数）
    tool_executor.py    驱动链 + 真正调 service 方法

为什么不在 registry 里实现
------------------------
registry 是**声明层**，任何带工具的 service 模块 import 它时都会
走一遍（装饰器要注册）。让它反向依赖 service 工厂（`factory.create`）
与处理器链，就把"声明一个工具"的代价从"建一个 pydantic 模型"
变成"拖进整个执行栈"。而且 registry 属于 services 侧，
本包属于 infra 侧 —— 依赖方向本来就是 services <- infra，反过来
会让 services 自成一个环。

为什么不在 graph/react/tools.py 里实现
------------------------------------
节点的职责是图语义（读 tool_calls、写 ToolMessage、记观测）。
执行一条链是**与图无关**的能力：将来 MCP 化、后台任务、非图调用
路径都要用它。塞进节点等于把可复用的那部分锁在 LangGraph 里。
"""
