"""
组合工具：用 langchain 的 @tool 写的、跨多个 service 的工具。

与 app/services/**/application/ 里那些 @tool 的区别
--------------------------------------------------
    服务方法工具    一个方法 = 一个工具，绑定在一个 service 上。
                    执行时由工厂建出那个 service 的 self。
    组合工具        一个函数可以调任意多个 service，
                    把它们的输出拼成一个结果给模型。

两者的**声明方式**不同（一个是我那个 @tool，一个是 langchain 的），
但都会被同步扫进同一张表 —— 它们在 tool_definition 里是同一类东西，
都是 source_type=local_method。

拿 service 的唯一入口
---------------------
    from app.common.factories import get_service_factory

工厂是**应用级**的，不依赖 FastAPI Request，因此在被模型调用的
工具里也能直接取到。这是它能在这里工作的前提。

事务怎么算
----------
每个 `factory.scope(契约)` 是一段**独立**事务。组合工具如果调了
三个服务，就是三段事务 —— 只读无所谓（用 begin=False 连事务都
不开），但**写操作跨服务就不是原子的**：A 提交了、B 失败，A 不会
回滚。要写多个服务时，得先想清楚这一点。

source_ref 怎么指定
-------------------
langchain 的 @tool 不接受我们的参数，声明不到装饰器上。
在模块顶部写一行即可：

    TOOL_SOURCE_REF = "agent.platform"

不写就按模块路径推导（app.agent.tools.x -> agent）。

本包在 TOOL_SOURCE_PACKAGES 里，所以放进来的工具会被同步自动扫到。
"""
