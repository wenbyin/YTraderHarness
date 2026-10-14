"""
Agent 运行时 + 开发验证脚本。

⚠️ 本包有两类完全不同的东西，别放错
-----------------------------------
    **运行时子包**（被 app 代码 import，是应用的一部分）
        context/    Context Builder 契约与注册表
        graph/      ReAct 图（AgentNode）
        prompt/     提示词（节点功能性：按 model_role 取）与 PromptProvider
        tools/      工具的处理器实现（拦截器 / 转换器 / 观察者）

    **检查脚本**（没有任何 app 代码 import，跑一次就退出）
        下面那张清单里的 .py 文件

判据很简单：**会不会被 app 代码 import**。会，就是运行时，放子包；
不会，就是脚本，放包根。

⚠️ 这里**仍然不是**写业务工具的地方
-----------------------------------
业务工具（Agent 能调用的那些）是**服务方法上的 @tool 声明**，
写在各自模块里：

    app/services/<模块>/application/<xxx>_service_impl.py

它们的入口与治理在 app/services/tool/。三者的区别：

    app/agent/tools/      工具**被调用时**挂上去的处理器逻辑
    app/services/tool/    工具的定义、发现、同步与文档管理
    app/infra/agent/      把工具接到 langchain / langgraph 的执行适配层

⚠️ 子包里必须有 __init__.py
---------------------------
`context/`、`graph/` 与 `prompt/` 下有按包反射的机制（见
context/registry.py、prompt/registry.py），而 `pkgutil.walk_packages`
**不会走进没有 `__init__.py` 的目录** —— 删掉会让实现无声消失。

脚本清单
--------
    retrieve_smoke.py  检索链路的端到端回归（假仓储，不需要数据库）
    pipeline_smoke.py  流水线拓扑自检 + 桩节点跑通全图
    tool_smoke.py      工具模块回归（自注册、发现、同步状态机）
    prompt_smoke.py    提示词回归（目录、渲染、职责提示词的降级边界）
    storage_check.py   文件存储的路径穿越等安全边界
    spec_check.py      OpenAPI yaml 与路由的契约一致性
    import_bisect.py   二分定位 import 循环 / 慢导入

除 spec_check.py（只读 yaml，失败项是既有的、与代码无关）外，
其余脚本以退出码 0 表示通过。
"""
