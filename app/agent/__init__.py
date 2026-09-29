"""
开发与验证脚本。

⚠️ 这里**不是**写业务工具的地方
--------------------------------
本包装的是"手动跑一次、看一眼结果"的脚本，不是应用运行时的一部分
（没有任何 app 代码 import 它们，全靠 `python -m app.agent.xxx` 跑）。

业务工具（Agent 能调用的那些）是**服务方法上的 @tool 声明**，
写在各自模块里：

    app/services/<模块>/application/<xxx>_service_impl.py

它们的入口与治理在 app/services/tool/。两者的区别：

    app/agent/            跑一次就退出的检查脚本
    app/services/tool/    工具的定义、发现、同步与文档管理
    app/infra/agent/      把工具接到 langchain / langgraph 的适配层

脚本清单
--------
    retrieve_smoke.py  检索链路的端到端回归（假仓储，不需要数据库）
    pipeline_smoke.py  流水线拓扑自检 + 桩节点跑通全图
    tool_smoke.py      工具模块回归（自注册、发现、同步状态机）
    storage_check.py   文件存储的路径穿越等安全边界
    spec_check.py      OpenAPI yaml 与路由的契约一致性
    import_bisect.py   二分定位 import 循环 / 慢导入

除 spec_check.py（只读 yaml，失败项是既有的、与代码无关）外，
其余脚本以退出码 0 表示通过。
"""
