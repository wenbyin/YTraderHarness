"""
开发用的 Mock 模型服务。

这个包只服务本地开发 / 前端联调，**不参与生产路径**：

    settings.py      配置模型 + 是否启用
    model_server.py  OpenAI-compatible aiohttp 服务（守护线程）
    runtime.py       启动 / 停止 + 与 app.yaml、providers.yaml 的接线

设计上的取舍见 model_server.py 的模块 docstring。最重要的两条：

    1. 它是**进程内守护线程**，不是独立服务 —— 一个进程、一个端口，
       启动主服务就能用，前端不需要额外起进程。
    2. 它只绑定回环地址；生产环境不要打开。
"""
