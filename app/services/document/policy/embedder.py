"""
（本模块已退役，保留此文件只为让旧的 import 路径留下指路牌。）

Embedder 去哪了
---------------
向量化能力已并入 DocumentProcessor 体系（role=EMBED）：

    契约      embedder/base.py       DocumentEmbedder
    中间层    embedder/ollama.py     OllamaDocumentEmbedder（按推理后端分族）
    实现      embedder/bge_small_zh.py  OllamaBgeSmallZhEmbedder
    注册      PipelineRegistry.processors（CAPABILITY_PACKAGES 里登记了包名）

原 `Embedder` ABC 与 `registry.embedders` 字典 / `registry.embedder()`
一并退役 —— 保留并列的第二套会让"向量化实现在哪"有两个答案。

维度声明的变化
--------------
原 `dimensions` 是抽象 property（"未知时返回 None，由调用方在首次
embed 后回填"）。现在它是 embedder/base.py 上的 `ClassVar`：建 Milvus
collection 的 schema 需要维度，也就是**任何一次调用之前**就要拿到，
让它变成"首次调用后才知道"的运行时值没有意义。

尚未迁移的同类
--------------
`FusionStrategy` / `RerankStrategy` 还留在 policy/base.py（阶段 8 与
查询期的 role，尚未并入体系）。本文件不含任何类 —— 不要在这里重新
定义它们，否则会和 policy/base.py 的两份定义打架。
"""
