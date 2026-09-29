"""
Document 服务共用的常量。

**模型身份集中在这里 —— 换模型只改这一个文件。**

为什么需要它
------------
模型名原本散在各个实现里（embedder 一个、reranker 一个），
"系统里到底用了哪些模型"这个问题没有单一答案。更麻烦的是
它们之间有**隐含约束**：

    bge-small-zh-v1.5   512 维   写入与查询**必须同一个** ——
                        不一致时维度照样对得上、检索照常返回结果，
                        只是排序没有意义
    bge-reranker-v2-m3            只服务精排，与上面的向量空间无关

把它们放在一起，这层关系才看得见。

关于编码器名（cl100k_base / o200k_base）
---------------------------------------
它们**刻意不在这里**，因为性质不同：

    - 它们是 tiktoken 的编码器名，同时也是**对外配置项**
      （chunker 的 encoding_model 字段，取值是 Literal）
    - Literal 类型要求字面量，抽成变量会破坏类型约束
    - 且它们的取值与 dimensions 绑定，各自留在实现里更清楚
      （见 embedder/sparse.py 的 _ENCODING_MODEL）

服务器地址为什么也不该做成配置
------------------------------
放在这里是为了集中，但**不要**把它挪进 config/*.yaml ——
服务器地址是**部署事实**，一旦进配置就会被冻结进
PipelineConfigSnapshot，以后换地址时历史快照还指着旧地址。
（embedder/ollama.py 的模块说明也写了同一条理由。）
"""

from __future__ import annotations

__all__ = [
    "OLLAMA_HOST",
    "BGE_SMALL_ZH_LOCAL_MODEL",
    "BGE_RERANKER_MODEL",
    "QA_OUTPUT_DIR",
]


#: Ollama 服务地址。
#:
#: 与 embedder 侧一致：langchain 的 OllamaEmbeddings 在 base_url 为
#: None 时会读 OLLAMA_HOST 环境变量（默认也是这个地址），所以改部署
#: 地址有两个入口，这里是显式的那一个。
OLLAMA_HOST = "http://127.0.0.1:11434"


# ============================================================
# 模型身份
# ============================================================


#: 稠密向量模型（**本地目录**，不再走 Ollama）。
#:
#: ⚠️ 写入期与查询期**必须是同一个** —— 它决定向量空间。换它等于
#: 已有向量全部作废（维度可能还对得上，但空间不同，检索结果没有意义
#: 且**不报错**），必须重建索引。
#:
#: ⚠️ 换模型时**还要换 embedder 的 name（能力键）** ——
#: `LocalBgeSmallZhEmbedder.name = "TEXT_BGE_SMALL_ZH_LOCAL_EMBEDDER"`
#: 会冻结进 PipelineConfigSnapshot，历史运行靠它解释"当时用的是哪个
#: 模型"。只改这里而不改键，历史快照就会撒谎：键还是那个键，模型却
#: 已经换了。
#:
#: 从 Ollama 换到本地时**键也跟着换了**（旧键
#: TEXT_BGE_SMALL_ZH_EMBEDDER 已随 Ollama 实现一起删除），因为
#: Ollama 里那份是 GGUF **量化**版、这里是 HF 原版 —— 量化会改变
#: 向量输出，同键混两种空间是最难查的那类问题。
#:
#: 512 维；输入上限 512 token。注意这个 512 是**模型自己词表**的
#: token，与 chunker 用 tiktoken 量的 token 不是一回事 —— 实测中文下
#: bge 比 tiktoken 多吃 16%~33%，所以 tiktoken 口径的 1200 token
#: 在模型看来远超 512，尾部会被静默截断。
#:
#: 下载记录（换机器时照做）：
#:     来源   ModelScope（实测 11.2 MB/s）
#:     文件   config.json / model.safetensors (95827648 字节) /
#:            tokenizer.json / tokenizer_config.json / vocab.txt /
#:            special_tokens_map.json / sentence_bert_config.json /
#:            modules.json / config_sentence_transformers.json
BGE_SMALL_ZH_LOCAL_MODEL = r"G:\models\bge-small-zh-v1.5"


#: 精排模型（**HuggingFace 模型名**，不是 Ollama 模型名）。
#:
#: 与上面的向量模型**无关**：它只在精排阶段读"问题 + 候选正文"，
#: 不参与向量召回，换它不影响已入库的向量，也不需要重建索引。
#:
#: ⚠️ 它**不能走 Ollama**。bge-reranker-v2-m3 是 cross-encoder
#: （序列分类）模型 —— Ollama 只服务生成模型与嵌入模型。实测拿它打
#: /api/chat 与 /api/embed 都会让 llama-server 直接崩（exit 0xc0000409），
#: /api/rerank 这个端点根本不存在（Ollama 0.33.3）。
#:
#: 所以由 FlagEmbedding 在**进程内**加载。首次用到时下载+加载要几秒
#: （模型约 2.3GB），之后复用同一份（见 retrieve/rerank.py 的
#: _load_model）。
#:
#: 国内下载不动的话设 HF_ENDPOINT=https://hf-mirror.com，或先把模型
#: 拉到本地再把这里改成目录路径。
BGE_RERANKER_MODEL = r"G:\models\bge-reranker-v2-m3"


#: QA 测试数据集的落盘目录。
#:
#: 这是**部署事实**（哪台机器、哪块盘），所以与模型路径一样放在常量里，
#: 不进任何业务配置。
#:
#: 为什么不走 File 模块的 artifact_store：那套是**内容寻址**的产物存储
#: （按 (tenant, checksum) 去重，落在 ./data/files/<uuid>/source），
#: 适合"中间产物"，不适合"我要在资源管理器里打开 G:\QA 看一眼"这种用法。
#: 数据集是给人用、给外部工具（ragas）读的，需要一个稳定、可预期的
#: 文件名。
QA_OUTPUT_DIR = r"G:\QA"
