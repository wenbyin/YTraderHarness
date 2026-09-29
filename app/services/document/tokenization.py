"""
分词与长度度量 —— 跨 role 共用的文本工具。

为什么单独一个模块
------------------
这套东西**不属于任何一个 role**：

    chunker    用它度量 chunk 长短（chunk_size / overlap 按 token 算）
    embedder   稀疏实现用它分词（词项 = token）
    将来        清洗、抽取如果需要词数统计，同样用它

放在 chunker 包里会让 embedder 反向依赖 chunker —— 两个平级的 role
包不该互相 import。放在这里，两边都只依赖这个中立模块。

不依赖任何 role 包，也不依赖 app.services 里的其它模块。
"""

from __future__ import annotations

from functools import lru_cache

__all__ = [
    "encoding_for",
    "tokenizer_for",
    "token_length",
    "model_token_length",
]


@lru_cache(maxsize=4)
def encoding_for(name: str):
    """
    惰性加载 tiktoken 编码器，按名字缓存。

    惰性有两个理由：
        1. 按字符度量的实现**完全不需要** tiktoken，
           不该为没用到的东西付启动成本
        2. 未安装时给出能照着修的提示，而不是一个指向
           module 内部的 ImportError

    ⚠️ tiktoken 首次调用会**联网下载** BPE 词表并缓存到本地。
    离线 / 镜像环境需要预先下载并设置 TIKTOKEN_CACHE_DIR，
    否则第一次调用会很慢甚至失败。
    """

    try:
        import tiktoken

    except ImportError as exc:
        raise RuntimeError(
            "按 token 处理需要 tiktoken，但它没有安装。"
            "请 pip install tiktoken（版本见 requirements.txt），"
            "或改用按字符度量的实现。"
        ) from exc

    return tiktoken.get_encoding(name)


def token_length(text: str, encoding_name: str) -> int:
    """text 在指定编码下的 token 数。空文本是 0。"""

    if not text:
        return 0

    return len(encoding_for(encoding_name).encode(text))


def token_ids(text: str, encoding_name: str) -> list[int]:
    """
    text 在该编码下的 token id 序列。

    稀疏 embedder 用它把文本变成词项序列 —— 编码器给出的 id 本身就是
    稳定的整数，可以直接当"词表维度"用。
    """

    if not text:
        return []

    return list(encoding_for(encoding_name).encode(text))


# ============================================================
# 按**向量模型自己的词表**度量
# ============================================================
#
# 为什么需要另一套：tiktoken 的 o200k_base 与向量模型的词表**不是
# 一回事**。模型对单次输入的 token 上限是在**它自己词表**里算的，
# 拿 tiktoken 的量去比会系统性低估 —— 实测中文下 bge-small-zh 要比
# o200k_base 多吃 16%~33%（数字与拉丁字符差得最多）。
#
# 后果不是"稍微超一点"，而是模型**静默截断**：不报错、向量照常产出、
# 维度照常正确，只是 chunk 的尾巴从来没有被向量化过。所以凡是"要跟
# 模型上限比大小"的地方，都必须用这一套。


@lru_cache(maxsize=4)
def tokenizer_for(name: str):
    """
    惰性加载 HuggingFace tokenizer，按名字缓存。

    惰性的理由与 encoding_for 相同（不为没用到的东西付启动成本、
    未安装时给出能照着修的提示），另外多一条：`transformers` 是重
    依赖，而本模块被 chunker 在**能力注册表构造期**间接导入 ——
    在模块顶层 import 它会拖慢每一次注册表构建。

    name 可以是本地目录，也可以是仓库名（离线环境下后者会失败）。
    """

    try:
        from transformers import AutoTokenizer

    except ImportError as exc:
        raise RuntimeError(
            "按向量模型词表度量需要 transformers，但它没有安装。"
            "请 pip install transformers（版本见 requirements.txt），"
            "或改用 char / token 度量。"
        ) from exc

    return AutoTokenizer.from_pretrained(name)


def model_token_length(text: str, model_name: str) -> int:
    """
    text 在指定向量模型**自己词表**下的 token 数。

    ⚠️ 含特殊 token（BERT 系的 [CLS] / [SEP]，共 2 个），与
    token_length 的口径**不同**。这是有意的：这里回答的问题是
    "这条能不能塞进模型的 512 上限"，而模型看到的就是带特殊 token
    的那一串，漏掉它们会让边界情况恰好超限。
    """

    if not text:
        return 0

    return len(tokenizer_for(model_name).encode(text))
