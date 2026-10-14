"""
脱敏：任何要落日志 / 落库 / 进观测的东西都先过这里。

为什么需要它，而不是"注意别打密钥"
-----------------------------------
"注意别打密钥"是一条挡不住的约定：真正泄漏的那一次，通常是**异常
消息**里带着 key（`Incorrect API key provided: sk-...`），或者某个人
往 `ModelRequest.metadata` 里塞了一个 `api_key` 想"顺便带上"。两者都
不经过任何人的手写 log 语句。

因此这里提供两个入口，**在写出之前**统一过滤：

    redact_mapping(dict)   键命中敏感词 -> 值替换成 `***`
    redact_text(str)       文本里的 sk-xxx / Bearer xxx / key=xxx 抹掉

⚠️ 为什么不能简单地"见到 token 就抹"
------------------------------------
本模块观测里最重要的两个字段恰好叫 `input_tokens` / `total_tokens`。
用子串匹配（`"token" in key`）会把它们一起抹掉 —— 于是脱敏把**成本
数据**也毁了，而日志里看起来一切正常。

所以判据是**带边界的**：

    key == "token"            ✅ 敏感（一个裸的 token 字段）
    key == "access_token"     ✅ 敏感（以 `_token` 结尾）
    key == "input_tokens"     ❌ 不敏感（复数，且不是以 `_token` 结尾）
    key == "credential_ref"   ❌ 不敏感（它是**名字**，不是秘密）

这与 `_SENSITIVE_KEYS` 里的 `credential` 条目配合：它只匹配
`credential` / `xxx_credential`，不匹配 `credential_ref`。
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

__all__ = [
    "REDACTED",
    "is_sensitive_key",
    "redact_mapping",
    "redact_text",
]


#: 命中敏感词的值被替换成它。
REDACTED = "***"

#: 递归脱敏的最大深度（防止自引用结构把栈打爆）。
_MAX_DEPTH = 8

#: 敏感的**键名**（小写、`-` 归一成 `_` 之后比较）。
#:
#: 匹配规则有两条，见 is_sensitive_key：
#:     完全相等                     "api_key"、"token"、"secret"
#:     以 `_<词>` 结尾              "openai_api_key"、"access_token"
#:
#: 故意**不**用 `in`（子串）匹配 —— 那会让 `credential_ref` 与
#: `input_tokens` 一起阵亡，见模块 docstring。
_SENSITIVE_KEYS: tuple[str, ...] = (
    "api_key",
    "apikey",
    "key",
    "secret",
    "client_secret",
    "password",
    "passwd",
    "credential",
    "credentials",
    "token",
    "authorization",
    "auth",
    "bearer",
    "private_key",
    "access_key",
    "session_key",
)


#: 文本里的密钥形态。
#:
#: 三条各自对应一种真实出现过的形态：
#:     sk-…            OpenAI / DeepSeek / Moonshot 的密钥前缀
#:     Bearer …        放进 Authorization 头的形态
#:     key=… / key: …  异常消息与 URL query 里的形态
#:
#: ⚠️ 第三条带 `\b` 与单数词：`total_tokens: 120` 里的 "tokens" 不该被
#: 抹掉（`\btoken\b` 不匹配 "tokens"），理由与上面同一条。
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bsk-[A-Za-z0-9_\-]{6,}"),
    re.compile(r"(?i)\b(bearer)\s+[A-Za-z0-9._\-]{6,}"),
    re.compile(
        r"(?i)\b(api[_-]?key|secret|password|token|authorization)"
        r"(\s*[=:]\s*)(\S+)"
    ),
)


def is_sensitive_key(key: str) -> bool:
    """
    这个键名是否指向一个秘密。

    只认"完全相等"与"以 `_<敏感词>` 结尾"两种形态 —— 理由见模块
    docstring（子串匹配会误伤 `credential_ref` / `input_tokens`）。
    """

    normalized = str(key).strip().lower().replace("-", "_")

    return any(
        normalized == candidate
        or normalized.endswith("_" + candidate)
        for candidate in _SENSITIVE_KEYS
    )


def redact_mapping(
        value: Any,
        *,
        _depth: int = 0,
) -> Any:
    """
    递归脱敏一个结构，返回**新的**对象（不就地改）。

    规则：
        Mapping          键敏感 -> 值换成 `***`；否则递归
        序列（非 str）   逐项递归
        str              过 redact_text
        其他             原样返回

    不就地改是刻意的：调用方给的常常是它自己还要用的 metadata
    （比如节点往 span 里写的那份），脱敏不该把原对象改花。
    """

    if _depth >= _MAX_DEPTH:
        # 到了深度上限就不再往下走，整体折成一个占位符而不是原样吐出 ——
        # "太深了所以不脱敏"会让深层藏着的密钥顺利通过。
        return REDACTED

    if isinstance(value, Mapping):
        return {
            key: (
                REDACTED
                if is_sensitive_key(str(key))
                else redact_mapping(item, _depth=_depth + 1)
            )
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [
            redact_mapping(item, _depth=_depth + 1)
            for item in value
        ]

    if isinstance(value, str):
        return redact_text(value)

    return value


def redact_text(text: str) -> str:
    """
    抹掉一段文本里的密钥形态。

    用在**异常消息**上：provider SDK 会把 key 原样写进报错
    （`Incorrect API key provided: sk-abc…`），而异常消息会被记日志、
    写进观测行的 error_message —— 不处理就是一条真实的泄漏路径。
    """

    result = text

    for pattern in _SECRET_PATTERNS:

        if pattern.groups >= 2:
            # 保留"哪个字段泄漏了"这个信息（`api_key=***`），
            # 它对人排查有用，而值本身没有用。
            result = pattern.sub(
                lambda match: f"{match.group(1)}{match.group(2)}{REDACTED}",
                result,
            )
            continue

        if pattern.groups == 1:
            result = pattern.sub(
                lambda match: f"{match.group(1)} {REDACTED}",
                result,
            )
            continue

        result = pattern.sub(REDACTED, result)

    return result
