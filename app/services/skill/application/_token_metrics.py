"""
Skill 容量度量：在写路径测出 token 数，连口径一起落库。

两个数字、两种用途
------------------
    instructions_tokens   只有 instructions 原文，回答「这份 SOP 多长」
    prompt_tokens         SkillSnapshot.render_prompt() 渲染出的完整
                          system prompt，回答「实际占掉多少上下文」

后者才是前端卡片上那个数：真正注入子 agent 的不止 instructions，还有
标题、description 和能力清单。所以这里**直接复用运行时的
SkillSnapshot.render_prompt()**，不另写一份拼接 —— 两份定义一旦漂移，
展示出来的 token 数就再也对应不上实际注入的内容。

为什么不复用 document.tokenization
----------------------------------
`app/services/document/tokenization.py` 也是同一套 tiktoken 精确计数，
但它面向的是 chunk / embed 这类**可以失败重来**的离线任务：编码器首次
调用会联网下载 BPE 词表，失败就抛。这里是管理端 CRUD —— 一次词表下载
失败不能让「保存技能」整单失败。因此本模块自己做了三件那边的通用工具
不该做的事：

    1. 缓存「失败」这个结论（lru_cache 不缓存异常，会每次重试联网）
    2. 精确不可用时降级启发式，并**把降级口径写进 token_encoding**，
       绝不把估算伪装成精确值
    3. 提供 SKILL_TOKEN_FORCE_HEURISTIC=1，让离线 / 镜像环境显式关掉
       联网探测

口径永远和数字成对存在：读 `prompt_tokens` 必须同时读 `token_encoding`，
否则数字无法解释。运维若要精确值，预先下载词表并设置 TIKTOKEN_CACHE_DIR。
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from app.agent.context.skills import SkillCapability, SkillSnapshot

__all__ = [
    "FALLBACK_ENCODING",
    "SKILL_TOKEN_ENCODING",
    "UNMEASURED_ENCODING",
    "SkillTokenMetrics",
    "measure_skill_tokens",
]

#: 精确口径（与全项目 document/eval/windowing.py 的默认编码一致）。
SKILL_TOKEN_ENCODING = "o200k_base"

#: 降级口径：chars / 3.0，与运行时 HeuristicTokenCounter 一致。
FALLBACK_ENCODING = "heuristic:3.0"

#: 存量行 / 尚未测量。
UNMEASURED_ENCODING = "unmeasured"

_CHARS_PER_TOKEN = 3.0

# 缓存成功**或失败**：lru_cache 遇到异常不会记住，那样每个请求都会
# 重新尝试一次可能超时的词表下载。None = 还没探测过。
_encoder_cache: tuple[Any, str] | None = None


def _encoder() -> tuple[Any, str]:
    global _encoder_cache

    if _encoder_cache is None:
        if os.environ.get("SKILL_TOKEN_FORCE_HEURISTIC") == "1":
            _encoder_cache = (None, FALLBACK_ENCODING)
        else:
            try:
                import tiktoken

                _encoder_cache = (
                    tiktoken.get_encoding(SKILL_TOKEN_ENCODING),
                    SKILL_TOKEN_ENCODING,
                )
            except Exception:  # noqa: BLE001 - 测量永不阻断保存
                # 未安装 / 词表下载失败 / 缓存目录不可写，都走降级。
                _encoder_cache = (None, FALLBACK_ENCODING)

    return _encoder_cache


def _heuristic(text: str) -> int:
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def _measure(texts: tuple[str, ...]) -> tuple[list[int], str]:
    """一次拿到编码器，保证同一条记录的两个数字口径相同。"""

    encoder, encoding = _encoder()

    if encoder is not None:
        try:
            return [len(encoder.encode(t)) for t in texts], encoding
        except Exception:  # noqa: BLE001 - encode 本身也可能失败
            pass

    return [_heuristic(t) for t in texts], FALLBACK_ENCODING


@dataclass(frozen=True, slots=True)
class SkillTokenMetrics:
    """一条 skill 的容量度量。字段名与 DB / 响应契约逐字一致。"""

    instructions_tokens: int
    prompt_tokens: int
    token_encoding: str
    token_computed_at: datetime

    def as_entity_fields(self) -> dict[str, object]:
        """给 SkillEntity(...) / model_copy(update=...) 直接展开用。"""

        return {
            "instructions_tokens": self.instructions_tokens,
            "prompt_tokens": self.prompt_tokens,
            "token_encoding": self.token_encoding,
            "token_computed_at": self.token_computed_at,
        }


def measure_skill_tokens(
        *,
        skill_key: str,
        name: str,
        description: str,
        instructions: str,
        capabilities: Sequence[tuple[str, str]] = (),
) -> SkillTokenMetrics:
    """
    测量一份 skill 的容量。

    capabilities 是 (code, name) 二元组序列，由服务层从能力实体转换，
    本模块不认识 CapabilityEntity —— 保持纯计算、不碰数据库。

    新建的 skill 还没有绑定，传空序列即可；后续换绑定会重算一次。
    """

    snapshot = SkillSnapshot(
        skill_key=skill_key,
        name=name,
        description=description,
        instructions=instructions,
        capabilities=tuple(
            SkillCapability(code=code, name=display_name)
            for code, display_name in capabilities
        ),
    )

    (instructions_tokens, prompt_tokens), encoding = _measure(
        (instructions, snapshot.render_prompt())
    )

    return SkillTokenMetrics(
        instructions_tokens=instructions_tokens,
        prompt_tokens=prompt_tokens,
        token_encoding=encoding,
        token_computed_at=datetime.now(timezone.utc),
    )
