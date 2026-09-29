"""
Chunker 身份：决定"哪几路 retrieval 该共用同一份 chunk"。

为什么需要它
------------
切分的产出只取决于两件事：

    chunker 实现 + 它的配置

与 retrieval_type / retriever / storage_backend **都无关**。但流水线
原本按 DocumentRetrievalDefinition 扇出，于是"一路 keyword + 一路
dense_vector 共用同一个 chunker"这个最常见的配置会把同一份正文切两遍，
产出两组内容完全相同、document_chunk_id 不同的 chunk。

后果在查询期：融合按 document_chunk_id 归并，同一块被两路命中却认不出
是同一块 —— 重复结果，且"两路都命中"这个最强的相关性信号丢失。

所以切分阶段改为按**本模块算出的身份**扇出；后三个阶段仍按 retrieval
扇出（向量形态与目标容器确实是 per-retrieval 的）。

白名单，不是黑名单
------------------
身份只取上表那两项。**不要**改成"拿整个解析后的对象哈希、把
document_retrieval_definition_id 置空" —— RetrievalBranch 里还有
retrieval_type / retriever_config / storage_backend / index_identifier
等 6 个字段会随 retrieval 变化，其中 retrieval_type 恰好就是
"一路 keyword、一路 dense_vector"里不同的那一维。把它们带进哈希，
去重必然 0 命中。

白名单的失效方向是"该分的没分"（错），黑名单在当前字段集下是
"该合的没合"（只是没优化）—— 所以这里必须用白名单，并且：
**将来若给切分引入新的输入维度，必须同步加进 identity()。**

配置要归一化
------------
`{"chunk_size": 512}` 与 `{}`（默认就是 512）是同一个切分，字典却不等。
因此把配置过一遍实现自己的 config_model，比的是**真正影响产出的那组
参数**：默认值补齐、字段集合稳定、未知键按模型的 extra 策略处理。

这同时保证了"落库的配置"与"查 chunk 时用来比对的配置"是同一份规范
形式 —— 两边都用本函数算，不可能漂移。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..domain.schemas.enum import ProcessorRole

if TYPE_CHECKING:
    from ..pipeline.registry import PipelineRegistry

__all__ = [
    "ChunkerIdentity",
    "chunker_identity",
    "identity_digest",
    "normalize_chunker_config",
    "resolve_chunker",
]


def identity_digest(
        chunker_key: str,
        config: dict[str, Any] | None,
) -> str:
    """
    从**已经归一化**的 (key, config) 直接算 digest。

    给文档 chunk 用：它落库时就存着归一化后的两项，查的时候不需要
    再去解析 chunker 实现、更不需要注册表。要拿到"某一路 retrieval
    的身份"才需要 registry（见 chunker_identity），因为定义里存的是
    原始配置，得先过实现自己的 config_model。

    ⚠️ 传进来的 config 必须已经是归一化的。传原始配置会得到与
    chunker_identity 不同的 digest —— 表现是"这条 chunk 不属于任何
    retrieval"，且不报错。chunk 行上的值满足这个前提（见 text.py
    的组装处），原始配置**不**满足。
    """

    return hashlib.sha256(
        _canonical(
            {"key": chunker_key, "config": dict(config or {})}
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, eq=False)
class ChunkerIdentity:
    """
    一份切分产出的身份。

    eq=False 是刻意的：`config` 是 dict，不可哈希，自动生成的
    __eq__ / __hash__ 会在放进 set 时炸。要比较就比 `digest`。
    """

    #: chunker 的契约键（如 TEXT_TXT_RECURSIVE_CHUNKER）。
    chunker_key: str

    #: 归一化之后的配置。可直接落 jsonb —— Postgres 的 jsonb 相等是
    #: **语义相等**（键序无关），因此它能直接进唯一约束。
    config: dict[str, Any]

    #: chunker_key + config 的 sha256。用作产物通道的键
    #: （chunk_refs）与日志/观测里的短标识。
    digest: str

    @property
    def config_json(self) -> str:
        """规范化 JSON 文本，便于日志与调试输出。"""

        return _canonical(self.config)


def _canonical(value: Any) -> str:
    """
    规范化 JSON 文本。

    sort_keys 是必须的：否则同一份配置换个键序就是另一个 digest，
    去重会随写入顺序时灵时不灵。
    """

    return json.dumps(
        value, sort_keys=True, ensure_ascii=False
    )


def resolve_chunker(
        registry: "PipelineRegistry",
        chunker_name: str | None,
):
    """
    取本路要用的 chunker：**点名优先，没点名才按 role 取默认**。

    与 pipeline/nodes/chunking.py 原本内联的逻辑是同一套，提到这里
    是为了让"算身份"与"真去切"用的是**同一个** chunker —— 两处各
    写一遍、哪天选择规则改了就只改一处，身份与实际产出会对不上。
    """

    # 局部 import：pipeline.registry 在模块级会反向 import chunker
    # 的实现类，放在顶层会形成循环。
    from ..pipeline.registry import CapabilityNotFound

    if chunker_name:

        chunker = registry.processor(chunker_name)

        if type(chunker).role is not ProcessorRole.CHUNK:
            raise CapabilityNotFound(
                f"chunker_name={chunker_name!r} 指向的是 "
                f"role={type(chunker).role.value} 的能力，"
                "不是 chunker —— 配置与阶段对不上。"
            )

        return chunker

    chunker = registry.processor_by_role(ProcessorRole.CHUNK)

    if chunker is None:
        raise RuntimeError(
            "No capability registered for role=chunk. "
            "分块是流水线的必经阶段，缺实现应当显式失败，"
            "而不是产出一份没有 chunk 的结果。"
        )

    return chunker


def normalize_chunker_config(
        chunker,
        raw_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    把原始配置过一遍该实现自己的 config_model。

    `config_model is None` 是**真实存在**的情形（例如
    TEXT_NO_CHUNK_CHUNKER：它没有任何可调项），不是防御性代码。
    这类实现的所有配置都归一到 `{}` —— 它们本来就等价。
    """

    model = getattr(type(chunker), "config_model", None)

    if model is None:
        return {}

    return model.model_validate(
        dict(raw_config or {})
    ).model_dump(mode="json")


def chunker_identity(
        registry: "PipelineRegistry",
        chunker_name: str | None,
        raw_config: dict[str, Any] | None,
) -> ChunkerIdentity:
    """
    算出本路切分的身份。

    注意 `resolve_chunker` 会显式失败（配置点名的 chunker 不存在、
    或该 role 没有实现）—— 这是刻意的：拿一个"默认 chunker"顶上去
    会让身份与实际产出对不上，而去重的错误表现是**静默的**。
    """

    chunker = resolve_chunker(registry, chunker_name)

    chunker_key = chunker.get_key()

    config = normalize_chunker_config(chunker, raw_config)

    return ChunkerIdentity(
        chunker_key=chunker_key,
        config=config,
        digest=identity_digest(chunker_key, config),
    )
