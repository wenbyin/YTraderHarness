"""
出题的"被截断"必须与"真写坏了 JSON"分开报。

为什么值得单列一条
------------------
两者在 `json.loads` 眼里**完全一样**（都是 `Unterminated string`），处置
却相反：截断要抬额度或拆请求，写坏要改提示词、重试有意义。此前日志把两类
都写成"模型输出不是合法 JSON"，指向的是解析器 —— 2026-10-16 实测那次
（出题窗口输出 456 字符断在字符串中间）就是被这句话带偏的。

判据是 provider 的 `finish_reason == "length"`（权威信号）；拿不到它就
返回 None，宁可退回"无法解析"也不猜。
"""

from __future__ import annotations

from types import SimpleNamespace

from app.services.document.eval.generator import (
    _truncation_evidence,
    build_model_request,
)


def _response(*, finish_reason, completion=None, reasoning=None):
    """一条假的 ModelResponse —— 只带这个判定要看的那几个字段。"""

    token_usage = {"completion_tokens": completion}

    if reasoning is not None:
        token_usage["completion_tokens_details"] = {
            "reasoning_tokens": reasoning
        }

    message = SimpleNamespace(
        response_metadata={
            "finish_reason": finish_reason,
            "token_usage": token_usage,
        }
    )

    return SimpleNamespace(message=message)


def test_length_finish_reason_is_reported_as_truncation() -> None:
    """`finish_reason=length` + 用量 → 一句能直接定位问题的证据。"""

    evidence = _truncation_evidence(
        _response(finish_reason="length", completion=8192, reasoning=7800)
    )

    assert evidence is not None
    assert "length" in evidence
    assert "8192" in evidence
    assert "7800" in evidence, "推理占了多少是判断'额度被谁吃了'的关键"


def test_normal_stop_is_not_reported_as_truncation() -> None:
    """正常结束就是正常结束 —— 别把"写坏了"也算到额度头上。"""

    assert _truncation_evidence(_response(finish_reason="stop")) is None


def test_missing_finish_reason_falls_back_to_parse_error() -> None:
    """
    拿不到 `finish_reason` 时不猜。

    provider 没报的话，宁可让调用方按"无法解析"处理（至少不误导），也不要
    把"看起来像截断"当成截断 —— 那会让人去调一个没问题的额度。
    """

    assert _truncation_evidence(_response(finish_reason=None)) is None
    assert _truncation_evidence(SimpleNamespace(message=None)) is None


def test_usage_fields_are_optional() -> None:
    """用量缺失时仍然给得出证据（只是少两个数）。"""

    message = SimpleNamespace(
        response_metadata={"finish_reason": "length"}
    )

    evidence = _truncation_evidence(SimpleNamespace(message=message))

    assert evidence == "finish_reason=length"


def test_generation_request_asks_for_the_full_window() -> None:
    """
    出题一次要的条数（5 chunk × 3 题）—— 那个量决定了可见输出的需求，
    也就决定了"额度被推理吃掉之后还够不够"。
    """

    request = build_model_request(
        window_chunks=[{"chunk_id": "c", "content": "正文"}] * 5,
        questions_per_chunk=3,
        document_context="标题：某某公司",
    )

    assert request.metadata["chunk_count"] == 5
    assert request.metadata["questions_per_chunk"] == 3
