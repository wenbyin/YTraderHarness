"""
工具返回值 -> (给模型看的正文, 入库的结构) 这两条路的形状。

`_render` 与 `_ledger_result` **不是同一件事**，但它们判断的"值能
变成什么"有重叠：

    _render         -> ToolMessage 的正文（一个字符串）
    _ledger_result  -> tool_execution.result 列（JSONB，可查询）

本文件守三件事：

    1. `_render` 的行为**一个字都没变** —— 它是模型的输入，
       动了就改了模型看到的东西（扇出那条用例只覆盖到"字符串返回值"）
    2. `_ledger_result` 尽最大努力保住**结构**：容器要递归下去，
       而不是"试一次 dumps、失败就整份降级"
    3. 真降级时**不静默** —— 存进去的是一段文本而不是结构化数据时，
       必须留下 WARNING，否则读库的人无从分辨
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime, timezone
from decimal import Decimal
from uuid import uuid4

import pytest
from pydantic import BaseModel

from app.agent.graph.react.tools import _ledger_result, _render


class _Payload(BaseModel):
    """服务方法常见的返回形状：一个 pydantic 模型。"""

    request_id: str
    total: int


class _Explodes:
    """连 repr 都会抛的对象 —— 降级那条路自己也会炸的那种。"""

    def __repr__(self) -> str:
        raise RuntimeError("连 repr 都炸")


# ============================================================
# _render：模型看到的正文
# ============================================================


def test_render_keeps_its_old_behaviour() -> None:
    """
    ⚠️ 这几条是**锁定既有行为**的，不是新需求。

    `_render` 的输出直接进 ToolMessage，也就是模型的输入。任何一条
    变了都意味着模型看到的东西变了 —— 那是要单独做的决定，不该
    随着"给账本加个 result 列"顺带发生。
    """

    # None 是**合法的成功结果**，渲染成空串。
    assert _render(None) == ""

    # str 直用，不被引号包起来。
    assert _render("已经查完了") == "已经查完了"

    # 其余 JSON 化，且中文不转义（ensure_ascii=False）。
    assert _render({"total": 3}) == '{"total": 3}'
    assert json.loads(_render({"名称": "沪深300"})) == {"名称": "沪深300"}


def test_render_does_not_truncate() -> None:
    """
    长结果**不截断** —— 悄悄砍一刀会让模型看到一段被截断的内容却
    不知道它被截断了（该由 compaction / 预算那条路解决）。
    """

    long_text = "x" * 50_000

    assert _render(long_text) == long_text


# ============================================================
# _ledger_result：入库的结构
# ============================================================


def test_ledger_result_keeps_json_native_values_as_structure() -> None:
    """
    ⚠️ 这是本函数存在的理由：值**原样存**，不是渲染成字符串。

    存成字符串的话 `result` 那一列就没法查询了（`result->>'total'`
    之类的用法全部落空），而它作为一个 JSONB 列的意义正是可查询。
    """

    assert _ledger_result(None) is None
    assert _ledger_result("文本") == "文本"
    assert _ledger_result(7) == 7
    assert _ledger_result(True) is True

    container = {"items": [1, 2], "total": 2}
    assert _ledger_result(container) == container


def test_ledger_result_dumps_pydantic_models() -> None:
    """服务方法返回实体 / DTO 是最常见的一种，它必须是结构而不是 repr。"""

    stored = _ledger_result(_Payload(request_id="r1", total=3))

    assert stored == {"request_id": "r1", "total": 3}
    assert not isinstance(stored, str)


def test_ledger_result_recurses_into_containers_of_models() -> None:
    """
    ⚠️ 这条来自**真实数据**，不是假想的边角。

    第一版是"试一次 `json.dumps`，失败就整份降级"。而本仓最常见的
    返回恰好撞上那条路 —— `capability.list_capabilities` 返回
    `list[Capability]`（pydantic 实体列表），于是那一行的 result 存成了：

        "[\\"id=UUID('8af0325a-...') code='CAPABILI..."

    一段 repr 文本，`result->0->>'code'` 这种查询完全没法用。
    """

    stored = _ledger_result([
        _Payload(request_id="r1", total=1),
        _Payload(request_id="r2", total=2),
    ])

    assert stored == [
        {"request_id": "r1", "total": 1},
        {"request_id": "r2", "total": 2},
    ]
    # 嵌在字典里同样要递归下去。
    assert _ledger_result({"rows": [_Payload(request_id="r3", total=3)]}) == {
        "rows": [{"request_id": "r3", "total": 3}]
    }


def test_ledger_result_normalizes_json_adjacent_scalars() -> None:
    """
    标量那几条的判据锚在 pydantic 上：`model_dump(mode="json")`
    会把 datetime 变成 ISO 串、UUID 变成 str —— 本函数给出同样的
    形态，于是"同一个值裹不裹在 pydantic 里"存进库都是一样的。
    """

    moment = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
    identity = uuid4()

    assert _ledger_result(moment) == moment.isoformat()
    assert _ledger_result(date(2026, 1, 1)) == "2026-01-01"
    assert _ledger_result(identity) == str(identity)
    assert _ledger_result(Decimal("1.5")) == "1.5"

    # 嵌在容器里也一样 —— 否则一个 datetime 会把整份结果拖去降级。
    assert _ledger_result({"at": moment, "id": identity}) == {
        "at": moment.isoformat(),
        "id": str(identity),
    }


def test_ledger_result_handles_tuples_and_sets_as_arrays() -> None:
    """与 `model_dump(mode="json")` 一致：都当数组。"""

    assert _ledger_result((1, 2)) == [1, 2]
    assert sorted(_ledger_result({1, 2})) == [1, 2]


def test_ledger_result_warns_when_it_degrades(
        caplog: pytest.LogCaptureFixture,
) -> None:
    """
    认不出来的叶子会让**整份**值降级 —— 所以必须留痕。

    存进去的是一段 JSON **文本**而不是结构，读库的人从值本身分辨
    不出来。WARNING 里带上类型名，配工具的人据此知道要改返回形状。
    与 `select_specs` 跳过不可执行工具时的"跳过要留痕"同一条立场。
    """

    with caplog.at_level(
            logging.WARNING, logger="app.services.tool.application._execution"
    ):
        degraded = _ledger_result({"fine": 1, "bad": object()})

    assert isinstance(degraded, str)          # 确实是降级形态
    assert "object" in caplog.text
    assert "不知道怎么变成 JSON" in caplog.text


def test_ledger_result_warns_for_unknown_top_level_types(
        caplog: pytest.LogCaptureFixture,
) -> None:
    """既不是 JSON 原生、也不是容器 / pydantic / 已知标量 —— 同样降级。"""

    with caplog.at_level(
            logging.WARNING, logger="app.services.tool.application._execution"
    ):
        degraded = _ledger_result(object())

    assert isinstance(degraded, str)
    assert "不知道怎么变成 JSON" in caplog.text


def test_ledger_result_falls_back_for_self_referencing_containers() -> None:
    """
    自引用的容器不会让它 RecursionError。

    `json.dumps` 遇到这种结构会自己抛 ValueError，而这里是**手写
    递归** —— 不设深度上限就是 RecursionError 一路冒到节点外，把一次
    成功的调用变成崩图。与 registry/schema.py 的 MAX_SCHEMA_DEPTH
    是同一类防御。
    """

    loop: list = [1]
    loop.append(loop)

    stored = _ledger_result(loop)

    # 降级成了文本（而不是抛 RecursionError）。
    assert isinstance(stored, str)


def test_ledger_result_never_raises(
        caplog: pytest.LogCaptureFixture,
) -> None:
    """
    ⚠️ 它**永不抛**，哪怕降级那条路本身也炸。

    走到这里工具已经跑过了。为了记一笔账把一次成功的调用变成崩节点，
    是拿观测去推翻事实。

    这不是假想的边角：降级走的是 `_render`，而它的 `default=str`
    会调 `str()` —— **一个 repr 会抛的对象能让它炸**。写这条用例时
    它是真的会抛，所以 `_ledger_result` 里补了一层兜底。
    """

    with caplog.at_level(
            logging.WARNING, logger="app.services.tool.application._execution"
    ):
        stored = _ledger_result(_Explodes())

    assert isinstance(stored, str)
    assert "_Explodes" in stored
    # 两次降级都留了痕：外层与内层各一条。
    assert "不知道怎么变成 JSON" in caplog.text
    assert "连降级渲染也失败了" in caplog.text
