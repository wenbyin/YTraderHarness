"""
`model_role=memory.atom.dedup` 的提示词：L1 记忆冲突检测（去重）。

文本来自 `memory_l1.md`（原项目 `CONFLICT_DETECTION_SYSTEM_PROMPT`），
**一字未改** —— 与抽取那份同一条理由：它是一份被调过的提示词。

谁用它
------
`MemoryAtomServiceImpl._dedup`（`memory.l1.extract` 那条任务的第二步）。

⚠️ 一次调用 = 整批新记忆 + 一个统一的候选池
--------------------------------------
提示词要的是"批量比较"：**一次**调用里同时给出这批新记忆与合并后的候选
池，输出是"每条新记忆一个决策"的数组。不是每条记忆调一次 —— 前者能让
模型看到彼此（两条新记忆之间也可能重复），后者看不到。

⚠️ 可变部分（新记忆与候选池）走 HumanMessage
--------------------------------------
与抽取那份同分工，见 app/agent/prompt/contracts.py 的模块 docstring。
`render` 因此没有参数。

⚠️ 这份提示词的输出**没有 scene_name**
-------------------------------
它只给 `merged_content` / `merged_type` / `merged_priority` /
`merged_timestamps`。所以合并之后那条记忆的情境取**新记忆**的（服务侧
决定，见 memory_atom_service_impl 的 `_apply`）—— 这是原地更新语义下
一个必须显式处理的地方，不处理会静默保留旧值。
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from app.agent.model import ModelRole

from ..contracts import PromptDefinition

__all__ = ["PROMPT", "TEXT", "AtomLike", "render_user_message"]


TEXT = """\
你是记忆冲突检测器。批量比较多条【新记忆】与【统一候选记忆池】中的已有记忆，逐条决定如何处理。

**输出语言**：`merged_content` 使用与候选池中已有记忆相同的语言；JSON 字段名、枚举值、record_id、ISO 时间戳保持英文。

## 核心规则

- **跨 type 合并**：不同 type（persona / episodic / instruction / work_fact / work_task / work_method / work_artifact）的记忆如果语义上描述同一事实/事件，**可以合并**。
- **多对多合并**：一条新记忆可以同时替换/合并候选池中的**多条**已有记忆（通过 target_ids 数组指定）。
- 合并后你必须判断新记忆的最佳 type（merged_type）。

## 判断逻辑

1. **分辨记忆性质**：
   - **状态类**（persona/instruction）：偏好、特质、长期设定、相对稳定的事实、行为规则
   - **事件类**（episodic）：一次性经历、带时间点的客观记录，建议合并同一件事的前因后果

2. **判断是否同一事实/事件**：主体相同、主题一致、时间接近、scene_name 相似

3. **选择动作**：
   - "store"：视为新信息，新增当前记忆。
   - "skip"：已有记忆更好，新记忆无增量或更模糊，忽略当前记忆。
   - "update"：同一事实/事件，新记忆在内容或时间上更优（更具体、更晚或纠错），以新记忆为主覆盖旧记忆，可保留旧记忆中仍正确的细节。
   - "merge"：同一事实或同一演化过程，多条记忆信息互补且不矛盾，合并成一条更完整记忆，信息尽量不冗余。

4. **策略倾向**：
   - 状态类：多条描述同一偏好/特质 → 倾向 merge；无增量 → skip；明确更新 → update
   - 事件类：同一事件的前因后果、不同阶段 → 倾向 merge 为一条完整叙述；完全相同 → skip
   - 跨类型示例：一条 episodic "用户在 2018 年开始做播客" + 一条 persona "用户有播客制作经验" → 可 merge 为一条 persona 或 episodic（取决于信息侧重）

5. **timestamp 处理**：
   - merge / update 时，merged_timestamps 应包含**所有相关记忆的时间戳并集**（去重排序）
   - 这样可以保留事件发生的完整时间线

## 输出格式

严格输出 JSON 数组，每个元素对应一条新记忆的决策。不输出任何其他内容：

[
  {
    "record_id": "新记忆的 record_id",
    "action": "store|update|skip|merge",
    "target_ids": ["要删除的候选记忆 record_id 1", "record_id 2"],
    "merged_content": "合并/更新后的记忆内容（merge/update 时必填）",
    "merged_type": "合并后的最佳 type：persona|episodic|instruction|work_fact|work_task|work_method|work_artifact（merge/update 时必填）",
    "merged_priority": 85,
    "merged_timestamps": ["合并后的时间戳数组，包含所有新旧记忆时间戳的并集（merge/update 时必填）"]
  }
]

字段说明：
- target_ids：要删除替换的旧记忆 ID **数组**（可以 1 条或多条）。store/skip 时省略或为空。
- merged_content：merge/update 时的最终记忆文本。store/skip 时省略。
- merged_type：merge/update 后记忆应归属的 type。根据合并后内容本质判断。
- merged_priority：merge/update 后的新优先级（0-100 整数，merge/update 时必填）。合并后信息更完整、更确定，通常应**酌情提升** priority（例如两条 priority 70 的记忆合并后可提升到 80）。参考标准：80-100（核心特质/重要事件），60-79（一般偏好/普通活动），<60（次要信息）。
- merged_timestamps：合并后的时间戳数组。收集新记忆 + 所有被合并旧记忆的时间戳，去重排序。"""


def _render() -> str:
    """无参数 —— 可变部分（新记忆 / 候选池）走 `render_user_message`。"""

    return TEXT


# ============================================================
# 用户消息（HumanMessage）：新记忆 + 统一候选池
# ============================================================
#
# ⚠️ 排版归本文件，取数归服务
# ----------------------
# 见 prompts/__init__.py 的约定。这一段原先散在
# `MemoryAtomServiceImpl._build_conflict_request` 里（`new_lines` /
# `pool_lines` 两处拼串），搬过来是为了让"这条记忆在提示词里长什么样"
# 只有一处 —— 它与 TEXT 里那句"批量比较【新记忆】与【统一候选记忆池】"
# 是同一件事的两半，分开写就会漂。


class AtomLike(Protocol):
    """
    渲染新记忆 / 候选池需要的**最小字段集**（duck-typed，不 import 领域实体）。

    新记忆与候选池用的是同一组字段，所以共用一个 Protocol —— 它们本来就是
    同一种东西（`memory_atom` 的行），只是一个"刚抽出来"、一个"已经在库里"。
    """

    record_id: Any
    type: str
    priority: int
    scene_name: str
    timestamp_str: str
    meta: Mapping[str, Any]
    content: str


def render_user_message(
        *,
        atoms: Sequence[AtomLike],
        pool: Sequence[AtomLike],
        related: Mapping[Any, Sequence[Any]],
) -> str:
    """
    新记忆 + 统一候选池 + **每条新记忆各自的候选 id** -> 一次调用的用户消息。

    ⚠️ 四块的内容与顺序照抄原项目的 `formatBatchConflictPrompt`
    -----------------------------------------------------
        1  统一候选池（含条数）     所有新记忆召回结果的**并集**，去重
        2  （隔断）
        3  待判断的新记忆（含条数）
        4  收尾那一句判据           "候选列表为空时直接 store"

    ⚠️ `related` 是这一版**新补上的**东西（2026-10-08 那版没有）
    ----------------------------------------------------
    原实现只给一个并集池，模型得自己猜"这几条候选里哪些跟这条新记忆有关"
    —— 而那个对应关系**召回时本来就算出来了**，只是被丢掉了（见服务侧
    `_candidate_pool`，它现在返回两样东西）。

    为什么它值得单独渲染：池子是并集，一条新记忆的候选可能与另一条的
    完全无关；模型看着一池几十条候选去判"是不是同一件事"，很容易抓错
    对象（尤其两条新记忆主题相近时）。把"这条的候选是这几条"直接给它，
    判定的搜索空间从"整个池子"缩到"这几条"。

    ⚠️ 池子为空的分支在**当前调用链上走不到**
    ------------------------------------
    服务侧在池子空时**根本不调模型**（没有可比的东西，每条只能是 store，
    调一次是纯浪费）。这里留着它是因为本函数的契约是"任何输入都渲染得出
    一份合法的用户消息" —— 直接调它的人（测试、将来的批量工具）不该拿到
    一段空标题。
    """

    records = [
        {
            "record_id": atom.record_id,
            "content": atom.content,
            "type": atom.type,
            "priority": atom.priority,
            "scene_name": atom.scene_name,
            "timestamps": _timestamps(atom),
        }
        for atom in pool
    ]

    if records:
        pool_section = (
            f"## 统一候选记忆池（共 {len(records)} 条已有记忆）\n\n"
            + _json(records)
        )
    else:
        pool_section = (
            "## 统一候选记忆池\n\n"
            "（空，没有已有记忆，所有新记忆直接 store）"
        )

    parts: list[str] = []

    for index, atom in enumerate(atoms, start=1):

        ids = [str(value) for value in related.get(atom.record_id, ())]

        # 空列表**不是**"[ ]"就算了 —— 提示词的收尾明确要求"候选为空时直接
        # store"，把这句话贴在数据旁边（和 L1 抽取那份把"只从这里提取"
        # 贴在数据旁边是同一个理由：贴着数据的那句才会被照着做）。
        note = (
            _json(ids)
            if ids
            else "[]（无相似候选，直接 store）"
        )

        parts.append(
            f"### 第 {index} 条新记忆 (record_id: {atom.record_id})\n"
            + _json({
                "record_id": atom.record_id,
                "content": atom.content,
                "type": atom.type,
                "priority": atom.priority,
                "scene_name": atom.scene_name,
            })
            + f"\n\n【关联候选 ID】{note}"
        )

    return "\n\n".join([
        pool_section,
        _RULE,
        f"## 待判断的新记忆（共 {len(atoms)} 条）",
        _SEPARATOR.join(parts),
        "请逐条判断并输出决策 JSON 数组。"
        "当某条新记忆的候选列表为空时，该条直接输出 action=store。",
    ])


#: 两个隔断。照抄原项目 —— 它们在这里是有用的：一段池子 JSON 与一段新记忆
#: JSON 挨在一起时，模型容易把两者的边界看错（都是 `{` 开头的对象）。
_RULE = "═" * 50
_SEPARATOR = "\n\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"


def _json(value: Any) -> str:
    """
    渲染用的 JSON。

    `ensure_ascii=False`：中文原样进去（记忆正文多数是中文），转成 `\\uXXXX`
    会让提示词里那一大段完全不可读 —— 排查时读日志的人只能看到一片转义。
    `default=str`：id 可能是 UUID / 别的不可序列化类型，转成它的字符串形式
    就行（模型要的就是那个可回显的串）。
    """

    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _timestamps(atom: AtomLike) -> list[str]:
    """
    一条记忆的时间线 —— 优先取 `meta` 里那个并集，否则退回 `timestamp_str`。

    ⚠️ 两个来源都要看
    ------------
    `meta['timestamps']` 只在**被合并过**的记忆上才有（合并时把新旧并集写
    进去，见 `_merge`）；一条从没合并过的记忆只有 `timestamp_str`。只看前者
    会让绝大多数候选的 timestamps 是空数组 —— 而提示词的判断逻辑第 2 条
    要拿"时间接近"当判据之一，空数组等于把那一条废掉。
    """

    stored = (atom.meta or {}).get("timestamps")

    if stored:
        return [str(value) for value in stored]

    value = (atom.timestamp_str or "").strip()

    return [value] if value else []


PROMPT = PromptDefinition(
    key=ModelRole.MEMORY_ATOM_DEDUP,
    description=(
        "L1 记忆冲突检测：把一批新记忆与候选池里的已有记忆逐条比对，"
        "决定 store / skip / update / merge 并给出合并后的内容。"
    ),
    render=_render,
)
