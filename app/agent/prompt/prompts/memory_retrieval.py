"""
记忆上下文（L3 画像 + L2 场景索引）—— 拼进主模型上下文的那一段。

⚠️ 这份提示词**不发起模型调用**
------------------------
其他四份 memory 提示词各是一次后台调用（抽取 / 去重 / 整合 / 生成）。
这一份的产物是**上下文的一个片段**：每轮组装时贴在 system 里给主模型看。
所以它的消费者有两条，都不是"调模型"：

    主模型上下文    用 `render_blocks(...)` **逐层**注入成四段 system 分节
                    （见 context/processor/processors/memory.py 的
                    MemoryProcessor）—— 标题由 `add_section(label=...)`
                    给，所以走的是分块的那个函数，不是 `render_user_message`
    管理端预览      `POST /memory/retrieval` —— 用同一份排版拼出一个整串
                    （`render_user_message`），给运维看"这个 scope 的记忆
                    拼出来长什么样"

⚠️ 文本当前是**占位**，不是定稿
------------------------
`SYSTEM_PROMPT` 是空串，`render_user_message` 的小节标题与分隔符是
**最小可用**的写法。它们等的是提示词的正文（那是产品决定，不是实现
细节）—— 换掉时只改本文件：

    SYSTEM_PROMPT         空串 -> 管理端预览呈现 `prompt.system: null`
                          （契约里写着"尚未实现时为 null"）
    render_user_message   小节标题 / 行格式 / 分隔符

调用方（服务层）**不认识**这些排版：它只把数据交进来、把字符串交出去
（见本目录 docstring 的第 3、4 条约定）。所以换文本不必碰服务层。

⚠️ 四段，但**默认只给两段**
--------------------
L3 画像与 L2 场景索引是每轮都带的；L0/L1 是**真召回**的结果（贵、按需），
只有调用方显式开了才会传进来（见 `MemoryRetrievalRequest.retrieve`）。
所以本函数里后两段是可选的，缺省是空序列 —— 它与"召回了但一条没命中"
在渲染上长得一样（都不打那一段），而**区分它们**是调用方的事
（`degraded` 与 `returned: 0` 分得开）。

⚠️ 为什么排版不能写在服务里
--------------------
与那四份提示词同一条理由：排版是提示词的正文。写在服务里时，改提示词的
人翻不到它，而"共 N 篇场景"这类计数会在两处各写一遍、改一处另一处就漂。

⚠️ 这一层的输入用 `Protocol` 声明，不 import 领域实体
--------------------------------------------
本文件不认识 `app.services.*`（与 `classification.py` 对候选清单同一手法）：
它只要求"给我一个有 filename / summary / heat 的东西"。
"""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, Sequence

from app.agent.model.types import ModelRole
from app.agent.prompt.contracts import PromptDefinition

__all__ = [
    "PROMPT",
    "SYSTEM_PROMPT",
    "BLOCK_CONVERSATIONS",
    "BLOCK_FACTS",
    "BLOCK_PERSONA",
    "BLOCK_SCENES",
    "CONVERSATIONS_TITLE",
    "FACTS_TITLE",
    "PERSONA_TITLE",
    "SCENES_TITLE",
    "ConversationHitLike",
    "FactHitLike",
    "PersonaLike",
    "SceneLike",
    "render_blocks",
    "render_user_message",
]


#: 系统提示词正文。
#:
#: ⚠️ **空 = 还没写**，不是"不需要"。占位状态在管理端预览里呈现为
#: `prompt.system: null`（见契约 `MemoryRetrievalPromptText.system`）。
#: 写成一句"你是一个助手…"之类的通用话反而更糟：它会看起来像定稿，
#: 而这段文本每一轮都占着主模型的上下文预算。
SYSTEM_PROMPT: str = ""


#: 各小节的标题。与正文一样是**占位**（改这里就够，调用方不认它们）。
#:
#: ⚠️ `PERSONA_TITLE` 与 `UserProfileProcessor` 的 label 曾经撞过车
#: （两边都叫"用户画像"）—— 已改那边（见 processors/profile.py 的说明）。
#: 这两个是**不同**的东西：这里是 L3 长期画像（从对话抽象出来的），
#: 那边是 ProfileSource 给的租户 / 偏好 / 权限摘要。
PERSONA_TITLE = "用户画像"
SCENES_TITLE = "记忆场景索引"
CONVERSATIONS_TITLE = "相关原话"
FACTS_TITLE = "相关记忆"

#: 四段的**稳定标识** —— 取值与契约里 `*.block` 是同一套词
#: （见 app/services/memory/domain/schemas/query.py 的 `PersonaSection` 等）。
#:
#: ⚠️ 与 `*_TITLE` **刻意分开**：标题是给人看的排版（文案随时会改），
#: 标识是调用方接数据用的（`source_ids` 按它取）。合成一个的话，改一次
#: 文案就会动到调用方的判断 —— 而那是静默的。
BLOCK_PERSONA = "persona"
BLOCK_SCENES = "scenes"
BLOCK_CONVERSATIONS = "conversations"
BLOCK_FACTS = "facts"


class PersonaLike(Protocol):
    """渲染需要的最小字段集（`PersonaRecord` 满足它）。"""

    content: str
    version: int


class SceneLike(Protocol):
    """渲染需要的最小字段集（`SceneSummary` 满足它）。"""

    filename: str
    summary: str
    heat: int | None
    created_at: datetime | None
    updated_at: datetime | None


class ConversationHitLike(Protocol):
    """L0 命中渲染需要的最小字段集（`ConversationHit` 满足它）。"""

    role: str
    text: str
    created_at: datetime | None


class FactHitLike(Protocol):
    """L1 命中渲染需要的最小字段集（`FactHit` 满足它）。"""

    content: str
    type: str
    created_at: datetime | None
    updated_at: datetime | None


def _stamp(value: object) -> str:
    """
    一个时间 -> 提示词里那一小段。**None 给空串**（整节省掉）。

    ⚠️ 用 `isoformat` 而不是 `strftime`
    -------------------------------
    仓库里把时间喂进提示词的既有口径就是它（见
    `prompts/memory_atom_extract.py` 的 `time=...`、以及两份抽取提示词里
    的 `Current Timestamp`）。跟着走，读提示词的人不必在两套格式之间切换。

    `timespec="minutes"`：这一段的用途是"这条记忆什么时候的事"，秒与
    微秒对判断没有增量，而它每轮都占主模型的上下文预算。

    ⚠️ 带时区偏移（值从库里出来就是 UTC，会渲染成 `+00:00`）
    ----------------------------------------------------
    **不做本地化换算**：换算会引入一个看不见的行为（同一句话在不同部署
    时区下渲染出不同文本），而回答里提到具体日期时，那个偏差很难归因到
    "提示词做了什么时区转换"。偏移量原样带着，读的人自己能判。
    取不到（None / 不是 datetime）时返回空串，调用方据此整节省掉。
    """

    if not isinstance(value, datetime):
        return ""

    return value.isoformat(timespec="minutes")


def _render() -> str:
    """系统提示词的 render（静态文本 -> `lambda: TEXT` 形态）。"""

    return SYSTEM_PROMPT


def render_blocks(
        *,
        persona: PersonaLike | None,
        scenes: Sequence[SceneLike],
        scenes_total: int,
        conversations: Sequence[ConversationHitLike] = (),
        facts: Sequence[FactHitLike] = (),
) -> list[tuple[str, str, str]]:
    """
    四层 -> 一组 `(段标识, 标题, 正文)`，**正文不带 `## 标题`**。

    标题与正文分开，是因为两个消费者的需要不同：

        管理端预览      要一个整串 -> `render_user_message` 把它拼回去
        主模型上下文    每层各占一段 system 分节，标题由
                        `add_section(label=...)` 给，再由
                        `system_message()` 渲染成不可信引用数据块
                        `<ref:memory desc="{label}, …">`（见
                        context/processor/reference.py）

    若正文里自带标题，注入时就渲染成两层 `## 用户画像` —— 所以分节注入的
    那条路**必须**用本函数而不是 `render_user_message`。

    两个函数共用这一处排版，因此管理端预览与主模型实际看到的**不可能漂**
    （原规格担心的正是"给你看一个 Agent 其实拿不到的结果"）。

    ⚠️ 为什么要带段标识（`persona` / `scenes` / `conversations` / `facts`）
    --------------------------------------------------------------
    分节注入的调用方还要给每段配 `source_ids`（画像是文件名、L0 是
    message_id、L1 是 record_id），而它只能按"这是哪一段"去取。让调用方
    自己再数一遍"哪几层非空"来对齐下标的话，两处条件一旦分叉就**静默**
    错位（每段的 id 挂到隔壁那段上，没人会去核）。所以标识由本函数给出，
    取值就是契约里 `*.block` 用的同一套词（见 domain/schemas/query.py）。

    `persona` 为 None 时**整段略去**（不是写一句"没有画像"）：这一段的
    用途是给模型提供事实，而"还没有画像"是调用方（与 degraded 列表）该
    说的事 —— 往上下文里塞一句无关信息会占预算。同理，`conversations` /
    `facts` 为空时那两段也不出现。

    `scenes_total` 与 `len(scenes)` 可能不同（列表被 `limits.l2` 截断）：
    计数写**总数**、条数写实际给的，于是"截断了"这件事在提示词里也看得见
    （原规格在 L2 那一层修的就是这个 —— 模型看不到被截掉的部分时，
    它会以为清单就是全部）。

    ⚠️ 两个参数必须由**同一个**数据源给出：调用方把 `list_scenes` 的结果
    先算总数、再截断，不要把"总数"另取一次（两次读之间有人写入时，两边
    会对不上，而症状是提示词里的计数与清单条数不符 —— 没人会去核）。

    ⚠️ L0/L1 那两段**不打分数**
    ----------------------
    `score` 是给管理端排查用的（"为什么这条排前面"），它对模型没有意义
    —— 名次就是它需要的信息，而两个数字（分数 + 名次）会让人以为分数
    有绝对含义（它只在本次响应内可比）。顺序本身就带着它。
    """

    blocks: list[tuple[str, str, str]] = []

    if persona is not None:
        # 版本进**标题**而不是正文：正文是整篇 persona.md（它自己带 META
        # 块），往开头插一行会混进那篇文档的结构里。标题是排版层的东西，
        # 挂在那里既不打扰正文，又让人一眼看到"这是第几版"。
        blocks.append(
            (
                BLOCK_PERSONA,
                f"{PERSONA_TITLE}（v{persona.version}）",
                persona.content.strip(),
            )
        )

    if scenes:
        blocks.append(
            (
                BLOCK_SCENES,
                f"{SCENES_TITLE}（共 {scenes_total} 篇，"
                f"以下 {len(scenes)} 篇）",
                "\n".join(_render_scene(scene) for scene in scenes),
            )
        )

    if conversations:
        blocks.append(
            (
                BLOCK_CONVERSATIONS,
                f"{CONVERSATIONS_TITLE}（{len(conversations)} 条）",
                "\n".join(
                    f"- [{_role_text(hit.role)}]{_at(hit.created_at)} {hit.text}"
                    for hit in conversations
                ),
            )
        )

    if facts:
        blocks.append(
            (
                BLOCK_FACTS,
                f"{FACTS_TITLE}（{len(facts)} 条）",
                "\n".join(_render_fact(hit) for hit in facts),
            )
        )

    return blocks


def render_user_message(
        *,
        persona: PersonaLike | None,
        scenes: Sequence[SceneLike],
        scenes_total: int,
        conversations: Sequence[ConversationHitLike] = (),
        facts: Sequence[FactHitLike] = (),
) -> str:
    """
    画像 + 场景索引（+ 可选的 L0/L1 命中）-> 用户消息。

    管理端的 prompt 预览就是这一段；主模型上下文将来也拼这一段。

    `persona` 为 None 时**整段略去**（不是写一句"没有画像"）：这一段的
    用途是给模型提供事实，而"还没有画像"是调用方（与 degraded 列表）该
    说的事 —— 往上下文里塞一句无关信息会占预算。同理，`conversations` /
    `facts` 为空时那两段也不出现。

    `scenes_total` 与 `len(scenes)` 可能不同（列表被 `limits.l2` 截断）：
    计数写**总数**、条数写实际给的，于是"截断了"这件事在提示词里也看得见
    （原规格在 L2 那一层修的就是这个 —— 模型看不到被截掉的部分时，
    它会以为清单就是全部）。

    ⚠️ 两个参数必须由**同一个**数据源给出：调用方把 `list_scenes` 的结果
    先算总数、再截断，不要把"总数"另取一次（两次读之间有人写入时，两边
    会对不上，而症状是提示词里的计数与清单条数不符 —— 没人会去核）。

    ⚠️ L0/L1 那两段**不打分数**
    ----------------------
    `score` 是给管理端排查用的（"为什么这条排前面"），它对模型没有意义
    —— 名次就是它需要的信息，而两个数字（分数 + 名次）会让人以为分数
    有绝对含义（它只在本次响应内可比）。顺序本身就带着它。
    """

    return "\n\n".join(
        f"## {title}\n{body}"
        for _block, title, body in render_blocks(
            persona=persona,
            scenes=scenes,
            scenes_total=scenes_total,
            conversations=conversations,
            facts=facts,
        )
    )


def _at(value: object) -> str:
    """
    " `时间`" —— 带前导空格，取不到时是空串。

    单列出来是为了让拼接处读起来是 `f"...]{_at(t)} 正文"`：时间缺失时
    整段（含那个空格）一起消失，不会留下一个孤零零的分隔符。
    """

    stamp = _stamp(value)

    return f" {stamp}" if stamp else ""


def _render_scene(scene: SceneLike) -> str:
    """
    一行场景：

        - 音乐爱好-吉他Blues练习.md｜热度 4｜2026-10-10T03:20 → 2026-10-12T02:41：…

    `heat` 为 None（META 里没写或解析不出）时**不打那一节**，而不是打
    "热度 null" —— 后者会让模型把 null 当成一个有意义的取值。时间同此：
    取不到就整节省掉。

    ⚠️ 两个时间用 `→` 连，且**只在不同时才打后半段**
    -------------------------------------------------
    场景是"被整合改写过"的东西，两个时间相等只说明"写过之后再没动过"，
    把同一个值打两遍纯属占预算。不同时打出 `创建 → 更新`，读的人一眼
    看到这条场景最近被动过 —— 那正是判断"它还新不新"要的东西。
    """

    heat = "" if scene.heat is None else f"｜热度 {scene.heat}"

    created = _stamp(scene.created_at)
    updated = _stamp(scene.updated_at)

    if created and updated and created != updated:
        when = f"｜{created} → {updated}"
    else:
        stamp = created or updated
        when = f"｜{stamp}" if stamp else ""

    return f"- {scene.filename}{heat}{when}：{scene.summary}"


def _role_text(role: object) -> str:
    """
    一条原话的角色 -> 提示词里那个词（`user` / `assistant`）。

    ⚠️ 不能直接 `f"{role}"`
    --------------------
    `ConversationHitLike` 声明的是 `str`，而真实 DTO（`ConversationHit`）
    里它是 `MessageRole` —— 一个 `str` 混入的 Enum。混入枚举的 `__format__`
    给的是 **`MessageRole.USER`** 而不是 `user`，于是模型读到的是
    `- [MessageRole.USER] 我上次说过…` 这种噪音（既有行为，不是本次引入的）。

    取 `.value`；拿不到就当它本来就是字符串（协议就是这么声明的）。
    """

    return str(getattr(role, "value", role))


def _render_fact(hit: FactHitLike) -> str:
    """
    一条 L1 命中：

        [persona] 2026-10-10T03:20 → 2026-10-12T02:41 用户弹电吉他，…

    `type` 空串（词表外的类型、或模型没给）时把中括号一并省掉，不留一对
    空括号 —— 与场景那行的 `heat` 同一条取舍。

    ⚠️ `type` 是这一层**唯一**能说明"这条记忆是什么性质"的东西
    ------------------------------------------------------
    persona（稳定属性）/ episodic（发生过的事）/ instruction（用户提的
    长期要求）—— 三条同权重地列在一起，不给类型，模型就分不出"用户**是**
    这样的人"与"用户**要求**这样做"，而后者是约束、不能被当成描述复述。

    ⚠️ 两个时间与场景同一条取舍：相等就只打一个
    -----------------------------------------
    合并（dedup）会改写 `updated_at` 而 `created_at` 不动，所以两者相等
    只说明"写下之后没被合并过"。不同时打 `创建 → 更新`，读的人据此看得出
    "这条刚被合并改写过" —— 而那是判断它可不可信的一个信号。
    """

    kind = f"[{hit.type}] " if hit.type else ""

    created = _stamp(hit.created_at)
    updated = _stamp(hit.updated_at)

    if created and updated and created != updated:
        when = f"{created} → {updated} "
    else:
        stamp = created or updated
        when = f"{stamp} " if stamp else ""

    return f"{kind}{when}{hit.content}"


PROMPT = PromptDefinition(
    key=ModelRole.MEMORY_RETRIEVAL,
    description=(
        "记忆上下文（L3 画像 + L2 场景索引）—— 拼进主模型 system 的那一段。"
        "**不发起模型调用**，所以没有 model.yaml 绑定指向它。"
    ),
    render=_render,
)
