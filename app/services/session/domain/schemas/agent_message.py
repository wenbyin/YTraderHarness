"""
会话消息的对外契约。

逐字段对应 messages.yaml 的 components/schemas：
    ContentBlock / Message / Pagination / MessagePage

⚠️ 类名是 AgentMessage，规格里的组件叫 Message
------------------------------------------
类名与表名一起改成了 agent_message / AgentMessage（见
models/sqlalchemy/agent_message.py），但**对外契约一个字没动**：生成的文档
里组件仍然叫 `Message`，`MessagePage.messages` 的 $ref 也仍然指向它 —— 与
messages.yaml 继续逐字段对得上。

两件事配合才够，缺一不可：

    title="Message"   组件**体内**那个 title 字段与契约名一致（下面那行）
    改名表            app/common/openapi.py 在文档生成之后，把组件键与
                      全部 $ref 从 AgentMessage 换成 Message

为什么不能在 DTO 上一步声明完：FastAPI 拿类的 `__name__` 当组件名，
Pydantic 的 title 管不着 $ref 的名字 —— 试过，生成的仍是 AgentMessage
（实测 fastapi 0.141.1 / pydantic 2.13.5，见 app/common/openapi.py）。

⚠️ 注意这个契约在**另一个文件**里（messages.yaml），不是 session.yaml
------------------------------------------------------------------
同一个模块、同一个 server 前缀（/api/v1），但规格分了两份文件。实现上不
分家：DTO 都在本包，路由都在 app/api/v1/sessions.py。

⚠️ 分页是**游标**，仓库其余列表接口都是 page/page_size
--------------------------------------------------
这是刻意的，理由与代价：

    好处    聊天历史是"不断朝尾部追加"的。页码分页下，翻到第 2 页的过程中
            又来了一条新消息，第 2 页的内容就整体位移了 —— 客户端会重复
            渲染或直接漏掉一条。keyset（按 seq）没有这个问题：游标指向一条
            具体的消息，它后面的内容不会因为前面加了东西而变。
    代价    给不出 total。keyset 算总数要额外一次 COUNT，而它对"还能翻多久"
            这个问题的答案本来就不稳定。因此契约里没有 total，用的是
            has_more_before / has_more_after。

⚠️ 游标为什么不透明
----------------
契约只说 "Opaque cursor"，没说它是什么。这里编码成
`base64url("v1:<seq>")`，于是：

    - 客户端不该解析它（没有承诺）；
    - 实现可以换编码而不破坏契约 —— 把版本号前缀换成 v2 即可。

游标**不加密**，它只是一个序号 —— 消息的内容与归属都不在其中。谁能调这个
接口谁就能给出任意 seq，但那只是分页位置，取不到没权限的会话（会话 id
本身就是入参）。
"""

from __future__ import annotations

import base64
import binascii
from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.common.schemas import APIModel

__all__ = [
    "AgentMessage",
    "ContentBlock",
    "MessagePage",
    "Pagination",
    "decode_cursor",
    "encode_cursor",
]


#: 游标前缀。换编码时改它（老游标会因此解析失败 -> 400，客户端重新取一页即可）。
_CURSOR_PREFIX = "v1"


def encode_cursor(seq: int) -> str:
    """把 seq 编成不透明游标。见模块 docstring。"""

    raw = f"{_CURSOR_PREFIX}:{seq}".encode("utf-8")

    # urlsafe + 去 padding：游标要能直接放进 query string，
    # '=' 在 URL 里虽然合法但会被某些客户端转义成 %3D，读起来很脏。
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def decode_cursor(token: str) -> int:
    """
    游标 -> seq。

    Raises:
        ValueError: 不是本实现产出的游标（-> 400）。

    ⚠️ 坏游标必须报错，不能"当成没给"
    ------------------------------
    静默退回第一页会让客户端陷入死循环：它拿着一个坏游标反复请求，每次
    都拿到最新的那一页，且永远不知道自己给错了东西。
    """

    # 补回 padding：去 padding 是为了 URL 干净，解码时要还原。
    padded = token + "=" * (-len(token) % 4)

    try:
        decoded = base64.urlsafe_b64decode(padded).decode("utf-8")

    except (binascii.Error, UnicodeDecodeError, ValueError) as exc:
        raise ValueError(f"游标无法解析：{token!r}") from exc

    prefix, separator, value = decoded.partition(":")

    if separator != ":" or prefix != _CURSOR_PREFIX:
        raise ValueError(f"游标格式不认识：{token!r}")

    try:
        seq = int(value)

    except ValueError as exc:
        raise ValueError(f"游标里的序号不是整数：{token!r}") from exc

    # seq 从 1 起（表上有 seq > 0 的 CHECK）。0 或负数只能是编造的。
    if seq < 1:
        raise ValueError(f"游标里的序号越界：{token!r}")

    return seq


class ContentBlock(APIModel):
    """
    消息内容里的一块。

    ⚠️ 这一个 DTO **允许额外键**（与其余 DTO 相反）
    ------------------------------------------
    规格写的是 `additionalProperties: true`，且是必须的：内容块的种类是开放的
    （今天有 text 与 tool_call，将来可能有图片、引用、产物）。写死成几个字段
    会让新块类型要么进不来、要么被 `extra="forbid"` 当场拒掉。

    APIModel 默认 forbid，因此这里显式放开 —— 一个刻意的例外，不是漏配。
    """

    model_config = ConfigDict(
        extra="allow",
        from_attributes=True,
    )

    type: str = Field(
        description="内容块的种类，如 text / tool_call。",
    )

    text: str | None = Field(
        default=None,
        description="文本块的内容。非文本块没有它。",
    )

    artifact_id: str | None = Field(
        default=None,
        description=(
            "指向产物的块（附件 / 工具产物）。当前**没有产出方** —— "
            "字段先按规格留着。"
        ),
    )

    source_id: str | None = Field(
        default=None,
        description=(
            "指向来源的块（引用了哪条检索结果 / 哪份资料）。"
            "当前**没有产出方**。"
        ),
    )


class AgentMessage(APIModel):
    """
    一条完整的会话消息。

    ⚠️ 没有 session_id
    -----------------
    它是查询的输入（URL 里那个），每条都一样，回显只是噪声。

    ⚠️ updated_at 目前**恒等于** created_at
    ------------------------------------
    消息写完就不再变（见 domain/entities/agent_message.py），两个写入点填的
    都是同一个时刻，所以这两个字段现在永远相等。它进契约是为了给前端一个
    统一的"最后有效时间"；真正分叉要等出现编辑消息的能力，而契约里还没有
    那条端点。

    ⚠️ `metadata` 而不是 `meta`
    -------------------------
    契约边界是**唯一**改名的地方：实体与库列都叫 meta（SQLAlchemy 保留了
    `metadata` 这个属性名），契约按规格叫 metadata。转换在
    application/_assembly.py 里做一次。

    ⚠️ title 是**接口的一部分**，不是装饰
    --------------------------------
    类名改了（Message -> AgentMessage），但契约里的组件名必须留在 Message
    —— messages.yaml 声明的是 `Message`，`MessagePage.messages` 的 $ref 也
    按那个名字解析。

    这里钉住的是组件**体内**的 title；组件**键**与 $ref 由
    app/common/openapi.py 的改名表在生成之后换掉 —— 两处缺一不可，
    理由见模块 docstring。删掉这一行不会改变接口（改名表仍在），但会让
    schema 里出现 `"title": "AgentMessage"`，与它自己的键对不上。
    """

    model_config = ConfigDict(
        title="Message",
    )

    message_id: UUID = Field(description="消息 ID。")

    seq: int = Field(
        description=(
            "会话内单调递增的序号。**允许有空洞**（一次写入失败会留洞），"
            "因此不要拿它当「第几条消息」用。"
        ),
    )

    run_id: UUID | None = Field(
        default=None,
        description="产出这条消息的运行。用户消息也有值。",
    )

    role: str = Field(description="user / assistant。")

    kind: str = Field(
        description=(
            "chat / interaction / tool_activity / notice / "
            "task_instruction / task_result。\n\n"
            "任务消息（task_instruction / task_result）的 metadata 里带 "
            "`task_id` / `plan_id` / `skill_key`，前端按它归组到任务卡片。"
        ),
    )

    business_agent_id: str | None = Field(
        default=None,
        description=(
            "产出这条消息的职责（本轮运行用的职责提示词 key，如 "
            "`stock_research`）。同一轮的 user / assistant 消息上是同一个值；"
            "null 表示那一轮没有职责信息（存量数据）。"
        ),
    )

    content: list[ContentBlock] = Field(
        description="内容块数组。纯文本也包成 [{type: text, ...}]。",
    )

    reasoning_content: str | None = Field(
        default=None,
        description=(
            "模型的思维链（CoT）原文。\n\n"
            "**取值规则**（存量数据的形状与这里不同，必须按这条读）："
            "`reasoning_content ?? content 里 type=reasoning 的块`。\n\n"
            "新写入的消息推理只在这一列（content 只放可见内容块，两个容器"
            "不重叠）。存量行没有这一列 —— 实测库里 169 条 assistant 消息的"
            "推理躺在 content 的块数组里，刻意不回填（见 messages.yaml 与库表"
            "的说明）。那个 `??` 因此是长期的读法，不是临时兼容。\n\n"
            "null 表示这条消息没有思维链：非推理模型、用户消息（那边是"
            "人说的话，没有草稿纸）、取消公告。"
        ),
    )

    status: str = Field(
        description=(
            "completed / interrupted / failed / cancelled。"
            "只表示最终结果，没有 streaming 这一档 —— 一条消息要么完整地"
            "出现，要么不出现。"
        ),
    )

    reply_to_message_id: UUID | None = Field(
        default=None,
        description="被回复的消息。assistant 指向本轮 user 消息。",
    )

    created_at: datetime = Field(description="创建时间。")

    updated_at: datetime = Field(
        description=(
            "最后更新时间。**当前恒等于 created_at** —— 消息不可变，"
            "写下去之后没有代码改它（见类 docstring）。"
        ),
    )

    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="扩展元数据。库里那一列叫 meta，见类 docstring。",
    )


class Pagination(APIModel):
    """
    一次翻页的位置信息。

    ⚠️ 没有 total，见模块 docstring
    -----------------------------
    两个 has_more_* 才是"还能不能往下翻"的答案。

    ⚠️ 两个游标什么时候有值
    ---------------------
    页非空时**两个都给**：before_cursor 指向最旧那条、after_cursor 指向最新
    那条 —— 客户端因此可以拿着任意一个往两个方向翻，不必先记住自己是从哪个
    方向来的。页为空时两个都是 null（连同两个 false），客户端据此收手，
    而不是拿着 null 再去请求一次。
    """

    limit: int = Field(description="本次请求的每页条数。")

    has_more_before: bool = Field(
        description="还存在更旧的消息（拿 before_cursor 往旧翻）。",
    )

    before_cursor: str | None = Field(
        default=None,
        description="指向本页最旧那条。页为空时是 null。",
    )

    has_more_after: bool = Field(
        description="还存在更新的消息（拿 after_cursor 往新翻）。",
    )

    after_cursor: str | None = Field(
        default=None,
        description="指向本页最新那条。页为空时是 null。",
    )


class MessagePage(APIModel):
    """
    一页会话历史。

    ⚠️ messages 按 seq **升序**（从旧到新）
    ------------------------------------
    与"默认返回最新一页"不矛盾：取的是最新的那一段，但段内按时间正序排 ——
    这样客户端可以直接顺序渲染，不必自己反转。
    """

    session_id: UUID = Field(description="会话 ID。")

    messages: list[AgentMessage] = Field(
        description="这一页的消息，按 seq 升序（旧 -> 新）。",
    )

    pagination: Pagination = Field(description="翻页位置。")
