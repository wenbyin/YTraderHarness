"""
L2（场景）的**工具契约**：暴露给主模型的 `memory.scene.get`。

⚠️ 这一层为什么需要一个工具
-------------------
L2 进上下文的是**索引**（文件名 + 摘要 + 热度 + 时间），不是全文 —— 全文
十几篇、每篇上千字，每轮都塞进 system 是不划算的（原规格里 L2"只给索引"
就是为这个）。而"这件事具体是怎么说的"要靠全文，所以给模型一个按文件名
取全文的工具，让它**自己要**。

⚠️ 为什么没有对应的 HTTP 端点
----------------------
L0 / L1 的两个工具各自有一份 yaml（memory.yaml / memory_atom.yaml）与一个
HTTP 端点，因为那两个服务方法同时是 HTTP 路径的实现。L2 的这一次读**已经
有** HTTP 入口了 —— 管理端的 `POST /memory/query/l2/detail`
（见 memory_api_v2.yaml，同一个服务方法）。所以这里只补它对**模型**那一侧
的入参形状：

    HTTP    `L2DetailRequest{target, filename}`    身份（target）在请求体里
    工具    `MemorySceneGetRequest{filename}`      身份从**调用者上下文**取

差别只在身份从哪来 —— 与 `memory.search` / `memory_atom.search` 那一对是
同一条纪律：工具的入参 schema 是给**模型**看的，`user_id` / `agent_id`
绝不能进去（模型能填一个别人的 id，那就是一条跨租户读）。

返回形状复用 `L2DetailResponse`（`scope_kind` + 场景全文）：它就是"一篇场景"
的契约，工具与管理端看的是同一个东西。**这不算契约外泄** —— 两个入口本来
就共用同一个服务方法。
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["MemorySceneGetRequest"]


class MemorySceneGetRequest(BaseModel):
    """
    取一篇场景的全文。

    ⚠️ 只有 `filename` —— 没有身份，也没有 scope
    ----------------------------------------
    模型手上有的就是上下文里那份场景索引（每行以文件名开头），而 scope
    （agent + user|org）由**运行时**决定：它必须与"注入那份索引时用的
    scope"完全一致，否则模型会照着清单点名、却被告知"场景不存在"。
    身份走 `app/common/tool_caller.py`，形状与 `memory_atom.search` 同。
    """

    model_config = ConfigDict(extra="forbid")

    filename: str = Field(
        min_length=1,
        description=(
            "场景文件名，**逐字**取自上下文里场景索引那一行的开头"
            "（例如 `音乐爱好-吉他Blues练习.md`）。不做归一化、不做模糊匹配。"
        ),
    )
