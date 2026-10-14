"""
鉴权通过之后的"调用者"。

为什么定义在 app/common/auth 而不是 app/services/user
--------------------------------------------------
方向是**服务依赖公共**，不是反过来：`app/services/user` 的 DTO 从
这个模型构造，而 `app/common` 一旦反过来 import `app.services.*`，
就破坏了它作为纯叶子的前提（见 app/common/exceptions.py 开头那条）。
所以这个模型住在这里，`app/common/auth/__init__.py` **不要**为了
方便去 re-export 服务层的 DTO —— 那会把刚绕开的反向依赖重新接上。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

__all__ = ["AuthenticatedUser"]


class AuthenticatedUser(BaseModel):
    """
    一个通过了签名校验与群组校验的调用者。

    字段全部来自 JWT 的 claim，**没有一行查库** —— Casdoor 是用户的
    唯一真相，本地不留副本（见计划里"不落库"的取舍：省掉的是一套
    同步逻辑，以及"两边对不上时该信谁"这个问题）。
    """

    model_config = ConfigDict(from_attributes=True)

    subject: str = Field(
        description="Casdoor 用户 ID（JWT 的 `sub`）。",
    )

    name: str = Field(
        default="",
        description="登录名（JWT 的 `name`）。",
    )

    display_name: str = Field(
        default="",
        description="展示名。Casdoor 用 `displayName`，可能为空。",
    )

    email: str = Field(
        default="",
        description="邮箱，可能为空（不是所有组织都要邮箱）。",
    )

    groups: list[str] = Field(
        default_factory=list,
        description=(
            "**归一化之后**的群组名 —— 每个元素都是可以直接与配置里 "
            "`required_groups` 逐一比对的形式。\n\n"
            "归一化（拆数组 / 拆对象）发生在"
            " app/common/auth/component.py，因为那是 Casdoor claim "
            "形状的知识；这里只保管结果，不猜形状。\n\n"
            "⚠️ 名字**可能带 owner 前缀**（`y-trader/Y-Trader-User`）——"
            "归一化只处理值的形状，不去前缀。所以比对时不能只做相等，"
            "见 component._has_required_group 的两段匹配。"
        ),
    )

    roles: list[str] = Field(
        default_factory=list,
        description=(
            "**归一化之后**的角色名，与 `groups` 同一套形状容错。\n\n"
            "与 groups 的分工：groups 是**登录门槛**（在 auth 层按"
            " `required_groups` 拦一次，拦不住就 403），roles 不进那道"
            "门槛，它被带到这里是为了**工具级**的鉴权 —— 挂在工具处理器"
            "链上的 checkauth 拦截器按\"这个工具要求哪些角色\"去判。"
            "两者都在 token 里、都用同一套归一化，但用途不同。"
        ),
    )

    org_id: str = Field(
        default="",
        description=(
            "调用者所属的组织（JWT 的 `org` claim，claim 名可配）。\n\n"
            "用途只有一个：写进 `agent_session.org_id`，作为**团队 / 组织"
            "归属的记录**。它不参与鉴权门槛（那是 groups 的事），"
            "也不参与可见性过滤（那是 user_id 的事）。\n\n"
            "⚠️ 空串是**正常值**，表示这个 token 里没有组织信息 —— "
            "Casdoor 各版本与各应用上 \"Token fields\" 的配置不一样，"
            "取不到就空置。空串在落库时归一成 NULL。\n\n"
            "⚠️ 只认字符串：claim 是数组 / 对象时同样落空串。不做 str() "
            "兜底 —— 那种字符串写进库之后没人认得出它是什么。"
        ),
    )

    raw_claims: dict[str, Any] = Field(
        default_factory=dict,
        repr=False,
        description=(
            "原始 claim，给排查问题用（比如\"群组到底叫什么名字\"）。\n\n"
            "⚠️ **不要放进对外响应**：里面是 Casdoor 签发的全部内容，"
            "可能含权限、角色、密钥类字段。`repr=False` 只是防止它"
            "意外出现在日志的 repr 里，不是防线。"
        ),
    )
