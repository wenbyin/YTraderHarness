"""
凭据引用（credential reference）的解析。

规格里的一句话定了这个模块的全部职责：

    "Registry 可以保存模型使用的 credential reference，但不要将 API Key
     散落在 Agent / Node 中。实际 secret 的读取应该由 Model 层或
     Secret Provider 完成。"

因此：

    ModelDefinition.credential_ref   一个**名字**（`deepseek-production`）
    CredentialResolver.resolve(ref)  这个名字 -> 秘密本身
    谁都不写明文密钥

与既有 llm.yaml 的关系（**不重复造配置系统**）
--------------------------------------------
本项目的秘密一直只有一条来源：`app/common/llm.yaml` 里的
`api_key: "${DEEPSEEK_API_KEY:-}"`，由 `app/common/llm.py` 在构造
客户端时展开。本模块**不另建一份秘密配置**，只加一个可选的覆盖口子：

    credential_ref = None（默认）  -> 什么都不做，用 llm.yaml 里那份
    credential_ref = "xxx"        -> 解析出秘密，**覆盖** provider 的默认值

覆盖的用途是多账号 / 多租户（同一个 provider 有两把 key）与"某个模型
用专用 key"。默认路径完全不碰它，因此不引入任何新的配置面。

⚠️ 解析不到不算错（默认）
------------------------
`EnvVarCredentialResolver` 找不到环境变量时返回 None，而不是抛错：
那表示"没说要用哪份专用凭据"，于是回落到 llm.yaml 的默认凭据 ——
这与本模块之前的行为一致。真要"声明的凭据必须存在"，构造时传
`required=True`，缺了会抛 `ModelConfigurationError`（fail fast）。

日志安全
--------
`resolve()` 的任何日志都**只打 ref 名**，不打值；值的脱敏由
redaction.py 负责（异常消息、metadata、观测行）。
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from .errors import ModelConfigurationError

__all__ = [
    "CredentialResolver",
    "EnvVarCredentialResolver",
    "LiteralCredentialResolver",
    "NullCredentialResolver",
]

logger = logging.getLogger(__name__)


@runtime_checkable
class CredentialResolver(Protocol):
    """
    `credential_ref -> 秘密`。

    返回 None 的语义是"**没有覆盖**，让 provider 用它自己配置里那份"，
    不是"取不到所以失败"。要失败请抛 `ModelConfigurationError`。
    """

    def resolve(self, ref: str) -> str | None:
        ...


class NullCredentialResolver:
    """
    永远不覆盖 —— 全部凭据走 llm.yaml / provider 自己的配置。

    这是 Registry 的默认。没有多账号需求时它就该是唯一在用的实现：
    多一层"名字到秘密"的间接，只在该有间接的地方有。
    """

    def resolve(self, ref: str) -> str | None:

        return None


class EnvVarCredentialResolver:
    """
    从**环境变量**取秘密。

    ref -> 环境变量名有两条路：

        refs 映射里登记过    用登记的那个名字
        没登记过             ref 归一化后当环境变量名
                             （`deepseek-production` -> `DEEPSEEK_PRODUCTION`）

    第二条是"约定优于配置"：临时试一把时不必先改映射表。第一条留给
    "名字与环境变量名对不上"的正式部署。

    Parameters
    ----------
    refs
        `credential_ref -> 环境变量名`。**里面不能有秘密**，只有名字
        —— 它就是 model.yaml 里 `credentials:` 那一段。
    required
        True 时解析不到就抛 `ModelConfigurationError`。默认 False
        （回落 provider 默认凭据，见模块 docstring）。
    """

    def __init__(
            self,
            refs: Mapping[str, str] | None = None,
            *,
            required: bool = False,
    ) -> None:

        self._refs = dict(refs or {})
        self._required = required

    # --------------------------------------------------------

    def resolve(self, ref: str) -> str | None:

        env_name = self._refs.get(ref) or _to_env_name(ref)

        secret = os.getenv(env_name) or None

        if secret is not None:
            return secret

        if self._required:
            raise ModelConfigurationError(
                f"credential_ref {ref!r} 对应的环境变量 {env_name!r} "
                "没有设置。\n"
                "本解析器被要求「声明的凭据必须存在」（required=True）——"
                "要么把该环境变量配上，要么去掉这个模型的 credential_ref"
                "（那样会回落到 app/common/llm.yaml 里的 provider 凭据）。"
            )

        # 只打名字，**绝不打值** —— 这里本来也没有值可打。
        logger.debug(
            "credential_ref %r 未在环境变量 %r 中解析到；"
            "回落到 provider 自己配置的凭据。",
            ref,
            env_name,
        )

        return None


class LiteralCredentialResolver:
    """
    直接给出秘密值 —— **只给测试 / 已由外部解密后注入的场景**。

    名字里带 Literal 是提醒：用它的地方，秘密**已经在进程内存里**了。
    生产路径应当走 EnvVarCredentialResolver（或接一个真正的 Secret
    Provider），而不是把秘密写进代码或配置文件。

    ⚠️ `__repr__` 被覆盖成只显示 ref 名 —— 否则"打一条日志"或
    "pytest 失败时打印 fixture"就会把秘密带出去。
    """

    def __init__(self, secrets: Mapping[str, str]) -> None:
        self._secrets = dict(secrets)

    def resolve(self, ref: str) -> str | None:
        return self._secrets.get(ref)

    def __repr__(self) -> str:
        return (
            f"LiteralCredentialResolver(refs={sorted(self._secrets)!r}, "
            f"values=<{len(self._secrets)} secret(s) hidden>)"
        )


def _to_env_name(ref: str) -> str:
    """把 `deepseek-production` 归一成 `DEEPSEEK_PRODUCTION`。"""

    return ref.strip().upper().replace("-", "_").replace(".", "_")
