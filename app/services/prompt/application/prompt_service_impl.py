"""
`PromptService` 的实现：读 `catalog.py` 那张硬编码表。

为什么这个实现**没有仓储参数**
---------------------------
它现在没有表可读。而构造参数里出现仓储契约注解会让
`ServiceFactory.needs_session()` 判成 True —— 那时"取一段硬编码文本"就
要求一个数据库会话（FastAPI 依赖会开事务，脚本里则直接不可用）。

提示词进库时给本类加一个仓储参数即可：needs_session 会跟着变成 True，而
`PromptProvider` 那边把 `factory.create(...)` 换成 `factory.scope(...)`
（一行，注释里已标）。**契约与调用方都不动。**

文件名的 `_impl` 后缀不是风格：`resolve_impl` 靠它找实现
（app/common/introspect.py），改掉它服务就解析不出来了。
"""

from __future__ import annotations

from app.common.exceptions import NotFoundError

from ..catalog import DEFAULT_DUTY, DUTY_PROMPTS
from .prompt_service import PromptService

__all__ = ["PromptServiceImpl"]


class PromptServiceImpl(PromptService):

    async def get_duty_prompt(self, duty: str | None = None) -> str:

        key = duty or DEFAULT_DUTY

        try:
            return DUTY_PROMPTS[key]

        except KeyError as exc:
            # 抛 NotFoundError 而不是回落到默认职责：拼错的 key 若被静默
            # 兜底成默认提示词，"我配的职责没生效"就变成一件看不出来的事
            # —— 而提示词错了只会表现为"模型答得不太对"。
            raise NotFoundError(
                f"没有登记名为 {key!r} 的职责提示词。"
                f"已登记：{sorted(DUTY_PROMPTS)}。"
            ) from exc

    async def default_duty_key(self) -> str:
        """
        今天恒返回 `DEFAULT_DUTY` —— 还没有任何依据可选（见契约的说明）。

        仍然做一次"这个 key 确实在表里"的校验：`DEFAULT_DUTY` 与
        `DUTY_PROMPTS` 是同一个文件里的两个声明，删掉表里那一行而忘了改
        常量，症状会是"建运行成功、跑到主循环时提示词取不到"—— 那离原因
        很远。在这里炸，报错直接指着这两个名字。
        """

        if DEFAULT_DUTY not in DUTY_PROMPTS:
            raise NotFoundError(
                f"DEFAULT_DUTY={DEFAULT_DUTY!r} 不在 DUTY_PROMPTS 里"
                f"（已登记：{sorted(DUTY_PROMPTS)}）。"
                "两者都在 catalog.py，改的时候要一起改。"
            )

        return DEFAULT_DUTY

    async def duty_keys(self) -> tuple[str, ...]:

        return tuple(sorted(DUTY_PROMPTS))
