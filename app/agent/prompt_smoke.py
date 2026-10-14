"""
提示词模块的冒烟测试。

不连数据库、不调模型。三件事：

    1. 节点提示词的目录与渲染（键、取值域、候选排版、截断、降级边界）
    2. 摘要提示词确实还在（它搬过一次家，别搬丢了）
    3. 职责提示词的两条路：有 ServiceFactory 取到真文本、没工厂时**只**
       降级不炸

跑法：
    python -m app.agent.prompt_smoke

⚠️ 断言口径说明
-------------
这份脚本**不**断言提示词的完整正文（那是给写提示词的人改的东西，锁死它
只会让每次润色都要改测试）。锁的是三类东西：**结构**（谁能取到、取不到时
怎么样）、**唯一出口**（capability_needs 不提 modules 这类约定）、
**取值域来自枚举**（派生，不是手抄）。正文的措辞由 review 管。
"""

from __future__ import annotations

import asyncio
import sys
from typing import Any

from langchain_core.messages import HumanMessage

from app.agent.context.processor.dependencies import ContextDependencies
from app.agent.model import ModelRole
from app.agent.prompt import (
    DefaultPromptProvider,
    PromptNotFoundError,
    PromptUnavailableError,
    found_prompts,
    get_prompt_provider,
    reset_prompt_provider,
    reset_prompt_registry,
    set_prompt_provider,
)
from app.agent.prompt.prompts.classification import BEHAVIORS, PLANS
from app.agent.state import BehaviorCategory, ExecutionPlan
from app.common.factories import build_factories, reset_factories
from app.services.prompt.catalog import DEFAULT_DUTY, DUTY_PROMPTS


# ============================================================
# 断言工具（与 agent_smoke.py 同一套）
# ============================================================


def check(label: str, condition: Any) -> None:
    print(f"  [{'OK ' if condition else 'FAIL'}] {label}")
    if not condition:
        raise AssertionError(label)


def section(title: str) -> None:
    print(f"\n{title}")


def _raises(exc_type: type[BaseException], fn: Any, *args: Any, **kwargs: Any) -> bool:
    """跑一下，看它抛的是不是那个类型。"""

    try:
        fn(*args, **kwargs)

    except exc_type:
        return True

    except BaseException as other:  # noqa: BLE001 - 只为把"抛错了"说清楚
        print(f"        （期望 {exc_type.__name__}，实际 {type(other).__name__}）")
        return False

    return False


class _Candidate:
    """提示词侧只认 code / name / description 三项（见 CandidateLike）。"""

    def __init__(self, code: str, name: str, description: str | None = None):
        self.code = code
        self.name = name
        self.description = description


# ============================================================
# 用例
# ============================================================


async def main() -> int:

    provider = get_prompt_provider()

    # --------------------------------------------------------
    section("节点提示词目录")

    keys = set(found_prompts())

    check(
        "两个 model_role 都登记了",
        keys == {ModelRole.CLASSIFICATION, ModelRole.SUMMARIZATION},
    )
    check("目录被缓存（进程一份）", found_prompts() is found_prompts())

    check(
        "⚠️ reasoning（主循环）**不在**这张表里",
        ModelRole.REASONING not in keys,
    )
    check(
        "查 reasoning 报的是 PromptNotFoundError，且提示走 duty_prompt",
        _raises(
            PromptNotFoundError, provider.node_prompt, ModelRole.REASONING
        ),
    )

    # --------------------------------------------------------
    section("分类提示词：取值域来自枚举")

    check(
        "BEHAVIORS 由 BehaviorCategory 派生",
        BEHAVIORS == "、".join(m.value for m in BehaviorCategory),
    )
    check(
        "PLANS 由 ExecutionPlan 派生",
        PLANS == "、".join(m.value for m in ExecutionPlan),
    )

    # --------------------------------------------------------
    section("分类提示词：候选清单的排版")

    text = provider.node_prompt(
        ModelRole.CLASSIFICATION,
        candidates=[
            _Candidate("financial", "财务", "季度财报与比较"),
            _Candidate("document", "文档", None),
        ],
    )

    check("标识与名称带上了", "financial：财务" in text)
    check("描述带上了", "季度财报与比较" in text)
    check(
        "描述缺失时不留空破折号",
        "- document：文档" in text and "文档 ——" not in text,
    )
    check(
        "明确要求回填的是标识（code）",
        "capability_needs 只填上面每项**开头的那个标识（code）**" in text,
    )
    check("提示词里不提 modules（只有一个出口）", "modules" not in text)
    check(
        "候选为空时给的是「没有清单」，不是一个空列表",
        "当前没有可用的能力清单"
        in provider.node_prompt(ModelRole.CLASSIFICATION, candidates=[]),
    )

    long_description = "描" * 200
    truncated = provider.node_prompt(
        ModelRole.CLASSIFICATION,
        candidates=[_Candidate("x", "长描述", long_description)],
    )

    check("超长描述被截断并留下 …（看得出被截过）", "…" in truncated)
    check(
        "截断到 80 字（不是别处某个数）",
        long_description[:80] in truncated
        and long_description[:81] not in truncated,
    )

    check(
        "同样的输入渲染两次完全一致（context_hash 靠这个稳定）",
        text
        == provider.node_prompt(
            ModelRole.CLASSIFICATION,
            candidates=[
                _Candidate("financial", "财务", "季度财报与比较"),
                _Candidate("document", "文档", None),
            ],
        ),
    )

    # 变量契约由 render 的签名表达：写错名字/少传都是 TypeError，不静默。
    check(
        "参数名写错 -> TypeError（不静默）",
        _raises(
            TypeError,
            provider.node_prompt,
            ModelRole.CLASSIFICATION,
            capability_candidates=[],
        ),
    )
    check(
        "漏传参数 -> TypeError",
        _raises(TypeError, provider.node_prompt, ModelRole.CLASSIFICATION),
    )

    # --------------------------------------------------------
    section("摘要提示词")

    summary = provider.node_prompt(ModelRole.SUMMARIZATION)

    check("非空", bool(summary.strip()))
    check("仍写着「不要推测」", "不要推测" in summary)
    check("仍要求合并上一版摘要", "合并" in summary)
    check(
        "不接参数（它不随运行时数据变化）",
        _raises(
            TypeError,
            provider.node_prompt,
            ModelRole.SUMMARIZATION,
            candidates=[],
        ),
    )

    # --------------------------------------------------------
    section("摘要压缩器：提示词由 Provider 供给（不是它自带的）")

    # ⚠️ 这一节盯的是一个刚发生过的**搬家**：压缩器原来的构造参数默认值就是
    # 那份文本，于是"用哪份提示词"取决于谁构造了它。默认值已删，现在唯一
    # 来源是 Provider —— 这条断言把那条线钉住（不再有第二份正文可漂移）。
    class _FakeRegistry:
        def get_runtime(self, **kwargs: Any) -> Any:
            raise AssertionError("这一步不该真去调模型。")

    # `_FakeRegistry.get_runtime` 一被调用就抛 —— 因此"走到下面那两条断言"
    # 本身就证明了建压缩器与组装请求都不碰模型（真调用发生在 compact() 里）。
    compactor = ContextDependencies(
        model_registry=_FakeRegistry()
    ).resolved_compactor(agent_type="react")

    request = compactor.build_request(
        previous=None,
        messages=[HumanMessage("用户说了一句话")],
        max_output_tokens=256,
    )

    check(
        "压缩器发出的 system 消息就是 Provider 给的那份",
        request.messages[0].content == summary,
    )

    # --------------------------------------------------------
    section("职责提示词：没有 ServiceFactory 时只降级，不炸")

    # 先清干净：本进程里可能已经有别的用例装过工厂（比如 agent_smoke）。
    reset_factories()
    set_prompt_provider(DefaultPromptProvider())

    bare = get_prompt_provider()

    check(
        "取职责提示词抛的是 PromptUnavailableError",
        await _araises(PromptUnavailableError, bare.duty_prompt),
    )
    check(
        "取 duty_key 同样抛 PromptUnavailableError（同一条降级边界）",
        await _araises(PromptUnavailableError, bare.duty_key),
    )
    check(
        "同一时刻节点提示词照常可用（两边独立）",
        bool(bare.node_prompt(ModelRole.SUMMARIZATION).strip()),
    )

    # --------------------------------------------------------
    section("职责提示词：有 ServiceFactory 时取到真文本")

    build_factories(lambda: None)   # 本路径不该碰 session（见 provider 的说明）

    try:
        duty = await get_prompt_provider().duty_prompt()

        check(
            "默认职责 = catalog 里登记的那份",
            duty == DUTY_PROMPTS[DEFAULT_DUTY],
        )
        # 「选职责」的落点：建运行那一步取它落 agent_run.duty_key。
        # 今天恒为默认，因此它必须与上面那份文本**指向同一个 key** ——
        # 两处漂了的话，运行行上写着 A、模型拿到的是 B 的文本。
        check(
            "duty_key() 给出的是同一个 key",
            await get_prompt_provider().duty_key() == DEFAULT_DUTY,
        )
        check(
            "显式指定默认职责，结果一致",
            await get_prompt_provider().duty_prompt(DEFAULT_DUTY) == duty,
        )
        check(
            "没登记的职责 -> PromptNotFoundError（不静默兜底成默认）",
            await _araises(
                PromptNotFoundError,
                get_prompt_provider().duty_prompt,
                "查无此职责",
            ),
        )
        check(
            "职责文本非空",
            bool(duty.strip()),
        )

    finally:
        reset_factories()

    # --------------------------------------------------------
    section("端口可替换（组合根与测试的接缝）")

    class _FakeProvider:
        def node_prompt(self, model_role: str, /, **variables: Any) -> str:
            return "<假的节点提示词>"

        async def duty_prompt(self, duty_key: str | None = None) -> str:
            return "<假的职责提示词>"

    fake = _FakeProvider()

    set_prompt_provider(fake)

    check("装进去的就是取出来的", get_prompt_provider() is fake)
    check(
        "取用走的是假的",
        get_prompt_provider().node_prompt(ModelRole.SUMMARIZATION)
        == "<假的节点提示词>",
    )

    reset_prompt_provider()

    check(
        "reset 之后回落默认实现（不是 None）",
        isinstance(get_prompt_provider(), DefaultPromptProvider),
    )

    # 目录缓存也是进程级事实：清掉之后仍应发现同样两份提示词。
    reset_prompt_registry()

    check(
        "清掉缓存后重新发现，结果一致",
        set(found_prompts()) == keys,
    )

    print("\n全部通过。")
    return 0


async def _araises(exc_type: type[BaseException], fn: Any, *args: Any) -> bool:
    """`_raises` 的异步版（await 型的调用点）。"""

    try:
        await fn(*args)

    except exc_type:
        return True

    except BaseException as other:  # noqa: BLE001 - 只为把"抛错了"说清楚
        print(f"        （期望 {exc_type.__name__}，实际 {type(other).__name__}）")
        return False

    return False


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
