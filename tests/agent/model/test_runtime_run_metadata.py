"""
调用身份（`agent_type` / `model_role`）必须随**每一次**调用传下去。

为什么它值得一条测试
------------------
它没有任何**功能**上的回报：写漏了不报错、不跑错，只是下游
（`run_event_projector`）再也分不出"这次调用是上下文压缩还是主循环"——于是
压缩结果会以正文的形式流进聊天窗。那种 bug 只有用户看得见（实测发生过，
见 tests/agent/event/test_run_event_projector.py 的说明）。

这一条管的是链路上**运行时那一跳**：role 由 Runtime 给适配器。适配器把它
挂进 langchain 的运行元信息那一跳由上面那个投影层测试覆盖（那里跑真的图）。
"""

from __future__ import annotations

from fakes import FakeAdapter, make_definition, make_registry

from app.agent.model import (
    ModelProfile,
    ModelRequest,
    ModelRole,
    ModelRoleBinding,
)


def _runtime(adapter: FakeAdapter):
    """单候选、单一绑定的 runtime（角色就是这次要断言的那个）。"""

    registry = make_registry(
        models=[make_definition("primary")],
        profiles=[ModelProfile(id="p", primary="fake:primary")],
        bindings=[
            ModelRoleBinding(
                agent_type="react",
                model_role=ModelRole.SUMMARIZATION,
                profile="p",
            )
        ],
        adapter=adapter,
    )

    return registry.get_runtime(
        agent_type="react", model_role=ModelRole.SUMMARIZATION
    )


async def test_runtime_passes_its_own_identity_not_the_callers() -> None:
    """
    Runtime 按**自己**的身份（构造时定的 agent_type / model_role）注入，
    而不是按请求里的什么东西 —— 请求里根本没有身份可读。
    """

    adapter = FakeAdapter()
    runtime = _runtime(adapter)

    await runtime.ainvoke(ModelRequest(messages=("你好",)))

    assert adapter.run_metadata == [
        {"agent_type": "react", "model_role": ModelRole.SUMMARIZATION}
    ]


async def test_identity_is_sent_on_every_attempt() -> None:
    """
    重试与降级**不改变身份**：三次 attempt 都是同一次调用，消费端看到的
    应该是"这一次调用是谁"，而不是"第几次尝试"。漏在中途（比如只在第一次
    带上）会让最后一次真的产生内容的调用失去角色，于是它又被当成主循环。
    """

    from fakes import Outcome

    adapter = FakeAdapter(
        script={
            "fake:primary": [
                Outcome(error=TimeoutError("第一次超时")),
                Outcome(message="好了", deltas=("好", "了")),
            ]
        }
    )

    runtime = _runtime(adapter)

    await runtime.ainvoke(ModelRequest(messages=("你好",)))

    assert len(adapter.run_metadata) == 2
    assert all(
        item == {"agent_type": "react", "model_role": ModelRole.SUMMARIZATION}
        for item in adapter.run_metadata
    )
