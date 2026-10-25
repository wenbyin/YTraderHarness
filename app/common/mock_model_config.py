"""
``app.yaml`` 的 ``mock_model`` 段配置模型。

放在 ``app/common`` 而不是 ``app/mock``：``app/common/app_config.py`` 要
用这些类型校验整份 app.yaml，而 common 层不应该反过来 import 一个功能包。
``app/mock`` 只是这些配置的消费者。

分布参数用 ``mean / std / min / max``：真实模型的 TTFT 与输出长度都不是
固定值，截断正态分布能表达"大多数请求落在区间内、偶尔慢/长"，同时避免
正态尾部产生负延迟或 0 字符回复。
"""

from __future__ import annotations

import random
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

__all__ = [
    "MOCK_EXECUTION_PLANS",
    "MockCapabilityConfig",
    "MockClassificationConfig",
    "MockModelConfig",
    "MockModeConfig",
    "MockToolCallConfig",
    "NormalSpec",
]

#: 分类结果里 execution_plan 的合法值。
#:
#: 与 app/mock/model_server.py 的随机候选集保持一致。刻意不 import
#: app.agent.state：mock 只依赖配置与 aiohttp。
#: ``plan`` 不在其中 —— planner 节点尚未接通。
MOCK_EXECUTION_PLANS: tuple[str, ...] = (
    "react",
    "other",
)


class NormalSpec(BaseModel):
    """一个截断正态分布：``N(mean, std²)`` 裁到 ``[min, max]``。"""

    model_config = ConfigDict(extra="forbid")

    mean: float = Field(description="均值。")
    std: float = Field(default=0.0, ge=0.0, description="标准差；0 = 固定值。")
    min: float = Field(default=0.0, description="下界（采样结果不会低于它）。")
    max: float = Field(default=1e9, description="上界（采样结果不会高于它）。")

    @model_validator(mode="after")
    def _check_bounds(self) -> "NormalSpec":

        if self.min > self.max:
            raise ValueError(
                f"正态分布下界 {self.min} 大于上界 {self.max}。"
            )

        return self

    def sample(self, rng: random.Random) -> float:
        """采样一个值；``std=0`` 时直接返回均值。"""

        value = (
            self.mean
            if self.std <= 0
            else rng.gauss(self.mean, self.std)
        )

        return min(max(value, self.min), self.max)

    def sample_int(self, rng: random.Random) -> int:
        """采样后四舍五入为整数，并按整数边界再夹一次。"""

        value = round(self.sample(rng))
        low = int(round(self.min))
        high = int(round(self.max))

        return min(max(value, low), high)


class MockModeConfig(BaseModel):
    """一类回复（普通聊天 / 分类）的时序与长度分布。"""

    model_config = ConfigDict(extra="forbid")

    ttft_ms: NormalSpec = Field(
        description="首 token 延迟（毫秒），从请求开始到第一个 SSE 数据块。",
    )

    response_chars: NormalSpec = Field(
        description="整段回复的字符数。分类调用按序列化后的参数长度近似。",
    )

    chars_per_second: NormalSpec = Field(
        description="流式输出速率（字符/秒）；非流式也用它折算总耗时。",
    )


class MockToolCallConfig(BaseModel):
    """工具调用的随机策略。"""

    model_config = ConfigDict(extra="forbid")

    probability: float = Field(
        default=0.05,
        ge=0.0,
        le=1.0,
        description="带 tools 的自动调用中，返回 tool_calls 的概率。",
    )

    count: NormalSpec = Field(
        default_factory=lambda: NormalSpec(
            mean=1.0, std=0.4, min=0.0, max=2.0
        ),
        description=(
            "命中工具调用时每次返回的 tool_call 数量。数量用正态分布并裁到 "
            "[min, max]；具体选哪个工具是分类选择，无法服从正态分布，"
            "按等概率抽取。"
        ),
    )


class MockClassificationConfig(MockModeConfig):
    """分类调用的分布；与普通回复分开，避免分类也等 8 秒。"""


class MockCapabilityConfig(BaseModel):
    """
    mock 能力 -> 工具的解析策略。

    Phase 0 不依赖数据库里的 capability/tool 关联：分类器返回什么 code
    都不影响结果，统一从固定工具池里随机选 N 个。这样前端联调时一定
    有工具可暴露，不会出现"分类说要用 retrieval，但库里没挂工具，于是
    主模型一个工具都看不到"的静默空转。
    """

    model_config = ConfigDict(extra="forbid")

    tool_keys: list[str] = Field(
        default_factory=lambda: [
            "mock.echo",
            "mock.lookup",
            "mock.calc",
            "mock.current_time",
            "mock.ping",
        ],
        description="固定工具池；无论能力 code 是什么都从这里选。",
    )

    count: NormalSpec = Field(
        default_factory=lambda: NormalSpec(
            mean=2.0, std=1.0, min=1.0, max=5.0
        ),
        description=(
            "每次暴露多少个工具。用截断正态分布取 1~5；"
            "最终不会超过工具池大小。"
        ),
    )


class MockModelConfig(BaseModel):
    """``app.yaml`` 的 ``mock_model`` 段。"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description=(
            "是否启用进程内 mock 服务。启动参数 --mock / --no-mock "
            "优先于这里。"
        ),
    )

    host: str = Field(
        default="127.0.0.1",
        min_length=1,
        description="监听地址。只建议 127.0.0.1；0.0.0.0 仅供容器/局域网联调。",
    )

    port: int = Field(
        default=18080,
        ge=1,
        le=65535,
        description="监听端口；providers.yaml 的 llm.mock.base_url 默认指向 18080。",
    )

    seed: int | None = Field(
        default=None,
        description=(
            "随机种子。None = 每次启动不同；固定值用于复现同一条回复。"
        ),
    )

    execution_plan: str | None = Field(
        default=None,
        description=(
            "强制分类结果的 execution_plan。None / random = 从 schema "
            "枚举里随机抽；stress_eval 固定为 react，避免压测流量"
            "分叉到 other。"
        ),
    )

    chunk_chars: int = Field(
        default=24,
        ge=1,
        le=4096,
        description="流式回复每块字符数。越大事件越少、越不像逐 token。",
    )

    log_requests: bool = Field(
        default=True,
        description="每个请求打一条 INFO 日志（只含角色/长度/耗时，不含 prompt）。",
    )

    stream: MockModeConfig = Field(
        default_factory=lambda: MockModeConfig(
            ttft_ms=NormalSpec(mean=700, std=250, min=100, max=3000),
            response_chars=NormalSpec(mean=500, std=160, min=40, max=2000),
            chars_per_second=NormalSpec(
                mean=60, std=12, min=15, max=150
            ),
        ),
        description="普通对话 / 推理调用的分布。",
    )

    classification: MockClassificationConfig = Field(
        default_factory=lambda: MockClassificationConfig(
            ttft_ms=NormalSpec(mean=250, std=90, min=50, max=1200),
            response_chars=NormalSpec(mean=160, std=30, min=40, max=400),
            chars_per_second=NormalSpec(
                mean=200, std=40, min=50, max=500
            ),
        ),
        description="分类调用的分布。",
    )

    tool_call: MockToolCallConfig = Field(
        default_factory=MockToolCallConfig,
        description="工具调用的随机策略。",
    )

    capability: MockCapabilityConfig = Field(
        default_factory=MockCapabilityConfig,
        description="能力 code -> mock 工具的解析策略。",
    )

    capability_pool: list[str] = Field(
        default_factory=list,
        description=(
            "分类结果 capability_needs 的候选池。为空时从 system prompt "
            "里的候选清单解析；两者都没有就返回空列表。"
        ),
    )

    @field_validator("enabled", mode="before")
    @classmethod
    def _parse_enabled(cls, value: Any) -> Any:
        """允许 app.yaml 里用 ``${MOCK_MODEL_ENABLED:-false}`` 这种字符串。"""

        if isinstance(value, str):
            normalized = value.strip().lower()

            if not normalized:
                return False

            if normalized in {"1", "true", "yes", "on", "y"}:
                return True

            if normalized in {"0", "false", "no", "off", "n"}:
                return False

            raise ValueError(
                f"mock_model.enabled 不是合法布尔值：{value!r}"
            )

        return value

    @field_validator("seed", mode="before")
    @classmethod
    def _parse_seed(cls, value: Any) -> Any:
        """``${MOCK_MODEL_SEED:-}`` 展开成空串时按未设置处理。"""

        if isinstance(value, str) and not value.strip():
            return None

        return value

    @field_validator("execution_plan", mode="before")
    @classmethod
    def _parse_execution_plan(cls, value: Any) -> Any:
        """
        ``random`` / 空串 = 恢复随机；其余必须是合法执行方案。

        stress_eval 只需要 react；保留 other 是为了以后能单独复现
        那条分支，但默认配置把压测钉在 react 上。
        """

        if value is None:
            return None

        normalized = str(value).strip().lower()

        if normalized in {"", "random", "none"}:
            return None

        if normalized not in MOCK_EXECUTION_PLANS:
            raise ValueError(
                "mock_model.execution_plan 必须是 "
                f"{', '.join(MOCK_EXECUTION_PLANS)} 或 random，"
                f"收到 {value!r}。"
            )

        return normalized
