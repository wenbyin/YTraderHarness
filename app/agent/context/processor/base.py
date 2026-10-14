"""
Context Processor 的契约（规格 §六）。

与 `BaseToolProcessor`（tool 侧）刻意保持同一套形状
--------------------------------------------------
    类级自描述（spec / display_name / description）
    `__init_subclass__` 做**类定义期**校验
    零参可构造（注册表按类反射实例化）
    实例是**共享的**，因此 self 上不许有"这次调用的东西"

最后一条是并发正确性的要求，不是风格：注册表每个 key 只 new 一次
（见 registry.py），并发的每一次上下文构建都会用到同一个实例。

⚠️ 为什么这里的 `process` 是 `@abstractmethod`
------------------------------------------
tool 侧的 `before/after/around` 不做抽象，是因为每个实现只服务
`supported_hooks` 里的那几个，抽象会逼出空壳。Processor 不一样：
**每一个**都必须实现 `process`（它就是处理器本身），因此抽象是准确的
描述。附带一个好处：`inspect.isabstract()` 能把 `ContextProcessor`
基类自己排除掉，反射发现不必再靠结构规则。

⚠️ Processor 拿不到完整 State（规格 §六 / §十七）
---------------------------------------------
`process(inputs, context)` 里的 `inputs` 是 ContextBuilder 按 spec.inputs
解析好的**扁平 dict**，`context` 是本次构建的工作集。Processor 想做任何
与 State 有关的事，都必须先在 spec 里声明那条路径 —— 这是"自描述依赖"
变成机器保证的地方。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .context import ContextBuildContext
from .spec import ContextProcessorSpec

__all__ = ["ContextProcessor"]


class ContextProcessor(ABC):
    """
    所有 Context Processor 的公共契约。

    子类要做三件事（且只有这三件）：

        1. 声明 `spec`（name / order / stage / when / inputs）
        2. 实现 `process(inputs, context)`
        3. 需要对外说明时填 `display_name` / `description`
    """

    #: 自描述（见 spec.py）。**必须**是 ContextProcessorSpec 实例。
    spec: ClassVar[ContextProcessorSpec]

    #: 给人看的名字（目录接口用）。
    display_name: ClassVar[str] = ""

    #: 一句话说明（目录接口用）。
    description: ClassVar[str] = ""

    # ========================================================
    # 类定义期校验
    # ========================================================

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        类体执行完的那一刻校验声明。

        与 `BaseToolProcessor.__init_subclass__` 同一条理由：这是**静态**
        约束，违反它的类从定义那一刻就是错的。等运行期发现，报错只会
        指向某一次请求，而根因在几天前写下的那个类。
        """

        super().__init_subclass__(**kwargs)

        spec = getattr(cls, "spec", None)

        if spec is None:

            # 抽象中间基类不必声明 spec；只有具体实现必须。
            if not getattr(cls, "__abstractmethods__", frozenset()):

                raise TypeError(
                    f"{cls.__name__} 没有声明 spec。"
                    "每个具体 Processor 都必须自描述：name（我是谁）、"
                    "order（什么顺序）、stage（服务哪个固定阶段）、"
                    "when（从哪个 State 路径读是否执行）、inputs（需要"
                    "哪些 State 输入）。\n"
                    "见 app/agent/context/processor/spec.py。"
                )

            return

        if not isinstance(spec, ContextProcessorSpec):

            raise TypeError(
                f"{cls.__name__}.spec 必须是 ContextProcessorSpec，"
                f"得到 {type(spec).__name__}。"
            )

    # ========================================================
    # 身份
    # ========================================================

    @classmethod
    def get_spec(cls) -> ContextProcessorSpec:
        """自描述（**类级**）。注册表与目录接口走这里。"""

        spec = getattr(cls, "spec", None)

        if spec is None:
            raise TypeError(
                f"{cls.__name__} 没有 spec（它可能是抽象中间基类）。"
            )

        return spec

    def spec_of(self) -> ContextProcessorSpec:
        """
        这个**实例**的有效自描述。

        与 `get_spec()` 的差别只有一处：允许实例级覆盖
        （`self.spec = ...`）。用途是**参数化**的处理器 —— 同一个类
        用不同 order / inputs 装配两次（比如两个不同来源的检索器），
        以及测试里造变体。

        流水线用的是**这一个**（它面对的是实例）。反射发现与目录接口
        读的仍然是类级声明 —— 没有实例时只能读类。
        """

        spec = self.__dict__.get("spec") or getattr(type(self), "spec", None)

        if spec is None:
            raise TypeError(
                f"{type(self).__name__} 没有 spec（它可能是抽象中间基类）。"
            )

        return spec

    def get_key(self) -> str:
        """注册键（= 有效 spec 的 name）。"""

        return self.spec_of().name

    # ========================================================
    # 执行
    # ========================================================

    @abstractmethod
    async def process(
            self,
            inputs: dict[str, Any],
            context: ContextBuildContext,
    ) -> ContextBuildContext:
        """
        执行这个 Processor。

        Parameters
        ----------
        inputs
            按 `spec.inputs` 解析好的扁平 dict。缺失的**可选**输入以
            None 出现（必需缺失根本到不了这里 —— 流水线会先让整次
            构建失败，见 pipeline.py）。
        context
            本次构建的工作集：注入片段、历史、工具、overlay、告警都在
            这里累积。实现就地改它并返回（返回同一个对象即可；返回一个
            新的也可以，流水线会用返回值继续）。

        ⚠️ 不要在这里访问完整 Graph State
        ---------------------------------
        需要什么就在 spec.inputs 里声明。忍不住想 `context.base_state[...]`
        时，先问"这条路径能不能写进 spec" —— 能就写进去（那样它会被
        记进观测、会被 `when` 那套 MISSING/INVALID 校验保护），不能
        就说明这个 Processor 的职责划错了。

        ⚠️ 不要在这里做 Provider 调用
        ----------------------------
        需要模型就走 `context.dependencies` 里的 ModelRegistry
        （见 processors/summary.py：全仓库唯一一个在上下文构建里调模型的
        地方；它用的实现在 summarization.py。compaction 阶段现在只做
        message 列表的切片，**不调模型**）。
        """
