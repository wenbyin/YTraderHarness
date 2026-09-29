"""
工具处理器的契约。

处理器 = 挂在一次工具调用上的一段可配置逻辑
--------------------------------------------
链由 `tool_processor` 表里的行组成（一个 Tool 一条链，按 sequence 排）。
每行指向一个**代码里的实现**，行上的 `config` 是这个实现的一份参数。

与 DocumentProcessor 同一套元数据约定
------------------------------------
    name           注册键，进 tool_processor.processor_key 列
    config_model   参数结构的 pydantic 模型；get_config_schema()
                   把它变成前端的表单描述

照着 `app/services/document/parser/processor.py` 写，连
`get_config_schema()` 的缓存方式都一样 —— 能力目录（`/processors`）
与 document 的能力目录是同一类接口，两边的字段语义应当一致。

三层结构
--------
    BaseToolProcessor          共性：身份 / 参数模型 / 执行入口
      ├── ToolInterceptor      processor_type=INTERCEPTOR
      ├── ToolTransformer      processor_type=TRANSFORMER
      └── ToolObserver         processor_type=OBSERVER

**具体实现必须挂在三个 type 基类之下**，不能直接继承
BaseToolProcessor —— `processor_type` 由 type 基类定死，直接继承会
拿到 None，注册表据此判定"尚未归类"并跳过它（这与 document 那边
`role is None` 的能力不进目录是同一条规则）。

所以 processor_type 不是作者声明的，而是**继承来的**：声明与实际
能力不符时，没有第二处可以撒谎。

为什么 action 方法放基类而不是各自 role
---------------------------------------
`before` / `after` / `around` 是**调用点**的划分，不是某一类处理器
专属的：一个 TRANSFORMER 可以挂 BEFORE（改参数）也可以挂 AFTER
（改结果）。放进 role 基类会逼着每个 role 把三种都写一遍。

三个方法默认抛 NotImplementedError，而不是声明成 @abstractmethod：
每个实现只服务自己 `supported_hooks` 里的那几个 hook，声明成抽象
会逼着"只做 AROUND 的重试"也写一遍用不到的 before / after 空壳。
调错 hook 时的失败信息因此要**指明去查 supported_hooks** ——
见 `_unsupported()`。
"""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, ClassVar

from pydantic import BaseModel

if TYPE_CHECKING:
    # 只在类型检查时导入：base.py 会被**服务侧**的注册表 import
    # （registry/processor_registry.py），而管理面的"列处理器、看
    # config_schema"这些功能不该被 langgraph 绑住 —— 与 scanner.py
    # 里 _import_base_tool 的惰性导入是同一条立场。
    #
    # 标注本身无成本：本模块有 `from __future__ import annotations`，
    # 注解不求值。真要用的处理器自己 import（那时它已经在
    # langgraph 里跑了）。
    from langchain.tools import ToolRuntime

from app.services.tool.domain.schemas.enum import (
    ExecutionMode,
    ProcessorHook,
    ProcessorType,
)

from .context import (
    AsyncToolCall,
    ToolCallDenied,
    ToolInvocation,
    ToolOutcome,
)

__all__ = [
    "BaseToolProcessor",
    "ToolInterceptor",
    "ToolObserver",
    "ToolTransformer",
]


class BaseToolProcessor(ABC):
    """
    处理器实现的公共契约。

    ⚠️ 本类**故意没有抽象方法**。它不能靠 `inspect.isabstract()`
    与具体实现区分 —— 区分靠注册表的结构规则（直接子类 = type 基类，
    见 `registry/processor_registry.py`）。因此不要在子类里指望
    "忘了实现某个方法会在实例化时炸"：真正的闸门是注册表的
    `processor_type` 检查与这里的 `__init_subclass__`。

    runtime 怎么传：**参数，不是属性**
    ----------------------------------
    `process()` 与三个 hook 方法都收一个 `runtime`（langgraph 的
    ToolRuntime），由执行链那一层注入 —— 与 langchain 给工具函数
    注入 runtime 是同一套约定（声明一个 `runtime: ToolRuntime`
    参数即可，框架负责填）。

    用它拿得到 state / store / context / config / stream_writer /
    tool_call_id：处理器要靠它读写图状态与跨轮存储。

    ⚠️ **绝对不要把它存成 `self.runtime`**
    -------------------------------------
    注册表里的处理器实例是**共享的**（`found_processors()` 每个
    key 只 new 一次，`get_processor_registry()` 是进程级单例）。
    也就是说同一个实例会被**所有挂载它的 Tool、所有并发调用**同时
    使用。把这次调用的 runtime 写到 self 上，等于让 A 调用读到
    B 调用的 state 与 store —— 而症状是"偶发地写错了地方"，
    几乎不可能从日志里看出来。

    `self` 上只该有**从配置与类声明推导出来的不可变状态**；
    凡是"这次调用的东西"，都走参数。

    runtime 可能是 None
    ------------------
    不在 langgraph 图里执行时（例如脱离图的直接调用、冒烟脚本）
    它不存在。实现要么用它、要么明确回落到别的途径，**不要**
    假设它一定有：写 `runtime.store` 而不判空，会在图外直接炸。
    """

    #: 注册键，落进 tool_processor.processor_key。
    #:
    #: ⚠️ 一旦发布**永不复用** —— 它会冻结进已经挂载的链里，
    #: 复用它等于让历史配置指向另一个实现。改名 = 新键 + 迁移。
    name: str = ""

    #: 给人看的名字，进 ProcessorDefinition.name。
    display_name: str = ""

    #: 一句话说明这个处理器做什么。没有就留 None。
    description: str | None = None

    #: 该处理器接受的参数结构。无参数的实现留 None。
    config_model: ClassVar[type[BaseModel] | None] = None

    #: 由三个 type 基类定死。None = 尚未归类，注册表跳过。
    processor_type: ClassVar[ProcessorType | None] = None

    #: 它能挂在调用链的哪几段。服务层据此校验挂载请求。
    supported_hooks: ClassVar[frozenset[ProcessorHook]] = frozenset()

    #: 它支持的执行方式。AROUND 必然只支持 SYNC（见 __init_subclass__）。
    supported_execution_modes: ClassVar[
        frozenset[ExecutionMode]
    ] = frozenset()

    #: get_config_schema() 的缓存。纯函数，按类缓存一份即可。
    config_model_schema: ClassVar[dict[str, Any] | None] = None

    # ========================================================
    # 类定义期校验
    # ========================================================

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        类体执行完的那一刻校验声明是否自洽。

        放在这里而不是服务层：这是一条**静态**约束，违反它的类从
        定义那一刻起就是错的。等服务层发现要等到有人挂载它，
        而那时错误信息只会指向某次请求。

        与 ORM 的 ck_tool_processor_around_sync 是同一条规则的两次
        表达 —— DB 那道拦的是绕过应用的写入，这道拦的是写错的实现。
        """

        super().__init_subclass__(**kwargs)

        # 中间基类（三个 type 基类自己）还没声明 hooks，跳过。
        if not cls.supported_hooks:
            return

        if (
                ProcessorHook.AROUND in cls.supported_hooks
                and ExecutionMode.SYNC
                not in cls.supported_execution_modes
        ):
            raise TypeError(
                f"{cls.__name__} 声明了 AROUND 却不支持 SYNC。"
                "AROUND 包住整次调用，一个不被 await 的包装器无法"
                "决定何时、是否调用工具，因此 AROUND 必然同步。"
                "（DB 的 ck_tool_processor_around_sync 是同一条规则。）"
            )

        # 转换器只支持 ASYNC = 它改了数据但没人接返回值 —— 等于
        # 什么都没改，而且**一点迹象都没有**：链上看着挂了脱敏，
        # 实际参数与结果原样流出去。这种错必须在这里拦住。
        if (
                cls.processor_type is ProcessorType.TRANSFORMER
                and ExecutionMode.SYNC
                not in cls.supported_execution_modes
        ):
            raise TypeError(
                f"{cls.__name__} 是 TRANSFORMER 却不支持 SYNC。"
                "转换器靠**返回值**交付改过的数据，而 ASYNC 的返回值"
                "会被丢弃 —— 那样它就是一个什么都不做的处理器，"
                "却在链上看起来很正常。"
            )

    # ========================================================
    # 身份与参数（能力目录用的访问器）
    # ========================================================

    def get_key(self) -> str:
        """全局唯一注册键。能力目录与挂载校验都走这里。"""

        return self.name

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """
        参数结构的 JSON Schema（draft 2020-12），给前端生成表单。

        没有参数模型时返回**空 object**（properties 为空、
        additionalProperties 为 false），而不是 None —— 前端据此
        隐藏表单，不必再处理"这个字段可能不存在"。

        与 DocumentProcessor.get_config_schema() 逐字同形，
        连按类缓存的方式都一样。
        """

        if cls.config_model is None:
            return {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            }

        if cls.config_model_schema is None:
            cls.config_model_schema = (
                cls.config_model.model_json_schema()
            )

        return cls.config_model_schema

    def supports_hook(self, hook: ProcessorHook) -> bool:
        """这个实现能挂在 hook 这一段吗。"""

        return hook in self.supported_hooks

    def supports_execution_mode(
            self,
            mode: ExecutionMode,
    ) -> bool:
        """这个实现支持这种执行方式吗。"""

        return mode in self.supported_execution_modes

    # ========================================================
    # 执行入口
    # ========================================================

    async def process(
            self,
            hook: ProcessorHook,
            invocation: ToolInvocation,
            next_call: AsyncToolCall,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:
        """
        统一入口 —— 调用方只认这一个方法。

        与 DocumentProcessor.process 同一角色：各 role 把它覆写成
        自己的语义方法，调用方换实现不影响。这里按 hook 分发给
        before / after / around 三个语义方法。

        `runtime` 是 langgraph 的 ToolRuntime，带 state / store /
        context / config / stream_writer / tool_call_id —— 处理器
        要靠它读写图状态与跨轮存储（见类 docstring 的"runtime 怎么
        传"）。

        ⚠️ 本方法**完全不校验 hook 是否在 supported_hooks 里**。
        校验是挂载期的事（服务层），执行期再查一次只会把
        "配置写错了"变成"运行时才炸"。调用方按挂载的 hook 调即可。

        ASYNC 的那几种由调用方 `asyncio.create_task` 起掉、不等
        返回值 —— 本方法本身不区分 SYNC / ASYNC：那是**怎么调**的
        问题，不是处理器怎么做的问题。用 create_task 时 runtime
        仍然可用：任务会继承创建时的 contextvars。
        """

        # 归一成枚举再比。链上的 hook 存在 VARCHAR 列里，调用方
        # （比如把链编译成 handler 嵌套的那一层）手上可能是个字符串；
        # 而下面的分支用 `is` 比 —— 字符串与枚举成员**不 is**，
        # 于是会一路落到最后的 around()，表现是"配了 BEFORE，
        # 跑的却是 AROUND"，而且一点报错都没有。
        #
        # 认不出来的值让 ProcessorHook() 直接抛 ValueError，而不是
        # 兜到某个分支上：能静态发现的错误不要留到运行期。
        hook = ProcessorHook(hook)

        if hook is ProcessorHook.BEFORE:

            decided = await self.before(
                invocation, config=config, runtime=runtime
            )

            # 短路：处理器直接给出了这次调用的结果（缓存命中、
            # 审批拒绝……），工具**不会被调用**。
            if isinstance(decided, ToolOutcome):
                return decided

            return await next_call()

        if hook is ProcessorHook.AFTER:

            # 先跑调用，再把结果交给处理器 —— 顺序不能反。
            # 让 AFTER 自己调 next_call 会逼着每个"只想看看结果"的
            # 实现都去写一遍这个模板，而且很容易写成"忘了调"或
            # "调了两次"（后者会**真的执行两遍工具**）。
            outcome = await next_call()

            return await self.after(
                invocation,
                outcome,
                config=config,
                runtime=runtime,
            )

        return await self.around(
            invocation,
            next_call,
            config=config,
            runtime=runtime,
        )

    # ========================================================
    # 语义方法
    # ========================================================

    async def before(
            self,
            invocation: ToolInvocation,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolInvocation | ToolOutcome:
        """
        调用前。两种返回：

            ToolInvocation  放行，按它（可能已被改写）去调用
            ToolOutcome     **短路**：直接把它当作这次调用的结果，
                            工具不会被调用

        短路的两种典型用法：

            缓存命中     ToolOutcome(value=缓存值) —— 一次成功的调用
            审批拒绝     ToolOutcome(error=ToolCallDenied(...))

        "拒绝"不单独占一个返回值：它本来就是"短路并给出一个失败
        结果"，与缓存命中是同一种形状的不同内容。多一个 None 分支
        只会让每个实现都要处理三种情况，而其中两种是同一件事。

        改参数就**就地改** `invocation.arguments`（它是本次调用私有
        的副本，见 context.py），不必返回一个新对象。
        """

        raise self._unsupported(ProcessorHook.BEFORE)

    async def after(
            self,
            invocation: ToolInvocation,
            outcome: ToolOutcome,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:
        """
        调用后。拿得到 invocation 与它产生的结果（成功的值或失败
        的异常），返回可能被改写的 outcome。

        **调用已经发生**，因此这里没有 next_call：一个 AFTER 处理器
        不该有"再执行一遍工具"的能力 —— 想要"结果不满足就重来"
        的实现属于 AROUND（它从一开始就包住整次调用），
        混进 AFTER 会让"工具被调了几次"变成没人说得清的事。

        最常见的用法是**只读**（指标、事件）：看一遍 outcome，
        原样返回。
        """

        raise self._unsupported(ProcessorHook.AFTER)

    async def around(
            self,
            invocation: ToolInvocation,
            next_call: AsyncToolCall,
            *,
            config: dict[str, Any],
            runtime: ToolRuntime | None = None,
    ) -> ToolOutcome:
        """
        包住整次调用。可以不调 next_call、也可以调多次。

        重试、超时、熔断都挂在这里：`await next_call()` 拿到
        `ToolOutcome`，据 `outcome.error` 决定下一步。
        """

        raise self._unsupported(ProcessorHook.AROUND)

    # ========================================================

    def _unsupported(self, hook: ProcessorHook) -> Exception:
        """
        调了一个自己没声明的 hook。

        信息里必须带上 supported_hooks：这条错误的实际成因是
        **链上的挂载写错了**（挂了 AFTER 但实现只做 AROUND），
        不指明"这个实现支持什么"的话，看日志的人只能去翻代码。
        """

        return NotImplementedError(
            f"{type(self).__name__}（{self.get_key()}）没有实现 "
            f"{hook.value}。它支持的 hook 是 "
            f"{sorted(h.value for h in self.supported_hooks)}。"
        )


# ============================================================
# 三个 type 基类
# ============================================================
#
# processor_type 在这里定死，实现不再声明 —— 与 document 模块
# 把 role 定死在 role 基类是同一个理由：这是**分类**，不是选项。
#
# ⚠️ 三者都必须**有子类**才有意义。注册表按"BaseToolProcessor 的
# 直接子类 = type 基类"把它们排除掉（见 registry/processor_registry.py），
# 因此删光某个 type 基类下所有实现不会让它被误当成实现注册。


class ToolInterceptor(BaseToolProcessor, ABC):
    """
    拦截器：**改变控制流**。

    可以不让调用发生（before 返回 None），也可以让它发生多次
    （around 里重复 await next_call）—— 重试、超时、熔断、
    人工审批都属于这一类。

    因为它能决定"调不调"，所以不可能异步：ASYNC 的拦截器起了就
    不等回来，没有机会做决定。
    """

    processor_type: ClassVar[ProcessorType] = (
        ProcessorType.INTERCEPTOR
    )


class ToolTransformer(BaseToolProcessor, ABC):
    """
    转换器：**只改数据，不改控制流**。

    改参数（BEFORE）或改结果（AFTER），但调用该发生还是发生、
    该发生几次还是几次。

    ⚠️ 这条约束**没有代码强制**：Python 拦不住一个 TRANSFORMER
    在 around 里把 next_call 调两次。它靠的是实现自律与评审 ——
    声明成 TRANSFORMER 就是承诺"我不动控制流"。分类的价值在于
    让读链的人一眼看出哪些处理器会改变执行次数。
    """

    processor_type: ClassVar[ProcessorType] = (
        ProcessorType.TRANSFORMER
    )


class ToolObserver(BaseToolProcessor, ABC):
    """
    观察者：**只读旁路**。

    看得到调用与结果，但不改任何东西，返回值被丢弃（因此通常挂
    ASYNC：起了就不等）。指标、审计、trace 属于这一类。

    与变更日志的区别：观察者拿得到的是**这次调用的实际参数与
    实际结果**，不是"配置变了"那种记录。
    """

    processor_type: ClassVar[ProcessorType] = (
        ProcessorType.OBSERVER
    )
