"""
Tool 模块的枚举。

与 document 模块同一套约定：枚举只有这一处定义，
models/sqlalchemy 那边按 String 列存值（见 tool_definition.source_type），
需要转发时再建一个只做 re-export 的兼容层 —— 不并列定义第二份。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ExecutionMode",
    "ProcessorHook",
    "ProcessorType",
    "RETRYABLE_EXECUTION_STATUSES",
    "SourceState",
    "SourceType",
    "ToolExecutionStatus",
]


class SourceType(str, Enum):
    """
    Tool 定义的**来源种类**。

    LOCAL_METHOD
        本进程里被 @tool 标记过的 service 方法。
        发现方式是反射扫描（见 registry/discoverer.py），
        entrypoint 形如 app.services.x:Class.method。

    MCP
        外部 MCP server 暴露的工具。
        entrypoint 是 server 上的工具名，source_ref 是 server 标识。

    OPENAPI
        从一份 OpenAPI spec 里抽出来的 operation。
        entrypoint 是 operationId，source_ref 是 spec 的位置。

    ⚠️ 本期只实现了 LOCAL_METHOD（见 discoverer.py 的注册表）。
    另外两个先把值占好：它们是"同一件事的另一些来源"，不是"以后
    可能要支持"—— 表结构、sync 的比对逻辑、source_state 的语义
    对三种来源完全一样，区别只在扫描器。占好值可以避免将来
    加一个来源就要动一次枚举与已有数据。
    """

    LOCAL_METHOD = "local_method"
    MCP = "mcp"
    OPENAPI = "openapi"


class SourceState(str, Enum):
    """
    Tool 定义相对**代码现状**的状态。

    ACTIVE
        最近一次同步时，代码里还能发现它。

    MISSING
        库里曾经有，但最近一次同步在代码里找不到了 ——
        方法被删了、改名了，或者 @tool 被摘了。

    刻意不物理删除：Tool 可能已经被编排进某个 Agent / 工作流，
    行没了，引用就断了、报错也指不到原因。标成 MISSING 让调用方
    自己决定是提示、降级还是清理（见 ToolSyncRequest.remove_missing）。

    与 document 模块的 DefinitionStatus（active/deleted）是两个概念：
    那个是"人主动删的"，这个是"代码先没了的"。因此不共用枚举。

    ⚠️ 取值是**大写**，与 document 模块那批小写枚举不一致 ——
    这是照规格来的，不是笔误：models.sql 写的是
    `DEFAULT 'ACTIVE'`，tool.yaml 的枚举也列 `ACTIVE` / `MISSING`。
    两者都是对外契约，以它们为准。
    """

    ACTIVE = "ACTIVE"
    MISSING = "MISSING"


# ============================================================
# Processor
# ============================================================
#
# 三个枚举都照 tool_processor.yaml 的大写取值 —— 它们是**对外契约**
# （前端按字面值传 hook / execution_mode）。与上面 SourceState 同样
# 是大写，与 document 模块那批小写枚举不一致，这是照规格来的。


class ProcessorType(str, Enum):
    """
    处理器**怎么改变工具调用**——由实现所继承的 type 基类决定。

        INTERCEPTOR    拦截器：能改控制流（不调 / 调多次 next_call）
        TRANSFORMER    转换器：只改数据，必须且只能调一次 next_call
        OBSERVER       观察者：只读旁路，返回值被丢弃

    它不是作者声明的，是从基类推出来的（见
    app/agent/tools/processor/base.py 的三个 type 基类）—— 与
    document 模块的 ProcessorRole 由 role 基类定死是同一个做法：
    声明与实际能力不符时，没有第二处可以撒谎。
    """

    INTERCEPTOR = "INTERCEPTOR"
    TRANSFORMER = "TRANSFORMER"
    OBSERVER = "OBSERVER"


class ProcessorHook(str, Enum):
    """
    处理器挂在工具调用的哪一段。

        BEFORE    调用前 —— 校验参数、审批、拒绝（不调用工具）
        AFTER     调用后 —— 看得到调用与结果，可改写结果
        AROUND    包住整次调用 —— 重试、超时、埋点

    ⚠️ AROUND 在链上是**两条记录**（先进 + 后出），各带自己那份
    config；两条记录之间必须与其它 AROUND 的区间正确嵌套（洋葱
    模型），校验见 application/tool_processor_service_impl.py。
    """

    BEFORE = "BEFORE"
    AFTER = "AFTER"
    AROUND = "AROUND"


class ExecutionMode(str, Enum):
    """
    处理器与工具调用是否**同步**。

        SYNC     参与结果：调用方等它，它能改控制流或数据
        ASYNC    旁路：起了就不等，返回值丢弃，只能是观察者

    ⚠️ AROUND 不可能是 ASYNC —— 一个不被 await 的包装器无法决定
    何时、是否调用工具。因此 `AROUND ⇒ SYNC`，这条在 ORM 的
    ck_tool_processor_around_sync 与服务层各拦一次（DB 的 CHECK
    抛的是 IntegrityError 而非 ValueError，兜不成 409）。
    """

    SYNC = "SYNC"
    ASYNC = "ASYNC"


class ToolExecutionStatus(str, Enum):
    """
    一次工具调用在 `tool_execution` 里的一生。

    **判重只看"能不能重跑"，不看语义好听不好听。** 每个取值下面的
    "重试"那一行就是它在 `先查` 策略里的后果
    （见 graph/react/tools.py 的 `_reconcile` 与 application 的
    `decide`）—— 改这里的取值之前先看那张表。

        PENDING     记录已建、工具还没跑           重试：拒绝（有别人在跑，或崩溃遗留）
                    ⚠️ 它也可能是**上一次**崩溃留下的
        RUNNING     执行中，或**无法确认**是否执行过   重试：拒绝
                    （fail-closed：不确定就当副作用可能已发生）
        SUCCESS     执行成功，结果已落库            重试：拒绝（工具确实跑过了）
        FAILED      确认失败，**且没有产生副作用**   重试：允许
        UNKNOWN     无法确认操作是否成功            重试：拒绝
        CANCELLED   执行前已取消，确认无副作用       重试：允许
        REJECTED    执行前被权限 / 参数 / 业务规则拒绝 重试：允许（让链再判一次）
        INTERRUPT   HITL 中断，**工具未执行**        重试：允许

    ⚠️ 本期有**两个取值没有任何写入方**：FAILED 与 UNKNOWN
    ----------------------------------------------------
    与 SourceType 把 MCP / OPENAPI 先占好是同一套做法
    （见上面那个枚举）—— 表结构与策略对它们已经完全成立，少的
    只是一个产生它们的时机：

        FAILED    它背着"确认无副作用"这个承诺，而我们判不出它是否
                  成立（工具可能已经产生外部副作用，例如下单走了外部
                  接口——那不在 factory.scope 的事务里，回滚不了）。
                  要写它，得先有一个"这个工具没有外部副作用"的声明。
        UNKNOWN   进程被杀才会产生（没有异常可抓）。要写它，得有一个
                  扫描遗留行的 reaper，而"多久算死"是个必须有人拍的数。

    在那之前，未分类的异常一律落 RUNNING（不确定），而不是 FAILED
    —— 后者会错误地**许可**重试一个有副作用的操作。

    ⚠️ INTERRUPT 等价于"未执行"，只因为今天所有中断都在执行之前
    ---------------------------------------------------------
    审批是 BEFORE（interceptor/approval.py 的 supported_hooks），
    `interrupt()` 在工具跑之前抛出。因此中断时工具**一定**没执行，
    放行是安全的。

    将来若加一个 AFTER 中断（"确认结果"那类），同一个 except 会把
    **已经执行过**的中断也记成 INTERRUPT → resume 放行 → 重复执行
    一个有副作用的工具，正好把这张表的用途反过来。

    靠状态名分不开这两件事。要根治，得让"已开始执行"成为一个
    独立可查的事实 —— 即在工具方法真正被调用的前一刻把 PENDING
    推成 RUNNING（那一步在 infra/agent/tool_executor.py 里，不在
    本表这一层）。**本期没做**，所以在那里加 AFTER 中断之前，
    先把这条读完。

    取值是**大写**，与同模块的 ProcessorHook / ExecutionMode 一致，
    也与 models.sql 的 DB 约束一致。
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    INTERRUPT = "INTERRUPT"


#: 命中「先查」时**允许重跑**的状态；不在这里的一律拒绝（fail-closed）。
#:
#: 这份集合与上面每个取值标注的「重试：允许 / 拒绝」是同一份判据 ——
#: 改这里之前先回去看那张表。
#:
#: ⚠️ 被拒绝的那一组才是要害：
#:
#:     PENDING   别人正在跑，或者是上一次崩溃遗留的
#:     RUNNING   执行中，或**无法确认**是否执行过
#:     SUCCESS   确实跑过了
#:     UNKNOWN   不知道跑没跑
#:
#: 四个都不是能安全重跑的情形。而这个集合一旦被误改宽，后果是
#: 「一个副作用被执行两遍」，且不会有任何报错。
RETRYABLE_EXECUTION_STATUSES: frozenset[ToolExecutionStatus] = frozenset({
    ToolExecutionStatus.FAILED,
    ToolExecutionStatus.CANCELLED,
    ToolExecutionStatus.REJECTED,
    ToolExecutionStatus.INTERRUPT,
})
