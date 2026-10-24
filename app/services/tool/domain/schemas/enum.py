"""
Tool 模块的枚举。

与 document 模块同一套约定：枚举只有这一处定义，
models/sqlalchemy 那边按 String 列存值（见 tool_definition.source_type），
需要转发时再建一个只做 re-export 的兼容层 —— 不并列定义第二份。
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "ExecutionDecision",
    "ExecutionMatch",
    "ExecutionMode",
    "ExecutionScope",
    "GuardAction",
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

    LOCAL_METHOD 与 MCP 都已有扫描器：前者扫代码（见
    registry/scanner.py），后者按 mcp.yaml 连 server 取
    `tools/list`（见 registry/mcp_scanner.py）。OPENAPI 仍只占
    枚举值。

    三种来源是"同一件事的另一些来源"：表结构、sync 的比对逻辑、
    source_state 的语义完全一样，区别只在扫描器与执行入口。
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


# ============================================================
# 执行守卫（Execution Guard）
# ============================================================
#
# 守卫回答的是「同一个维度上已经发生过几次、这次还让不让做」。
# 它不是幂等判重：幂等是「这件事只能发生一次」，守卫是「同一维度
# 最多发生 N 次」。两者可以用同一套配置表达（limit=1 + invoke），
# 但概念上必须分开 —— 混在一起会让「为什么第 2 次被拦」有两种
# 完全不同的答案。


class ExecutionScope(str, Enum):
    """
    计数按什么维度聚合。

        SESSION   同一会话
        RUN       同一次运行（pipeline_run_id）
        USER      同一个用户（caller_identity.user_id）
        GLOBAL    所有用户

    ⚠️ GLOBAL 是跨租户的：一个用户刷满额度会把所有人挡住。
    除非业务明确要「全局熔断」，否则不要把它当默认值。
    """

    SESSION = "SESSION"
    RUN = "RUN"
    USER = "USER"
    GLOBAL = "GLOBAL"


class ExecutionMatch(str, Enum):
    """
    哪些调用算「同一个调用」。

        METHOD    只看工具（方法），不同参数合并计数
        INVOKE    工具 + 参数摘要，只有完全相同才合并

    INVOKE 的参数摘要由服务层用 `_execution.param_hash_of` 计算，
    与执行记录的 param_hash 是同一个算法。
    """

    METHOD = "METHOD"
    INVOKE = "INVOKE"


class GuardAction(str, Enum):
    """
    守卫命中后的动作。

        BLOCK   不执行工具，返回一条拒绝结果
        ALLOW   放行；计数仍然发生，只是不拦（用于灰度 / 告警模式）

    「超过上限」「已有 UNKNOWN」「已有 ERROR」各有一个独立的动作，
    因此同一个守卫可以做到「上限拦截、错误放行、未知拦截」。
    """

    BLOCK = "BLOCK"
    ALLOW = "ALLOW"


class ExecutionDecision(str, Enum):
    """
    一次预占请求的判定结果。

        EXECUTE  允许执行；调用方拿到一条 PENDING 记录，跑完要 settle
        BLOCK    不执行；reason 说明原因
        REPLAY   同一次工具调用（run_id + tool_call_id）已经成功过，
                 直接回放当时的结果，**不再执行**

    REPLAY 不是「拦截」：它返回的是一条真实成功过的结果。把它记成
    失败会让模型基于一个不存在的结果继续编。
    """

    EXECUTE = "EXECUTE"
    BLOCK = "BLOCK"
    REPLAY = "REPLAY"


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
        FAILED      工具明确报错                    重试：由守卫的 error 配额决定
                    ⚠️ 它**不承诺**没有副作用 —— 新账本里它只表示
                    「工具抛了异常」，是否重试是配置项，不再是状态本身
        UNKNOWN     无法确认操作是否成功            重试：由守卫的 unknown 配额决定
                    来源：执行开始后被取消、进程崩溃遗留、审批之后的
                    中断（见下）
        BLOCKED     守卫**主动**拦下，工具未执行      重试：允许（重算一次决策）
        CANCELLED   执行前已取消，确认无副作用       重试：允许
        REJECTED    执行前被权限 / 参数 / 业务规则拒绝 重试：允许（让链再判一次）
        INTERRUPT   HITL 中断，**工具未执行**        重试：允许

    ⚠️ FAILED 与 UNKNOWN 的写入方现在有了，但语义变了
    ------------------------------------------------
    旧账本把「未分类异常」一律落 RUNNING，因为 FAILED 背着
    「确认无副作用」的承诺。新账本把「是什么状态」与「要不要重试」
    拆开：状态只描述**发生了什么**，重试由守卫的 success / error /
    unknown 三种独立配额决定。因此：

        FAILED    工具抛了异常（可能已经产生副作用，配置方自己判断）
        UNKNOWN   连异常都没有：执行开始后被取消、进程崩溃遗留；
                  这是真正的「不知道跑没跑」

    这个拆分是刻意的：旧模型没有地方表达「部署方想对 error 放行、
    对 unknown 拦截」，而守卫的配置正好需要这两个独立的档位。

    ⚠️ INTERRUPT 等价于"未执行"，只因为今天所有中断都在执行之前
    ---------------------------------------------------------
    审批是 BEFORE（interceptor/approval.py 的 supported_hooks），
    `interrupt()` 在工具跑之前抛出。因此中断时工具**一定**没执行，
    放行是安全的。

    守卫会额外检查 `ToolInvocation.context["tool.started"]`：
    只有**没有**开始执行的中断才记 INTERRUPT；已经开始的中断记
    UNKNOWN（可能已产生副作用）。这样将来加 AFTER 中断时不会把
    「已经执行过」误记成「未执行」。

    取值是**大写**，与同模块的 ProcessorHook / ExecutionMode 一致，
    也与 models.sql 的 DB 约束一致。
    """

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"
    UNKNOWN = "UNKNOWN"
    BLOCKED = "BLOCKED"
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
#:
#: ⚠️ 新守卫**不再用这个集合做判定**：它用的是每次记录上的
#: `counted` 位与配置里的 success / error / unknown 三种独立配额。
#: 保留这个集合是
#: 为了旧调用方与测试的兼容，别把它当成新逻辑的判据。
RETRYABLE_EXECUTION_STATUSES: frozenset[ToolExecutionStatus] = frozenset({
    ToolExecutionStatus.FAILED,
    ToolExecutionStatus.BLOCKED,
    ToolExecutionStatus.CANCELLED,
    ToolExecutionStatus.REJECTED,
    ToolExecutionStatus.INTERRUPT,
})
