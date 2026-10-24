"""
改控制流的处理器。

    approval.py        调用前要人点头（BEFORE），三个选项
    auth.py            按调用者的角色 / 群组放行或拒绝（BEFORE）
    cache.py           结果缓存，命中直接短路（只支持 AROUND）
    execution_guard.py 按维度计数，超限 / unknown / error 时拦截（AROUND）
    timeout.py         单次调用超时（只支持 AROUND）
    retry.py           失败重试 + 退避（只支持 AROUND）

执行守卫（execution_guard）替代了旧版写在 graph/react/tools.py 里的
tool_execution 账本。它需要一次执行一条记录来计数，因此旧表上的
UNIQUE(session_id, run_id, tool_key, param_hash) 被删掉、改用
(run_id, tool_call_id) 做 replay 去重；表名仍是 tool_execution。
它只在**挂了这条处理器**的工具上生效 —— 没有链就没有守卫。

审批是 BEFORE：它要决定的是"这次调用要不要发生"，AFTER 时工具
已经跑完了。缓存 / 守卫 / 超时 / 重试都是 AROUND：它们要包住整次
调用，在执行前占位、执行后结算。

⚠️ 超时与重试**默认不挂**
--------------------------
图节点层已经有重试与超时；工具级再来一遍会多一层"到底是谁在重跑"
的歧义。因此它们不是默认配置，而是**按工具显式挂载**的处理器：

    timeout   只管单次等多久
    retry     只管失败后再试几次、哪些错误才试

两级都能重试时，出了"怎么跑了三遍"这种问题，日志里得先分清是谁
在重跑。只给确实需要的工具挂，并且 retry 的 `retry_on` 必须显式
列出可重试错误（默认只有超时），不要对所有异常一把抓。
"""
