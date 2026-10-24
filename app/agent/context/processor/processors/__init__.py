"""
Context Processor 的**具体实现**（反射发现的根）。

一眼看清整条流水线（`order` 与固定阶段模板**严格一致**）：

    order  阶段                      处理器                 gate
    ----------------------------------------------------------------------
    10     context_resolver          context_resolver      ——（required）
    15     task_resolver             task_resolver         task_mode
    16     skill_loader              skill_loader          task_mode
    20     profile_loader            user_profile          include_user_profile
    30     summary_loader            summary               include_summary
    40     history_selector          history               include_history
    45     memory_retriever          memory_profile        include_memory
    50     memory_retriever          memory                include_memory
    60     knowledge_retriever       retrieval             include_retrieved_context
    70     capability_loader         capability            include_capabilities
    80     prompt_assembler          prompt_assembler      ——（required）
    90     token_budgeter            token_budget          enforce_token_budget
    95     context_compactor         compaction            allow_compaction
    99     context_validator         validation            validate

⚠️ 记忆那两层拆成两个处理器（一个阶段，45 在 50 前）
--------------------------------------------
`memory_retriever` 阶段上有**两个**处理器，按取数代价分：

    45  memory_profile    L3 画像 + L2 场景索引   **每轮现读**（纯 SQL，便宜）
    50  memory            L0 原话 + L1 原子记忆   一个运行只召回一次（贵，
                                              结果交 state 保活）

阶段顺序只要求"按 order 排"与"按阶段排"是同一个序列，同阶段多个处理器
只要 order 全局唯一即可（见 pipeline.py 的校验）。排在 50 之前是有意的：
system 里四段的顺序（画像 -> 场景 -> 原话 -> 记忆）与拆分之前逐字节相同。

⚠️ 顺序为什么是 profile -> summary -> history，而不是别的
----------------------------------------------------
阶段顺序由**服务端定义且不可变**（`app/services/context/context_builder.yaml`
里那 13 个 `StageId` 的枚举顺序）。Processor 的 `order` 只是把同一个顺序
再念一遍，pipeline 会校验两者一致 —— 于是"谁先跑"只有一个来源。

（这也解释了为什么这里是 profile 在 summary 之前：那是规格里的顺序，
不是随手排的。）

⚠️ 两个 `when=None`
-----------------
`context_resolver` 与 `prompt_assembler` 是无条件执行的（它们所在的
阶段是 required）：入口先有鸡才有蛋，出口少它就没有产出。pipeline
会双向校验这条规则 —— 给它们配 gate 会在**启动期**报错。

⚠️ 各阶段的依赖是怎么表达的
------------------------
不靠 ContextBuilder 里的 `if processor.name == ...`，而靠**路径**：
resolver 阶段把 `context_policy` / `context_budget` 写进 overlay，
后面的处理器绑 `context_policy.*` / `context_budget.*` 就能读到。见
各文件顶部的说明。

⚠️ 本包必须有 __init__.py
------------------------
`pkgutil.walk_packages` 不会走进没有它的目录：删掉它，所有 Processor
会**无声消失**（发现结果为空，而不是报错）。
"""
