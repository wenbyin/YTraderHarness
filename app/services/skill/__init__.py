"""
Skill 管理模块。

Skill 是一份**行为定义 + 能力绑定**：

    skill               稳定标识、给主 agent 看的描述、给子 agent 的提示词
    skill_capability    skill 绑定的能力节点

工具可见性**不直接挂在 Skill 上**，而是沿能力树推导：

    skill_capability.capability_id
      -> capability.code
      -> tool_definition_classification
      -> tool_definition.tool_key（ACTIVE）

这条推导复用 tool 模块现有的
`ToolQueryService.tool_keys_for_capabilities`，本模块不重复实现。

明确不做
--------
- 版本表：更新就地生效，运行时在 spawn 子 agent 时自己取快照
- 分类（taxonomy）绑定：主 agent 按 name/description 选 skill
- token / A2A Agent Card：子 agent 是进程内图节点，不走外部 ingress
- 任何层级的鉴权：管理接口若挂在 protected_router 上，只沿用平台登录

与 capability 的关系
--------------------
本模块**引用** capability，不拥有它。删除 capability 时的引用检查由
capability 模块通过 `CapabilityRepository.count_skill_links` 完成 ——
那里已经有一条读 tool 关联表的前例，这里沿用同一种形状。
"""
