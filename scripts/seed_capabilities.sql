-- ============================================================================
-- 能力树（capability）全量 Seed
--
-- 本脚本是当前 capability 表的**全量快照**：8 个一级节点 + 20 个二级节点，
-- 共 28 个。新环境初始化后跑一遍，即可得到与开发库一致的能力目录。
--
-- 形状
-- ----
--   * 自引用树：parent_id 指向父节点的 id。
--   * 二级节点按父节点的 **code 反查**，不硬编码 UUID —— 一级节点即使
--     在别的环境里生成了不同的 id，父子关系仍然成立。
--   * sort_order 与 is_active 都保留，排序 / 停用状态与快照一致。
--
-- 幂等
-- ----
--   * 全部 ON CONFLICT (code) DO NOTHING：重跑安全。
--   * 冲突时**不更新**已有行 —— 线上 / 联调库里人工改过的 name、
--     description、is_active 不会被这份种子覆盖。
--   * 整体一个事务，失败全回滚。
--   * 只建能力节点，不建工具、不建 Skill、不建任何关联。
--
-- 维护
-- ----
--   能力树的增删改走接口 / 数据库后，请重新生成这份快照，
--   避免脚本与真实目录漂移。
-- ============================================================================

BEGIN;


-- ----------------------------------------------------------------------------
-- 1. 一级能力（parent_id IS NULL）
-- ----------------------------------------------------------------------------

INSERT INTO capability (
    id, code, name, description, parent_id, sort_order, is_active
) VALUES
    (gen_random_uuid(), 'AGENT_RUNTIME', '智能体运行时', 'agent运行时相关', NULL, 0, TRUE),
    (gen_random_uuid(), 'CAPABILITY', '能力域', '描述当前系统拥有哪些能力域', NULL, 0, TRUE),
    (gen_random_uuid(), 'DOCUMENT', '文档', '业务文档相关', NULL, 0, TRUE),
    (gen_random_uuid(), 'FILE', '文件', '文件相关', NULL, 0, TRUE),
    (gen_random_uuid(), 'MEMORY', '记忆', '记忆能力', NULL, 0, TRUE),
    (gen_random_uuid(), 'STOCK', '股票信息', 'TDX mcp接口，提供市场交易股票相关信息', NULL, 0, TRUE),
    (gen_random_uuid(), 'TOOL', '工具', '大模型本地工具', NULL, 0, TRUE),
    (
        gen_random_uuid(),
        'NOTIFICATION',
        '通知',
        '主动向人或外部系统发送消息的业务能力；与具体发送渠道的实现工具无关。',
        NULL,
        10,
        TRUE
    )
ON CONFLICT (code) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 2. 智能体运行时（AGENT_RUNTIME）下的能力
-- ----------------------------------------------------------------------------

INSERT INTO capability (
    id, code, name, description, parent_id, sort_order, is_active
)
SELECT
    gen_random_uuid(),
    v.code,
    v.name,
    v.description,
    p.id,
    v.sort_order,
    v.is_active
FROM (VALUES
    (
        'AGENT_INTERRUPT',
        '智能体HITL',
        '智能体人机交互
收集偏好/需求
澄清模糊指令
让用户做实现选择
给用户提供方向选项',
        0,
        TRUE
    ),
    (
        'AGENT_RUNTIME_ORCHESTRATION',
        '智能体运行时流程编排',
        '智能体运行时流程编排，控制节点跳转，
plan模式控制，chat类型控制相关工具集',
        0,
        TRUE
    ),
    (
        'AGENT_SKILL',
        '智能体技能',
        '智能体技能相关',
        0,
        TRUE
    ),
    (
        'AGENT_SUB_AGENT',
        '智能体子代理',
        '智能体子代理空置编排',
        0,
        TRUE
    ),
    (
        'AGENT_TASK',
        '智能体任务',
        '智能体任务创建，拾取，更新，完成等等流程空置相关',
        0,
        TRUE
    )
) AS v(code, name, description, sort_order, is_active)
JOIN capability p ON p.code = 'AGENT_RUNTIME'
ON CONFLICT (code) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 3. 记忆（MEMORY）下的能力
-- ----------------------------------------------------------------------------

INSERT INTO capability (
    id, code, name, description, parent_id, sort_order, is_active
)
SELECT
    gen_random_uuid(),
    v.code,
    v.name,
    v.description,
    p.id,
    v.sort_order,
    v.is_active
FROM (VALUES
    (
        'MEMORY_L0_CONVERSATION',
        '对话',
        'L0级别记忆 - 对话原数据',
        0,
        TRUE
    ),
    (
        'MEMORY_L1_ATOM',
        '原子记忆',
        '结构化的核心记忆，类型persona, episodic, instruction',
        0,
        TRUE
    ),
    (
        'MEMORY_L2_SCENE',
        '情景记忆',
        NULL,
        0,
        TRUE
    )
) AS v(code, name, description, sort_order, is_active)
JOIN capability p ON p.code = 'MEMORY'
ON CONFLICT (code) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 4. 通知（NOTIFICATION）下的渠道
--
-- 只列平台常见渠道；没有对应工具时这些节点不会暴露任何工具，但可以先
-- 建好供 Skill 绑定与分类器识别。渠道节点是**叶子**：工具挂在叶子下，
-- 绑父节点 NOTIFICATION 不会递归带出它们。
-- ----------------------------------------------------------------------------

INSERT INTO capability (
    id, code, name, description, parent_id, sort_order, is_active
)
SELECT
    gen_random_uuid(),
    v.code,
    v.name,
    v.description,
    p.id,
    v.sort_order,
    v.is_active
FROM (VALUES
    (
        'EMAIL',
        '邮件通知',
        '把结果、提醒或摘要通过邮件发送给指定收件人。',
        10,
        TRUE
    ),
    (
        'SMS',
        '短信通知',
        '通过短信把验证码、告警或高时效提醒发送给指定手机号。',
        20,
        TRUE
    ),
    (
        'IM',
        '即时消息通知',
        '通过企业微信、飞书、钉钉、Slack 等即时消息渠道通知。',
        30,
        TRUE
    ),
    (
        'WEBHOOK',
        'Webhook 通知',
        '把事件以 HTTP 回调推送给外部系统。',
        40,
        TRUE
    )
) AS v(code, name, description, sort_order, is_active)
JOIN capability p ON p.code = 'NOTIFICATION'
ON CONFLICT (code) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 5. 工具（TOOL）下的能力
--
-- 与 DATETIME_TOOL / WEB_TOOL 同粒度：一个节点代表一组域内工具，而不是
-- 一个工具。EMAIL_ATTACHMENT_TOOL 刻意与 EMAIL_TOOL 平级而不是它的子
-- 节点 —— 附件解析工具属于邮件域，也属于「附件处理」这条横切能力，
-- 让它在关联表里同时挂两个能力，比放进父子树更灵活。
--
-- SMS 按同一逻辑镜像：
--   SMS_TOOL           对应 EMAIL_TOOL，负责发送与状态查询
--   SMS_TEMPLATE_TOOL  对应 EMAIL_ATTACHMENT_TOOL，负责模板 / 签名 / 变量渲染
--   STANDARDIZE_TOOL   两者共用，不重复建（绑定多个能力即可）
-- ----------------------------------------------------------------------------

INSERT INTO capability (
    id, code, name, description, parent_id, sort_order, is_active
)
SELECT
    gen_random_uuid(),
    v.code,
    v.name,
    v.description,
    p.id,
    v.sort_order,
    v.is_active
FROM (VALUES
    (
        'DATETIME_TOOL',
        '时间日期工具',
        '时间日常通用工具，用来做时间查询和计算
涉及相对时间的问题（今天 / 今晚 / 现在 / 本周 / 最近 / 还有几天）都需要它',
        0,
        TRUE
    ),
    (
        'TEST_TOOL',
        '测试工具',
        '用于测试系统能力，测试Tool工具集，测试Tool Processor',
        0,
        TRUE
    ),
    (
        'WEB_TOOL',
        '网络工具',
        '接入网络工具
可以实现对网络数据的检索浏览',
        0,
        TRUE
    ),
    (
        'EMAIL_TOOL',
        '邮件工具',
        '邮件的读取、检索、发送、回复与归档；附件能力另见 EMAIL_ATTACHMENT_TOOL。',
        10,
        TRUE
    ),
    (
        'EMAIL_ATTACHMENT_TOOL',
        '邮件附件工具',
        '下载、解析邮件附件，并从附件中抽取正文与结构化内容。',
        20,
        TRUE
    ),
    (
        'STANDARDIZE_TOOL',
        '标准化工具',
        '把非结构化内容归一为统一字段、格式与模板，供下游复用。',
        30,
        TRUE
    ),
    (
        'SMS_TOOL',
        '短信工具',
        '单发、批量发送短信并查询发送状态；模板与签名另见 SMS_TEMPLATE_TOOL。',
        40,
        TRUE
    ),
    (
        'SMS_TEMPLATE_TOOL',
        '短信模板工具',
        '管理短信模板与签名，并做变量渲染；发送能力见 SMS_TOOL。',
        50,
        TRUE
    )
) AS v(code, name, description, sort_order, is_active)
JOIN capability p ON p.code = 'TOOL'
ON CONFLICT (code) DO NOTHING;


-- ----------------------------------------------------------------------------
-- 6. 验收
-- ----------------------------------------------------------------------------

SELECT
    child.code,
    child.name,
    parent.code AS parent_code,
    child.sort_order,
    child.is_active
FROM capability child
LEFT JOIN capability parent ON parent.id = child.parent_id
ORDER BY
    COALESCE(parent.code, ''),
    child.sort_order,
    child.code;

COMMIT;
