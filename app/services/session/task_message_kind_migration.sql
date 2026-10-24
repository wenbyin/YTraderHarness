-- 2026-10-22：agent_message.kind 增加任务消息类型。
--
-- 已有库必须在新代码上线前执行；create_all 不会改既有 CHECK。
-- 存量行不受影响，不需要回填。

BEGIN;

ALTER TABLE agent_message
    DROP CONSTRAINT IF EXISTS ck_agent_message_kind;

ALTER TABLE agent_message
    ADD CONSTRAINT ck_agent_message_kind
    CHECK (kind IN (
        'chat',
        'interaction',
        'tool_activity',
        'notice',
        'task_instruction',
        'task_result'
    ));

COMMIT;
