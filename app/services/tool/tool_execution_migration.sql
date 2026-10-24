-- ============================================================
-- tool_execution 扩展迁移：旧幂等账本 -> 执行守卫账本
--
-- 适用：
--   已有旧版 tool_execution 表的数据库。旧结构是
--   UNIQUE(session_id, run_id, tool_key, param_hash)（一个判重键一行）。
--   新版要「一次执行一行」，所以必须删掉那条唯一约束，并补上
--   scope / execution / counted / tool_call_id 等列。
--
-- 用法：
--   psql -h <host> -U <user> -d <db> -f app/services/tool/tool_execution_migration.sql
--   或者把整个文件粘进 DBeaver / Navicat 执行。
--
-- 幂等：
--   重复执行安全。全新库（create_all 已按新结构建表）上执行时，
--   除最后的 DROP TABLE IF EXISTS 外都是 no-op。
--
-- ⚠️ 执行前先让应用跑一次（或者确认表已存在）。create_all 只建缺失的
--    表，不会改已存在的表；本脚本负责那部分。
--
-- ⚠️ 旧行 counted = FALSE：
--    旧行表达「这个判重键曾经跑过」，不是「守卫窗口内执行过一次」。
--    把它们计入新配额会让上线瞬间消耗一堆额度，因此保留为历史、
--    不参与计数。它们的 match_key 用 md5(id) 占位，不会与新算法
--    算出的 sha256 相同。
-- ============================================================

BEGIN;

-- ------------------------------------------------------------
-- 0. 前置检查
-- ------------------------------------------------------------
DO $$
BEGIN
    IF to_regclass('public.tool_execution') IS NULL THEN
        RAISE EXCEPTION
            'tool_execution 不存在：先启动一次应用让 create_all 建表，'
            '或在空库上执行完整的 models.sql';
    END IF;
END
$$;

-- ------------------------------------------------------------
-- 1. 删掉旧的判重约束
--
-- 旧库可能是 UNIQUE CONSTRAINT，也可能是 UNIQUE INDEX：
-- 先删 constraint，再删 index，两种都能清理干净。
-- ------------------------------------------------------------
ALTER TABLE tool_execution
    DROP CONSTRAINT IF EXISTS uq_tool_execution_identity;

DROP INDEX IF EXISTS uq_tool_execution_identity;

-- ------------------------------------------------------------
-- 2. 新列
-- ------------------------------------------------------------
ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS tool_id UUID;

ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);

ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS scope_type VARCHAR(16);

ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS scope_key VARCHAR(255);

ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS execution_type VARCHAR(16);

ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS match_key VARCHAR(64);

ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS counted BOOLEAN DEFAULT TRUE;

ALTER TABLE tool_execution
    ADD COLUMN IF NOT EXISTS tool_call_id VARCHAR(255);

-- ------------------------------------------------------------
-- 3. 图外直调没有会话 / 运行 —— 新模型允许为空
-- ------------------------------------------------------------
ALTER TABLE tool_execution
    ALTER COLUMN session_id DROP NOT NULL;

ALTER TABLE tool_execution
    ALTER COLUMN run_id DROP NOT NULL;

-- ------------------------------------------------------------
-- 4. 回填
--
-- 兼容两种状态：
--   旧行                 scope_type 为 NULL -> 全部补上、counted=FALSE
--   部分迁移过的行       个别新列为 NULL -> COALESCE 补齐
-- ------------------------------------------------------------
UPDATE tool_execution
   SET scope_type = COALESCE(scope_type, 'SESSION'),
       scope_key = COALESCE(scope_key, session_id::text, '*'),
       execution_type = COALESCE(execution_type, 'INVOKE'),
       match_key = COALESCE(match_key, md5(id::text)),
       counted = CASE
                     WHEN scope_type IS NULL THEN FALSE
                     ELSE COALESCE(counted, TRUE)
                 END
 WHERE scope_type IS NULL
    OR scope_key IS NULL
    OR execution_type IS NULL
    OR match_key IS NULL
    OR counted IS NULL;

-- ------------------------------------------------------------
-- 5. 回填之后再收紧
-- ------------------------------------------------------------
ALTER TABLE tool_execution
    ALTER COLUMN scope_type SET NOT NULL;

ALTER TABLE tool_execution
    ALTER COLUMN scope_key SET NOT NULL;

ALTER TABLE tool_execution
    ALTER COLUMN execution_type SET NOT NULL;

ALTER TABLE tool_execution
    ALTER COLUMN match_key SET NOT NULL;

ALTER TABLE tool_execution
    ALTER COLUMN counted SET NOT NULL;

ALTER TABLE tool_execution
    ALTER COLUMN counted SET DEFAULT TRUE;

-- ------------------------------------------------------------
-- 6. 状态约束：旧集合没有 BLOCKED，整体换掉
-- ------------------------------------------------------------
ALTER TABLE tool_execution
    DROP CONSTRAINT IF EXISTS ck_tool_execution_status;

ALTER TABLE tool_execution
    ADD CONSTRAINT ck_tool_execution_status
    CHECK (status IN ('PENDING', 'RUNNING', 'SUCCESS', 'FAILED',
                      'UNKNOWN', 'BLOCKED', 'CANCELLED',
                      'REJECTED', 'INTERRUPT'));

-- ------------------------------------------------------------
-- 7. 其余 CHECK（PostgreSQL 没有 ADD CONSTRAINT IF NOT EXISTS）
-- ------------------------------------------------------------
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conname = 'ck_tool_execution_scope'
           AND conrelid = 'tool_execution'::regclass
    ) THEN
        ALTER TABLE tool_execution
            ADD CONSTRAINT ck_tool_execution_scope
            CHECK (scope_type IN ('SESSION', 'RUN', 'USER', 'GLOBAL'));
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conname = 'ck_tool_execution_execution'
           AND conrelid = 'tool_execution'::regclass
    ) THEN
        ALTER TABLE tool_execution
            ADD CONSTRAINT ck_tool_execution_execution
            CHECK (execution_type IN ('METHOD', 'INVOKE'));
    END IF;
END
$$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM pg_constraint
         WHERE conname = 'ck_tool_execution_error_object'
           AND conrelid = 'tool_execution'::regclass
    ) THEN
        ALTER TABLE tool_execution
            ADD CONSTRAINT ck_tool_execution_error_object
            CHECK (error IS NULL OR jsonb_typeof(error) = 'object');
    END IF;
END
$$;

-- ------------------------------------------------------------
-- 8. 索引
--
-- uq_tool_execution_call_id  : replay / HITL resume 去重
-- idx_tool_execution_match   : 守卫计数（只覆盖 counted）
-- ------------------------------------------------------------
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_execution_call_id
    ON tool_execution(run_id, tool_call_id)
    WHERE tool_call_id IS NOT NULL AND run_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_tool_execution_match
    ON tool_execution(match_key, created_at)
    WHERE counted;

-- 旧版本的 on_prior_success 专用索引。现在按状态分组计数，
-- 不再需要它；已有库里如果跑过上一版，这里顺手清掉。
DROP INDEX IF EXISTS idx_tool_execution_success;

CREATE INDEX IF NOT EXISTS idx_tool_execution_run_created
    ON tool_execution(run_id, created_at);

CREATE INDEX IF NOT EXISTS idx_tool_execution_tool_key
    ON tool_execution(tool_key, created_at);

CREATE INDEX IF NOT EXISTS idx_tool_execution_unsettled
    ON tool_execution(created_at)
    WHERE status IN ('PENDING', 'RUNNING');

-- ------------------------------------------------------------
-- 9. 清理中途版本留下的表
--
-- 上一版实现曾短暂用过 tool_execution_attempt。现在语义只有一张表；
-- 如果它存在（例如中途跑过那个版本），清掉以免留下第二份没人读的
-- 账本。⚠️ 这张表里的数据会被丢弃。
-- ------------------------------------------------------------
DROP TABLE IF EXISTS tool_execution_attempt;

COMMIT;
