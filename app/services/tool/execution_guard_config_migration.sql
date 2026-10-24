-- ============================================================
-- tool_processor.execution_guard_interceptor 配置迁移
--
-- 背景：
--   旧守卫配置是扁平但**另一套**键：
--     limit / on_exceeded / on_unknown / on_error
--   新守卫按状态拆成三套独立配额：
--     success_limit / success_action
--     error_limit   / error_action
--     unknown_limit / unknown_action
--
--   库里已经挂载的旧配置如果不迁移：
--     - 前端按新 schema 读不到 success_limit 等字段；
--     - 运行时 ExecutionGuardConfig(extra="forbid") 会直接拒绝。
--
-- 映射规则：
--   scope / execution / window_seconds   原样保留
--   success_limit                        沿用旧的 limit（没有就 3）
--   error_limit / unknown_limit          取新默认 3 / 1
--   *_action                             全部 BLOCK（按当前要求）
--
-- 幂等：已经迁过（存在 success_limit、且没有旧键）的行不会被再次修改。
--
-- 用法：
--   psql ... -f app/services/tool/execution_guard_config_migration.sql
-- ============================================================

BEGIN;

UPDATE tool_processor
   SET config = jsonb_build_object(
           'scope', COALESCE(config->>'scope', 'SESSION'),
           'execution', COALESCE(config->>'execution', 'METHOD'),
           'window_seconds',
               COALESCE(config->'window_seconds', 'null'::jsonb),
           'success_limit',
               COALESCE(NULLIF(config->>'limit', '')::int, 3),
           'success_action', 'BLOCK',
           'error_limit', 3,
           'error_action', 'BLOCK',
           'unknown_limit', 1,
           'unknown_action', 'BLOCK'
       ),
       updated_at = now()
 WHERE processor_key = 'execution_guard_interceptor'
   AND (
           config ? 'limit'
        OR config ? 'on_exceeded'
        OR config ? 'on_unknown'
        OR config ? 'on_error'
        OR NOT (config ? 'success_limit')
   );

COMMIT;
