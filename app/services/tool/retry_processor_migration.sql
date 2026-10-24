-- ============================================================
-- retry_interceptor 配置迁移：retry_on 白名单 -> retry_on_error 开关
--
-- 背景：
--   旧配置用 retry_on: ["类名", ...] 的多选白名单。前端要为它做
--   多选下拉框，而实际使用只有两种模式：
--
--     只重试超时       retry_on_error = false
--     重试所有错误     retry_on_error = true
--
--   映射规则：
--     旧 retry_on 为空 / 只含 ToolTimeout / TimeoutError
--         -> retry_on_error = false
--     旧 retry_on 含其它异常类名
--         -> retry_on_error = true（比原来宽，但至少会重试）
--     已经有 retry_on_error 的行
--         -> 保留原值，只删掉旧 retry_on
--
-- 幂等：迁移后不再有 retry_on 键，脚本再跑是 no-op。
--
-- 用法：
--   psql ... -f app/services/tool/retry_processor_migration.sql
-- ============================================================

BEGIN;

UPDATE tool_processor
   SET config = (config - 'retry_on')
                || jsonb_build_object(
                    'retry_on_error',
                    CASE
                        WHEN config ? 'retry_on_error'
                            THEN config->'retry_on_error'
                        WHEN config ? 'retry_on'
                             AND jsonb_array_length(config->'retry_on') > 0
                             AND NOT (
                                 config->'retry_on'
                                 <@ '["ToolTimeout", "TimeoutError"]'::jsonb
                             )
                            THEN 'true'::jsonb
                        ELSE 'false'::jsonb
                    END
                ),
       updated_at = now()
 WHERE processor_key = 'retry_interceptor'
   AND (config ? 'retry_on' OR NOT (config ? 'retry_on_error'));

COMMIT;
