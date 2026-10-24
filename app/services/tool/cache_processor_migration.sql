-- ============================================================
-- cache_interceptor 挂载迁移：BEFORE + AFTER -> AROUND
--
-- 背景：
--   旧版缓存拆成 BEFORE（查） + AFTER（回填）两条记录。
--   现在处理器只支持 AROUND（查 + 回填在同一个实现里），
--   旧的 BEFORE/AFTER 记录在运行时会走到未实现的 hook。
--
-- 本脚本对每个挂了旧版缓存的 Tool：
--   1. 从 BEFORE（没有就 AFTER）取 ttl_seconds；
--   2. 删除旧的 cache BEFORE/AFTER 两行；
--   3. 把该 Tool 其余的行整体后移 2 位；
--   4. 在最外层插入两条 AROUND 记录（sequence 1、2，SYNC）。
--
-- ⚠️ 新缓存不再接受 key_fields：判重键固定为「方法名 + 全部参数
--    的 sha256」。这里只保留 ttl_seconds，旧配置里的 key_fields
--    会被丢弃。
--
-- 为什么放最外层：
--   缓存应该看到**处理完之后**的结果（例如 privacy AFTER 已生效），
--   并且应该在其它 BEFORE 改写参数之前先查。放最外层是唯一同时
--   满足这两点的位置。
--
-- 幂等：
--   迁移后不再有 cache 的 BEFORE/AFTER 行，脚本再跑是 no-op。
--
-- 用法：
--   psql ... -f app/services/tool/cache_processor_migration.sql
-- ============================================================

BEGIN;

-- 每个有旧版缓存的 Tool，以及要沿用的配置。
CREATE TEMP TABLE _cache_migrate_tools ON COMMIT DROP AS
SELECT tool_id,
       jsonb_build_object(
           'ttl_seconds',
           COALESCE(
               (
                   (ARRAY_AGG(config) FILTER (WHERE hook = 'BEFORE'))[1]
                   ->> 'ttl_seconds'
               )::int,
               (
                   (ARRAY_AGG(config) FILTER (WHERE hook = 'AFTER'))[1]
                   ->> 'ttl_seconds'
               )::int,
               300
           )
       ) AS config
  FROM tool_processor
 WHERE processor_key = 'cache_interceptor'
   AND hook IN ('BEFORE', 'AFTER')
 GROUP BY tool_id;

-- 先把不是 cache 的行抄出来。
-- 注意排除**所有** cache 行（BEFORE / AFTER / 可能已有的 AROUND）：
-- 否则同一 Tool 上会留下旧的 AROUND 对，再加上新插入的一对，变成
-- 四条 AROUND，链不合法。
CREATE TEMP TABLE _cache_migrate_chain ON COMMIT DROP AS
SELECT tp.id,
       tp.tool_id,
       tp.processor_key,
       tp.sequence,
       tp.execution_mode,
       tp.hook,
       tp.config,
       tp.enabled,
       tp.created_at,
       tp.updated_at
  FROM tool_processor tp
 WHERE tp.tool_id IN (SELECT tool_id FROM _cache_migrate_tools)
   AND tp.processor_key <> 'cache_interceptor';

-- 整条链重建：先清空这些 Tool 的挂载行。
DELETE FROM tool_processor
 WHERE tool_id IN (SELECT tool_id FROM _cache_migrate_tools);

-- 其余行后移两位，给最外层的 AROUND 让出 1、2。
INSERT INTO tool_processor
    (id, tool_id, processor_key, sequence, execution_mode,
     hook, config, enabled, created_at, updated_at)
SELECT id, tool_id, processor_key, sequence + 2, execution_mode,
       hook, config, enabled, created_at, now()
  FROM _cache_migrate_chain;

-- 新的 AROUND 一对（先进 + 后出），配置沿用旧缓存。
INSERT INTO tool_processor
    (id, tool_id, processor_key, sequence, execution_mode,
     hook, config, enabled, created_at, updated_at)
SELECT gen_random_uuid(), tool_id, 'cache_interceptor', 1,
       'SYNC', 'AROUND', config, TRUE, now(), now()
  FROM _cache_migrate_tools
UNION ALL
SELECT gen_random_uuid(), tool_id, 'cache_interceptor', 2,
       'SYNC', 'AROUND', config, TRUE, now(), now()
  FROM _cache_migrate_tools;

-- 已经迁成 AROUND、但配置里还残留 key_fields 的行：只留 ttl_seconds。
UPDATE tool_processor
   SET config = jsonb_build_object(
           'ttl_seconds',
           COALESCE((config->>'ttl_seconds')::int, 300)
       ),
       updated_at = now()
 WHERE processor_key = 'cache_interceptor'
   AND hook = 'AROUND'
   AND tool_id NOT IN (SELECT tool_id FROM _cache_migrate_tools)
   AND (config ? 'key_fields' OR NOT (config ? 'ttl_seconds'));

COMMIT;
