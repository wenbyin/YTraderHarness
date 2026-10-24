-- ============================================================
-- Stress Eval 表结构重建（多轮会话版）
-- ============================================================
--
-- ⚠️ 破坏性操作：会删除 stress_runs 与 stress_run_metrics。
--
-- 为什么需要它
-- ------------
-- 「单请求版 -> 多轮会话版」是破坏性变更：
--   - stress_runs.config 的 JSON 形状从 payload_template 变成了
--     target / session / turn / load；
--   - stress_run_metrics 的 requests_* 列改成了 turns_*，并新增
--     sessions_*、turns_per_session 等列。
--
-- 仓库没有 Alembic，initializer 的 create_all() 只会建新表、不会改
-- 已有表。旧表如果留着，ORM 的 SELECT 会带上新列，症状是**每次查询
-- 都 500**，而不是启动时报错。
--
-- 当前是开发期，建议直接重建；重建后启动应用（create_all）或执行
-- design_agent_eval/schema.sql 即可得到新结构。
--
-- 如需保留旧的 stress_runs 行，请先自行导出，再手工做一次 config 转换。

BEGIN;

DROP TABLE IF EXISTS stress_run_metrics CASCADE;
DROP TABLE IF EXISTS stress_runs CASCADE;

COMMIT;
