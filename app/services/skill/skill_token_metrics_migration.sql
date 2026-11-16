-- 把已存在的 skill 表补齐到「容量度量」四列。
--
-- 本仓没有 Alembic，create_all() 也不会修改已存在的表：
-- 对于在补字段之前就建好的库，必须手工跑这一份。
-- 新建的库由 SQLAlchemy 模型直接建出同样的列，不需要本文件。
--
-- 四列全是**派生只读列**：不出现在 SkillCreate / SkillUpdate 写请求里，
-- 只由服务层在保存 / 改绑定时测量。字符数（char）刻意不落库 —— 前端
-- 用 instructions.length 实时算，多存一份只会和原文漂移。
--
-- 存量行落进 DEFAULT 的「未测量」态：
--     0 / 0 / 'unmeasured' / NULL
-- 前端据此显示「未测量」，不会把 0 当成真实测量结果。
-- 任一条 skill 被保存一次、或能力绑定被替换一次，就会被刷成实测值。
--
-- 幂等：全部 IF NOT EXISTS，可以重复执行。

ALTER TABLE skill
    ADD COLUMN IF NOT EXISTS instructions_tokens
        INTEGER NOT NULL DEFAULT 0;

ALTER TABLE skill
    ADD COLUMN IF NOT EXISTS prompt_tokens
        INTEGER NOT NULL DEFAULT 0;

ALTER TABLE skill
    ADD COLUMN IF NOT EXISTS token_encoding
        VARCHAR(32) NOT NULL DEFAULT 'unmeasured';

ALTER TABLE skill
    ADD COLUMN IF NOT EXISTS token_computed_at
        TIMESTAMPTZ NULL;

-- 存量行若历史上被别的分支写入过这几列、却没写口径，统一收成
-- 「未测量」，避免前端把口径不明的历史数字当成精确值展示。
UPDATE skill
SET instructions_tokens = 0,
    prompt_tokens       = 0,
    token_encoding      = 'unmeasured',
    token_computed_at   = NULL
WHERE token_computed_at IS NULL
  AND (
        instructions_tokens <> 0
        OR prompt_tokens <> 0
        OR token_encoding <> 'unmeasured'
      );
