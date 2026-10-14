-- 统一异步任务基础设施的两张表：tasks 与 task_outbox。
--
-- ⚠️ 本文件是**人读的 DDL**，不是启动路径的一部分。
-- 建表实际由 app/infra/datasource/initializer.py 的
-- Base.metadata.create_all() 完成，它以 models/sqlalchemy/ 下的 ORM 声明为准。
-- 两边必须保持一致 —— 不一致时 ORM 赢，而症状是运行期才发现的
-- "column does not exist"。
--
-- 本仓库没有迁移工具：create_all() 幂等但**不会改已存在的表**。
-- 已有库上新增本模块时，下面这些 CREATE 会自动生效（表不存在）；
-- 但**给已有表加列**必须手动 ALTER，见文件末尾。


-- ============================================================
-- tasks —— Task 的事实来源（spec §5）
-- ============================================================
--
-- 状态、参数、lineage、重试计数全以本表为准；MQ 只是投递手段，
-- Worker 只是执行手段。因此每次状态推进都是一条原子 UPDATE（CAS），
-- 而不是"读出来改完整体写回"—— 后者在多 Worker 并发下会覆盖别人的结果。

CREATE TABLE tasks (
    task_id         UUID PRIMARY KEY,

    -- 逻辑 Handler Name（如 hitl.timeout）。不是 import path：
    -- 解析走显式注册表，见 app/infra/task/registry.py。
    handler         VARCHAR(128) NOT NULL,

    -- Pydantic params 序列化后的结果。只有业务 ID 与最小参数，
    -- 不放 Agent State / Message History / Tool Result ——
    -- 那些在执行阶段按 ID 重新加载。理由见 ORM 的类 docstring。
    params_json     JSONB NOT NULL,

    -- pending / running / retryable / success / failed / cancelled
    -- 定义在 app/common/task/types.py::TaskStatus，这里存 .value。
    status          VARCHAR(16) NOT NULL,

    -- NULL = 立即执行。延迟与定时执行都用它，
    -- 不为任何业务单独造 scheduler（spec §7）。
    execute_at      TIMESTAMPTZ,

    attempt         INTEGER NOT NULL DEFAULT 0,
    max_attempts    INTEGER NOT NULL DEFAULT 3,

    -- 定时器代际：提交时 1，每重排一次 +1。
    --
    -- 给"防抖的延迟任务"用：执行时间被反复重排时，必须让上一个已经
    -- 生成的定时失效，否则它到点就会把任务跑掉。定时器醒来把库里的
    -- generation 与它绑定的那个比对，不一致就改按新时间重排。
    --
    -- 为什么不只比 execute_at：重排到**同一时刻**是合法的一次重排，
    -- 那时只有计数器能区分新旧定时。
    generation      INTEGER NOT NULL DEFAULT 1,

    -- 自引用外键，单父，不做 fan-in（spec §9）。
    -- 刻意不加 CASCADE：删父时级联删掉整条子链会让"某次 HITL 触发的
    -- 任务"无声消失，而那种消失没有任何地方会记录。
    parent_task_id  UUID REFERENCES tasks(task_id),

    -- 以下两列**不是**外键，理由不同：
    --   root_task_id  是 lineage 的缓存（查整条链），不是结构约束；
    --                 建成外键会给根任务加上"被所有后代钉住"的隐含语义。
    --   causation_id  可以是业务事件 ID（spec §9），那类 ID 不在本表里 ——
    --                 外键会立刻拦下它。
    root_task_id    UUID,
    causation_id    UUID,

    idempotency_key VARCHAR(255),

    -- 这个键在任务结束后可以被复用吗（终态时释放它）。
    -- 默认 false = 老语义（一次性键）；见 ORM 上那一列的注释。
    reusable_key    BOOLEAN NOT NULL DEFAULT false,

    created_at      TIMESTAMPTZ NOT NULL,
    updated_at      TIMESTAMPTZ NOT NULL,

    started_at      TIMESTAMPTZ,

    -- 租约：claim 时写 now() + 租期；Sweeper 收回崩溃 Worker 留下的
    -- RUNNING 行（否则那条任务永远停在 RUNNING，既不被重试也没有告警）。
    -- 这两列是 spec 建议字段之外的加法，理由见 ORM 的类 docstring。
    lease_expires_at TIMESTAMPTZ,
    leased_by       VARCHAR(128),

    finished_at     TIMESTAMPTZ,

    -- 最近一次失败的原因。重试成功后**仍然保留** ——
    -- 排查"它为什么重试过"要有据可查。
    error           TEXT
);


-- 轮询与 claim 的主索引，对应最热的那条查询：
--
--   WHERE status IN ('pending','retryable')
--     AND (execute_at IS NULL OR execute_at <= now())
--   ORDER BY execute_at NULLS FIRST, created_at
--
-- 列顺序 (status, execute_at)：status 是等值条件，放前面才能把范围扫描
-- 限制在"待跑"的那一小部分行上；反过来会先按 execute_at 扫全表。
CREATE INDEX idx_tasks_due
    ON tasks(status, execute_at);


-- 幂等的**部分**唯一索引。
--
-- WHERE 子句不能省：它把"不传 idempotency_key 就不参与幂等"写成了
-- 约束本身，而不是依赖 NULL 的比较规则 —— 否则一旦有人写了空字符串，
-- 所有这样的行会互相冲突。
CREATE UNIQUE INDEX uq_tasks_idempotency
    ON tasks(idempotency_key)
    WHERE idempotency_key IS NOT NULL;


CREATE INDEX idx_tasks_parent
    ON tasks(parent_task_id);

-- 查整条执行链：WHERE root_task_id = :id ORDER BY created_at
CREATE INDEX idx_tasks_root
    ON tasks(root_task_id);

-- 租约回收：WHERE status = 'running' AND lease_expires_at <= now()
CREATE INDEX idx_tasks_lease
    ON tasks(status, lease_expires_at);


-- ============================================================
-- task_outbox —— 投递意图（spec §5）
-- ============================================================
--
-- "DB commit 成功但 MQ publish 失败"不会丢任务：要投递这件事与 Task
-- 本身写在**同一个事务**里，投递失败的那行还是 pending，Relay 会一直捞。
--
-- 第一版没有 MQ，"投递"落在 Dispatcher 的进程内唤醒队列上，
-- 因此本表不是为将来预留的空壳：进程重启会丢掉内存队列里的投递，
-- 那些投递**只有本表能补**。

CREATE TABLE task_outbox (
    -- 与 task_id 分开：一条任务理论上可投多次（投完、标记 published
    -- 之前崩掉，就会重投一次）。
    id              UUID PRIMARY KEY,

    -- CASCADE：投递记录离开它要投的任务没有意义。
    task_id         UUID NOT NULL
                    REFERENCES tasks(task_id) ON DELETE CASCADE,

    -- pending / published / failed
    -- 定义在 app/common/task/types.py::OutboxStatus，这里存 .value。
    -- 注意与 tasks.status 是两件事：这里说"投出去了没有"，
    -- 那里说"跑得怎么样了"。
    status          VARCHAR(16) NOT NULL,

    created_at      TIMESTAMPTZ NOT NULL,

    published_at    TIMESTAMPTZ,

    retry_count     INTEGER NOT NULL DEFAULT 0,

    -- "本该被执行、却一直没被交出去"这件事的唯一痕迹。
    last_error      TEXT
);


-- Relay 的扫描条件：WHERE status = 'pending' ORDER BY created_at
-- 列顺序与之一致，可以直接走索引有序读出，不必再排一次序。
CREATE INDEX idx_task_outbox_pending
    ON task_outbox(status, created_at);

-- 按任务反查投递记录（排查"这条为什么没被跑"）。
CREATE INDEX idx_task_outbox_task
    ON task_outbox(task_id);


-- ============================================================
-- 已有库：新增列必须手动 ALTER
-- ============================================================
--
-- create_all() 只建缺失的表、**不改已存在的表**。所以上面那个
-- CREATE TABLE 只对全新库生效；tasks 表已经存在的库必须手动补列。
--
-- 目前有两批列是后加的，按加入顺序：
--
--   1) 租约（撤销崩溃 Worker 留下的 running）
--
--   ALTER TABLE tasks ADD COLUMN IF NOT EXISTS
--       lease_expires_at TIMESTAMPTZ;
--   ALTER TABLE tasks ADD COLUMN IF NOT EXISTS
--       leased_by VARCHAR(128);
--
--   CREATE INDEX IF NOT EXISTS idx_tasks_lease
--       ON tasks(status, lease_expires_at);
--
--   2) 定时器代际（防抖的延迟任务）
--
--   ALTER TABLE tasks ADD COLUMN IF NOT EXISTS
--       generation INTEGER NOT NULL DEFAULT 1;
--
-- NOT NULL + DEFAULT 是必须的：存量行的 generation 会被填成 1，也就是
-- "从未被重排过"——那时还没有任何定时器绑定过它们，语义正好吻合。
-- 少了 DEFAULT 这条 ALTER 会直接失败（表里已有行）。
--
--   3) 幂等键可复用（L1 的"每会话一条、反复重排"）
--
--   ALTER TABLE tasks ADD COLUMN IF NOT EXISTS
--       reusable_key BOOLEAN NOT NULL DEFAULT false;
--
-- DEFAULT false 是**语义**，不只是为了让 ALTER 通过：存量行全是老语义
-- （键一次性），一条都不该被改成可复用。要复用必须**提交时显式声明**
-- （L1 的触发就是这么做的）—— 见 ORM 上那一列的注释。
--
-- ⚠️ 必须**先跑 ALTER 再上新代码**：ORM 一旦声明了列，SELECT 就会带上它，
-- 顺序反了的话每一次任务查询都是 500（"column tasks.generation
-- does not exist"）。
