-- Tool Processor persistence model
-- PostgreSQL
--
-- Design:
--   1. Processor implementation/metadata is code-owned and comes from runtime registry.
--   2. This database stores only the processor instance mounted on a Tool.
--   3. One Tool currently has exactly one ordered processor chain.
--   4. The chain is represented by tool_processor rows ordered by sequence.
--   5. execution_mode and hook are structural runtime properties; config contains
--      processor-specific business configuration only.
--
-- Assumption:
--   tool_definition(id) already exists in the Tool Definition domain.

BEGIN;

CREATE TABLE IF NOT EXISTS tool_processor (
    id                  UUID PRIMARY KEY,

    tool_id             UUID NOT NULL
                        REFERENCES tool_definition(id)
                        ON DELETE CASCADE,

    -- Stable key of a processor implementation registered in code.
    -- Example: approval_interceptor, cache_interceptor, metrics_observer
    processor_key       VARCHAR(255) NOT NULL,

    -- Zero-based or one-based is an application convention; this design uses 1..N.
    sequence            INTEGER NOT NULL
                        CHECK (sequence >= 1),

    -- Whether the processor must complete before runtime continues.
    execution_mode      VARCHAR(16) NOT NULL
                        CHECK (execution_mode IN ('SYNC', 'ASYNC')),

    -- Where the processor is attached to a single Tool invocation.
    hook                VARCHAR(16) NOT NULL
                        CHECK (hook IN ('BEFORE', 'AFTER', 'AROUND')),

    -- Processor-specific configuration. Do not place sequence, hook or execution_mode here.
    config              JSONB NOT NULL DEFAULT '{}'::jsonb
                        CHECK (jsonb_typeof(config) = 'object'),

    enabled             BOOLEAN NOT NULL DEFAULT TRUE,

    created_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,

    -- An AROUND processor must be synchronous because it wraps the Tool invocation.
    CONSTRAINT ck_tool_processor_around_sync
        CHECK (hook <> 'AROUND' OR execution_mode = 'SYNC')
);

-- One Tool currently owns one ordered chain, so sequence is unique per Tool.
CREATE UNIQUE INDEX IF NOT EXISTS uq_tool_processor_tool_sequence
    ON tool_processor(tool_id, sequence);

CREATE INDEX IF NOT EXISTS idx_tool_processor_tool_enabled_sequence
    ON tool_processor(tool_id, enabled, sequence);

CREATE INDEX IF NOT EXISTS idx_tool_processor_processor_key
    ON tool_processor(processor_key);

-- Optional helper index for operational queries over async processors.
CREATE INDEX IF NOT EXISTS idx_tool_processor_async
    ON tool_processor(tool_id, sequence)
    WHERE execution_mode = 'ASYNC' AND enabled = TRUE;

-- Generic updated_at trigger function. If your project already has a shared
-- trigger function, replace this function with the project-wide implementation.
CREATE OR REPLACE FUNCTION set_tool_processor_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_tool_processor_updated_at ON tool_processor;

CREATE TRIGGER trg_tool_processor_updated_at
BEFORE UPDATE ON tool_processor
FOR EACH ROW
EXECUTE FUNCTION set_tool_processor_updated_at();

COMMIT;
