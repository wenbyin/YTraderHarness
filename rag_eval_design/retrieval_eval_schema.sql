-- First-layer deterministic Retrieval Eval schema for PostgreSQL.
-- Aligns with the existing UUID-based Document/DocumentVersion/DocumentChunk model.
-- Source chunk IDs are intentionally snapshots (no FK to document_chunk): a bundle
-- must retain its ground-truth identity even if the source document is later reprocessed.

CREATE TABLE retrieval_eval_bundle (
    bundle_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT NULL,
    status VARCHAR(24) NOT NULL DEFAULT 'draft'
        CHECK (status IN ('draft', 'generating', 'ready', 'partial_failed', 'failed', 'archived')),
    schema_version VARCHAR(24) NOT NULL DEFAULT '1.0',
    selection_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    generation_config JSONB NOT NULL DEFAULT '{}'::jsonb,
    dataset_sha256 VARCHAR(64) NULL,
    source_chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (source_chunk_count >= 0),
    case_count INTEGER NOT NULL DEFAULT 0 CHECK (case_count >= 0),
    created_by UUID NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ NULL
);
CREATE INDEX ix_retrieval_eval_bundle_tenant_status
    ON retrieval_eval_bundle (tenant_id, status, created_at DESC);

CREATE TABLE retrieval_eval_generation_run (
    generation_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bundle_id UUID NOT NULL REFERENCES retrieval_eval_bundle(bundle_id) ON DELETE RESTRICT,
    status VARCHAR(24) NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'partial_failed', 'failed', 'cancelled')),
    sample_seed BIGINT NOT NULL,
    config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    model_role VARCHAR(128) NOT NULL DEFAULT 'eval.qa_generation',
    model_name VARCHAR(255) NULL,
    total_chunks INTEGER NOT NULL DEFAULT 0 CHECK (total_chunks >= 0),
    total_windows INTEGER NOT NULL DEFAULT 0 CHECK (total_windows >= 0),
    completed_windows INTEGER NOT NULL DEFAULT 0 CHECK (completed_windows >= 0),
    failed_windows INTEGER NOT NULL DEFAULT 0 CHECK (failed_windows >= 0),
    generated_cases INTEGER NOT NULL DEFAULT 0 CHECK (generated_cases >= 0),
    error_message TEXT NULL,
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_retrieval_eval_generation_bundle
    ON retrieval_eval_generation_run (bundle_id, created_at DESC);

CREATE TABLE retrieval_eval_generation_window (
    generation_window_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    generation_run_id UUID NOT NULL REFERENCES retrieval_eval_generation_run(generation_run_id) ON DELETE CASCADE,
    window_index INTEGER NOT NULL CHECK (window_index >= 0),
    status VARCHAR(24) NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'partial_failed', 'failed', 'cancelled')),
    -- JSON array of {chunk_id, document_type_definition_id, document_version_id,
    -- content, content_hash, token_count}. Content snapshot enables safe retry/review.
    input_chunks JSONB NOT NULL,
    input_token_estimate INTEGER NULL CHECK (input_token_estimate IS NULL OR input_token_estimate >= 0),
    request_sha256 VARCHAR(64) NULL,
    raw_response JSONB NULL,
    attempt_count INTEGER NOT NULL DEFAULT 0 CHECK (attempt_count >= 0),
    valid_case_count INTEGER NOT NULL DEFAULT 0 CHECK (valid_case_count >= 0),
    error_code VARCHAR(100) NULL,
    error_message TEXT NULL,
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_retrieval_eval_generation_window UNIQUE (generation_run_id, window_index)
);
CREATE INDEX ix_retrieval_eval_window_status
    ON retrieval_eval_generation_window (generation_run_id, status, window_index);

CREATE TABLE retrieval_eval_case (
    eval_case_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bundle_id UUID NOT NULL REFERENCES retrieval_eval_bundle(bundle_id) ON DELETE RESTRICT,
    generation_window_id UUID NULL REFERENCES retrieval_eval_generation_window(generation_window_id) ON DELETE SET NULL,
    document_type_definition_id UUID NOT NULL,
    document_version_id UUID NULL,
    document_retrieval_definition_id UUID NULL,
    -- 源文档身份（快照，无 FK）。出题时主体取自这里：正文常写「本公司」，
    -- 不写出公司名的问题在索引里与其它几十家公司完全同形。
    -- 反过来，问题一旦写了公司名，答案的正确性就依赖「这块 chunk 属于
    -- 那家公司」—— 所以不记它，这类题事后无法复核。
    document_id UUID NULL,
    source_document_title VARCHAR(255) NULL,
    source_document_key VARCHAR(255) NULL,
    -- Intentionally not an FK: this is the immutable gold ID captured at dataset-build time.
    source_chunk_id UUID NOT NULL,
    source_content_hash VARCHAR(128) NOT NULL,
    question TEXT NOT NULL,
    -- Verbatim substring from the source chunk, never an LLM summary.
    answer_original TEXT NOT NULL,
    answer_start_offset INTEGER NULL CHECK (answer_start_offset IS NULL OR answer_start_offset >= 0),
    answer_end_offset INTEGER NULL CHECK (answer_end_offset IS NULL OR answer_end_offset >= 0),
    question_order SMALLINT NOT NULL CHECK (question_order >= 0),
    question_sha256 VARCHAR(64) NOT NULL,
    validation_status VARCHAR(24) NOT NULL DEFAULT 'validated'
        CHECK (validation_status IN ('pending', 'validated', 'rejected')),
    validation_error TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_retrieval_eval_case_order UNIQUE (bundle_id, source_chunk_id, question_order),
    CONSTRAINT uq_retrieval_eval_case_question UNIQUE (bundle_id, source_chunk_id, question_sha256),
    CONSTRAINT ck_retrieval_eval_answer_offsets CHECK (
        answer_start_offset IS NULL OR answer_end_offset IS NULL OR answer_end_offset >= answer_start_offset
    )
);
CREATE INDEX ix_retrieval_eval_case_bundle_dtd
    ON retrieval_eval_case (bundle_id, document_type_definition_id);
CREATE INDEX ix_retrieval_eval_case_source_chunk
    ON retrieval_eval_case (bundle_id, source_chunk_id);

-- One question may be answerable from multiple chunks. source_chunk_id on
-- retrieval_eval_case is the originating chunk; this table is the complete,
-- validated gold set discovered during offline cross-chunk adjudication.
CREATE TABLE retrieval_eval_case_gold_chunk (
    eval_case_gold_chunk_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_case_id UUID NOT NULL REFERENCES retrieval_eval_case(eval_case_id) ON DELETE CASCADE,
    chunk_id UUID NOT NULL,
    is_source_chunk BOOLEAN NOT NULL DEFAULT FALSE,
    validation_method VARCHAR(40) NOT NULL DEFAULT 'source_chunk',
    -- e.g. source_chunk, llm_cross_chunk_judge, human_review
    evidence_excerpt TEXT NULL,
    source_content_hash VARCHAR(128) NULL,
    judgment JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_retrieval_eval_case_gold_chunk UNIQUE (eval_case_id, chunk_id)
);
CREATE INDEX ix_retrieval_eval_case_gold_chunk_case
    ON retrieval_eval_case_gold_chunk (eval_case_id);
CREATE INDEX ix_retrieval_eval_case_gold_chunk_id
    ON retrieval_eval_case_gold_chunk (chunk_id);

CREATE TABLE retrieval_eval_run (
    eval_run_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    bundle_id UUID NOT NULL REFERENCES retrieval_eval_bundle(bundle_id) ON DELETE RESTRICT,
    status VARCHAR(24) NOT NULL DEFAULT 'queued'
        CHECK (status IN ('queued', 'running', 'completed', 'partial_failed', 'failed', 'cancelled')),
    max_k INTEGER NOT NULL DEFAULT 10 CHECK (max_k >= 1 AND max_k <= 1000),
    query_processor_override JSONB NOT NULL DEFAULT '{}'::jsonb,
    effective_config_snapshot JSONB NOT NULL DEFAULT '{}'::jsonb,
    include_stage_trace BOOLEAN NOT NULL DEFAULT TRUE,
    total_cases INTEGER NOT NULL DEFAULT 0 CHECK (total_cases >= 0),
    completed_cases INTEGER NOT NULL DEFAULT 0 CHECK (completed_cases >= 0),
    failed_cases INTEGER NOT NULL DEFAULT 0 CHECK (failed_cases >= 0),
    -- Cached summary; source of truth remains result, stage observation and stage candidate rows.
    metrics JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_by UUID NULL,
    started_at TIMESTAMPTZ NULL,
    completed_at TIMESTAMPTZ NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX ix_retrieval_eval_run_bundle
    ON retrieval_eval_run (bundle_id, created_at DESC);

CREATE TABLE retrieval_eval_case_result (
    eval_case_result_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_run_id UUID NOT NULL REFERENCES retrieval_eval_run(eval_run_id) ON DELETE CASCADE,
    eval_case_id UUID NOT NULL REFERENCES retrieval_eval_case(eval_case_id) ON DELETE RESTRICT,
    query_id UUID NULL,
    -- 本轮**实际发给检索**的文本（改写之后）。不等于 case 的 question：
    -- 逐轮评测里每轮可以用不同改写器，而 LLM 改写不可复现 —— 不存这列，
    -- 事后「这条命中的是哪句话」只能靠翻日志，而日志会轮转。
    query_text TEXT NULL,
    gold_hit BOOLEAN NULL,
    gold_rank INTEGER NULL CHECK (gold_rank IS NULL OR gold_rank >= 1),
    -- Ordered final candidates: [{chunk_id, rank, score, score_name, source_key, metadata}].
    retrieved_chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
    -- 第几轮（从 1 起）。评测是逐轮放宽的：第 N 轮只跑第 N-1 轮没命中
    -- 的 case，且每轮参数不同（第 3 轮才开精排）。所以同一条 case 会有多行。
    -- 没有这一列时第 2 轮会被下面的唯一约束挡住，而插入是
    -- ON CONFLICT DO NOTHING —— 一行都写不进去且不报错。
    attempt_no SMALLINT NOT NULL DEFAULT 1 CHECK (attempt_no >= 1),
    -- 本轮没能参与的召回路线。没有它，一次**降级跑的** miss 与一次正常跑的
    -- miss 在数据上完全一样，而前者是「那一路当时不可用」。
    degraded_routes JSONB NOT NULL DEFAULT '[]'::jsonb,
    latency_ms INTEGER NULL CHECK (latency_ms IS NULL OR latency_ms >= 0),
    status VARCHAR(24) NOT NULL DEFAULT 'completed'
        CHECK (status IN ('completed', 'failed', 'skipped')),
    error_code VARCHAR(100) NULL,
    error_message TEXT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- 三元组：一条 case 在一轮评测里**每轮一行**。
    CONSTRAINT uq_retrieval_eval_case_result UNIQUE (eval_run_id, eval_case_id, attempt_no)
);
CREATE INDEX ix_retrieval_eval_case_result_run_status
    ON retrieval_eval_case_result (eval_run_id, status);

CREATE TABLE retrieval_eval_stage_observation (
    stage_observation_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_case_result_id UUID NOT NULL REFERENCES retrieval_eval_case_result(eval_case_result_id) ON DELETE CASCADE,
    stage_key VARCHAR(100) NOT NULL,
    -- Examples: dense_vector, keyword, graph; empty string for non-branch stages.
    source_key VARCHAR(100) NOT NULL DEFAULT '',
    executed BOOLEAN NOT NULL DEFAULT TRUE,
    input_count INTEGER NULL CHECK (input_count IS NULL OR input_count >= 0),
    output_count INTEGER NULL CHECK (output_count IS NULL OR output_count >= 0),
    gold_present_before BOOLEAN NULL,
    gold_present_after BOOLEAN NULL,
    gold_rank_before INTEGER NULL CHECK (gold_rank_before IS NULL OR gold_rank_before >= 1),
    gold_rank_after INTEGER NULL CHECK (gold_rank_after IS NULL OR gold_rank_after >= 1),
    gold_score_before DOUBLE PRECISION NULL,
    gold_score_after DOUBLE PRECISION NULL,
    gold_score_name VARCHAR(100) NULL,
    gold_score_direction VARCHAR(16) NULL CHECK (gold_score_direction IS NULL OR gold_score_direction IN ('higher_is_better', 'lower_is_better', 'not_comparable')),
    drop_reason VARCHAR(100) NULL CHECK (drop_reason IS NULL OR drop_reason IN (
        'not_recalled', 'top_k_cutoff', 'score_threshold', 'metadata_filter', 'ownership_filter',
        'rerank_cutoff', 'final_top_k_cutoff', 'deduplicated', 'permission_filter',
        'processor_error', 'unknown', 'not_applicable'
    )),
    details JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT uq_retrieval_eval_stage_observation
        UNIQUE (eval_case_result_id, stage_key, source_key)
);
CREATE INDEX ix_retrieval_eval_stage_observation_stage
    ON retrieval_eval_stage_observation (stage_key, source_key, drop_reason);
CREATE INDEX ix_retrieval_eval_stage_observation_result
    ON retrieval_eval_stage_observation (eval_case_result_id);

-- Keep top candidates and raw scores entering/leaving each observed stage (normally top 5).
-- Score is stage-specific; never compare dense, BM25, RRF and reranker scores
-- as if they shared a scale. Persist only top-N plus gold candidates when needed.
CREATE TABLE retrieval_eval_stage_candidate (
    stage_candidate_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    eval_case_result_id UUID NOT NULL REFERENCES retrieval_eval_case_result(eval_case_result_id) ON DELETE CASCADE,
    stage_key VARCHAR(100) NOT NULL,
    source_key VARCHAR(100) NOT NULL DEFAULT '',
    chunk_id UUID NOT NULL,
    -- Rank/score on entry and exit allow explicit diagnosis of threshold/cutoff drops.
    rank_before INTEGER NULL CHECK (rank_before IS NULL OR rank_before >= 1),
    rank_after INTEGER NULL CHECK (rank_after IS NULL OR rank_after >= 1),
    score_before DOUBLE PRECISION NULL,
    score_after DOUBLE PRECISION NULL,
    score_name_before VARCHAR(100) NULL,
    score_name_after VARCHAR(100) NULL,
    score_direction_before VARCHAR(16) NULL CHECK (score_direction_before IS NULL OR score_direction_before IN ('higher_is_better', 'lower_is_better', 'not_comparable')),
    score_direction_after VARCHAR(16) NULL CHECK (score_direction_after IS NULL OR score_direction_after IN ('higher_is_better', 'lower_is_better', 'not_comparable')),
    selected BOOLEAN NOT NULL DEFAULT TRUE,
    drop_reason VARCHAR(100) NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT ck_retrieval_eval_stage_candidate_has_rank CHECK (rank_before IS NOT NULL OR rank_after IS NOT NULL),
    CONSTRAINT uq_retrieval_eval_stage_candidate_chunk UNIQUE (eval_case_result_id, stage_key, source_key, chunk_id)
);
CREATE INDEX ix_retrieval_eval_stage_candidate_stage_rank
    ON retrieval_eval_stage_candidate (eval_case_result_id, stage_key, source_key, rank_after, rank_before);
CREATE INDEX ix_retrieval_eval_stage_candidate_chunk
    ON retrieval_eval_stage_candidate (eval_case_result_id, chunk_id);

COMMENT ON TABLE retrieval_eval_bundle IS 'Immutable first-layer retrieval QA dataset bundle.';
COMMENT ON TABLE retrieval_eval_generation_run IS 'One reproducible random chunk sampling + LLM QA generation attempt for a bundle.';
COMMENT ON TABLE retrieval_eval_generation_window IS 'One context-window-limited LLM batch, including source chunk snapshots and retry metadata.';
COMMENT ON TABLE retrieval_eval_case IS 'One deterministic retrieval QA case; answer_original is an exact source-chunk excerpt.';
COMMENT ON TABLE retrieval_eval_case_gold_chunk IS 'All validated chunks that independently answer a QA, including source and cross-chunk adjudicated gold chunks.';
COMMENT ON TABLE retrieval_eval_stage_candidate IS 'Top-N scored candidates captured per retrieval stage/branch for score-based drop diagnosis and audit.';
COMMENT ON TABLE retrieval_eval_run IS 'One retrieval benchmark execution over a fixed bundle; one query targets one DTD only.';
COMMENT ON TABLE retrieval_eval_case_result IS 'Per-case hit/miss result for an eval run.';
COMMENT ON TABLE retrieval_eval_stage_observation IS 'Compact per-case/per-stage gold-chunk trace; aggregate this for macro drop statistics without persisting all candidates.';


-- ============================================================
-- 增量：已建库怎么办
-- ============================================================
--
-- ⚠️ 上面的 CREATE TABLE 是**新建库**用的。而本仓库没有 Alembic，建表走
-- 启动时的 `Base.metadata.create_all` —— 它**只建缺失的表，不加列、也
-- 不改约束**。所以既有的库必须手写下面这几条。
--
-- 顺序不能变：先加列（带默认值，旧行才填得上），再换约束。

ALTER TABLE retrieval_eval_case
    ADD COLUMN IF NOT EXISTS document_id UUID NULL,
    ADD COLUMN IF NOT EXISTS source_document_title VARCHAR(255) NULL,
    ADD COLUMN IF NOT EXISTS source_document_key VARCHAR(255) NULL;

ALTER TABLE retrieval_eval_case_result
    ADD COLUMN IF NOT EXISTS attempt_no SMALLINT NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS degraded_routes JSONB NOT NULL DEFAULT '[]'::jsonb;

-- 唯一约束改不动，只能删了重建。
ALTER TABLE retrieval_eval_case_result
    DROP CONSTRAINT IF EXISTS uq_retrieval_eval_case_result;

ALTER TABLE retrieval_eval_case_result
    ADD CONSTRAINT uq_retrieval_eval_case_result
    UNIQUE (eval_run_id, eval_case_id, attempt_no);
