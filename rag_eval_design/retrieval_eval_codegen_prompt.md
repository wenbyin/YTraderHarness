# Coding-agent prompt: First-layer Retrieval Eval for Document Service

You are implementing the **first-layer, deterministic retrieval benchmark** in the existing Document Service. Before coding, inspect the actual repository and follow existing module boundaries, SQLAlchemy conventions, migrations, authentication, service/repository patterns, task system, ModelRegistry/ModelRuntime, and OpenAPI layout. Do not create a parallel architecture if the repository already has the relevant abstraction.

## Repository / model constraints

- The supplied architecture snapshot uses PostgreSQL + SQLAlchemy, UUID primary keys, `document_type_definition`, `document`, `document_version`, `document_retrieval_definition`, `document_version_retrieval_binding`, and `document_chunk`.
- DTD has a stable UUID `document_type_definition_id`; `type_key` is a unique business key. `DocumentChunk.document_chunk_id` is UUID and references `document_version_id` plus `document_retrieval_definition_id`.
- `DocumentChunk` is the source-text/canonical chunk record; retrieval types such as graph/vector/keyword may have their own projections. Only sample eligible canonical text chunks from the correct document retrieval definition, following the actual enum/constants in the repository. Do not sample arbitrary graph nodes as if they were canonical text chunks.
- If the current checkout does not contain Document Service/DTD code, stop before adding these tables to an unrelated package. Report the mismatch explicitly; do not graft this feature into an unrelated trading/backtesting engine.
- Keep IDs as UUID throughout API, ORM, and tests. Do not copy older API examples that use numeric IDs if the current entities use UUID.
- DTD selection is explicit and fixed for a bundle. Each generated QA case stores exactly one DTD ID and is executed against that single DTD. Do not implement cross-DTD retrieval or LLM routing in this first-layer feature.
- Respect the established model abstraction. Register/use a dedicated internal `model_role` such as `eval.qa_generation` through the existing ModelRegistry/ModelRuntime; do not instantiate a provider SDK directly from domain/service code, and do not expose arbitrary provider credentials/model endpoints to the frontend.

## Goal

Build an offline QA dataset bundle and a deterministic retrieval-only evaluator. Use the LLM only to prepare the dataset. During evaluation, do not ask an LLM to generate answers or judge relevance. Compute hit/recall/ranking metrics from gold `chunk_id` membership in retrieved IDs.

## Dataset generation behavior

1. `POST /api/v1/retrieval-eval/bundles` creates a bundle and launches a generation run asynchronously using the existing background task mechanism. Return HTTP 202 with `bundle_id` and `generation_run_id`.
2. Request fields:
   - `name`, optional `description`
   - `document_type_definition_ids`: non-empty unique UUID list; only active/authorized DTDs
   - `sampled_chunk_count` (N): total source chunks sampled across the selected DTDs
   - `questions_per_chunk` (M): default 3
   - `chunks_per_window` (L): default 5, meaning max number of source chunks grouped into one LLM input window
   - `max_input_tokens`: configured safe input budget; splitting must obey both `chunks_per_window` and token budget
   - optional deterministic `seed`; create/store one if omitted
   - `sampling_strategy`: use a documented default (prefer balanced sampling across selected DTDs where possible) and persist it
3. Randomly sample eligible source chunks without replacement. Persist the selected chunk IDs, DTD/version/retrieval-definition IDs, source content hash, content snapshot per generation window, seed, and the complete generation config. The same seed/config plus unchanged source set should reproduce the same chunk sample.
4. Chunk windows must be split by **both** L and the estimated input-token budget. Character count alone is not an adequate context-window limit. Keep a safety margin for instructions and expected output. A single chunk exceeding the budget must be skipped with an explicit reason or handled by a deterministic safe policy; never silently truncate source evidence.
5. Call the internal `eval.qa_generation` model role through ModelRuntime. The prompt asks for exactly M *distinct* direct factual questions per chunk, only answerable from that individual chunk. No multi-chunk synthesis, external knowledge, causal inference, implications, prediction, hidden assumptions, or multi-hop reasoning.
6. The model output is strict JSON with one item per QA:
   ```json
   {
     "items": [
       {
         "chunk_id": "<UUID from this input window>",
         "question": "A standalone question directly supported by that chunk.",
         "answer_original": "A verbatim substring copied from that chunk."
       }
     ]
   }
   ```
   The final exported dataset must contain `question`, `answer_original`, `chunk_id` (origin/source chunk) and `gold_chunk_ids` (one or more independently validated chunk IDs that can answer the same question); it may also contain `document_type_definition_id`, `document_version_id`, and metadata needed for reproducibility.
7. Validate every item server-side. Reject unknown chunk IDs, missing/extra schema fields when strict mode is used, blank questions/answers, duplicates, answers not found verbatim in the relevant source text, and questions that are obviously unsupported by the source. Never trust the LLM merely because it returned valid JSON. Store the answer offsets when available. Use a small bounded retry policy for invalid/model-failed windows. Failed windows must be retryable without regenerating already validated QAs or creating duplicates.
8. `answer_original` is the source excerpt, not a generated summary. It must match a substring of the source chunk exactly. Store a content hash and preserve source snapshot in generation-window input. Bundle export should be deterministic and include schema version + dataset fingerprint.
9. The bundle becomes `ready` only when all expected chunks/windows have reached a terminal successful state and validation produced the configured M valid distinct questions per sampled chunk. Otherwise set `partial_failed`/`failed`; expose failures and allow retrying failed windows. Do not silently mark an incomplete bundle ready.
10. Bundle contents are immutable after `ready`. To change sample size, DTD scope, M, L, or the question-generation model role, create a new bundle or generation run version; never mutate cases under an existing completed benchmark bundle.

## QA generation prompt: strict semantic boundaries

Use this policy verbatim in the `eval.qa_generation` model-role prompt. The model must generate questions whose answer is explicitly stated in the individual source chunk. This is a retrieval benchmark, not a reasoning benchmark.

### Allowed question patterns (examples are illustrative)

1. **Explicit entity/property**: source says “The company was founded in 1998.” Ask “In what year was the company founded?” Answer must be the exact text `1998` (or a larger exact source excerpt if needed).
2. **Explicit value/unit**: source says “Revenue reached 12.4 billion yuan in 2024.” Ask “What revenue figure does the chunk report for 2024?”
3. **Explicit named item/list**: source explicitly lists products A, B and C. Ask “Which products are listed in the chunk?” Do not infer omitted list members.
4. **Explicit date/time/location/person/organization**: ask who/when/where only when the answer is literally present.
5. **Explicit stated definition or classification**: source says “X is classified as category Y.” Ask “How does the chunk classify X?” Do not ask for a broader taxonomy.
6. **Explicit comparison**: source directly says “A is higher than B by 3%.” A comparison question is allowed only if both values and the comparison are explicitly written. Do not calculate a new ratio/difference unless that exact computed value is stated.
7. **Explicitly stated cause/effect**: a causal question is allowed only when the chunk itself directly states the cause/effect relation, e.g. “The report attributes the decline to lower demand.” Ask “What reason does the report give for the decline?” Do not infer causality from two adjacent facts.
8. **Explicit negation/limitation/status**: if the chunk says “The trial did not include children under 12,” ask what age group was excluded. Preserve negation and scope exactly.
9. **Explicit procedure/sequence**: ask about an ordered step only when the chunk explicitly describes the step/order. Do not invent missing steps.
10. **Table/structured content**: a question may target a row/column value only if the source snapshot includes its header/context so the value is unambiguous.

### Forbidden question patterns

- **No multi-hop or cross-chunk questions**: the question must be answerable from this one source chunk alone.
- **No unstated causal inference**: do not turn correlation, chronology, co-occurrence or juxtaposition into “why/how caused”.
- **No logical jumps or implicit assumptions**: do not ask what a fact “means”, “implies”, “proves”, “will lead to”, or “suggests” unless that interpretation is explicitly stated.
- **No prediction/recommendation/opinion**: do not ask what will happen next, what should be done, whether something is good/bad, or what strategy follows.
- **No external/world knowledge**: do not fill in common knowledge, definitions, abbreviations, missing units, or background not present in the chunk.
- **No answer synthesis**: do not combine two unrelated statements into a new conclusion.
- **No calculation**: do not calculate totals, percentages, differences, rankings or conversions unless the result itself is explicitly stated in the chunk.
- **No ambiguous referents**: avoid “it/they/this company” if the chunk does not identify the referent clearly within itself.
- **No unsupported presuppositions**: do not ask “When did X stop…” unless the chunk states X stopped.
- **No answer beyond the chunk boundary**: never use another chunk, the DTD description, model memory or external sources to complete the answer.
- **No trivial templating/duplicates**: questions for one chunk must test distinct explicit facts, not merely paraphrase the same fact.

### Answer and evidence constraints

- `answer_original` must be an exact, contiguous substring of the source chunk, including exact punctuation/spacing where feasible. It is not a paraphrase or summary.
- Each generated question must be independently answerable using only its own source chunk.
- Output exactly the requested number of valid, distinct questions per source chunk when possible. If the chunk does not contain enough distinct explicit facts, return fewer and mark the chunk `insufficient_facts`; do not pad with speculative questions. The server must represent this state explicitly instead of silently claiming M questions were generated.
- Every item must include the source `chunk_id`. Never invent IDs.
- Return structured JSON only; no Markdown fences, explanations, or extra keys.

### Cross-chunk alternative-answer adjudication

A reverse-generated question may also be answerable by other chunks. Do not mark another chunk as gold merely because it is topically similar or contains the same entity. After initial QA generation, compare each question against other chunks already sampled into the same bundle (not the entire corpus by default), in bounded LLM windows. Add another chunk to `gold_chunk_ids` only if the candidate chunk independently contains enough explicit evidence to answer the exact question without relying on the source chunk, outside knowledge, or inference. Store the evidence excerpt and judgment. If the chunk is only related, incomplete, ambiguous, or requires combining it with another chunk, do not add it. Always include the originating chunk as a gold ID. State clearly in bundle metadata that alternative gold discovery is limited to the sampled chunk pool; do not claim corpus-wide exhaustive gold labels.

### Examples of acceptable vs unacceptable

Source chunk: “In 2023, Company A reported revenue of 8 billion yuan. The report attributed the 12% year-over-year decline to weaker domestic demand.”

Acceptable:
- “What revenue did Company A report in 2023?” → exact answer excerpt `8 billion yuan` (or source-exact surrounding text).
- “What reason did the report give for the year-over-year decline?” → exact answer excerpt `weaker domestic demand`.
- “What year-over-year decline did the report report?” → exact answer excerpt `12% year-over-year decline`.

Unacceptable:
- “Why did Company A’s revenue decline?” if the source only gives the revenue and says nothing about a reason.
- “Will domestic demand recover in 2024?” (prediction).
- “How much revenue did Company A lose?” (requires calculation unless the loss amount is stated).
- “Does weaker demand prove the company is losing market share?” (unstated inference).
- “How does Company A compare with Company B?” if Company B is not explicitly discussed in the chunk.

## Retrieval evaluation behavior

1. `POST /api/v1/retrieval-eval/runs` accepts `bundle_id`, `max_k` (default 10), `include_stage_trace` (default true), and an optional validated query-processor override only for controls that the existing public retrieval contract allows. Recaller/internal retrieval implementation stays server-controlled. The run stores the effective per-DTD retrieval config snapshot so results remain interpretable after DTD config changes.
2. Evaluate every validated case from the bundle. Execute exactly one retrieval round against the case's own DTD ID; never pass several DTD IDs for an individual case. Do not invoke response-generation LLM or judge LLM.
3. Ensure the retrieval pipeline can return enough candidates for `max_k`. Compute at least Hit@1, Hit@5 (if available), Hit@K, QA-level recall-success rate, MRR, and latency. Since each case has one gold chunk, `gold_hit` means the exact gold `chunk_id` exists in the returned result IDs; `gold_rank` is its 1-based rank or null. Use the same `max_k` cutoff consistently. Do not count a question as a hit merely because a different chunk has similar text.
4. Preserve the distinction between query-level success and chunk sampling coverage. Report total cases, unique gold chunks, hit cases, miss cases, failures, hit rate, MRR, and latency percentiles. Do not claim chunk-level recall unless the denominator/definition is explicit.
5. The existing public retrieval response may only expose final results. Add a typed internal `RetrievalTraceCollector`/trace callback to the pipeline, enabled only for eval/authorized diagnostics. It should observe stage boundaries without changing ranking behavior. Capture counts and every gold chunk's membership/rank/score before and after each stage. Also persist the top 5 candidates entering and leaving each stage/branch (including `chunk_id`, rank-before/rank-after, score-before/score-after, score names/directions, selected/dropped flag and drop reason); for the final response persist at least the top 5 ordered chunks and scores. A candidate rejected by a score threshold must be retained even if it is outside top 5, when the pipeline can expose that candidate and its score. If a gold chunk falls outside top 5 but is dropped by a score threshold, persist that gold candidate too. Keep this bounded; do not persist the full candidate list by default.
6. Record observations for every applicable stage/source branch, using the actual pipeline stages in the repository. Likely stage keys include `recall.dense_vector`, `recall.keyword`, `fusioner`, `filter`, `reranker`, and `cutter`. For each observation store input/output candidate counts; whether each gold chunk was present before/after; rank/score before/after when available; whether the stage executed; and a controlled drop reason such as `not_recalled`, `top_k_cutoff`, `score_threshold`, `metadata_filter`, `rerank_cutoff`, `final_top_k_cutoff`, `deduplicated`, `permission_filter`, `processor_error`, or `unknown`.
7. A stage can be blamed for dropping the gold chunk only when the gold chunk was present before that stage and absent after it. If it was already absent before a later stage, mark the later stage as not applicable for that chunk. Retrieval branches are parallel: being absent from dense retrieval alone is a branch miss, not a final failure if keyword retrieval found the gold chunk.
8. Generate aggregate stage statistics from the compact per-case stage observations: by stage/source, number observed, gold present before, gold retained after, gold dropped at this stage, drop-reason counts, average input/output candidate count, and aggregate transition/pass rates. Store aggregate metrics in the run for quick UI loading, while retaining result/stage rows as source of truth.
9. A trace collection failure must not silently turn a retrieval result into a miss. Record eval execution failure separately from gold miss. The evaluation path must not alter query order/scoring or add LLM calls.

## Tables / migrations

Implement the SQLAlchemy models and Alembic migration equivalent of `retrieval_eval_schema.sql` (provided alongside this prompt), adapting naming/import conventions to the actual repository:

- `retrieval_eval_bundle`: bundle metadata, status, selection/generation config, count/hash snapshot.
- `retrieval_eval_generation_run`: reproducible sampling/generation attempt and counters.
- `retrieval_eval_generation_window`: context-window request snapshot, input chunk snapshots, raw JSON output, attempt/error/status.
- `retrieval_eval_case`: one validated QA, gold chunk UUID, source DTD/version/retrieval-definition IDs, original answer excerpt, content hash, question hash/order.
- `retrieval_eval_run`: one evaluation over one immutable bundle, top-K, effective configuration snapshot, cached aggregate metrics, status.
- `retrieval_eval_case_result`: per-question result, final retrieved IDs, gold hit/rank, latency/error.
- `retrieval_eval_case_gold_chunk`: all independently validated gold chunk IDs per question; originating chunk is always included.
- `retrieval_eval_stage_observation`: compact per-question/per-stage gold membership, score and drop explanation.
- `retrieval_eval_stage_candidate`: top-5 scored candidates entering/leaving each stage/branch, with before/after rank and score, plus candidates explicitly rejected by a score threshold when observable; final top-5 is always retained.

Source chunk/DTD/version IDs copied into an immutable bundle are evidence snapshots, not ownership relationships. Avoid cascading deletion from documents into historical Eval bundles. If your repository has a standard snapshot/reference policy, follow it while preserving stable gold IDs.

## API contract

Implement `retrieval_eval_openapi.yaml` as the authoritative OpenAPI 3.1 contract (provided alongside this prompt), then make routes/DTOs conform to it. Required operations:

- `POST /retrieval-eval/bundles` — create bundle + start QA-generation run (202)
- `GET /retrieval-eval/bundles` — list/filter bundles for selector UI
- `GET /retrieval-eval/bundles/{bundle_id}` — bundle details, generation status, counts
- `GET /retrieval-eval/bundles/{bundle_id}/cases` — paginated QA preview
- `GET /retrieval-eval/bundles/{bundle_id}/export` — deterministic JSON export
- `GET /retrieval-eval/generation-runs/{generation_run_id}` — window/case generation progress
- `POST /retrieval-eval/generation-runs/{generation_run_id}/retry-failed-windows` — retry recoverable windows idempotently
- `POST /retrieval-eval/runs` — start first-layer retrieval run (202)
- `GET /retrieval-eval/runs/{eval_run_id}` — status + metrics
- `GET /retrieval-eval/runs/{eval_run_id}/stage-stats` — aggregate chunk-drop statistics
- `GET /retrieval-eval/runs/{eval_run_id}/cases` — paginated case-level hit/miss and gold rank

Use the repository's existing auth/error envelope and tenant/permission enforcement. Never let the client choose a DTD the caller cannot access. Validate that each bundle case has one DTD and that the evaluator makes one-DTD calls.

## Tests required

- Fixed random seed gives the same selected chunk IDs when source rows are unchanged.
- DTD scopes are validated; each case calls retrieval with exactly one DTD.
- Token-budget and chunks-per-window splitting both work; oversized individual chunks are handled explicitly.
- JSON schema parsing; unknown IDs; missing QA; answer not a verbatim substring; duplicate questions; valid exact source quote.
- Partial window failure and retry are idempotent and do not duplicate cases.
- Bundle cannot be evaluated until ready; a ready bundle's cases/config are immutable.
- Hit@K/gold rank/MRR calculations, failed-query separation, empty retrieval results.
- Stage trace does not affect retrieval results; drop attribution only occurs for present-before/absent-after transitions.
- Candidate trace preserves stage-specific raw score/rank for top 5, plus gold candidates removed by score thresholds; never compare scores across stages without score semantics.
- Multi-gold cases count as hit when any validated gold chunk is returned; best gold rank drives MRR.
- Parallel retriever branch miss is not falsely attributed as final miss when another branch finds the gold chunk.
- Aggregate stage statistics match hand-calculated fixtures.
- OpenAPI schema validation, API authorization, paging, and error cases.
- Run lint and the repository's documented test suite; report commands and results honestly. Do not claim tests passed unless they ran.

## Deliverables

1. Models + migration; 2. DTOs/schemas + service/repository/routes; 3. registered `eval.qa_generation` role and strict prompt; 4. trace collector and deterministic metrics; 5. OpenAPI alignment; 6. tests; 7. a short README documenting sampling, JSON output, statuses, metrics, trace drop attribution, and limitations.

Do not implement Ragas LLM judging, generated final answers, cross-DTD agentic routing, multi-hop QA, or document-service-wide trace UI as part of this task. Those are explicitly out of scope for Layer 1.
