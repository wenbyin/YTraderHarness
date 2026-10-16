-- ============================================================================
-- 事件大类 DTD Seed
--
-- 把原 COMPANY_EVENTS（上市公司事件）按 7 个 event_family 拆成 7 个 DTD，
-- 每个 DTD 复制 COMPANY_EVENTS 的完整配置簇：
--
--     document_type_definition             x1
--     document_processor_definition        x1   (.md / TEXT_TXT_COMMON_PARSER + CLEANER)
--     document_retrieval_definition        x2   (keyword->ES, dense_vector->Milvus)
--     document_processor_retrieval_binding x2
--     ---------------------------------------------------------------
--     合计 x7 = 42 行
--
-- 配置来源：直接取自库中 COMPANY_EVENTS 的 processor_config / chunker_config，
-- 字面一致，未做任何改写。
--
-- 分类映射（2026-10-16 修正）：以 main.EVENT_FAMILIES 导出的
-- 《大事提醒事件分类映射表》为准 —— 7 个大类共 **45 个**子类
-- （旧 COMPANY_EVENTS 的 description 只列了 28 个，不全；
--   本文件上一版按口述只写了 30 个，漏 15 个，已补齐）。
--
-- 本文件里的族序号 f=1..7 与映射表中**固定不变**的「输出文件序号 01~07」一一对应：
--
--     f=1  investor_relations      投资者关系          2 子类
--     f=2  financial_reporting     财务业绩与信息披露  3 子类
--     f=3  shareholder_governance  股东结构与公司治理 14 子类
--     f=4  capital_operations      融资与资本运作      9 子类
--     f=5  market_activity         市场交易与资金流向  5 子类
--     f=6  business_development    经营活动与项目进展  3 子类
--     f=7  risk_compliance         风险、法律与上市状态 9 子类
--                                                       ----
--                                                       45
--
-- 每个 DTD 的 description 列出该大类的全部子类，顺序 = 映射表里的**条数降序**。
--
-- 特性：
--   * 幂等 —— 全部 ON CONFLICT DO NOTHING，重跑安全。
--   * 子表按 type_key 反查父 DTD，而不是硬编码父 UUID；
--     因此即使某个 DTD 已由别的途径（如 API）建过，子表也会挂到既有的那条上。
--   * 整体在一个事务内，中途失败全部回滚，不会留下半套配置。
--
-- 不包含：document_index_binding（该表当前为空，检索侧不依赖显式绑定）。
-- 不动：原 COMPANY_EVENTS 及其 101 篇 document。
-- ============================================================================

BEGIN;

-- ============================================================================
-- 1/7  COMPANY_EVENT_INVESTOR_RELATIONS
--       上市公司事件-投资者关系
--       子分类（2）：机构调研 / 投资互动
-- ============================================================================

-- 1.1 DocumentTypeDefinition
INSERT INTO document_type_definition (
    document_type_definition_id, tenant_id, type_key, name, description,
    modality_type, analysis_mode, revision, allow_update, version_mode,
    fusion_strategy, fusion_config, rerank_strategy, rerank_config,
    metadata, status
) VALUES (
    'ef000001-0001-4000-8000-000000000001', NULL, 'COMPANY_EVENT_INVESTOR_RELATIONS', '上市公司事件-投资者关系',
    $desc$机构调研
投资互动$desc$,
    'text', 'config', 1, TRUE, 'versioned',
    NULL, '{}'::jsonb, NULL, '{}'::jsonb,
    '{}'::jsonb, 'active'
)
ON CONFLICT (type_key) DO NOTHING;

-- 1.2 DocumentProcessorDefinition (.md)
INSERT INTO document_processor_definition (
    document_processor_definition_id, document_type_definition_id,
    file_extension, mime_types, loader_name, loader_config,
    processor_name, processor_config, status, created_by
)
SELECT
    'ef000001-0001-4000-8000-000000000002', d.document_type_definition_id,
    '.md', '[]'::jsonb, 'TEXT_TXT_COMMON_PARSER', '{}'::jsonb,
    'TEXT_TXT_COMMON_CLEANER', $cleaner${"separator": {"enabled": true, "pattern": "^([-=_*~#·•])\\1*$", "min_length": 3}, "watermark": {"enabled": true, "patterns": ["草稿", "机密", "内部资料", "样本", "复印无效", "禁止外传", "confidential", "draft", "sample", "internal use only"]}, "line_number": {"enabled": false, "pattern": "^\\s*\\d{1,4}[.、)）]\\s*"}, "ocr_garbage": {"enabled": true, "pattern": "^[\\W_]{3,}$", "min_length": 4, "normal_punct": "，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-", "symbol_ratio": 0.6, "garbage_chars": "�□■◆●○◇¤§¶†‡※…", "bad_char_ratio": 0.3}, "page_number": {"enabled": true, "patterns": ["^\\s*[-—–\\[\\(【]*\\s*\\d{1,4}\\s*[-—–\\]\\)】]*\\s*$", "^\\s*第\\s*\\d+\\s*页(\\s*[/共]\\s*\\d+\\s*页)?\\s*$", "(?i)^\\s*Page\\s+\\d+(\\s+of\\s+\\d+)?\\s*$", "(?i)^\\s*[ivxlcdm]{2,7}\\s*$"]}, "table_separator": {"enabled": true, "pattern": "^\\s*[+|][\\s\\-+|:=]{3,}[+|]\\s*$"}, "repeated_boilerplate": {"enabled": true, "min_count": 3, "max_length": 80}}$cleaner$::jsonb,
    'active', NULL
FROM document_type_definition d
WHERE d.type_key = 'COMPANY_EVENT_INVESTOR_RELATIONS'
ON CONFLICT (document_type_definition_id, file_extension) DO NOTHING;

-- 1.3 DocumentRetrievalDefinition (keyword -> Elasticsearch / dense_vector -> Milvus)
INSERT INTO document_retrieval_definition (
    document_retrieval_definition_id, document_type_definition_id,
    retrieval_type, chunker_name, chunker_config,
    retriever_name, retriever_config, storage_backend, storage_config,
    status, created_by
)
SELECT
    v.rid, d.document_type_definition_id,
    v.retrieval_type, 'TEXT_TXT_RECURSIVE_CHUNKER', $chunker${"chunk_size": 512, "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ",", " "], "chunk_overlap": 64, "keep_separator": true, "min_chunk_size": 0, "length_function": "char", "strip_whitespace": true, "is_separator_regex": false}$chunker$::jsonb,
    v.retriever_name, '{}'::jsonb, v.storage_backend, '{}'::jsonb,
    'active', NULL
FROM document_type_definition d
CROSS JOIN (VALUES
    ('ef000001-0001-4000-8000-000000000003'::uuid, 'keyword',      'bm25_retriever',         'elasticsearch'),
    ('ef000001-0001-4000-8000-000000000004'::uuid, 'dense_vector', 'dense_vector_retriever', 'milvus')
) AS v(rid, retrieval_type, retriever_name, storage_backend)
WHERE d.type_key = 'COMPANY_EVENT_INVESTOR_RELATIONS'
ON CONFLICT (document_type_definition_id, retrieval_type) DO NOTHING;

-- 1.4 DocumentProcessorRetrievalBinding (processor -> 每条 retrieval)
INSERT INTO document_processor_retrieval_binding (
    document_processor_retrieval_binding_id,
    document_processor_definition_id,
    document_retrieval_definition_id
)
SELECT b.bid, p.document_processor_definition_id, r.document_retrieval_definition_id
FROM document_type_definition d
JOIN document_processor_definition p
  ON p.document_type_definition_id = d.document_type_definition_id
 AND p.file_extension = '.md'
JOIN document_retrieval_definition r
  ON r.document_type_definition_id = d.document_type_definition_id
JOIN (VALUES
    ('ef000001-0001-4000-8000-000000000005'::uuid, 'keyword'),
    ('ef000001-0001-4000-8000-000000000006'::uuid, 'dense_vector')
) AS b(bid, retrieval_type)
  ON b.retrieval_type = r.retrieval_type
WHERE d.type_key = 'COMPANY_EVENT_INVESTOR_RELATIONS'
ON CONFLICT ON CONSTRAINT uq_document_processor_retrieval_binding DO NOTHING;

-- ============================================================================
-- 2/7  COMPANY_EVENT_FINANCIAL_REPORTING
--       上市公司事件-财务业绩与信息披露
--       子分类（3）：报表披露 / 业绩预告 / 业绩快报
-- ============================================================================

-- 2.1 DocumentTypeDefinition
INSERT INTO document_type_definition (
    document_type_definition_id, tenant_id, type_key, name, description,
    modality_type, analysis_mode, revision, allow_update, version_mode,
    fusion_strategy, fusion_config, rerank_strategy, rerank_config,
    metadata, status
) VALUES (
    'ef000002-0001-4000-8000-000000000001', NULL, 'COMPANY_EVENT_FINANCIAL_REPORTING', '上市公司事件-财务业绩与信息披露',
    $desc$报表披露
业绩预告
业绩快报$desc$,
    'text', 'config', 1, TRUE, 'versioned',
    NULL, '{}'::jsonb, NULL, '{}'::jsonb,
    '{}'::jsonb, 'active'
)
ON CONFLICT (type_key) DO NOTHING;

-- 2.2 DocumentProcessorDefinition (.md)
INSERT INTO document_processor_definition (
    document_processor_definition_id, document_type_definition_id,
    file_extension, mime_types, loader_name, loader_config,
    processor_name, processor_config, status, created_by
)
SELECT
    'ef000002-0001-4000-8000-000000000002', d.document_type_definition_id,
    '.md', '[]'::jsonb, 'TEXT_TXT_COMMON_PARSER', '{}'::jsonb,
    'TEXT_TXT_COMMON_CLEANER', $cleaner${"separator": {"enabled": true, "pattern": "^([-=_*~#·•])\\1*$", "min_length": 3}, "watermark": {"enabled": true, "patterns": ["草稿", "机密", "内部资料", "样本", "复印无效", "禁止外传", "confidential", "draft", "sample", "internal use only"]}, "line_number": {"enabled": false, "pattern": "^\\s*\\d{1,4}[.、)）]\\s*"}, "ocr_garbage": {"enabled": true, "pattern": "^[\\W_]{3,}$", "min_length": 4, "normal_punct": "，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-", "symbol_ratio": 0.6, "garbage_chars": "�□■◆●○◇¤§¶†‡※…", "bad_char_ratio": 0.3}, "page_number": {"enabled": true, "patterns": ["^\\s*[-—–\\[\\(【]*\\s*\\d{1,4}\\s*[-—–\\]\\)】]*\\s*$", "^\\s*第\\s*\\d+\\s*页(\\s*[/共]\\s*\\d+\\s*页)?\\s*$", "(?i)^\\s*Page\\s+\\d+(\\s+of\\s+\\d+)?\\s*$", "(?i)^\\s*[ivxlcdm]{2,7}\\s*$"]}, "table_separator": {"enabled": true, "pattern": "^\\s*[+|][\\s\\-+|:=]{3,}[+|]\\s*$"}, "repeated_boilerplate": {"enabled": true, "min_count": 3, "max_length": 80}}$cleaner$::jsonb,
    'active', NULL
FROM document_type_definition d
WHERE d.type_key = 'COMPANY_EVENT_FINANCIAL_REPORTING'
ON CONFLICT (document_type_definition_id, file_extension) DO NOTHING;

-- 2.3 DocumentRetrievalDefinition (keyword -> Elasticsearch / dense_vector -> Milvus)
INSERT INTO document_retrieval_definition (
    document_retrieval_definition_id, document_type_definition_id,
    retrieval_type, chunker_name, chunker_config,
    retriever_name, retriever_config, storage_backend, storage_config,
    status, created_by
)
SELECT
    v.rid, d.document_type_definition_id,
    v.retrieval_type, 'TEXT_TXT_RECURSIVE_CHUNKER', $chunker${"chunk_size": 512, "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ",", " "], "chunk_overlap": 64, "keep_separator": true, "min_chunk_size": 0, "length_function": "char", "strip_whitespace": true, "is_separator_regex": false}$chunker$::jsonb,
    v.retriever_name, '{}'::jsonb, v.storage_backend, '{}'::jsonb,
    'active', NULL
FROM document_type_definition d
CROSS JOIN (VALUES
    ('ef000002-0001-4000-8000-000000000003'::uuid, 'keyword',      'bm25_retriever',         'elasticsearch'),
    ('ef000002-0001-4000-8000-000000000004'::uuid, 'dense_vector', 'dense_vector_retriever', 'milvus')
) AS v(rid, retrieval_type, retriever_name, storage_backend)
WHERE d.type_key = 'COMPANY_EVENT_FINANCIAL_REPORTING'
ON CONFLICT (document_type_definition_id, retrieval_type) DO NOTHING;

-- 2.4 DocumentProcessorRetrievalBinding (processor -> 每条 retrieval)
INSERT INTO document_processor_retrieval_binding (
    document_processor_retrieval_binding_id,
    document_processor_definition_id,
    document_retrieval_definition_id
)
SELECT b.bid, p.document_processor_definition_id, r.document_retrieval_definition_id
FROM document_type_definition d
JOIN document_processor_definition p
  ON p.document_type_definition_id = d.document_type_definition_id
 AND p.file_extension = '.md'
JOIN document_retrieval_definition r
  ON r.document_type_definition_id = d.document_type_definition_id
JOIN (VALUES
    ('ef000002-0001-4000-8000-000000000005'::uuid, 'keyword'),
    ('ef000002-0001-4000-8000-000000000006'::uuid, 'dense_vector')
) AS b(bid, retrieval_type)
  ON b.retrieval_type = r.retrieval_type
WHERE d.type_key = 'COMPANY_EVENT_FINANCIAL_REPORTING'
ON CONFLICT ON CONSTRAINT uq_document_processor_retrieval_binding DO NOTHING;

-- ============================================================================
-- 3/7  COMPANY_EVENT_SHAREHOLDER_GOVERNANCE
--       上市公司事件-股东结构与公司治理
--       子分类（14）：股东户数 / 股东大会 / 高管及关联方增减持 / 股东增减持 / 解除质押 / 限售解禁 / 股权质押 / 总经理变更 / 法定代表人变更 / 名称变动 / 董事长变更 / 股权转让 / 增减持计划 / 代码变更
-- ============================================================================

-- 3.1 DocumentTypeDefinition
INSERT INTO document_type_definition (
    document_type_definition_id, tenant_id, type_key, name, description,
    modality_type, analysis_mode, revision, allow_update, version_mode,
    fusion_strategy, fusion_config, rerank_strategy, rerank_config,
    metadata, status
) VALUES (
    'ef000003-0001-4000-8000-000000000001', NULL, 'COMPANY_EVENT_SHAREHOLDER_GOVERNANCE', '上市公司事件-股东结构与公司治理',
    $desc$股东户数
股东大会
高管及关联方增减持
股东增减持
解除质押
限售解禁
股权质押
总经理变更
法定代表人变更
名称变动
董事长变更
股权转让
增减持计划
代码变更$desc$,
    'text', 'config', 1, TRUE, 'versioned',
    NULL, '{}'::jsonb, NULL, '{}'::jsonb,
    '{}'::jsonb, 'active'
)
ON CONFLICT (type_key) DO NOTHING;

-- 3.2 DocumentProcessorDefinition (.md)
INSERT INTO document_processor_definition (
    document_processor_definition_id, document_type_definition_id,
    file_extension, mime_types, loader_name, loader_config,
    processor_name, processor_config, status, created_by
)
SELECT
    'ef000003-0001-4000-8000-000000000002', d.document_type_definition_id,
    '.md', '[]'::jsonb, 'TEXT_TXT_COMMON_PARSER', '{}'::jsonb,
    'TEXT_TXT_COMMON_CLEANER', $cleaner${"separator": {"enabled": true, "pattern": "^([-=_*~#·•])\\1*$", "min_length": 3}, "watermark": {"enabled": true, "patterns": ["草稿", "机密", "内部资料", "样本", "复印无效", "禁止外传", "confidential", "draft", "sample", "internal use only"]}, "line_number": {"enabled": false, "pattern": "^\\s*\\d{1,4}[.、)）]\\s*"}, "ocr_garbage": {"enabled": true, "pattern": "^[\\W_]{3,}$", "min_length": 4, "normal_punct": "，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-", "symbol_ratio": 0.6, "garbage_chars": "�□■◆●○◇¤§¶†‡※…", "bad_char_ratio": 0.3}, "page_number": {"enabled": true, "patterns": ["^\\s*[-—–\\[\\(【]*\\s*\\d{1,4}\\s*[-—–\\]\\)】]*\\s*$", "^\\s*第\\s*\\d+\\s*页(\\s*[/共]\\s*\\d+\\s*页)?\\s*$", "(?i)^\\s*Page\\s+\\d+(\\s+of\\s+\\d+)?\\s*$", "(?i)^\\s*[ivxlcdm]{2,7}\\s*$"]}, "table_separator": {"enabled": true, "pattern": "^\\s*[+|][\\s\\-+|:=]{3,}[+|]\\s*$"}, "repeated_boilerplate": {"enabled": true, "min_count": 3, "max_length": 80}}$cleaner$::jsonb,
    'active', NULL
FROM document_type_definition d
WHERE d.type_key = 'COMPANY_EVENT_SHAREHOLDER_GOVERNANCE'
ON CONFLICT (document_type_definition_id, file_extension) DO NOTHING;

-- 3.3 DocumentRetrievalDefinition (keyword -> Elasticsearch / dense_vector -> Milvus)
INSERT INTO document_retrieval_definition (
    document_retrieval_definition_id, document_type_definition_id,
    retrieval_type, chunker_name, chunker_config,
    retriever_name, retriever_config, storage_backend, storage_config,
    status, created_by
)
SELECT
    v.rid, d.document_type_definition_id,
    v.retrieval_type, 'TEXT_TXT_RECURSIVE_CHUNKER', $chunker${"chunk_size": 512, "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ",", " "], "chunk_overlap": 64, "keep_separator": true, "min_chunk_size": 0, "length_function": "char", "strip_whitespace": true, "is_separator_regex": false}$chunker$::jsonb,
    v.retriever_name, '{}'::jsonb, v.storage_backend, '{}'::jsonb,
    'active', NULL
FROM document_type_definition d
CROSS JOIN (VALUES
    ('ef000003-0001-4000-8000-000000000003'::uuid, 'keyword',      'bm25_retriever',         'elasticsearch'),
    ('ef000003-0001-4000-8000-000000000004'::uuid, 'dense_vector', 'dense_vector_retriever', 'milvus')
) AS v(rid, retrieval_type, retriever_name, storage_backend)
WHERE d.type_key = 'COMPANY_EVENT_SHAREHOLDER_GOVERNANCE'
ON CONFLICT (document_type_definition_id, retrieval_type) DO NOTHING;

-- 3.4 DocumentProcessorRetrievalBinding (processor -> 每条 retrieval)
INSERT INTO document_processor_retrieval_binding (
    document_processor_retrieval_binding_id,
    document_processor_definition_id,
    document_retrieval_definition_id
)
SELECT b.bid, p.document_processor_definition_id, r.document_retrieval_definition_id
FROM document_type_definition d
JOIN document_processor_definition p
  ON p.document_type_definition_id = d.document_type_definition_id
 AND p.file_extension = '.md'
JOIN document_retrieval_definition r
  ON r.document_type_definition_id = d.document_type_definition_id
JOIN (VALUES
    ('ef000003-0001-4000-8000-000000000005'::uuid, 'keyword'),
    ('ef000003-0001-4000-8000-000000000006'::uuid, 'dense_vector')
) AS b(bid, retrieval_type)
  ON b.retrieval_type = r.retrieval_type
WHERE d.type_key = 'COMPANY_EVENT_SHAREHOLDER_GOVERNANCE'
ON CONFLICT ON CONSTRAINT uq_document_processor_retrieval_binding DO NOTHING;

-- ============================================================================
-- 4/7  COMPANY_EVENT_CAPITAL_OPERATIONS
--       上市公司事件-融资与资本运作
--       子分类（9）：资本运作 / 分红送转 / 股票回购 / 股权激励 / 增发 / 申购提示 / 可转债 / 配股 / 吸收合并
-- ============================================================================

-- 4.1 DocumentTypeDefinition
INSERT INTO document_type_definition (
    document_type_definition_id, tenant_id, type_key, name, description,
    modality_type, analysis_mode, revision, allow_update, version_mode,
    fusion_strategy, fusion_config, rerank_strategy, rerank_config,
    metadata, status
) VALUES (
    'ef000004-0001-4000-8000-000000000001', NULL, 'COMPANY_EVENT_CAPITAL_OPERATIONS', '上市公司事件-融资与资本运作',
    $desc$资本运作
分红送转
股票回购
股权激励
增发
申购提示
可转债
配股
吸收合并$desc$,
    'text', 'config', 1, TRUE, 'versioned',
    NULL, '{}'::jsonb, NULL, '{}'::jsonb,
    '{}'::jsonb, 'active'
)
ON CONFLICT (type_key) DO NOTHING;

-- 4.2 DocumentProcessorDefinition (.md)
INSERT INTO document_processor_definition (
    document_processor_definition_id, document_type_definition_id,
    file_extension, mime_types, loader_name, loader_config,
    processor_name, processor_config, status, created_by
)
SELECT
    'ef000004-0001-4000-8000-000000000002', d.document_type_definition_id,
    '.md', '[]'::jsonb, 'TEXT_TXT_COMMON_PARSER', '{}'::jsonb,
    'TEXT_TXT_COMMON_CLEANER', $cleaner${"separator": {"enabled": true, "pattern": "^([-=_*~#·•])\\1*$", "min_length": 3}, "watermark": {"enabled": true, "patterns": ["草稿", "机密", "内部资料", "样本", "复印无效", "禁止外传", "confidential", "draft", "sample", "internal use only"]}, "line_number": {"enabled": false, "pattern": "^\\s*\\d{1,4}[.、)）]\\s*"}, "ocr_garbage": {"enabled": true, "pattern": "^[\\W_]{3,}$", "min_length": 4, "normal_punct": "，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-", "symbol_ratio": 0.6, "garbage_chars": "�□■◆●○◇¤§¶†‡※…", "bad_char_ratio": 0.3}, "page_number": {"enabled": true, "patterns": ["^\\s*[-—–\\[\\(【]*\\s*\\d{1,4}\\s*[-—–\\]\\)】]*\\s*$", "^\\s*第\\s*\\d+\\s*页(\\s*[/共]\\s*\\d+\\s*页)?\\s*$", "(?i)^\\s*Page\\s+\\d+(\\s+of\\s+\\d+)?\\s*$", "(?i)^\\s*[ivxlcdm]{2,7}\\s*$"]}, "table_separator": {"enabled": true, "pattern": "^\\s*[+|][\\s\\-+|:=]{3,}[+|]\\s*$"}, "repeated_boilerplate": {"enabled": true, "min_count": 3, "max_length": 80}}$cleaner$::jsonb,
    'active', NULL
FROM document_type_definition d
WHERE d.type_key = 'COMPANY_EVENT_CAPITAL_OPERATIONS'
ON CONFLICT (document_type_definition_id, file_extension) DO NOTHING;

-- 4.3 DocumentRetrievalDefinition (keyword -> Elasticsearch / dense_vector -> Milvus)
INSERT INTO document_retrieval_definition (
    document_retrieval_definition_id, document_type_definition_id,
    retrieval_type, chunker_name, chunker_config,
    retriever_name, retriever_config, storage_backend, storage_config,
    status, created_by
)
SELECT
    v.rid, d.document_type_definition_id,
    v.retrieval_type, 'TEXT_TXT_RECURSIVE_CHUNKER', $chunker${"chunk_size": 512, "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ",", " "], "chunk_overlap": 64, "keep_separator": true, "min_chunk_size": 0, "length_function": "char", "strip_whitespace": true, "is_separator_regex": false}$chunker$::jsonb,
    v.retriever_name, '{}'::jsonb, v.storage_backend, '{}'::jsonb,
    'active', NULL
FROM document_type_definition d
CROSS JOIN (VALUES
    ('ef000004-0001-4000-8000-000000000003'::uuid, 'keyword',      'bm25_retriever',         'elasticsearch'),
    ('ef000004-0001-4000-8000-000000000004'::uuid, 'dense_vector', 'dense_vector_retriever', 'milvus')
) AS v(rid, retrieval_type, retriever_name, storage_backend)
WHERE d.type_key = 'COMPANY_EVENT_CAPITAL_OPERATIONS'
ON CONFLICT (document_type_definition_id, retrieval_type) DO NOTHING;

-- 4.4 DocumentProcessorRetrievalBinding (processor -> 每条 retrieval)
INSERT INTO document_processor_retrieval_binding (
    document_processor_retrieval_binding_id,
    document_processor_definition_id,
    document_retrieval_definition_id
)
SELECT b.bid, p.document_processor_definition_id, r.document_retrieval_definition_id
FROM document_type_definition d
JOIN document_processor_definition p
  ON p.document_type_definition_id = d.document_type_definition_id
 AND p.file_extension = '.md'
JOIN document_retrieval_definition r
  ON r.document_type_definition_id = d.document_type_definition_id
JOIN (VALUES
    ('ef000004-0001-4000-8000-000000000005'::uuid, 'keyword'),
    ('ef000004-0001-4000-8000-000000000006'::uuid, 'dense_vector')
) AS b(bid, retrieval_type)
  ON b.retrieval_type = r.retrieval_type
WHERE d.type_key = 'COMPANY_EVENT_CAPITAL_OPERATIONS'
ON CONFLICT ON CONSTRAINT uq_document_processor_retrieval_binding DO NOTHING;

-- ============================================================================
-- 5/7  COMPANY_EVENT_MARKET_ACTIVITY
--       上市公司事件-市场交易与资金流向
--       子分类（5）：新增概念 / 龙虎榜 / 大宗交易 / 融资融券 / 沪深港通
-- ============================================================================

-- 5.1 DocumentTypeDefinition
INSERT INTO document_type_definition (
    document_type_definition_id, tenant_id, type_key, name, description,
    modality_type, analysis_mode, revision, allow_update, version_mode,
    fusion_strategy, fusion_config, rerank_strategy, rerank_config,
    metadata, status
) VALUES (
    'ef000005-0001-4000-8000-000000000001', NULL, 'COMPANY_EVENT_MARKET_ACTIVITY', '上市公司事件-市场交易与资金流向',
    $desc$新增概念
龙虎榜
大宗交易
融资融券
沪深港通$desc$,
    'text', 'config', 1, TRUE, 'versioned',
    NULL, '{}'::jsonb, NULL, '{}'::jsonb,
    '{}'::jsonb, 'active'
)
ON CONFLICT (type_key) DO NOTHING;

-- 5.2 DocumentProcessorDefinition (.md)
INSERT INTO document_processor_definition (
    document_processor_definition_id, document_type_definition_id,
    file_extension, mime_types, loader_name, loader_config,
    processor_name, processor_config, status, created_by
)
SELECT
    'ef000005-0001-4000-8000-000000000002', d.document_type_definition_id,
    '.md', '[]'::jsonb, 'TEXT_TXT_COMMON_PARSER', '{}'::jsonb,
    'TEXT_TXT_COMMON_CLEANER', $cleaner${"separator": {"enabled": true, "pattern": "^([-=_*~#·•])\\1*$", "min_length": 3}, "watermark": {"enabled": true, "patterns": ["草稿", "机密", "内部资料", "样本", "复印无效", "禁止外传", "confidential", "draft", "sample", "internal use only"]}, "line_number": {"enabled": false, "pattern": "^\\s*\\d{1,4}[.、)）]\\s*"}, "ocr_garbage": {"enabled": true, "pattern": "^[\\W_]{3,}$", "min_length": 4, "normal_punct": "，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-", "symbol_ratio": 0.6, "garbage_chars": "�□■◆●○◇¤§¶†‡※…", "bad_char_ratio": 0.3}, "page_number": {"enabled": true, "patterns": ["^\\s*[-—–\\[\\(【]*\\s*\\d{1,4}\\s*[-—–\\]\\)】]*\\s*$", "^\\s*第\\s*\\d+\\s*页(\\s*[/共]\\s*\\d+\\s*页)?\\s*$", "(?i)^\\s*Page\\s+\\d+(\\s+of\\s+\\d+)?\\s*$", "(?i)^\\s*[ivxlcdm]{2,7}\\s*$"]}, "table_separator": {"enabled": true, "pattern": "^\\s*[+|][\\s\\-+|:=]{3,}[+|]\\s*$"}, "repeated_boilerplate": {"enabled": true, "min_count": 3, "max_length": 80}}$cleaner$::jsonb,
    'active', NULL
FROM document_type_definition d
WHERE d.type_key = 'COMPANY_EVENT_MARKET_ACTIVITY'
ON CONFLICT (document_type_definition_id, file_extension) DO NOTHING;

-- 5.3 DocumentRetrievalDefinition (keyword -> Elasticsearch / dense_vector -> Milvus)
INSERT INTO document_retrieval_definition (
    document_retrieval_definition_id, document_type_definition_id,
    retrieval_type, chunker_name, chunker_config,
    retriever_name, retriever_config, storage_backend, storage_config,
    status, created_by
)
SELECT
    v.rid, d.document_type_definition_id,
    v.retrieval_type, 'TEXT_TXT_RECURSIVE_CHUNKER', $chunker${"chunk_size": 512, "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ",", " "], "chunk_overlap": 64, "keep_separator": true, "min_chunk_size": 0, "length_function": "char", "strip_whitespace": true, "is_separator_regex": false}$chunker$::jsonb,
    v.retriever_name, '{}'::jsonb, v.storage_backend, '{}'::jsonb,
    'active', NULL
FROM document_type_definition d
CROSS JOIN (VALUES
    ('ef000005-0001-4000-8000-000000000003'::uuid, 'keyword',      'bm25_retriever',         'elasticsearch'),
    ('ef000005-0001-4000-8000-000000000004'::uuid, 'dense_vector', 'dense_vector_retriever', 'milvus')
) AS v(rid, retrieval_type, retriever_name, storage_backend)
WHERE d.type_key = 'COMPANY_EVENT_MARKET_ACTIVITY'
ON CONFLICT (document_type_definition_id, retrieval_type) DO NOTHING;

-- 5.4 DocumentProcessorRetrievalBinding (processor -> 每条 retrieval)
INSERT INTO document_processor_retrieval_binding (
    document_processor_retrieval_binding_id,
    document_processor_definition_id,
    document_retrieval_definition_id
)
SELECT b.bid, p.document_processor_definition_id, r.document_retrieval_definition_id
FROM document_type_definition d
JOIN document_processor_definition p
  ON p.document_type_definition_id = d.document_type_definition_id
 AND p.file_extension = '.md'
JOIN document_retrieval_definition r
  ON r.document_type_definition_id = d.document_type_definition_id
JOIN (VALUES
    ('ef000005-0001-4000-8000-000000000005'::uuid, 'keyword'),
    ('ef000005-0001-4000-8000-000000000006'::uuid, 'dense_vector')
) AS b(bid, retrieval_type)
  ON b.retrieval_type = r.retrieval_type
WHERE d.type_key = 'COMPANY_EVENT_MARKET_ACTIVITY'
ON CONFLICT ON CONSTRAINT uq_document_processor_retrieval_binding DO NOTHING;

-- ============================================================================
-- 6/7  COMPANY_EVENT_BUSINESS_DEVELOPMENT
--       上市公司事件-经营活动与项目进展
--       子分类（3）：项目投资 / 项目中标 / 重要合同
-- ============================================================================

-- 6.1 DocumentTypeDefinition
INSERT INTO document_type_definition (
    document_type_definition_id, tenant_id, type_key, name, description,
    modality_type, analysis_mode, revision, allow_update, version_mode,
    fusion_strategy, fusion_config, rerank_strategy, rerank_config,
    metadata, status
) VALUES (
    'ef000006-0001-4000-8000-000000000001', NULL, 'COMPANY_EVENT_BUSINESS_DEVELOPMENT', '上市公司事件-经营活动与项目进展',
    $desc$项目投资
项目中标
重要合同$desc$,
    'text', 'config', 1, TRUE, 'versioned',
    NULL, '{}'::jsonb, NULL, '{}'::jsonb,
    '{}'::jsonb, 'active'
)
ON CONFLICT (type_key) DO NOTHING;

-- 6.2 DocumentProcessorDefinition (.md)
INSERT INTO document_processor_definition (
    document_processor_definition_id, document_type_definition_id,
    file_extension, mime_types, loader_name, loader_config,
    processor_name, processor_config, status, created_by
)
SELECT
    'ef000006-0001-4000-8000-000000000002', d.document_type_definition_id,
    '.md', '[]'::jsonb, 'TEXT_TXT_COMMON_PARSER', '{}'::jsonb,
    'TEXT_TXT_COMMON_CLEANER', $cleaner${"separator": {"enabled": true, "pattern": "^([-=_*~#·•])\\1*$", "min_length": 3}, "watermark": {"enabled": true, "patterns": ["草稿", "机密", "内部资料", "样本", "复印无效", "禁止外传", "confidential", "draft", "sample", "internal use only"]}, "line_number": {"enabled": false, "pattern": "^\\s*\\d{1,4}[.、)）]\\s*"}, "ocr_garbage": {"enabled": true, "pattern": "^[\\W_]{3,}$", "min_length": 4, "normal_punct": "，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-", "symbol_ratio": 0.6, "garbage_chars": "�□■◆●○◇¤§¶†‡※…", "bad_char_ratio": 0.3}, "page_number": {"enabled": true, "patterns": ["^\\s*[-—–\\[\\(【]*\\s*\\d{1,4}\\s*[-—–\\]\\)】]*\\s*$", "^\\s*第\\s*\\d+\\s*页(\\s*[/共]\\s*\\d+\\s*页)?\\s*$", "(?i)^\\s*Page\\s+\\d+(\\s+of\\s+\\d+)?\\s*$", "(?i)^\\s*[ivxlcdm]{2,7}\\s*$"]}, "table_separator": {"enabled": true, "pattern": "^\\s*[+|][\\s\\-+|:=]{3,}[+|]\\s*$"}, "repeated_boilerplate": {"enabled": true, "min_count": 3, "max_length": 80}}$cleaner$::jsonb,
    'active', NULL
FROM document_type_definition d
WHERE d.type_key = 'COMPANY_EVENT_BUSINESS_DEVELOPMENT'
ON CONFLICT (document_type_definition_id, file_extension) DO NOTHING;

-- 6.3 DocumentRetrievalDefinition (keyword -> Elasticsearch / dense_vector -> Milvus)
INSERT INTO document_retrieval_definition (
    document_retrieval_definition_id, document_type_definition_id,
    retrieval_type, chunker_name, chunker_config,
    retriever_name, retriever_config, storage_backend, storage_config,
    status, created_by
)
SELECT
    v.rid, d.document_type_definition_id,
    v.retrieval_type, 'TEXT_TXT_RECURSIVE_CHUNKER', $chunker${"chunk_size": 512, "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ",", " "], "chunk_overlap": 64, "keep_separator": true, "min_chunk_size": 0, "length_function": "char", "strip_whitespace": true, "is_separator_regex": false}$chunker$::jsonb,
    v.retriever_name, '{}'::jsonb, v.storage_backend, '{}'::jsonb,
    'active', NULL
FROM document_type_definition d
CROSS JOIN (VALUES
    ('ef000006-0001-4000-8000-000000000003'::uuid, 'keyword',      'bm25_retriever',         'elasticsearch'),
    ('ef000006-0001-4000-8000-000000000004'::uuid, 'dense_vector', 'dense_vector_retriever', 'milvus')
) AS v(rid, retrieval_type, retriever_name, storage_backend)
WHERE d.type_key = 'COMPANY_EVENT_BUSINESS_DEVELOPMENT'
ON CONFLICT (document_type_definition_id, retrieval_type) DO NOTHING;

-- 6.4 DocumentProcessorRetrievalBinding (processor -> 每条 retrieval)
INSERT INTO document_processor_retrieval_binding (
    document_processor_retrieval_binding_id,
    document_processor_definition_id,
    document_retrieval_definition_id
)
SELECT b.bid, p.document_processor_definition_id, r.document_retrieval_definition_id
FROM document_type_definition d
JOIN document_processor_definition p
  ON p.document_type_definition_id = d.document_type_definition_id
 AND p.file_extension = '.md'
JOIN document_retrieval_definition r
  ON r.document_type_definition_id = d.document_type_definition_id
JOIN (VALUES
    ('ef000006-0001-4000-8000-000000000005'::uuid, 'keyword'),
    ('ef000006-0001-4000-8000-000000000006'::uuid, 'dense_vector')
) AS b(bid, retrieval_type)
  ON b.retrieval_type = r.retrieval_type
WHERE d.type_key = 'COMPANY_EVENT_BUSINESS_DEVELOPMENT'
ON CONFLICT ON CONSTRAINT uq_document_processor_retrieval_binding DO NOTHING;

-- ============================================================================
-- 7/7  COMPANY_EVENT_RISK_COMPLIANCE
--       上市公司事件-风险、法律与上市状态
--       子分类（9）：对外担保 / 诉讼仲裁 / 违规处罚 / 上市状态变动 / 停复牌 / 监管问询 / 破产重整 / 风险提示 / 非标审计意见
-- ============================================================================

-- 7.1 DocumentTypeDefinition
INSERT INTO document_type_definition (
    document_type_definition_id, tenant_id, type_key, name, description,
    modality_type, analysis_mode, revision, allow_update, version_mode,
    fusion_strategy, fusion_config, rerank_strategy, rerank_config,
    metadata, status
) VALUES (
    'ef000007-0001-4000-8000-000000000001', NULL, 'COMPANY_EVENT_RISK_COMPLIANCE', '上市公司事件-风险、法律与上市状态',
    $desc$对外担保
诉讼仲裁
违规处罚
上市状态变动
停复牌
监管问询
破产重整
风险提示
非标审计意见$desc$,
    'text', 'config', 1, TRUE, 'versioned',
    NULL, '{}'::jsonb, NULL, '{}'::jsonb,
    '{}'::jsonb, 'active'
)
ON CONFLICT (type_key) DO NOTHING;

-- 7.2 DocumentProcessorDefinition (.md)
INSERT INTO document_processor_definition (
    document_processor_definition_id, document_type_definition_id,
    file_extension, mime_types, loader_name, loader_config,
    processor_name, processor_config, status, created_by
)
SELECT
    'ef000007-0001-4000-8000-000000000002', d.document_type_definition_id,
    '.md', '[]'::jsonb, 'TEXT_TXT_COMMON_PARSER', '{}'::jsonb,
    'TEXT_TXT_COMMON_CLEANER', $cleaner${"separator": {"enabled": true, "pattern": "^([-=_*~#·•])\\1*$", "min_length": 3}, "watermark": {"enabled": true, "patterns": ["草稿", "机密", "内部资料", "样本", "复印无效", "禁止外传", "confidential", "draft", "sample", "internal use only"]}, "line_number": {"enabled": false, "pattern": "^\\s*\\d{1,4}[.、)）]\\s*"}, "ocr_garbage": {"enabled": true, "pattern": "^[\\W_]{3,}$", "min_length": 4, "normal_punct": "，。、；：？！“”‘’（）《》〈〉【】—…·,.!?;:\"'()<>[]-", "symbol_ratio": 0.6, "garbage_chars": "�□■◆●○◇¤§¶†‡※…", "bad_char_ratio": 0.3}, "page_number": {"enabled": true, "patterns": ["^\\s*[-—–\\[\\(【]*\\s*\\d{1,4}\\s*[-—–\\]\\)】]*\\s*$", "^\\s*第\\s*\\d+\\s*页(\\s*[/共]\\s*\\d+\\s*页)?\\s*$", "(?i)^\\s*Page\\s+\\d+(\\s+of\\s+\\d+)?\\s*$", "(?i)^\\s*[ivxlcdm]{2,7}\\s*$"]}, "table_separator": {"enabled": true, "pattern": "^\\s*[+|][\\s\\-+|:=]{3,}[+|]\\s*$"}, "repeated_boilerplate": {"enabled": true, "min_count": 3, "max_length": 80}}$cleaner$::jsonb,
    'active', NULL
FROM document_type_definition d
WHERE d.type_key = 'COMPANY_EVENT_RISK_COMPLIANCE'
ON CONFLICT (document_type_definition_id, file_extension) DO NOTHING;

-- 7.3 DocumentRetrievalDefinition (keyword -> Elasticsearch / dense_vector -> Milvus)
INSERT INTO document_retrieval_definition (
    document_retrieval_definition_id, document_type_definition_id,
    retrieval_type, chunker_name, chunker_config,
    retriever_name, retriever_config, storage_backend, storage_config,
    status, created_by
)
SELECT
    v.rid, d.document_type_definition_id,
    v.retrieval_type, 'TEXT_TXT_RECURSIVE_CHUNKER', $chunker${"chunk_size": 512, "separators": ["\n\n", "\n", "。", "！", "？", "；", ". ", "! ", "? ", "; ", "，", ",", " "], "chunk_overlap": 64, "keep_separator": true, "min_chunk_size": 0, "length_function": "char", "strip_whitespace": true, "is_separator_regex": false}$chunker$::jsonb,
    v.retriever_name, '{}'::jsonb, v.storage_backend, '{}'::jsonb,
    'active', NULL
FROM document_type_definition d
CROSS JOIN (VALUES
    ('ef000007-0001-4000-8000-000000000003'::uuid, 'keyword',      'bm25_retriever',         'elasticsearch'),
    ('ef000007-0001-4000-8000-000000000004'::uuid, 'dense_vector', 'dense_vector_retriever', 'milvus')
) AS v(rid, retrieval_type, retriever_name, storage_backend)
WHERE d.type_key = 'COMPANY_EVENT_RISK_COMPLIANCE'
ON CONFLICT (document_type_definition_id, retrieval_type) DO NOTHING;

-- 7.4 DocumentProcessorRetrievalBinding (processor -> 每条 retrieval)
INSERT INTO document_processor_retrieval_binding (
    document_processor_retrieval_binding_id,
    document_processor_definition_id,
    document_retrieval_definition_id
)
SELECT b.bid, p.document_processor_definition_id, r.document_retrieval_definition_id
FROM document_type_definition d
JOIN document_processor_definition p
  ON p.document_type_definition_id = d.document_type_definition_id
 AND p.file_extension = '.md'
JOIN document_retrieval_definition r
  ON r.document_type_definition_id = d.document_type_definition_id
JOIN (VALUES
    ('ef000007-0001-4000-8000-000000000005'::uuid, 'keyword'),
    ('ef000007-0001-4000-8000-000000000006'::uuid, 'dense_vector')
) AS b(bid, retrieval_type)
  ON b.retrieval_type = r.retrieval_type
WHERE d.type_key = 'COMPANY_EVENT_RISK_COMPLIANCE'
ON CONFLICT ON CONSTRAINT uq_document_processor_retrieval_binding DO NOTHING;

COMMIT;

-- ============================================================================
-- 校验：应看到 7 行，每行 proc=1 / kw=1 / dense=1 / bindings=2
-- ============================================================================
SELECT
    d.type_key,
    d.name,
    count(DISTINCT p.document_processor_definition_id) AS proc,
    count(DISTINCT r.document_retrieval_definition_id)
        FILTER (WHERE r.retrieval_type = 'keyword')      AS kw,
    count(DISTINCT r.document_retrieval_definition_id)
        FILTER (WHERE r.retrieval_type = 'dense_vector') AS dense,
    count(DISTINCT b.document_processor_retrieval_binding_id) AS bindings
FROM document_type_definition d
LEFT JOIN document_processor_definition p
       ON p.document_type_definition_id = d.document_type_definition_id
LEFT JOIN document_retrieval_definition r
       ON r.document_type_definition_id = d.document_type_definition_id
LEFT JOIN document_processor_retrieval_binding b
       ON b.document_processor_definition_id = p.document_processor_definition_id
      AND b.document_retrieval_definition_id = r.document_retrieval_definition_id
WHERE d.type_key LIKE 'COMPANY_EVENT\_%'
GROUP BY d.type_key, d.name
ORDER BY d.type_key;
