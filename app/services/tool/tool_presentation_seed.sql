-- 用户展示的初始配置（幂等）。
--
-- 默认是**隐藏**：没有行 = 不显示。本文件只录「确定要给用户看、
-- 且展示形态已经想清楚」的工具，不是全量清单。
--
-- ⚠️ 真正的用户交互（提问 / 审批）由 hitl.* 事件承担，它们不经过
--    presenter、永远显示。这里配的只是工具调用本身留痕的那一行。
--
-- 用法：应用启动（create_all 建表）之后执行一次。

INSERT INTO tool_presentation (
    tool_key,
    renderer,
    display_name,
    summary,
    show_input,
    show_output,
    props,
    updated_by,
    updated_at
)
VALUES (
    'agent.ask_user_question',
    'inline',
    '向用户提问',
    '向用户确认信息',
    FALSE,   -- 问题内容由 hitl.requested 的交互卡片展示，不在这里重复
    FALSE,   -- 用户答复由交互消息承载
    '{}'::jsonb,
    'seed',
    now()
)
ON CONFLICT (tool_key) DO UPDATE SET
    renderer     = EXCLUDED.renderer,
    display_name = EXCLUDED.display_name,
    summary      = EXCLUDED.summary,
    show_input   = EXCLUDED.show_input,
    show_output  = EXCLUDED.show_output,
    props        = EXCLUDED.props,
    updated_by   = EXCLUDED.updated_by,
    updated_at   = now();


-- ------------------------------------------------------------------
-- 行情查询（MCP: ytrader_market）。
--
-- 结果是包装对象：{bars: [...], rowCount, source, complete, ...}。
-- 表格要的是 bars 数组本身，所以用 output_path 先取段；summary 读的是
-- 取段**之前**的 rowCount（「共 N 条」），与表格互补 —— 表只画前 5 行。
--
-- columns 对齐 market.BarView 的 JSON 键（Go: internal/market/types.go）。
-- percentChange 已经是百分数（×100），标签里写 % 即可。
-- ------------------------------------------------------------------
INSERT INTO tool_presentation (
    tool_key,
    renderer,
    display_name,
    summary,
    show_input,
    show_output,
    props,
    updated_by,
    updated_at
)
VALUES (
    'ytrader_market.query_market_data',
    'table',
    '行情查询',
    '共 {output.rowCount} 条',
    FALSE,   -- 表格不展示入参
    TRUE,    -- 表格的行来自 output；为 false 会静默退化成 card
    '{
       "output_path": "/bars",
       "max_rows": 20,
       "columns": [
         {"key": "code",          "label": "代码"},
         {"key": "ts",            "label": "时间"},
         {"key": "close",         "label": "收盘"},
         {"key": "change",        "label": "涨跌"},
         {"key": "percentChange", "label": "涨跌幅%"},
         {"key": "volume",        "label": "成交量"}
       ]
     }'::jsonb,
    'seed',
    now()
)
ON CONFLICT (tool_key) DO UPDATE SET
    renderer     = EXCLUDED.renderer,
    display_name = EXCLUDED.display_name,
    summary      = EXCLUDED.summary,
    show_input   = EXCLUDED.show_input,
    show_output  = EXCLUDED.show_output,
    props        = EXCLUDED.props,
    updated_by   = EXCLUDED.updated_by,
    updated_at   = now();
