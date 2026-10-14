"""
Memory —— 对话记忆：L0 原文索引 + L1 原子记忆的提炼 + L2 场景文件。

本模块有三层（L3 只落了表，见下）。规格见 memory_l0.md / memory_l1.md /
memory_l2.md / memory_l3.md，但落库形态与那几份规格**不同**，差异是有意的。

    L0    `agent_message_fts` / `agent_message_vec`
          挂在 agent_message 上的两张索引表 —— 原始对话"原样可检索"，
          纯派生。

    L1    `memory_atom` 本体 + `memory_atom_fts` / `memory_atom_vec` 两张
          挂载表 + `memory_atom_cursor` 水位
          从 L0 提炼出来的结构化记忆。**"L1" 只是规格里的层名** ——
          落库与代码里一律叫 memory_atom（原子记忆）。

    L2    `memory_scene` 本体（**没有挂载表**）+ `memory_scene_cursor` 水位
          把 L1 记忆融合成的场景文件，一行 = 一个 `.md`（`scene_blocks/`
          那个目录的行化表示）。**"L2" 同样是规格里的层名** —— 落库与代码
          里一律叫 memory_scene。

    L3    `memory_persona` 本体（**当前只建了表**）
          用户画像（一整篇 `persona.md`，一个 scope 一行）。提示词与
          handler 都还没落地 —— 别把它当"能用但没接线的功能"看。

与规格的三处结构性差异
------------------
    l0_conversations 那张表**不建**
        它是腾讯 agent mem 的结构。本仓库消息的权威表已经是
        `agent_message`，再建一张 L0 表等于把同一份对话存两遍，而两份
        之间没有事务关系。所以索引**挂**在 agent_message 上。
        规格里 L0 比 L1 多出来的三列（role / recorded_at / timestamp）
        在 agent_message 上已经各有对应（role / created_at）。

    sqlite-vec 与 FTS5 用不了（L0/L1 都适用）
        本仓库是 Postgres：`vec0` 对应 pgvector 的 `vector(512)` + HNSW，
        `fts5` 对应 `tsvector` + GIN。而 FTS5 那个"存 jieba 分词后的文本"
        的设计**照搬**了 —— 只是分词器换成对齐 ES `cjk` 分析器的那一套
        （见 tokenization.py），因为 Postgres 侧没有中文分词配置。

    **L2/L3 没有虚表**（与 L1 最显眼的一处不同）
        它们当前不是"被检索的"，而是被**整段拼进提示词**的（场景清单 +
        全文）。所以只有本体表，没有关键词 / 向量挂载表 —— 也没有任何召回
        入口。真要做（给主模型一个"这个用户的画像是什么"的工具）再加，
        源就是本体表。

更细的偏离逐条写在各自的 docstring 里：L1 的（全列冗余 → 只放召回列、
record_id 不用 `rec_xxx`、metadata_json 在库里叫 meta、task_id 空置、
没有 JSONL 与 memory-cleaner）见 models/sqlalchemy/memory_atom.py；L2 的
（主键用内容派生的 hash、模型一轮定输赢而不是文件工具、`[DELETED]` 由
服务侧真删）见 memory_scene.py 与 application/memory_scene_service_impl.py。

非职责
------
    会话消息的读写                    归 session 模块。本模块通过它的
                                      MessageRepository 契约取消息，
                                      不直接查 agent_message
    L2/L3 的**消费侧**                画像怎么进主模型的上下文（检索 /
                                      注入）还没有落地 —— 当前只有生产
                                      侧（`memory.l2.extract` 那条任务）

数据的生命周期
------------
**三层不同，别混**：

    L0 的两张索引表        纯派生 —— 从 agent_message 删干净重建是安全的。
                          重建的抓手是 `memory.l0.extract` 任务（按 run_id 一轮
                          一轮跑），当前**没有**全量重跑的命令（见 models.sql
                          末尾）。

    L1 的 memory_atom      权威数据 —— 装的是 LLM 的结论，重跑一次会得到
    L1 的 memory_atom_cursor    **不同**的记忆/水位。删掉就真没了。
                          （水位严格说是权威状态，见 memory_atom_cursor.py 的说明。）
                          能重建的只有它们的两个派生品：memory_atom_fts /
                          memory_atom_vec，源都是 memory_atom。

    L2 的 memory_scene   同样权威 —— 场景文件也是 LLM 的结论（把碎片
    L2 的 memory_scene_cursor 记忆整合成的叙事），从同一批 L1 重跑得到的是
                          不同的文档。能重建的只有"重新整合一遍"（要先把
                          水位退回去），当前没有那条命令。
                          注意它与 L1 是**两条独立的链**：L2 只读 L1 的
                          结果，不参与写 L1。

    L3 的 memory_persona   同上，但它还没有生产者。
"""

__all__: list[str] = []
