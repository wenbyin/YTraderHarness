# Eval · 检索评测

> 这个子系统**只评召回**：给定一道题，正确答案所在的 chunk 有没有被检索链路捞回来、排在第几。
>
> 它最有价值的东西不是那几个指标，而是一张叫 `retrieval_eval_stage_observation` 的表 ——
> 它回答「**金标死在哪一段**」：召回没捞到、融合丢了、精排切了、还是裁剪砍了。

`9 张表` · `12 个端点` · `只评召回不调 LLM` · `三轮放阶梯` · `金标是下界不是唯一答案`

---

## 📚 本文结构

- 🎯 **一张表看懂**
- 🧩 **先分清三个「评测」**
- 🎬 **造题链路**
- 📏 **评测链路与指标**
- 🔬 **丢因分析** —— 最有用的一张表
- 📖 返回 **[主手册](README.md)** · ⚠️ 缺口见 **[todolist](todolist)**

---

## 🎯 一张表看懂

| 问题 | 一句话答案 |
|---|---|
| 这个子系统评什么？ | **只评召回**。金标所在 chunk 有没有被捞回来、排第几 |
| 评测阶段调 LLM 吗？ | **不调**。LLM 只用在**造题**与**跨 chunk 裁定**两处 |
| 怎么算分？ | `gold_hit = gold_rank <= max_k`；指标有 `hit_rate_at_1/5/k`、`mrr_at_k`、`hit_rate_curve`、`p50/p95_latency_ms` |
| 为什么要「三轮」？ | 逐轮**放宽**检索参数，只把上一轮**没命中**的题带入下一轮。三轮分开统计，所以是正当的 |
| 金标是「唯一答案」吗？ | **不是，是下界**。源 chunk 天然是金标，再用跨 chunk 裁判补上「别的 chunk 也能独立回答」的块 |
| 最有用的一张表？ | `retrieval_eval_stage_observation` —— 回答「金标死在哪一段」 |
| 数据集能改吗？ | **不能**。没有 update 端点，`ready` 后不可变，改参数要新建 bundle |
| 有多少张表？ | **9 张**（冒烟实测建表 9/9） |
| 实现完整度 | 12 个端点全部实现；2 个冒烟 + **68 个 pytest 全部通过** |

---

## 🧩 先分清三个「评测」

| 名字 | 是什么 | 是否调 LLM |
|---|---|---|
| **检索评测**（本手册） | 造题 → 跑检索 → 算命中与排序 | 造题/裁定调，**评测不调** |
| **QA 数据集**（`document/qa/`） | 另一套问答对生成，与 eval 只共享 `locate.py` | 调 |
| **压测**（`stress_*`） | 并发与稳定性，不关心答案对不对 | 不调 |

---

## 🏗️ 阅读地图

```
app/api/v1/retrieval_eval.py              ← HTTP 层

app/services/document/
├── application/
│   ├── retrieval_eval_service.py         ← 接口（14 个方法）
│   ├── retrieval_eval_service_impl.py    ← 实现（约 2954 行，编排主体）
│   └── retrieval_eval_evaluation.py      ← 评测执行 + 指标
├── domain/
│   ├── entities/retrieval_eval.py        ← 9 个领域实体
│   ├── schemas/retrieval_eval.py         ← DTO + 默认常量
│   ├── schemas/enum.py                   ← 8 个 eval 枚举
│   └── tasks.py                          ← 两个 handler 的参数结构
├── models/sqlalchemy/retrieval_eval.py   ← 9 个 ORM 模型
├── repository/retrieval_eval_{bundle,case,run}(+_impl).py
└── eval/                                 ← 造题算法层
    ├── sampling.py        ← 抽样
    ├── windowing.py       ← 切窗
    ├── generator.py       ← LLM 造题 + 校验
    ├── adjudication.py    ← 跨 chunk 裁判
    └── output.py          ← ⚠️ **模型输出解析**，不是「导出」
```

> ⚠️ `eval/output.py` 的名字有误导性：它是 `parse_window_questions` / `parse_adjudication_verdict` 的所在地，
> **不含** `canonical_json` / `dataset_sha256`。确定性导出在 `app/common/utils/digest.py`。

---

## 🎬 造题链路

```
POST /bundles
  → 选文档与版本
  → 抽样（sampling.py）      ← 挑哪些 chunk 出题
  → 切窗（windowing.py）     ← 把 chunk 切成人能读的窗口
  → LLM 造题（generator.py） ← 严格 JSON，自带校验
  → 跨 chunk 裁判（adjudication.py） ← 补「别的 chunk 也能独立回答」的块
  → bundle 状态 ready（**此后不可变**）
```

### 金标的两次来源

| 来源 | 说明 |
|---|---|
| **源 chunk** | 由构造天然是金标（题目就是从它出的） |
| **跨 chunk 裁判** | LLM 判定「另一个 chunk 是否也能独立回答这道题」，是则加入金标集 |

> 所以金标是**下界**：它保证「至少这些 chunk 算对」，但不声称「只有这些算对」。
> 这直接决定了指标的解释方式 —— `hit_rate` 偏低可能是金标不全，不一定是检索差。

---

## 📏 评测链路与指标

```
POST /runs
  → 取 bundle 的题
  → 按阶梯跑检索（每轮放宽参数）
  → 逐题判命中与名次
  → compute_metrics
  → 写 stage_observation（每段一行）
```

### 默认三档阶梯（`DEFAULT_ATTEMPT_LADDER`）

逐轮**放宽**，但**只把上一轮没命中的题带入下一轮** —— 所以三轮分开统计是正当的，不能混在一起算总命中率。

> ⚠️ 阶梯里 `cutter.top_k` 是 10/20，而 `max_k` 可能配成 50：
> `cutter.top_k = min(cutter.top_k, max_k)` **不保证二者相等**，
> 于是标签写 `at_50` 实际只量到 `at_10`，**且不报错**。见 [`todolist`](todolist)。

### 指标

| 指标 | 含义 |
|---|---|
| `hit_rate_at_1 / at_5 / at_k` | 金标是否出现在前 N 名 |
| `mrr_at_k` | 第一个金标名次的倒数均值 |
| `hit_rate_curve` | 命中率随 K 的曲线（比单点更能看出「差一点」还是「完全没戏」） |
| `p50 / p95_latency_ms` | 检索耗时 |

---

## 🔬 丢因分析

`retrieval_eval_stage_observation` 每个阶段一行，回答「金标死在哪一段」：

| 阶段 | 死了说明什么 |
|---|---|
| 召回（recall） | 索引里根本没有 / 向量模型不匹配 |
| 融合（fusion） | 某一路捞到了但被 RRF 排到后面 |
| 精排（rerank） | cross-encoder 判断错了 |
| 裁剪（cut） | 排到了但被 `top_k` 砍掉 |

**`GET /runs/{id}/stage-stats`** 做宏观统计 —— 一百道题里有多少死在精排，比看单题命中率有用得多。

### 确定性导出

`GET /bundles/{id}/export` 用 `canonical_json` + `dataset_sha256` 保证**同内容同摘要**，
让不同机器/不同时间的评测可比。

> ⚠️ `dataset_sha256` **只在响应里实时算，从不回写库** —— 生产环境 summary 里该字段恒为 `None`。
> 见 [`todolist`](todolist)。

---

## 📡 端点全清单（12 个 operation / 11 个 path）

| 组 | 端点 |
|---|---|
| **Bundle** | `POST /bundles`、`GET /bundles`、`GET /bundles/{id}`、`GET /bundles/{id}/export` |
| **Case / Window** | `POST /bundles/{id}/windows`、`GET /bundles/{id}/cases` |
| **Run** | `POST /runs`、`GET /runs`、`GET /runs/{id}`、`GET /runs/{id}/stage-stats` |
| **其它** | `GET /cases/{id}`、`DELETE /bundles/{id}` |

> ⚠️ **没有 update 端点**，`ready` 后不可变 —— 改参数必须新建 bundle。这是刻意的：可变的评测集不可比。

---

## 🚀 验证

```bash
.venv/Scripts/python.exe -m app.services.document.retrieval_eval_smoke          # 9 张表 + 端点
.venv/Scripts/python.exe -m app.services.document.retrieval_eval_evaluate_smoke # 评测执行 + 指标
.venv/Scripts/python.exe -m pytest tests/services/document/test_retrieval_eval_* -q
```

> 快速验证**不需要真模型 key** —— 冒烟用桩数据走完造题与评测全链。

---

## 🤝 相关

| 想深入看 | 去哪 |
|---|---|
| 被评测的检索链路本身 | [`README_DOCUMENT.md`](README_DOCUMENT.md) |
| 两个 `ModelRole`（造题 / 裁定） | [`README_MODEL.md`](README_MODEL.md) |
| 总览 | [`README.md`](README.md) |
