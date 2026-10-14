#!/usr/bin/env python
"""
把本地开发环境恢复到「刚建完表」的状态。

    python reset_data.py              # 演练：只报告 + 校验外键图，一个字节都不删
    python reset_data.py --yes        # 真删

默认是演练而不是真删
--------------------
这个脚本会 TRUNCATE 掉 27 张表和 195 个上传目录。写成"默认就干"的话，
一次手滑就是不可逆的；而演练模式不是摆设 —— 它会把外键图跑一遍
（见 _check_truncate）。被删的表之间存在互相引用，TRUNCATE 一张却漏掉
它的子表时，Postgres 会整条语句拒绝执行。演练用
BEGIN → TRUNCATE → ROLLBACK 提前把这件事撞出来，而不是等真删的那一刻。

什么算"原始状态"
----------------
只清**运行业务产生的数据**。定义与配置一律不动：

    留：capability、tool_definition / _documentation / _parameter_documentation
        / _processor / _definition_classification
        document_type_definition（DTD）、document_processor_definition、
        document_retrieval_definition、document_processor_retrieval_binding、
        document_index_binding
        checkpoint_migrations、store_migrations

其中两张 binding 表最容易搞混，模型的 docstring 把区别写死了：

    document_processor_retrieval_binding   配置层「这种前处理要不要进这种检索」
    document_version_retrieval_binding     执行层「这个具体版本处理到哪一步了」

前者留，后者清。document_index_binding 名字像执行层，但它**没有 version
外键**（只有 document_retrieval_definition_id + backend_type +
index_identifier），是"哪个定义用哪个 ES 索引 / Milvus collection"的配置，留。

checkpoint_migrations / store_migrations 也留着：它们是迁移水位，清掉会让
下次启动重跑迁移。清它们没有任何收益。

凭据从哪来
----------
复用 app/config/datasource/ 那份配置（DataSourceConfigLoader），不另抄一套主机名口令
—— 两份迟早会漂。所以本脚本必须**在项目根、用 .venv 的 python** 跑：

    cd <项目根> && ./.venv/Scripts/python.exe reset_data.py

外部存储
--------
ES 索引名 / Milvus collection 名从 app.services.document.pipeline.containers
的 EXTERNAL_INDEX 里 import —— 那个模块的 docstring 明说了"谁都不许再抄
一遍"，写入期与查询期共用一份。这里同理：抄一份的话，改了契约名字之后
脚本会去删一个不存在的容器，而且**不报错**。

⚠️ 清 ES 的后果：document_chunk 被清空后，那三个索引里的文档全指向不存在的
chunk。索引本身会在下次写入时由 ElasticsearchIndex._ensure_index 自动重建，
不需要手工建。
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

# ============================================================
# 路径引导
# ============================================================
#
# 与 app/main.py 同一个理由：`python reset_data.py` 时 sys.path[0] 是脚本
# 所在目录而不是项目根，下面的 `from app.*` 会 ModuleNotFoundError。
# 本脚本就在项目根，所以这里插的就是它自己那一层。

_PROJECT_ROOT = Path(__file__).resolve().parent

if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


from app.infra.datasource.loader import (  # noqa: E402
    DataSourceConfigLoader,
)

logger = logging.getLogger("reset_data")


CONFIG_DIR = _PROJECT_ROOT / "app" / "config" / "datasource"

#: 本地文件存储的根。默认值与
#: app/services/file/storage/local.py 的 DEFAULT_BASE_PATH 一致。
FILES_ROOT = _PROJECT_ROOT / "data" / "files"


# ============================================================
# 要清的表
# ============================================================
#
# ⚠️ 这张表是**硬编码**的，所以下面 _audit_tables 会在清之前拿库里的实际
# 表名对一遍：任何一张既不在 CLEAR 也不在 KEEP 里的表都会被大声报出来。
# 这是防"改了表名之后脚本悄悄漏清 / 悄悄多清" —— 硬编码清单过时不会报错，
# 只会安静地做错事（本仓库已经在别处踩过一次，见 memory 模块的裸 SQL）。
#
# 分组只是为了报告好读，执行时是**一条** TRUNCATE。

CLEAR_TABLES: dict[str, tuple[str, ...]] = {

    # ---- 会话与消息 ----
    # agent_message_fts / _vec 是 agent_message 的两张记忆索引表。
    # 它们与 agent_message **之间没有外键**（见 initializer 的说明），
    # 所以 TRUNCATE 基表不会连带清掉它们 —— 必须显式列出。
    "会话与消息": (
        "agent_session",
        "agent_message",
        "agent_message_fts",
        "agent_message_vec",
        "agent_run",
    ),

    # ---- 记忆 L0/L1/L2/L3 ----
    # *_cursor 是各层的抽取水位。不清的话，清完消息表之后水位还在原处，
    # 新一轮对话抽不出记忆（"增量从当前开始"，而当前比水位还早）。
    "记忆": (
        "memory_atom",
        "memory_atom_fts",
        "memory_atom_vec",
        "memory_atom_cursor",
        "memory_scene",
        "memory_scene_cursor",
        "memory_persona",
        "memory_persona_cursor",
    ),

    # ---- 文档：运行侧 ----
    # 不含 document_type_definition / document_retrieval_definition /
    # document_processor_definition / document_processor_retrieval_binding /
    # document_index_binding —— 那五张是配置。
    "文档": (
        "document",
        "document_version",
        "document_chunk",
        # 执行层：某个具体 version 在各 retrieval 上处理到哪一步。
        "document_version_retrieval_binding",
        "document_pipeline_run",
        "document_pipeline_node_run",
    ),

    # ---- 文件 ----
    # 与 data/files/ 下的目录**一一对应**（当前各 195 个）。
    # 清表不清盘 = 孤儿文件；清盘不清表 = 前端全死链。两处必须成对。
    "文件": (
        "file",
    ),

    # ---- 工具 ----
    # 只有执行账本。tool_definition / tool_documentation /
    # tool_parameter_documentation / tool_processor /
    # tool_definition_classification 是定义，一律不动。
    "工具": (
        "tool_execution",
    ),

    # ---- 任务 ----
    "任务": (
        "tasks",
        "task_outbox",
    ),

    # ---- LangGraph 框架表 ----
    #
    # checkpoints / checkpoint_blobs / checkpoint_writes 是 LangGraph
    # Postgres checkpointer 的三张表；store 是同一个池上的 BaseStore，
    # 目前唯一的消费者是审批账本（interceptor/approval.py）。
    #
    # store 里的批准记录**必须**一起清：不清的话，一个被删掉的 run
    # 留下的"这段已经批过了"会在下一个同名 run 上意外命中。
    #
    # checkpoint_migrations / store_migrations 不在这里 —— 见模块 docstring。
    "LangGraph": (
        "checkpoints",
        "checkpoint_blobs",
        "checkpoint_writes",
        "store",
    ),
}

#: 定义与配置：**不动**。列出来是为了让 _audit_tables 能算出"第三类"，
#: 也是给读的人一份"脚本故意没碰什么"的正面清单。
KEEP_TABLES: frozenset[str] = frozenset({
    # 能力树 + 工具定义
    "capability",
    "tool_definition",
    "tool_definition_classification",
    "tool_documentation",
    "tool_parameter_documentation",
    "tool_processor",
    # 文档配置
    "document_type_definition",
    "document_processor_definition",
    "document_retrieval_definition",
    "document_processor_retrieval_binding",
    "document_index_binding",
    # 迁移水位
    "checkpoint_migrations",
    "store_migrations",
})


# ============================================================
# Redis
# ============================================================
#
# 事件流的键形如 "<key_prefix>:<topic_id>"，即 agent_run:<run_id>
# （见 app/common/event/event.py 与 agent/event/event_stream.yaml）。
#
# probe: 是探针脚本的残留，没有 TTL。
#
# 只删**已知前缀**而不是 FLUSHDB：这个库是 dev 实例，但"清库脚本顺手
# 把别人（或另一个应用）的数据抹了"是那种事后才发现的事故。SCAN 一遍
# 把不认识的前缀报出来，比悄悄清掉有价值。

REDIS_PREFIXES: tuple[str, ...] = (
    "agent_run:",
    "probe:",
)


# ============================================================
# 命令行
# ============================================================

def _parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="把本地数据恢复到原始状态（默认只演练，不删）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--yes",
        action="store_true",
        help="真的执行删除。不加这个参数 = 演练。",
    )

    # 各个后端可以单独跳过 —— 只想清库、留着文件盘上的东西时有用。
    for name, help_text in (
        ("postgres", "跳过 PostgreSQL"),
        ("redis", "跳过 Redis"),
        ("elasticsearch", "跳过 Elasticsearch 索引"),
        ("milvus", "跳过 Milvus collection"),
        ("neo4j", "跳过 Neo4j 节点"),
        ("files", "跳过 data/files 本地文件"),
    ):
        parser.add_argument(
            f"--skip-{name}",
            action="store_true",
            help=help_text,
        )

    parser.add_argument(
        "--files-root",
        default=str(FILES_ROOT),
        help=f"本地文件存储根目录（默认 {FILES_ROOT}）",
    )

    return parser.parse_args()


# ============================================================
# 配置读取
# ============================================================

def _load_configs() -> dict:
    """
    读 app/config/datasource/ 下所有数据源定义，按 name 索引。

    复用应用自己的加载器（含 ${ENV:-default} 展开），这样脚本与应用
    永远看同一份主机名与口令。
    """

    definitions = DataSourceConfigLoader(CONFIG_DIR).load_all()

    return {d.name: d for d in definitions}


def _postgres_dsn(configs: dict) -> str:
    """
    取能直接交给 psycopg 的 DSN。

    **不自己写 `.replace("+asyncpg", "")`** —— PostgreSQLConfig 已经有
    `resolved_dsn_psycopg` 这个属性在做同一件事（走 SQLAlchemy 的
    make_url，比字符串替换稳），而且它还会优先采用显式配置的
    `dsn_psycopg`。手写替换的版本在"用户配了 dsn_psycopg"时会把它无视掉，
    然后连到一个看起来不相干的地方。
    """

    return configs["postgresql"].config.resolved_dsn_psycopg


# ============================================================
# PostgreSQL
# ============================================================

def _audit_tables(cursor) -> tuple[list[str], list[str]]:
    """
    拿库里**实际存在**的表对一遍硬编码清单。

    返回 (要清但库里没有的, 既不清也不留的)。

    为什么值得单独一趟：CLEAR_TABLES 是手写的，表改名之后它会安静地过时
    —— TRUNCATE 一张不存在的表会报错（这部分还算安全），但**新加一张运行
    表却忘了列进来**是完全静默的：脚本全绿，数据没清干净，而"清过了"这个
    结论会一直用下去。所以第三类必须显式报出来。
    """

    cursor.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = 'public'
        ORDER BY tablename
        """
    )

    actual = {row[0] for row in cursor.fetchall()}

    wanted = {t for tables in CLEAR_TABLES.values() for t in tables}

    missing = sorted(wanted - actual)

    unclassified = sorted(actual - wanted - KEEP_TABLES)

    return missing, unclassified


def _check_truncate(cursor, tables: list[str]) -> None:
    """
    在事务里跑一遍 TRUNCATE 再回滚，把外键问题提前撞出来。

    被清的表之间存在互相引用（document → document_version → chunk 等）。
    TRUNCATE 一条语句里列了一堆表时，只要有任何一张**没被列进来**的表用
    外键指向它们，整条语句就会被拒绝 —— 那次拒绝发生在真删的时候，而那时
    前面的步骤（ES、Redis）可能已经删完了，留下一个半清的环境。

    放在真删之前跑，失败时环境还是完整的。
    """

    statement = _truncate_sql(tables)

    cursor.execute(statement)


def _truncate_sql(tables: list[str]) -> str:
    """
    拼 TRUNCATE 语句。

    一条语句而不是逐张 TRUNCATE：这样表之间的外键引用是一次性解决的，
    也不必操心删除顺序。

    RESTART IDENTITY 对 UUID 主键是空操作，留着是为了万一哪天冒出个
    serial 列 —— 不清序列的话，"清干净"的库下次插进去的 id 还是接着涨。
    """

    joined = ", ".join(tables)

    return f"TRUNCATE TABLE {joined} RESTART IDENTITY"


def reset_postgres(configs: dict, *, execute: bool) -> None:

    import psycopg

    tables = [t for tables in CLEAR_TABLES.values() for t in tables]

    with psycopg.connect(_postgres_dsn(configs)) as connection:

        with connection.cursor() as cursor:

            missing, unclassified = _audit_tables(cursor)

            if missing:
                logger.warning(
                    "清单里有 %d 张表库里不存在（改名了？）：%s",
                    len(missing),
                    ", ".join(missing),
                )

            if unclassified:
                # 这条是**警告**而不是错误：它意味着脚本的认知过时了，
                # 但不该因此拦住"清运行数据"这件仍想做的事。
                logger.warning(
                    "⚠️ 有 %d 张表既不在清理清单也不在保留清单里，本次"
                    "**不会**清它们。请确认是否需要补进 CLEAR_TABLES：%s",
                    len(unclassified),
                    ", ".join(unclassified),
                )

            # 演练与真删走的是同一段代码，只差最后 commit 还是 rollback ——
            # 两套代码路径的话，"演练通过"就不再意味着"真删也能过"。
            _check_truncate(cursor, tables)

            if execute:
                connection.commit()
                logger.info("PostgreSQL：已清空 %d 张表。", len(tables))
            else:
                connection.rollback()
                logger.info(
                    "PostgreSQL：%d 张表的外键图校验通过（已回滚，未删）。",
                    len(tables),
                )


def report_postgres(configs: dict) -> None:
    """演练时把每张表的行数打出来 —— 删之前总该看一眼删的是什么。"""

    import psycopg

    with psycopg.connect(_postgres_dsn(configs)) as connection:

        with connection.cursor() as cursor:

            print("\n  PostgreSQL（将被清空）")
            print(f"  {'表':<40} 行数")
            print(f"  {'-' * 40} {'-' * 8}")

            total = 0

            for group, tables in CLEAR_TABLES.items():

                print(f"\n  [{group}]")

                for table in tables:

                    cursor.execute(
                        f'SELECT count(*) FROM "{table}"'
                    )

                    count = cursor.fetchone()[0]

                    total += count

                    print(f"  {table:<40} {count:>8}")

            print(f"\n  合计 {total} 行")

            print("\n  PostgreSQL（保留，不动）")

            for table in sorted(KEEP_TABLES):

                cursor.execute(
                    f'SELECT count(*) FROM "{table}"'
                )

                print(f"  {table:<40} {cursor.fetchone()[0]:>8}")


# ============================================================
# Redis
# ============================================================

def reset_redis(configs: dict, *, execute: bool) -> None:

    import redis

    definition = configs.get("redis")

    if definition is None:
        logger.warning("Redis：配置目录里没有 redis.yaml，跳过。")
        return

    client = redis.Redis.from_url(
        definition.config.url,
        decode_responses=True,
    )

    matched: dict[str, int] = {prefix: 0 for prefix in REDIS_PREFIXES}
    unknown: dict[str, int] = {}

    for key in client.scan_iter(count=500):
        for prefix in REDIS_PREFIXES:
            if key.startswith(prefix):
                matched[prefix] += 1
                break
        else:
            # 取第一段当分组，够看出"这批键是谁的"了。
            unknown[key.split(":", 1)[0]] = (
                unknown.get(key.split(":", 1)[0], 0) + 1
            )

    for prefix, count in matched.items():
        logger.info("Redis：%s* → %d 个键", prefix, count)

    if unknown:
        # 只报不删。见 REDIS_PREFIXES 上面那段。
        logger.warning(
            "Redis：还有 %d 组不认识的键，**未删除**：%s",
            len(unknown),
            ", ".join(
                f"{k}* ({v})" for k, v in sorted(unknown.items())
            ),
        )

    if not execute:
        return

    deleted = 0

    for prefix in REDIS_PREFIXES:

        # SCAN 游标边删边扫是安全的（不像 KEYS），但成批删更省往返。
        batch: list[str] = []

        for key in client.scan_iter(match=f"{prefix}*", count=500):
            batch.append(key)

            if len(batch) >= 200:
                deleted += client.delete(*batch)
                batch.clear()

        if batch:
            deleted += client.delete(*batch)

    logger.info("Redis：已删除 %d 个键。", deleted)


# ============================================================
# Elasticsearch
# ============================================================

def _external_containers() -> list[str]:
    """
    外部索引的容器名，来自契约模块。

    写成 import 而不是抄字符串：containers.py 的 docstring 明说了这几个
    名字是契约、谁都不许再抄一遍。抄一份的代价在这里很具体 —— 契约改名后
    脚本会去删一个不存在的容器，而且**不报错**（删除不存在的索引在 ES 里
    是 404，很容易被当成"本来就没有"）。
    """

    from app.services.document.pipeline.containers import (
        EXTERNAL_INDEX,
    )

    return sorted(EXTERNAL_INDEX.values())


def reset_elasticsearch(configs: dict, *, execute: bool) -> None:

    definition = configs.get("elasticsearch")

    if definition is None:
        logger.warning("Elasticsearch：配置目录里没有 elasticsearch.yaml，跳过。")
        return

    from elasticsearch import Elasticsearch

    config = definition.config

    client = Elasticsearch(
        hosts=[config.hosts[0]],
        basic_auth=(config.username, config.password),
        verify_certs=config.verify_certs,
        request_timeout=config.request_timeout,
    )

    for name in _external_containers():

        if not client.indices.exists(index=name):
            # 当前只有 document_instance_keyword 真的存在，sparse/dense 两路
            # 还没被写入过。不存在不算问题 —— 说明它本来就是空的。
            logger.info("Elasticsearch：%s 不存在，跳过。", name)
            continue

        count = client.count(index=name)["count"]

        if not execute:
            logger.info(
                "Elasticsearch：%s 有 %d 条文档（将删除整个索引）。",
                name,
                count,
            )
            continue

        client.indices.delete(index=name)

        logger.info(
            "Elasticsearch：已删除索引 %s（%d 条文档）。",
            name,
            count,
        )


# ============================================================
# Milvus
# ============================================================

def reset_milvus(configs: dict, *, execute: bool) -> None:

    definition = configs.get("milvus")

    if definition is None:
        logger.warning("Milvus：配置目录里没有 milvus.yaml，跳过。")
        return

    from pymilvus import MilvusClient

    config = definition.config

    client = MilvusClient(
        uri=config.uri,
        token=config.token,
        db_name=config.db_name or "default",
    )

    existing = set(client.list_collections())

    for name in _external_containers():

        if name not in existing:
            # 当前 Milvus 是空的（0 个 collection）：dense_vector 那一路还
            # 没接上。这一段是兜底，防止哪天数据进来了漏清。
            logger.info("Milvus：collection %s 不存在，跳过。", name)
            continue

        if not execute:
            logger.info("Milvus：collection %s（将删除）。", name)
            continue

        client.drop_collection(name)

        logger.info("Milvus：已删除 collection %s。", name)


# ============================================================
# Neo4j
# ============================================================

def reset_neo4j(configs: dict, *, execute: bool) -> None:

    definition = configs.get("neo4j")

    if definition is None:
        logger.warning("Neo4j：配置目录里没有 neo4j.yaml，跳过。")
        return

    from neo4j import GraphDatabase

    config = definition.config

    driver = GraphDatabase.driver(
        config.uri,
        auth=(config.username, config.password),
    )

    try:

        # database 留空（None）时用服务端默认库 —— 与应用侧
        # Neo4jDataSource 的取法一致，不要在这里硬写 "neo4j"：
        # Neo4j 4 及更早是单库模型，指定库名反而会连不上。
        with driver.session(database=config.database) as session:

            count = session.run(
                "MATCH (n) RETURN count(n) AS c"
            ).single()["c"]

            if not count:
                # 当前 Neo4j 是空的：图那一路的抽取与入库都还没做
                # （见 containers.py 里 graph 那条注释）。
                logger.info("Neo4j：没有节点，跳过。")
                return

            if not execute:
                logger.info("Neo4j：%d 个节点（将全部删除）。", count)
                return

            session.run("MATCH (n) DETACH DELETE n")

            logger.info("Neo4j：已删除 %d 个节点。", count)

    finally:
        driver.close()


# ============================================================
# 本地文件
# ============================================================

def reset_files(root: Path, *, execute: bool) -> None:

    if not root.exists():
        logger.warning("本地文件：%s 不存在，跳过。", root)
        return

    # 只删 root 下的**子目录**（每个上传是 <uuid>/source 一个目录），
    # 不删 root 自己：目录权限、以及与 file 表无关的东西留在原地。
    children = sorted(p for p in root.iterdir() if p.is_dir())

    if not children:
        logger.info("本地文件：%s 下没有子目录，跳过。", root)
        return

    total_bytes = sum(
        p.stat().st_size
        for child in children
        for p in child.rglob("*")
        if p.is_file()
    )

    if not execute:
        logger.info(
            "本地文件：%s 下 %d 个目录 / %.1f MB（将删除）。",
            root,
            len(children),
            total_bytes / 1024 / 1024,
        )
        return

    for child in children:
        shutil.rmtree(child, ignore_errors=False)

    logger.info(
        "本地文件：已删除 %d 个目录 / %.1f MB。",
        len(children),
        total_bytes / 1024 / 1024,
    )


# ============================================================
# 入口
# ============================================================

def main() -> int:

    # stream=stdout 是刻意的：本脚本一半输出走 logging、一半走 print
    # （那张行数表用 print 排版更省事）。两者分属 stderr/stdout 时，管道里
    # 的顺序会错乱到"=== 数据恢复 ==="跑到了它该概括的那些行后面。
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)-7s %(message)s",
        stream=sys.stdout,
    )

    # elasticsearch-py 把每条 HTTP 请求都按 INFO 打出来（HEAD ... status:404
    # 那种）。这些请求数正好等于被检查的容器数，读的人已经能从下面那几行
    # 结论里看到了 —— 留着只是把真正的结论埋掉。
    logging.getLogger("elastic_transport").setLevel(logging.WARNING)

    args = _parse_args()

    execute = args.yes

    print(
        "\n=== 数据恢复：%s ===\n"
        % ("真删" if execute else "演练（什么都不删）")
    )

    if not execute:
        print("  加 --yes 才会真的删。\n")

    configs = _load_configs()

    # 演练时先把每张表的行数摊开 —— 删之前总该看一眼删的是什么。
    if not execute and not args.skip_postgres:
        try:
            report_postgres(configs)
        except Exception:
            logger.exception("PostgreSQL：行数报告失败。")

    for label, skipped, action in (
        ("PostgreSQL", args.skip_postgres,
         lambda: reset_postgres(configs, execute=execute)),
        ("Redis", args.skip_redis,
         lambda: reset_redis(configs, execute=execute)),
        ("Elasticsearch", args.skip_elasticsearch,
         lambda: reset_elasticsearch(configs, execute=execute)),
        ("Milvus", args.skip_milvus,
         lambda: reset_milvus(configs, execute=execute)),
        ("Neo4j", args.skip_neo4j,
         lambda: reset_neo4j(configs, execute=execute)),
        ("本地文件", args.skip_files,
         lambda: reset_files(Path(args.files_root), execute=execute)),
    ):

        if skipped:
            logger.info("%s：已按参数跳过。", label)
            continue

        try:
            action()

        except Exception:
            # 一个后端挂了不该拦住其余的 —— 半清的环境比全清更难查，
            # 而把每个后端的真实结果都打出来，才知道停在哪一步。
            logger.exception("%s：失败（继续处理其余后端）。", label)

    print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
