"""
回填 skill 表的容量度量四列。

什么时候用
----------
跑完 `app/services/skill/skill_token_metrics_migration.sql` 之后：存量行
会落在 0 / 0 / 'unmeasured' / NULL，管理端卡片显示「未测量」。本脚本按
线上真实数据重算一遍，让已有 skill 立刻显示数字。

    .venv/Scripts/python.exe scripts/backfill_skill_tokens.py --dry-run
    .venv/Scripts/python.exe scripts/backfill_skill_tokens.py

    # 离线 / 镜像环境，跳过 tiktoken 词表下载，口径会写成 heuristic:3.0
    SKILL_TOKEN_FORCE_HEURISTIC=1 ./.venv/Scripts/python.exe scripts/backfill_skill_tokens.py

    # 覆盖默认 DSN
    YTRADER_DSN=postgresql://user:pwd@host:5432/db ...

只 UPDATE 那四个派生列，**不动 updated_at** —— 容量是派生值，回填不是
编辑行为，不该让列表页的「更新于」集体跳动。幂等，可重复跑。

口径与线上写路径共用 `measure_skill_tokens()`，不另写一份计数逻辑。
能力清单按 code 排序：顺序不影响 token 数，但可复现便于对账。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import psycopg
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.skill.application._token_metrics import (  # noqa: E402
    measure_skill_tokens,
)

_DSN_FILE = ROOT / "app" / "config" / "datasource" / "postgre.yaml"


def _dsn() -> str:
    """优先环境变量，其次项目配置；去掉 SQLAlchemy 的驱动后缀。"""

    raw = os.environ.get("YTRADER_DSN")

    if not raw:
        config = yaml.safe_load(_DSN_FILE.read_text(encoding="utf-8"))
        raw = config["config"]["dsn"]

    # psycopg 直连不吃 SQLAlchemy 的 +asyncpg 方言后缀
    # （postgre.yaml 里 checkpointer 也是这么推导的）。
    return raw.replace("postgresql+asyncpg://", "postgresql://")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Recompute skill capacity metrics.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只算不写库。",
    )
    args = parser.parse_args()

    conn = psycopg.connect(_dsn())

    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, skill_key, name, description, instructions "
            "FROM skill ORDER BY skill_key"
        )
        skills = cur.fetchall()

        cur.execute(
            "SELECT sc.skill_id, c.code, c.name "
            "FROM skill_capability sc "
            "JOIN capability c ON c.id = sc.capability_id "
            "ORDER BY c.code"
        )
        by_skill: dict[object, list[tuple[str, str]]] = {}
        for skill_id, code, name in cur.fetchall():
            by_skill.setdefault(skill_id, []).append(
                (code, name or "")
            )

    updates: list[tuple[object, ...]] = []

    for skill_id, key, name, description, instructions in skills:
        metrics = measure_skill_tokens(
            skill_key=key,
            name=name,
            description=description,
            instructions=instructions,
            capabilities=by_skill.get(skill_id, []),
        )

        print(
            f"{key:<36} {metrics.instructions_tokens:>7} "
            f"{metrics.prompt_tokens:>7}  {metrics.token_encoding}"
        )

        updates.append(
            (
                metrics.instructions_tokens,
                metrics.prompt_tokens,
                metrics.token_encoding,
                metrics.token_computed_at,
                skill_id,
            )
        )

    if updates and not args.dry_run:
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE skill SET "
                "  instructions_tokens = %s, "
                "  prompt_tokens = %s, "
                "  token_encoding = %s, "
                "  token_computed_at = %s "
                "WHERE id = %s",
                updates,
            )
        conn.commit()

    conn.close()

    verb = "dry-run" if args.dry_run else "backfilled"
    print(f"{verb}: {len(updates)} skills")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
