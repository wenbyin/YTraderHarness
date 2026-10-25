"""
数据源加载器：只认带顶层 `type` 的文件。

为什么要有这些用例
------------------
加载器原先假定被扫描的目录**只放数据源** —— 遇到没有 `type` 的 yaml
直接 `raise ValueError`，而那个 raise 在 try 之外，连一个空文件都能让
整个应用起不来。

于是当时的非数据源配置（`llm.yaml` / `event_stream.yaml` / `model.yaml`）为了躲开它，
各自被挪到了模块目录下（见 app/common/llm.py 开头的说明）。

现在判据改成"顶层有没有 `type`"：有 = 数据源，没有 = 别人的配置，
跳过。这些用例把这个判据焊住，免得日后又收紧回去。

数据源配置现在收在 `app/config/datasource/`，与 casdoor.yaml（不是
数据源）分开了，所以真实目录里已经**没有**非数据源的 yaml 可跳 ——
那个判据只剩临时目录在验（见下面 tmp_path 那几个用例）。

两种形态都覆盖：临时目录（判据本身）与真实的 datasource 目录（回归
—— 它必须在真实文件上仍然能加载出全部 6 个数据源）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from app.infra.datasource.loader import DataSourceConfigLoader


#: 真实的数据源配置目录。从 __file__ 往上推而不是写字符串：
#: 相对路径只在 CWD 恰好是仓库根时成立，IDE 里点单个用例、
#: 或从别处跑 pytest 都会失效（tests/conftest.py 已经负责把
#: 仓库根放进 sys.path，但不改 CWD）。
REAL_CONFIG_DIR = (
    Path(__file__).resolve().parents[2]
    / "app" / "config" / "datasource"
)

#: 数据源目录的父目录（app/config/）。
PARENT_CONFIG_DIR = REAL_CONFIG_DIR.parent


def _write_yaml(
        directory: Path,
        file_name: str,
        payload: Any,
) -> None:
    (directory / file_name).write_text(
        yaml.safe_dump(payload, allow_unicode=True),
        encoding="utf-8",
    )


def _redis(
        name: str = "redis",
) -> dict[str, Any]:
    """最小可用的数据源定义（RedisConfig 只有 url 是必填）。"""

    return {
        "name": name,
        "type": "redis",
        "required": False,
        "config": {"url": "redis://localhost:6379/0"},
    }


def test_non_datasource_yaml_is_skipped(
        tmp_path: Path,
) -> None:
    """casdoor.yaml 这一类（有 name、没 type）不该拦启动。"""

    _write_yaml(tmp_path, "redis.yaml", _redis())

    # 与数据源并排，但明确不是数据源 —— 有 name 也没有 type。
    _write_yaml(
        tmp_path,
        "casdoor.yaml",
        {"name": "casdoor", "endpoint": "http://localhost:8088"},
    )

    definitions = DataSourceConfigLoader(tmp_path).load_all()

    assert [d.name for d in definitions] == ["redis"]


def test_empty_file_is_skipped(
        tmp_path: Path,
) -> None:
    """
    空文件 safe_load 得 None。

    这条以前是**必炸**的路径 —— None 在 `_load_yaml` 里就被判为
    "不是 mapping" 抛掉了，而那个调用点在 try 之外。
    """

    _write_yaml(tmp_path, "redis.yaml", _redis())
    (tmp_path / "empty.yaml").write_text("", encoding="utf-8")

    definitions = DataSourceConfigLoader(tmp_path).load_all()

    assert [d.name for d in definitions] == ["redis"]


def test_non_mapping_yaml_is_skipped(
        tmp_path: Path,
) -> None:
    """顶层是列表/标量的 yaml 同样只是"不是我的" —— 跳过。"""

    _write_yaml(tmp_path, "redis.yaml", _redis())
    _write_yaml(tmp_path, "list.yaml", ["not", "a", "mapping"])

    definitions = DataSourceConfigLoader(tmp_path).load_all()

    assert [d.name for d in definitions] == ["redis"]


def test_datasource_with_bad_type_value_still_warns_and_skips(
        tmp_path: Path,
) -> None:
    """
    `type` 存在但值不认识 → 跳过该文件，其余数据源照常加载。

    宽松化只放宽了"缺 type 键"，没有把"type 的值写错"变成静默：
    这一条与宽松之前的行为一致（原先也是 warning + continue）。
    """

    _write_yaml(tmp_path, "redis.yaml", _redis())
    _write_yaml(
        tmp_path,
        "typo.yaml",
        {"name": "typo", "type": "postgresq", "config": {}},
    )

    definitions = DataSourceConfigLoader(tmp_path).load_all()

    assert [d.name for d in definitions] == ["redis"]


def test_postgresql_env_placeholder_without_default_is_fatal(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    postgresql 仍是唯一硬失败的数据源。

    加载器里的 `${NAME}`（无默认值）在变量缺失时会抛 ValueError，
    而 postgresql 的那条不走 warning 分支 —— 拦下启动是对的：
    没有库，整个应用没有意义。

    这是"宽松化没有误伤硬校验"的护栏用例。
    """

    monkeypatch.delenv("POSTGRES_DSN", raising=False)

    _write_yaml(
        tmp_path,
        "postgre.yaml",
        {
            "name": "postgresql",
            "type": "postgresql",
            "config": {"dsn": "${POSTGRES_DSN}"},
        },
    )

    with pytest.raises(ValueError):
        DataSourceConfigLoader(tmp_path).load_all()


def _has_top_level_type(path: Path) -> bool:
    """
    这个 yaml 是不是一份数据源定义（顶层有 `type`）。

    与加载器同一条判据，但**独立实现**：用例要验的正是加载器的判据，
    调加载器自己的私有方法就把被测对象当成了裁判。
    """

    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    return isinstance(raw, dict) and "type" in raw


def test_real_config_directory_loads() -> None:
    """
    回归：真实的 datasource 目录必须加载出全部 6 个数据源。

    这个用例的价值不在"能跑"，而在**它跑的是真的那个目录** ——
    配置目录被整目录扫描这件事，只有在真目录上才验得出来。

    顺带钉住"一共几个"：漏搬一个文件（比如把 neo4j.yaml 落在了旧位置）
    的表现是那个数据源起不来，而属性数据源全是 required: false ——
    只打一条 warning，应用照常启动。那是**静默降级**，必须在测试里响。
    """

    assert REAL_CONFIG_DIR.is_dir(), (
        f"找不到配置目录：{REAL_CONFIG_DIR}（测试位置变了？）"
    )

    definitions = DataSourceConfigLoader(
        REAL_CONFIG_DIR
    ).load_all()

    names = {d.name for d in definitions}

    assert names == {
        "postgresql",
        "checkpointer",
        "redis",
        "elasticsearch",
        "milvus",
        "neo4j",
    }


def test_parent_config_directory_holds_no_datasource() -> None:
    """
    反向断言：`app/config/` 根下**不许**再放数据源 yaml。

    这是"数据源收进 datasource/ 子目录"引入的**新失败模式**：以前
    app/config/*.yaml 一律被加载，现在只有 app/config/datasource/*.yaml
    被加载。于是"把配置放回父目录"从一个有效操作变成了一个**静默无效**
    的操作 —— 加载器只 glob 被交给它的那个目录，它根本不知道父目录里
    有东西，所以既不报错也不警告。

    断言必须落在**文件系统**上，不能落在加载结果上：加载结果无论如何
    都不会包含它们，看结果是看不出来的。
    """

    misplaced = sorted(
        path.name
        for path in PARENT_CONFIG_DIR.glob("*.yaml")
        if _has_top_level_type(path)
    )

    assert not misplaced, (
        f"{PARENT_CONFIG_DIR} 下有数据源定义 {misplaced}，"
        f"但它们不会被加载（加载器只扫 {REAL_CONFIG_DIR}）。"
        f"请挪进 datasource/。"
    )
