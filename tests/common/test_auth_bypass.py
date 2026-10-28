"""
DEV 下「不接 Casdoor」的边界。

验的是四件事：

    1. `CASDOOR_ENABLED=false` 能关掉鉴权（不必改被 gitignore 的 yaml）
    2. DEV 缺配置文件时自动按「关闭」处理；非 DEV 仍然失败
    3. 关闭后 current_user 直接给合成用户，但角色默认仍为空
    4. 必须问 Casdoor 的接口报 503，而不是 500

⚠️ 这些用例**不能**证明生产安全 —— 生产护栏单独测（见
test_prod_refuses_disabled_auth）。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

import app.bootstrap.auth as bootstrap_auth_module
import app.common.auth.dependencies as auth_dependencies_module
import app.services.user.application.user_service_impl as user_impl_module
from app.bootstrap.auth import initialize_auth
from app.common.auth import (
    CasdoorAuthComponent,
    CasdoorConfig,
    current_user,
    load_casdoor_config,
    reset_casdoor_config,
)
from app.common.exceptions import ServiceUnavailableError
from app.common.utils import to_http_error
from app.services.user.application.user_service_impl import UserServiceImpl


def _disabled_config(**overrides: object) -> CasdoorConfig:
    return CasdoorConfig(enabled=False, **overrides)


def test_enabled_switch_is_env_driven(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """yaml 里写 `${CASDOOR_ENABLED:-true}`，本地用环境变量关闭。"""

    path = tmp_path / "casdoor.yaml"
    path.write_text(
        'name: casdoor\n'
        'enabled: "${CASDOOR_ENABLED:-true}"\n',
        encoding="utf-8",
    )

    monkeypatch.setenv("CASDOOR_ENABLED", "false")
    reset_casdoor_config()

    assert load_casdoor_config(str(path)).enabled is False

    monkeypatch.delenv("CASDOOR_ENABLED", raising=False)
    reset_casdoor_config()

    assert load_casdoor_config(str(path)).enabled is True

    reset_casdoor_config()


def test_missing_config_falls_back_to_disabled_in_dev(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """DEV 下没有 casdoor.yaml = 不接 Casdoor，而不是起不来。"""

    monkeypatch.setenv("APP_ENV", "dev")
    reset_casdoor_config()

    config = load_casdoor_config(str(tmp_path / "missing.yaml"))

    assert config.enabled is False

    reset_casdoor_config()


def test_missing_config_still_fails_outside_dev(
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STAGING / PROD 缺文件仍然视为部署不完整。"""

    monkeypatch.setenv("APP_ENV", "prod")
    reset_casdoor_config()

    with pytest.raises(ValueError, match="不存在"):
        load_casdoor_config(str(tmp_path / "missing.yaml"))

    reset_casdoor_config()


def test_synthetic_user_roles_come_from_dev_roles() -> None:
    """
    角色默认空（与生产同形）；要本地调管理端，必须显式配 dev_roles。
    """

    component = CasdoorAuthComponent(
        _disabled_config(
            required_groups=["Y-Trader-User"],
            dev_roles=["Y-Trader-Admin"],
        )
    )

    user = component.synthetic_user()

    assert user.subject == "dev-bypass"
    assert user.groups == ["Y-Trader-User"]
    assert user.roles == ["Y-Trader-Admin"]


def test_synthetic_user_has_no_roles_by_default() -> None:
    component = CasdoorAuthComponent(_disabled_config())

    assert component.synthetic_user().roles == []


def test_current_user_bypasses_token_check(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """关闭鉴权时，不带 Authorization 也能进业务接口。"""

    component = CasdoorAuthComponent(
        _disabled_config(
            required_groups=["Y-Trader-User"],
        )
    )

    monkeypatch.setattr(
        auth_dependencies_module,
        "get_casdoor_component",
        lambda: component,
    )

    app = FastAPI()

    @app.get("/who")
    async def who(
            user=Depends(current_user),
    ) -> dict[str, str]:
        return {"subject": user.subject}

    response = TestClient(app).get("/who")

    assert response.status_code == 200
    assert response.json() == {"subject": "dev-bypass"}


def test_prod_refuses_disabled_auth(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """生产环境关掉鉴权必须响亮的失败，而不是裸奔。"""

    component = CasdoorAuthComponent(_disabled_config())

    monkeypatch.setattr(
        bootstrap_auth_module,
        "get_casdoor_component",
        lambda: component,
    )
    monkeypatch.setenv("APP_ENV", "prod")

    with pytest.raises(RuntimeError, match="拒绝"):
        initialize_auth()


def test_dev_allows_disabled_auth(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    component = CasdoorAuthComponent(_disabled_config())

    monkeypatch.setattr(
        bootstrap_auth_module,
        "get_casdoor_component",
        lambda: component,
    )
    monkeypatch.setenv("APP_ENV", "dev")

    assert initialize_auth() is component


def test_casdoor_backed_calls_report_503(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    登录 / 用户目录在关闭后是「依赖不可用」，不是 500。

    合成用户只解决取值，解决不了「Casdoor 上才有这些数据」。
    """

    component = CasdoorAuthComponent(_disabled_config())

    monkeypatch.setattr(
        user_impl_module,
        "get_casdoor_component",
        lambda: component,
    )

    service = UserServiceImpl()

    with pytest.raises(ServiceUnavailableError):
        service._require_enabled("登录相关接口")

    mapped = to_http_error(
        ServiceUnavailableError("Casdoor 不可用")
    )

    assert mapped.status_code == 503
