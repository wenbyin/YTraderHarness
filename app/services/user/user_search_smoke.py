"""
`GET /users/search` 的冒烟：**真打 HTTP**（ASGI），不是直接调服务。

跑法：
    python -m app.services.user.user_search_smoke

⚠️ 为什么要走 HTTP，不能只调服务
------------------------------
直接 `await service.search_users(...)` 会绕过 FastAPI 的 response_model
校验与序列化。那样即使组装函数把**原始 Casdoor 对象**当 DTO 返回、字段
名拼错、枚举没转成字符串，也一律全绿 —— 而那些问题在真实请求上就是
500，或者前端拿到一个带 accessToken 的用户对象。

所以这里用 httpx 的 ASGITransport 真发请求（不启真端口），让 FastAPI
把响应按 user.yaml 的契约校验一遍。断言也据此写：**先断类型，再断取值**。

⚠️ 挂的是 admin_router，且 current_user 被换掉
--------------------------------------------
protected_router 上挂着鉴权依赖，而本冒烟要验的是这个接口本身的行为与
它那**一道角色门槛**，不是 token 验签（那有单独的用例）。所以：

    app.include_router(user_admin_router)   # 角色门槛在它上面，要真跑
    app.dependency_overrides[current_user]  # 身份由用例直接给

换掉 current_user 之后，角色门槛仍在起作用 —— 403 那几条正是靠它验的。

⚠️ 一半用例要真的 Casdoor
------------------------
检索本身是**问 Casdoor 要数据**，没有替身可打（打替身就等于验"我对
Casdoor 的想象"，见 agent_run_repo_smoke.py 开头那段）。所以第二段用例
连真的 localhost:8088；连不上就 skip，并说清跳过了什么 —— 绿色的输出
不该让人以为它验过。
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx
from fastapi import FastAPI

from app.api.v1.users import admin_router
from app.common.auth import AuthenticatedUser, current_user
from app.common.factories import build_factories, reset_factories

FAILS: list[str] = []

#: 响应里**允许**出现的键。多一个都算漏了原始对象出去。
OPTION_KEYS = {"id", "name", "display_name", "email"}

#: 与 casdoor.yaml 的 admin_role / org_name 默认值一致。
ADMIN_ROLE = "Y-Trader-Admin"
ORG_NAME = "YTraderHarness"


def check(label: str, ok: bool, extra: str = "") -> None:
    print(f"  {'ok  ' if ok else 'FAIL'} {label}{(' — ' + extra) if extra else ''}")
    if not ok:
        FAILS.append(label)


def section(title: str) -> None:
    print(f"\n== {title} ==")


def skip(label: str, why: str) -> None:
    """
    明说这条**没验**。

    不写的话，绿色的输出会让人以为它验过 —— 与"连不上就 skip"同一个
    理由：跳过要说出来（见 tests/api/test_run_events_stream.py 的口径）。
    """

    print(f"  skip {label} — {why}")


def _unreachable_session_factory() -> Any:
    """
    本冒烟的服务**不碰库**，但 ServiceFactory 需要一个 session_factory
    才肯被构造。

    给个会炸的而不是真连接：万一将来有人给 UserServiceImpl 加了仓储参数，
    这里会立刻以一句看得懂的话失败，而不是静默地连上库跑起来 ——
    "用户模块不落库"这件事因此也被这条用例钉住了。
    """

    raise AssertionError(
        "UserService 不该开 session（它不落库）——"
        "若刚给它加了仓储参数，这个冒烟要跟着改。"
    )


def _user(roles: list[str]) -> AuthenticatedUser:
    return AuthenticatedUser(
        subject="smoke-user",
        name="smoke",
        display_name="冒烟",
        email="smoke@example.com",
        groups=["Y-Trader-User"],
        roles=roles,
    )


def _build_app(roles: list[str]) -> FastAPI:
    app = FastAPI()
    app.include_router(admin_router)
    app.dependency_overrides[current_user] = lambda: _user(roles)
    return app


async def _get(
        app: FastAPI,
        *,
        keyword: str | None = None,
        limit: int | None = None,
) -> httpx.Response:
    params: dict[str, Any] = {}

    if keyword is not None:
        params["keyword"] = keyword

    if limit is not None:
        params["limit"] = limit

    async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://smoke",
    ) as client:
        return await client.get("/users/search", params=params)


# ============================================================
# 一、角色门槛（不碰 Casdoor）
# ============================================================


async def _check_role_gate() -> None:
    section("角色门槛")

    # 没有角色。
    response = await _get(_build_app([]), keyword="x")
    check("没有角色 -> 403", response.status_code == 403, f"实际 {response.status_code}")

    # 有别的角色，但没有 admin_role。
    response = await _get(_build_app(["Y-Trader-User"]), keyword="x")
    check(
        "只有 Y-Trader-User -> 403",
        response.status_code == 403,
        f"实际 {response.status_code}",
    )

    # ⚠️ 别的 owner 下的同名角色**不算**：与群组同一套匹配，
    # 不做"取 / 之后那段"的宽松匹配（那会让任何组织都给过）。
    response = await _get(_build_app([f"evil-org/{ADMIN_ROLE}"]), keyword="x")
    check(
        f"evil-org/{ADMIN_ROLE} -> 403",
        response.status_code == 403,
        f"实际 {response.status_code}",
    )

    # 403 的文案要说清"缺什么"，否则前端只能干瞪眼。
    response = await _get(_build_app([]), keyword="x")
    check(
        "403 说明缺哪个角色",
        ADMIN_ROLE in response.text,
        response.text[:120],
    )

    # 用户传来的参数毛病要在**角色门槛之内**再判：这几条用带角色的身份
    # 发，否则 403 会先挡掉，验不到参数校验本身。
    app = _build_app([ADMIN_ROLE])

    for label, kwargs in (
        ("limit=0 -> 422", {"limit": 0}),
        ("limit=51 -> 422", {"limit": 51}),
        ("keyword 超长 -> 422", {"keyword": "x" * 129}),
    ):
        response = await _get(app, **kwargs)
        check(label, response.status_code == 422, f"实际 {response.status_code}")


# ============================================================
# 二、真检索（要 Casdoor）
# ============================================================


async def _check_search(app: FastAPI) -> bool:
    """
    跑真的检索。Casdoor 不可用时返回 False（调用方据此说明跳过了什么）。
    """

    section("检索（真 Casdoor）")

    # 先探一次：连不上 / 本应用没权限读用户时是 500。
    probe = await _get(app, keyword="yin_wen_bin")

    if probe.status_code != 200:
        check(
            "Casdoor 可达且本应用可读用户",
            False,
            f"实际 {probe.status_code}：{probe.text[:160]}",
        )
        return False

    rows = probe.json()

    check("命中至少一个用户", isinstance(rows, list) and len(rows) >= 1, f"{len(rows)} 条")

    if not rows:
        return True

    first = rows[0]

    check(
        "每一项只有白名单字段",
        all(set(row) == OPTION_KEYS for row in rows),
        f"实际 {sorted(set(first))}",
    )
    check(
        "字段类型都是字符串",
        all(
            isinstance(row.get(key), str)
            for row in rows
            for key in OPTION_KEYS
        ),
    )
    check("id 非空", bool(first.get("id")), repr(first.get("id")))

    matched_id = first.get("id")

    # ---- 三路匹配各自真的接上了 ----
    #
    # 这几条是**这个接口的核心主张**：name / email / id 都能近似匹配。
    # 断言到"命中的是同一个 id"，而不是"结果非空" —— 后者在只接上一路
    # 的情况下也会绿（另外两路返回空，合并后仍然非空）。
    #
    # ⚠️ 关键词要**只可能命中那一路**，否则验不出东西：本实例的邮箱是
    # `yin_wen_bin@163.com`，拿 `yin` 当关键词，三路都可能命中，绿了也
    # 说明不了邮箱那一路接上了。所以下面各挑一段，并在挑不出来时明说
    # 跳过（本实例的 name 恰好是 email 的前缀，name 那一路就只能弱验）。

    name = str(first.get("name") or "")
    substring = await _get(app, keyword=name[:3])
    check(
        "用户名子串命中同一个人",
        _contains_id(substring, matched_id),
        f"name={name!r}，{_ids(substring)}",
    )

    email = str(first.get("email") or "")

    #: `@` 之后那段只可能命中 email 那一路。
    domain = email.rsplit("@", 1)[-1].split(".")[0] if "@" in email else ""

    if not domain:
        skip("邮箱子串命中同一个人", "这个用户没有邮箱，或邮箱里没有域名段")

    elif domain in name:
        skip("邮箱子串命中同一个人", f"域名段 {domain!r} 也在用户名里，区分不开")

    else:
        by_email = await _get(app, keyword=domain)
        check(
            "邮箱子串命中同一个人",
            _contains_id(by_email, matched_id),
            f"关键词={domain!r}，{_ids(by_email)}",
        )

    #: id 前 8 位基本不可能出现在用户名或邮箱里。
    if matched_id:
        prefix = str(matched_id)[:8]
        by_id = await _get(app, keyword=prefix)
        check(
            "用户 id 子串命中同一个人",
            _contains_id(by_id, matched_id),
            f"关键词={prefix!r}，{_ids(by_id)}",
        )

    # ---- 边界 ----

    upper = await _get(app, keyword=str(first.get("name")).upper())
    check(
        "大写关键词不命中（大小写敏感，是 Casdoor 的行为）",
        not _contains_id(upper, matched_id),
        f"实际 {_ids(upper)}",
    )

    empty = await _get(app, keyword="zzz-不存在的人-zzz")
    check(
        "无匹配 -> 200 + 空列表（不是 404）",
        empty.status_code == 200 and empty.json() == [],
        f"实际 {empty.status_code} {empty.text[:80]}",
    )

    listed = await _get(app, keyword="", limit=5)
    check(
        "空关键词 = 不过滤，返回前 limit 个",
        listed.status_code == 200
        and isinstance(listed.json(), list)
        and len(listed.json()) <= 5,
        f"实际 {listed.status_code}，{len(listed.json())} 条",
    )

    capped = await _get(app, keyword="", limit=1)
    check(
        "limit=1 -> 最多 1 条",
        capped.status_code == 200 and len(capped.json()) <= 1,
        f"实际 {len(capped.json())} 条",
    )

    # 组织限定的角色形式同样算命中（与群组同一套规则）。
    qualified = await _get(_build_app([f"{ORG_NAME}/{ADMIN_ROLE}"]), keyword="")
    check(
        f"{ORG_NAME}/{ADMIN_ROLE} -> 放行",
        qualified.status_code == 200,
        f"实际 {qualified.status_code}",
    )

    return True


def _ids(response: httpx.Response) -> list[str]:
    if response.status_code != 200:
        return [f"<{response.status_code}>"]

    return [str(row.get("id"))[:8] for row in response.json()]


def _contains_id(response: httpx.Response, user_id: Any) -> bool:
    if response.status_code != 200:
        return False

    return any(row.get("id") == user_id for row in response.json())


# ============================================================


async def main() -> int:
    # 本服务不落库，但工厂是 Service[契约] 取服务的必经之路。
    build_factories(_unreachable_session_factory)

    try:
        await _check_role_gate()

        app = _build_app([ADMIN_ROLE])
        searched = await _check_search(app)

        if not searched:
            print(
                "\n  跳过：Casdoor 不可达（或本应用在那边读不了用户），"
                "检索那部分没验 —— 上面的 FAIL 说明是哪一种。"
            )

    finally:
        reset_factories()

    if FAILS:
        print(f"\n失败 {len(FAILS)} 项：")
        for failure in FAILS:
            print(f"  - {failure}")
        return 1

    print("\n全部通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
