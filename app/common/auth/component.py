"""
Casdoor 鉴权组件 —— 校验 token、拼授权链接、换/刷新 token，以及按
关键词查组织内的用户。

它为什么住在 app/common
----------------------
这些能力**与业务无关**：任何模块（接口、Agent 工具、后台任务）
都可能需要"这个 token 是谁的、他够不够格"，也可能需要"这个组织里
有哪些人"。放在 app/services/user 下面，别的模块就会反向依赖用户
服务，而 app/common 必须保持纯叶子（见 app/common/exceptions.py
开头那条）。用户**业务**（登录、回调、/me、用户选项）在
app/services/user，这里只有可复用的机制。

四个刻意的技术选择
------------------
1. **网络走 AsyncCasdoorSDK（aiohttp），绝不用同步 CasdoorSDK。**
   同步那个内部是 requests，在本机会被注册表里的系统代理
   127.0.0.1:7890 劫持（与 requirements.txt 里 ES 弃用 httpx 同一个
   坑），而且是阻塞调用 —— 跑在事件循环上正好是
   app/common/loop_watchdog.py 要抓的场景。

2. **验签自己用 jwt.decode，不用 SDK 的 parse_jwt_token。**
   后者每次都 `x509.load_pem_x509_certificate` 重新解析一遍 PEM；
   我们构造时解析一次拿公钥，之后每请求只是 RSA 验签。校验项与
   SDK 完全一致（同密钥 / 同算法 / 同 audience），**外加**要求 `exp`
   必须存在 —— 没有过期时间的 token 永不过期，这一点上没有理由
   跟随 SDK 的宽松。

3. **sdk 只是"怎么连"，不参与"怎么验"。** 因此 casdoor 包是惰性
   import 的：验签路径一个 SDK 都不需要，只有换 token 那两个方法才
   会 import 它。装不上 casdoor 时，受影响的只有登录流程。

4. **用户检索走 SDK 自己的 HTTP 客户端，不另起一个 session。**
   `GET /users/search` 要按 name / email / id 近似匹配，而 SDK 的
   `get_users()` **不接受任何参数**（它只会把整个组织的人全量拉回来，
   每人上百个字段）。所以那里用 SDK 的 `_session` 直接打
   `/api/get-users` 并带上 field / value —— 复用它的连接器（verify
   走同一份配置），也复用了本模块的超时与失败翻译。自己 new 一个
   aiohttp session 等于把那三样都绕开。
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from collections.abc import Callable, Coroutine, Sequence
from typing import Any, ClassVar
from urllib.parse import urlencode

import jwt
from cryptography import x509

from app.common.auth.config import (
    CasdoorConfig,
    load_casdoor_config,
)
from app.common.auth.models import AuthenticatedUser
from app.common.exceptions import (
    AuthenticationError,
    PermissionDeniedError,
)

__all__ = [
    "CasdoorAuthComponent",
    "get_casdoor_component",
    "reset_casdoor_component",
]

logger = logging.getLogger(__name__)


#: Casdoor 签 token 用的算法。SDK 里也是写死的 `["RS256"]`。
#:
#: ⚠️ 传给 `jwt.decode` 是**白名单**语义。绝不能因为它是个列表就
#: 把 `none` 之类塞进来 —— 那等于把签名校验关掉。
_ALGORITHMS: tuple[str, ...] = ("RS256",)

#: 派生 state 签名密钥时的盐。换个值会让所有在途登录失效。
_STATE_KEY_SALT = b"casdoor-state-v1"


class CasdoorAuthComponent:
    """
    进程级单例（见 get_casdoor_component）。

    无状态、线程/协程安全：持有的是配置、一个公钥、一个派生密钥，
    没有任何可变字段。
    """

    def __init__(
            self,
            config: CasdoorConfig,
    ) -> None:

        # 配置不全就在这里炸 —— 由 app/main.py 在启动期调用，
        # 因此症状是"起不来"，而不是"起得来但每个请求都 401"。
        config.require_ready()

        self._config = config

        self._public_key: Any = None
        self._state_key: bytes = b""

        if config.enabled:
            self._public_key = _load_public_key(
                config.resolve_certificate()
            )

            # state 的签名密钥**从 client_secret 派生**，而不是直接
            # 拿它当密钥：两者用途不同，派生出独立的密钥，将来任何
            # 一方泄漏都不会牵连另一方。
            self._state_key = hmac.new(
                config.client_secret.encode("utf-8"),
                _STATE_KEY_SALT,
                hashlib.sha256,
            ).digest()

    # --------------------------------------------------------
    # 只读属性
    # --------------------------------------------------------

    @property
    def config(self) -> CasdoorConfig:
        return self._config

    @property
    def enabled(self) -> bool:
        return self._config.enabled

    # --------------------------------------------------------
    # 校验
    # --------------------------------------------------------

    def verify_token(
            self,
            raw_token: str,
    ) -> AuthenticatedUser:
        """
        校验 token 并取出调用者。

        Raises:
            AuthenticationError:  token 缺失 / 签名不符 / 过期 /
                                  audience 不符 / 结构不可解析。
            PermissionDeniedError: 验签通过，但不在 required_groups 里。
        """

        if self._public_key is None:
            # 只可能是 enabled=false 时有人绕过了依赖直接调
            # （依赖在关闭鉴权时走 synthetic_user，不会到这里）。
            raise RuntimeError(
                "Casdoor 鉴权已关闭（casdoor.yaml 的 enabled: false），"
                "verify_token 不该被调用。"
            )

        token = raw_token.strip()

        if not token:
            raise AuthenticationError("访问令牌为空。")

        claims = self._decode(token)
        groups = self._extract_groups(claims)

        if not self._has_required_group(groups):
            self._log_group_mismatch(claims, groups)

            allowed = "、".join(self._config.required_groups)

            raise PermissionDeniedError(
                f"当前用户不在允许访问的群组中（需要：{allowed}）。"
            )

        return AuthenticatedUser(
            subject=str(claims.get("sub") or ""),
            name=str(claims.get("name") or ""),
            display_name=str(
                claims.get("displayName")
                or claims.get("display_name")
                or ""
            ),
            email=str(claims.get("email") or ""),
            groups=groups,
            roles=self._extract_roles(claims),
            org_id=self._extract_org(claims),
            raw_claims=dict(claims),
        )

    def _decode(
            self,
            token: str,
    ) -> dict[str, Any]:
        """验签 + 声明校验。异常按"客户端能不能自己修"分类。"""

        try:
            return jwt.decode(
                token,
                self._public_key,
                algorithms=list(_ALGORITHMS),
                audience=self._config.client_id,
                # exp 必须存在：PyJWT 默认只在"有 exp"时才校验它，
                # 于是没有 exp 的 token 等于永不过期。Casdoor 一直会
                # 带上它，所以这条要求不会伤到正常流程。
                options={"require": ["exp"]},
            )

        except jwt.ExpiredSignatureError as exc:
            raise AuthenticationError(
                "访问令牌已过期，请重新登录。"
            ) from exc

        except jwt.InvalidAudienceError as exc:
            # 这一条几乎总是配置问题，不是用户的问题。把话说到位，
            # 因为它自己报的是 "Invalid audience"，看不出该改哪里。
            raise AuthenticationError(
                "访问令牌的 audience 与本应用不符 —— "
                "casdoor.yaml 里的 application_name 多半填错了"
                "（Casdoor 按应用签发 token，audience 就是应用名）。"
            ) from exc

        except jwt.InvalidTokenError as exc:
            # 签名不符、结构损坏、算法不符等都归这里。
            raise AuthenticationError(
                f"访问令牌校验失败：{exc}"
            ) from exc

    # --------------------------------------------------------
    # 群组
    # --------------------------------------------------------

    def _extract_groups(
            self,
            claims: dict[str, Any],
    ) -> list[str]:
        """
        取归一化后的群组名。

        claim 名**可配**（groups_claim）：Casdoor 到底把群组放在哪个
        claim、值的形状如何，取决于版本与应用上 "Token fields" 的配置，
        不能假定。形状容错见 _group_names。
        """

        return _normalize_claim_names(
            claims.get(self._config.groups_claim)
        )

    def _extract_roles(
            self,
            claims: dict[str, Any],
    ) -> list[str]:
        """
        取归一化后的角色名。

        与群组同一套形状容错与去重（Casdoor 的 roles 也是"数组 / 裸
        字符串 / 对象数组"那几种形态之一），只是 claim 名独立可配。

        ⚠️ 角色**不做** required_* 校验 —— 那是 groups 的用途（见
        `_has_required_group` 与 config.required_groups）。角色在这里
        只是被带进 `AuthenticatedUser`，供**工具级**的 checkauth 拦截器
        按"这个工具要求哪些角色"去判 —— 那个要求是挂载在工具上的，
        不属于全局登录门槛。
        """

        return _normalize_claim_names(
            claims.get(self._config.roles_claim)
        )

    def _extract_org(
            self,
            claims: dict[str, Any],
    ) -> str:
        """
        取调用者所属的组织（写进 `agent_session.org_id` 的那个）。

        ⚠️ 与 `_extract_groups` / `_extract_roles` 的两点不同
        ---------------------------------------------------
        1. **不做形状归一化**。群组与角色在 Casdoor 里有数组 / 裸字符串 /
           对象数组几种形态（它甚至在只有一个元素时把数组退化成字符串），
           所以那边共用 `_normalize_claim_names`。组织是**单值标识**，
           只认字符串；别的形状一律当没有。刻意不做 `str(value)` 兜底 ——
           那样会把一个 dict 的 repr 写进 `agent_session.org_id`，而库里
           那一列看不出它是个转错型的东西。

        2. **取不到不报错**。claim 名可配（org_claim），Casdoor 各版本与
           各应用上 "Token fields" 的配置不一样，空置是**允许**的结果
           （见 AuthenticatedUser.org_id）。这里也不打日志：正常部署里
           "没有这个 claim"是常态，打一条每请求一条的 WARNING 只会把
           真问题淹掉；排查靠 `_log_group_mismatch` 打出的 claim 名单。
        """

        value = claims.get(self._config.org_claim)

        if isinstance(value, str):

            return value.strip()

        return ""

    def _has_required_group(
            self,
            groups: Sequence[str],
    ) -> bool:
        """
        命中 required_groups 里任意一个即通过。

        匹配规则见 `_matches_required`（与角色共用同一份实现 ——
        "什么算命中"只有一处定义，两边不会漂移）。
        """

        return _matches_required(
            groups,
            self._config.required_groups,
            org_name=self._config.org_name,
        )

    def has_role(
            self,
            user: AuthenticatedUser,
            role: str,
    ) -> bool:
        """
        这个调用者是否持有指定角色（`GET /users/search` 那道门槛）。

        与 required_groups **同一套匹配规则**（见 `_matches_required`）：
        裸名与 org_name 限定的形式都认，别的 owner 下的同名角色不认。

        ⚠️ 与 required_groups 的一点不同：那是**登录门槛**，本方法是
        给个别端点用的。角色刻意不进那道门槛（见 models.py 里 roles
        的说明），所以这里要显式地问，而不是像群组那样在 verify_token
        里顺手判掉。

        `user.roles` 已经由 `_normalize_claim_names` 去过重的形状容错，
        这里只做名字比对，不猜形状。
        """

        return _matches_required(
            user.roles,
            [role],
            org_name=self._config.org_name,
        )

    def _log_group_mismatch(
            self,
            claims: dict[str, Any],
            groups: Sequence[str],
    ) -> None:
        """
        403 的**客户端文案是刻意含糊的**（只说不符合），全部线索留在
        服务端日志里 —— 这里能说清"到底该改哪个配置"。

        `groups_claim` 配错是本配置最可能的失败：那会让**所有人**都
        403。所以下面把 token 里实际存在的 claim **名**打出来，照着
        grep 一下就知道该填什么。

        同时打出解析出的群组名（会落在服务端日志里）：这是排查时
        最有用的一条信息，而群组名不是秘密 —— 要按计划里"只打名字
        不打值"的严格口径，可以把它降到 debug，代价是排查变慢。
        """

        logger.warning(
            "Casdoor token 验签通过但没有要求的群组："
            "groups_claim=%r 解析出 %r，"
            "required_groups=%r，org_name=%r。"
            "token 里实际存在的 claim 名：%s",
            self._config.groups_claim,
            list(groups),
            self._config.required_groups,
            self._config.org_name,
            "、".join(sorted(claims.keys())) or "（无）",
        )

    # --------------------------------------------------------
    # 授权链接与 state
    # --------------------------------------------------------

    def authorize_url(
            self,
            *,
            redirect_uri: str,
            state: str,
            scope: str = "read",
    ) -> str:
        """
        拼 Casdoor 授权页地址（浏览器要跳过去的那个）。

        不使用 SDK 的 get_auth_link：**同步** SDK 的那个实现会为了
        拼 URL 真发一次请求（`requests.request("", url, params=...)`），
        白白走一趟网络、还顺带踩上系统代理。异步 SDK 那个不发请求，
        但要为此构造一个 SDK 实例 —— 不如四行 urlencode 来得直接。
        """

        params = urlencode(
            {
                "client_id": self._config.client_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": scope,
                "state": state,
            }
        )

        return (
            f"{self._config.resolved_front_endpoint}"
            f"/login/oauth/authorize?{params}"
        )

    def issue_state(self) -> tuple[str, str]:
        """
        生成一对 `(state, nonce)`。

        nonce 要由调用方写进 cookie —— 只有签名没有 cookie 绑定的
        state 挡不住登录 CSRF（攻击者可以自己走一遍 /auth/login 拿到
        合法 state，再把自己那次回调的 URL 喂给受害者）。理由详见
        casdoor.yaml 里 state_cookie_name 的注释。

        state 自带过期时间且**被签名覆盖**，所以不需要服务端存储。
        """

        nonce = secrets.token_urlsafe(24)
        expires_at = int(time.time()) + self._config.state_ttl_seconds

        return (
            f"{nonce}.{expires_at}.{self._sign_state(nonce, expires_at)}",
            nonce,
        )

    def verify_state(
            self,
            *,
            state: str,
            cookie_nonce: str | None,
    ) -> None:
        """
        校验回调带回来的 state。

        Raises:
            AuthenticationError: state 缺失 / 签名不符 / 已过期 /
                                 与当前浏览器的 cookie 对不上。
        """

        parts = state.split(".")

        if len(parts) != 3:
            raise AuthenticationError(
                "登录状态（state）格式不对，请重新发起登录。"
            )

        nonce, expires_raw, signature = parts

        try:
            expires_at = int(expires_raw)

        except ValueError as exc:
            raise AuthenticationError(
                "登录状态（state）格式不对，请重新发起登录。"
            ) from exc

        # 先验签，再相信里面任何一个字段（包括到期时间）。
        # compare_digest 而不是 == ：后者会在第一个不同的字节上早退，
        # 把签名变成可逐字节爆破的。
        if not hmac.compare_digest(
                signature,
                self._sign_state(nonce, expires_at),
        ):
            raise AuthenticationError(
                "登录状态（state）签名不符，请重新发起登录。"
            )

        if not cookie_nonce:
            raise AuthenticationError(
                "缺少登录状态 cookie，请重新发起登录。"
                "（回调必须由**发起登录的同一个浏览器**打开。）"
            )

        if not hmac.compare_digest(nonce, cookie_nonce):
            raise AuthenticationError(
                "登录状态（state）与当前浏览器不匹配，"
                "请重新发起登录。"
            )

        if expires_at < time.time():
            raise AuthenticationError(
                "登录请求已过期，请重新发起登录。"
            )

    def _sign_state(
            self,
            nonce: str,
            expires_at: int,
    ) -> str:
        return hmac.new(
            self._state_key,
            f"{nonce}.{expires_at}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    # --------------------------------------------------------
    # 换 token / 刷新
    # --------------------------------------------------------

    async def exchange_code(
            self,
            code: str,
    ) -> dict[str, Any]:
        """
        用授权码换 token（OAuth 回调那一步）。

        Raises:
            AuthenticationError: Casdoor 说这个 code 不行（过期/重复使用）。
            RuntimeError:        连不上 Casdoor，或它给了看不懂的东西。
        """

        result = await self._sdk_call(
            "换取访问令牌",
            lambda sdk: sdk.get_oauth_token(code=code),
        )

        return _require_access_token(result)

    async def refresh(
            self,
            refresh_token: str,
    ) -> dict[str, Any]:
        """
        用 refresh token 换一套新 token。

        Raises: 同 exchange_code。
        """

        result = await self._sdk_call(
            "刷新访问令牌",
            lambda sdk: sdk.refresh_token_request(
                refresh_token=refresh_token,
                scope="read",
            ),
        )

        return _require_access_token(result)

    async def _sdk_call(
            self,
            action: str,
            call: Callable[[Any], Coroutine[Any, Any, dict[str, Any]]],
    ) -> dict[str, Any]:
        """
        跑一次 SDK 网络调用，并把失败统一成两种语义。

        为什么要自己套超时：SDK 用的是 aiohttp 默认的 300 秒。Casdoor
        挂掉时，登录回调会把请求处理器挂满五分钟并一直占着连接。

        为什么 except Exception 而不是列举：SDK 抛的东西很杂 ——
        aiohttp 的连接类错误、拿不到 Content-Type 时的 KeyError、
        响应不是 JSON 时的 ContentTypeError。它们的共同点是"Casdoor
        那边没给出可用结果"，在这里区分毫无意义。
        **CancelledError 不受影响**（3.8 起它继承 BaseException），
        所以客户端断开仍能正常取消。

        失败为什么是 RuntimeError 而不是 AuthenticationError：
        这个错**不在客户端**，是服务端连不上身份提供方。报 401 会让
        前端以为"凭据不对"，从而反复让用户重新登录 —— 而重登多少次
        都一样。让它冒到 FastAPI 变成 500，才是诚实的信号。
        """

        sdk = self._build_sdk()

        try:
            return await asyncio.wait_for(
                call(sdk),
                timeout=self._config.request_timeout_seconds,
            )

        except Exception as exc:
            logger.error(
                "调用 Casdoor 失败（%s，endpoint=%s）：%s",
                action,
                self._config.resolved_endpoint,
                exc,
                exc_info=True,
            )

            raise RuntimeError(
                f"调用 Casdoor 失败（{action}）：{exc}。"
                f"请确认 {self._config.resolved_endpoint} 可达，"
                "以及 casdoor.yaml 里的 client_id / client_secret / "
                "application_name 是否正确。"
            ) from exc

    def _build_sdk(
            self,
    ) -> Any:
        """
        造一个 AsyncCasdoorSDK。

        ⚠️ **不要给它套 `async with`**：SDK 在 `__init__` 里就把
        `_session` 造好了，每个请求自己 `async with self._session`
        开一个 ClientSession 再关掉，生命周期是自洽的。外面再套一层
        只会多开一个用不到的空 session。

        每次调用都新建（登录/刷新是低频操作），代价是每次一条新的
        TCP 连接 —— 换取的是完全不碰连接池生命周期。

        惰性 import：验签路径不需要 casdoor 包，只有这里需要。
        """

        from casdoor import AsyncCasdoorSDK

        return AsyncCasdoorSDK(
            self._config.resolved_endpoint,
            self._config.client_id,
            self._config.client_secret,
            self._config.resolve_certificate(),
            self._config.org_name,
            self._config.application_name,
            front_endpoint=self._config.resolved_front_endpoint or None,
            verify=self._config.verify_ssl,
        )

    # --------------------------------------------------------
    # 组织内用户检索（管理端用）
    # --------------------------------------------------------

    #: 参与近似匹配的字段（按这个顺序请求、也按这个顺序合并）。
    #:
    #: ⚠️ 这三个都是**实测**过 Casdoor 服务端认的字段（见 search_users
    #: 的说明）。往里加字段之前先确认对方真的按 LIKE 过滤 —— Casdoor
    #: 对不认识的 field 返回的是**空集**而不是报错，加错了只会表现为
    #: "这个名字搜不到人"。
    _SEARCH_FIELDS: ClassVar[tuple[str, ...]] = (
        "name",
        "email",
        "id",
    )

    async def search_users(
            self,
            *,
            keyword: str,
            limit: int,
    ) -> list[dict[str, Any]]:
        """
        按 用户名 / 邮箱 / 用户 id 近似匹配本组织内的用户。

        返回的是 Casdoor 的**原始用户对象**（dict），字段裁剪是上层的事
        —— 这里只管"怎么问 Casdoor"。

        Args:
            keyword: 关键词。空串 = 不过滤，返回组织内前 limit 个。
            limit:   最多要几个。每路请求都按它取值，合并后再截断。

        Raises:
            RuntimeError: 连不上 Casdoor、它拒绝了这次查询（比如本应用
                          的凭据在那边没有读用户的权限），或返回了看不懂
                          的结构。

        ⚠️ 三个反直觉之处（都是实测出来的，别照着直觉改）
        ---------------------------------------------
        1. **必须带分页参数。** `/api/get-users` 在没有 `p` / `pageSize`
           时会**静默忽略** field / value，把整个组织的人全量返回 ——
           看起来像"过滤没生效"，其实是参数没给全。
        2. **一次只认一个字段。** 没有"多字段一起匹配"的写法，所以这里是
           三路并发请求再合并去重。
        3. **匹配是大小写敏感的子串匹配**（`yin` 命中，`YIN` 不命中）。
           直接传 `LIKE %keyword%` 的分隔符（`%` / `_`）会被 Casdoor 当
           通配符用，这里不拦：调用方是管理端，且查询范围始终是本组织。
        """

        # 三个字段各发一次，并发；都是同一个超时、同一套失败翻译。
        fields: tuple[str | None, ...] = (
            (None,) if not keyword.strip() else self._SEARCH_FIELDS
        )

        pages = await asyncio.gather(*(
            self._search_page(
                field=field,
                keyword=keyword.strip(),
                limit=limit,
            )
            for field in fields
        ))

        return _merge_users(pages, limit=limit)

    async def _search_page(
            self,
            *,
            field: str | None,
            keyword: str,
            limit: int,
    ) -> list[dict[str, Any]]:
        """跑一路查询，返回它那页原始用户对象。"""

        result = await self._sdk_call(
            f"检索用户（field={field or '全部'}）",
            lambda sdk: _fetch_users_page(
                sdk,
                owner=self._config.org_name,
                field=field,
                value=keyword,
                limit=limit,
            ),
        )

        return _require_user_rows(result)

    # --------------------------------------------------------
    # 关闭鉴权时的替身
    # --------------------------------------------------------

    def synthetic_user(self) -> AuthenticatedUser:
        """
        `enabled: false` 时 `current_user` 返回的替身。

        存在的理由很实际：接口上用 `Depends(current_user)` 取用户信息
        的地方，如果这里不返回东西就得写一堆 `| None` 分支。让关掉
        鉴权时"有个看起来正常的用户"，业务代码一行都不用改。

        它**不是**安全边界上的东西 —— 生产环境这个开关打不开
        （app/main.py 会拒绝启动）。
        """

        return AuthenticatedUser(
            subject="dev-bypass",
            name="dev-bypass",
            display_name="本地开发（Casdoor 鉴权已关闭）",
            groups=list(self._config.required_groups),
            #: 角色**刻意留空**：全局没有 `required_roles` 这种东西，
            #: 而工具级的 checkauth 要什么角色是挂载在工具上的 ——
            #: 这里凭空给几个，等于让本地开发"能跑"，生产却拒绝同一
            #: 个工具。那种差异只会在上线时才发现。
            roles=[],
            #: 组织同样留空，但理由不同：没有 token 就没有组织可言。
            #: 不拿配置里的 `org_name` 顶上 —— 那是**服务端**在哪个
            #: Casdoor 组织下（用于限定群组名前缀），不是调用者属于哪个
            #: 组织。两者混起来会让本地开发的 agent_session.org_id 里
            #: 出现一个生产上永远不会出现的值。
            org_id="",
        )


# ============================================================
# 群组 / 角色名的归一化与匹配
# ============================================================
#
# 这一段是"Casdoor 的 claim 形状与命名知识"，因此属于组件、不属于模型。


def _normalize_claim_names(
        raw: Any,
) -> list[str]:
    """
    claim 的值 -> 去重保序的候选名字。

    groups 与 roles 共用：两者的 claim 形状是同一套容错
    （见 `_group_names`），去重规则也该只有一份 —— 各写一遍的话，
    "数组里同一个名字出现两次"这件事在两边会有不同的表现。
    """

    if raw is None:
        return []

    seen: dict[str, None] = {}

    for name in _group_names(raw):
        seen.setdefault(name, None)

    return list(seen)


def _group_names(
        raw: Any,
) -> list[str]:
    """
    把 claim 的值摊平成一串候选名字。

    为什么要容错这么多种形状：Casdoor 在**只有一个元素**时会把数组
    退化成裸字符串（casdoor#4622，2.183.0 才修），而"群组只有一个"
    恰恰是最常见的部署形态 —— 不兼容的话，最该放行的那批人反而进不来。

    认这几种：

        ["Y-Trader-User"]           数组（正常形态）
        "Y-Trader-User"             裸字符串
        "A,B"                       逗号拼接（某些 Token fields 配置）
        [{"name": "Y-Trader-User"}] 对象数组
        {"name": "Y-Trader-User"}   单个对象
    """

    if isinstance(raw, str):
        # 只按逗号拆，**不按空白拆** —— 群组名里可以带空格，
        # 按空白拆会把一个名字劈成两个都不匹配的碎片。
        return [part.strip() for part in raw.split(",") if part.strip()]

    if isinstance(raw, dict):
        return _names_from_items([raw])

    if isinstance(raw, (list, tuple, set)):
        return _names_from_items(list(raw))

    return []


def _names_from_items(
        items: Sequence[Any],
) -> list[str]:
    """从数组里逐个取名字，元素可能是字符串或对象。"""

    names: list[str] = []

    for item in items:

        if isinstance(item, str):
            name = item.strip()

        elif isinstance(item, dict):
            name = str(
                item.get("name")
                or item.get("displayName")
                or ""
            ).strip()

        else:
            continue

        if name:
            names.append(name)

    return names


def _matches_required(
        names: Sequence[str],
        required: Sequence[str],
        *,
        org_name: str,
) -> bool:
    """
    `names` 里有没有命中 `required` 的任意一项。

    群组与角色**共用这一份实现** —— 两者在 Casdoor 里是同一套命名
    习惯，规则各写一遍迟早会不一致。

    匹配两种情况都算：

        name == required                      `Y-Trader-User`
        name == f"{org_name}/{required}"      `y-trader/Y-Trader-User`

    刻意不用 `endswith("/" + required)` 那种宽松写法：那会让**任何**
    owner 下的同名群组 / 角色都通过（`evil-org/Y-Trader-User` 也算），
    而名字是管理员造的，不该给这种越权留口子。

    owner 前缀不是配置里的 org_name 时怎么办：把 required 里那一项
    **写成完整限定名**即可（它走上面第一种精确匹配）。失败日志会把
    token 里的真实名字打出来，照着抄就行。

    `required` 里的空项跳过（配置里留了个空串不该等于"谁都能过"）。
    """

    org = org_name.strip()
    qualified = f"{org}/" if org else ""

    for item in required:

        target = item.strip()

        if not target:
            continue

        if target in names:
            return True

        if qualified and f"{qualified}{target}" in names:
            return True

    return False


def _require_access_token(
        result: dict[str, Any],
) -> dict[str, Any]:
    """
    确认 Casdoor 真的给了 access_token。

    必须显式查一次：SDK 的响应对非 200 的处理是 `status != 200 and
    "application/json" not in Content-Type` 才报错 —— 而 Casdoor 的
    错误响应**是** JSON，于是错误体（`{"status":"error","msg":...}`）
    会被原样返回，看起来像成功。不查就会拿到一个没有 token 的 dict，
    然后在更远的地方以看不懂的方式失败。

    归类为 AuthenticationError（401）：走到这里通常是 code 过期或
    已被用过，客户端重新发起登录即可。
    """

    if result.get("access_token"):
        return result

    message = result.get("msg") or result.get("error_description") or result

    raise AuthenticationError(
        f"Casdoor 未返回访问令牌：{message}"
    )


# ============================================================
# 用户检索 —— 与 Casdoor 的 HTTP 细节
# ============================================================
#
# 这一段是"`/api/get-users` 这个端点怎么用"的知识，因此属于组件、
# 不属于服务（服务那边只认识"关键词 -> 选项"）。


async def _fetch_users_page(
        sdk: Any,
        *,
        owner: str,
        field: str | None,
        value: str,
        limit: int,
) -> dict[str, Any]:
    """
    要一页用户。

    ⚠️ 为什么用 `sdk._session` 这个私有属性
    --------------------------------------
    SDK 的 `get_users()` **不接受任何参数** —— 它只会 `owner=<组织>`
    地把全量用户拉回来（每人上百个字段），既过滤不了也分不了页。要用
    field / value / pageSize 就只能自己打那个端点，而 `_session` 正是
    SDK 自己那个 aiohttp 客户端（SDK 内部也这么用它）。

    自己 new 一个 ClientSession 能做到同样的事，但会绕开 SDK 的
    connector（`verify` 的配置在那里生效）。代价是碰了一个私有属性 ——
    换来的是"这一层的出网行为与换 token 那条路完全一致"。

    `p` / `pageSize` **必须给**：不给时 Casdoor 会静默忽略 field /
    value（见 search_users 的第 1 条）。
    """

    params: dict[str, Any] = {
        "owner": owner,
        "p": 1,
        "pageSize": limit,
    }

    if field and value:
        params["field"] = field
        params["value"] = value

    async with sdk._session as session:

        return await session.get(
            "/api/get-users",
            headers=sdk.headers,
            params=params,
        )


def _require_user_rows(
        result: dict[str, Any],
) -> list[dict[str, Any]]:
    """
    从 Casdoor 的响应体里取用户数组。

    两种"拿不到"都抛 RuntimeError，理由是同一条：**它们都会被静默地
    当成"没人匹配"**，而"搜索永远返回空列表"这个症状跟真实原因（权限、
    版本）离得极远 ——

        status != "ok"   Casdoor **用 HTTP 200 回错误**（凭据没权限时
                         是 `{"status": "error", "msg": "Unauthorized
                         operation"}`）
        data 不是数组     对面换了形状，多半是 Casdoor 版本变了

    归类为 RuntimeError（500）而不是领域异常：这是**服务端**的配置 /
    兼容问题，客户端换什么凭据都没用。与 `_sdk_call` 对失败归类的理由
    一致。
    """

    if result.get("status") != "ok":

        message = result.get("msg") or result.get("error") or result

        raise RuntimeError(
            f"Casdoor 拒绝了用户查询：{message}。"
            "请确认本应用（casdoor.yaml 的 client_id）在 Casdoor 那边"
            "有权读取本组织的用户。"
        )

    data = result.get("data")

    if not isinstance(data, list):
        # status 是 ok 但 data 不是数组：这是"对面换了形状"，不是"没人
        # 匹配"。当空列表处理就会变成**搜索永远返回空**，而那是本文件
        # 反复警告的那类症状（见上面 status 那段）——所以这里响亮地失败。
        raise RuntimeError(
            f"Casdoor 的用户查询返回了看不懂的结构："
            f"{type(data).__name__}（期望数组）。"
            f"多半是 Casdoor 版本变了，/api/get-users 的响应形状跟着改了。"
        )

    # 单个元素不是对象就丢掉：它不是一条用户，凑不出选项，但它也不
    # 说明"对面变了"—— 一条脏数据不该把整个查询变成 500。
    return [row for row in data if isinstance(row, dict)]


def _merge_users(
        pages: Sequence[Sequence[dict[str, Any]]],
        *,
        limit: int,
) -> list[dict[str, Any]]:
    """
    把各路查询的结果并成一份选项候选。

    `pages` 里已经是 dict（`_require_user_rows` 过过一道），因此这里不
    再判形状 —— 一个函数里筛两遍，第二遍只会让人以为第一遍不管用。

    规则三条：

        按 id 去重（同一个"yin"可能同时命中 name 与 email）
        停用（isForbidden）/ 已删（isDeleted）的账号丢掉
        累到 limit 就停 —— 前端只要这么多，多取无益

    保序：先 name 那路的命中，再 email，再 id。**不是相关度排序** ——
    Casdoor 不给相关度，装作有只会让结果看起来比实际更可信。
    """

    seen: set[str] = set()
    merged: list[dict[str, Any]] = []

    for page in pages:

        for user in page:

            if user.get("isForbidden") or user.get("isDeleted"):
                continue

            # id 是 Casdoor 的主键；退到 name 只是为了让"没有 id"这种
            # 脏数据也能被去重，不至于把同一个人列两遍。
            key = str(user.get("id") or user.get("name") or "")

            if not key or key in seen:
                continue

            seen.add(key)
            merged.append(user)

            if len(merged) >= limit:
                return merged

    return merged


def _load_public_key(
        pem: str,
) -> Any:
    """
    从证书 PEM 里取出公钥。

    ⚠️ 是**证书**而不是裸公钥：Casdoor 控制台 Certs 页给的是 X.509
    证书，验签用的是证书里的公钥。传 `BEGIN PUBLIC KEY` 会在这里
    解析失败（config.require_ready 已经先挡了一道，报错更清楚）。
    """

    certificate = x509.load_pem_x509_certificate(
        pem.encode("utf-8")
    )

    return certificate.public_key()


# ============================================================
# 进程级单例
# ============================================================

_component: CasdoorAuthComponent | None = None


def get_casdoor_component() -> CasdoorAuthComponent:
    """
    取鉴权组件（懒构造 + 进程级缓存）。

    形状与 app/common/factories.py::get_service_factory、
    app/agent/model::get_model_registry 一致：模块级单例 + 一个
    reset（给测试）。

    由 app/main.py 在启动期调一次，因此"配置不全"在启动时就报出来
    （fail-closed），而不是等第一个请求。
    """

    global _component

    if _component is None:
        _component = CasdoorAuthComponent(
            load_casdoor_config()
        )

    return _component


def reset_casdoor_component() -> None:
    """
    丢掉单例（测试用）。

    ⚠️ 只丢组件，**不清配置缓存** —— 那是 reset_casdoor_config 的活。
    两个分开的原因是测试常常只想换组件（比如塞一份假配置），
    却不想让"配置读取"这件事也跟着变。
    """

    global _component

    _component = None
