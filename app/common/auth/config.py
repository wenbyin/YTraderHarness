"""
Casdoor 配置（app/config/casdoor.yaml）的读取与校验。

读取方式与仓库里其它非数据源配置一致（app/common/llm.py、
app/agent/event/config.py）：`${VAR:-default}` 展开 + pydantic
`extra="forbid"` + 按路径缓存。改 yaml 要重启才生效。

与 llm.yaml 不同的是验证时机
---------------------------
llm.py 刻意**不**在加载时校验密钥是否为空（"只跑检索、不跑 LLM 的
场景不该被一条缺失的 key 拦在门外"）。这里反过来：**启动期就校验**。

因为鉴权组件是**每个请求的必经之路** —— 没有"只跑一部分、用不到它"
的场景。配置不全时提前失败，症状是"起不来"，而不是"起得来但每个
请求都 401"——后者会被当成客户端或 Casdoor 的问题去查，离原因很远。

校验分两层：
    load_casdoor_config()   只解析，不判断能不能用（可被测试直接调）
    CasdoorConfig.require_ready()  判断能不能用（由组件构造时调用）
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field

from app.common.utils import expand_env

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "CasdoorConfig",
    "load_casdoor_config",
    "reset_casdoor_config",
]

#: 默认配置文件位置。
#:
#: ⚠️ **全仓唯一一个"配置不在读它的模块旁边"的非数据源配置**
#: —— llm.yaml / event_stream.yaml / model.yaml 都用 `Path(__file__)
#: .with_name(...)` 住在模块目录下。这里是刻意的例外：集成配置属于
#: "部署时要改的东西"，该和 redis.yaml / postgre.yaml 待在一起。
#:
#: parents 的下标：config.py → auth → common → app，所以 [2] 是 app/。
DEFAULT_CONFIG_PATH = (
        Path(__file__).resolve().parents[2] / "config" / "casdoor.yaml"
)


class CasdoorConfig(BaseModel):
    """整份 casdoor.yaml。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(
        default="casdoor",
        description="数据源加载器会跳过这一条（它没有 type），名字只为可读。",
    )

    enabled: bool = Field(
        default=True,
        description=(
            "总开关。false = 跳过全部校验（仅本地开发）。\n\n"
            "APP_ENV=prod 且这里是 false 时，app/main.py 会拒绝启动。"
        ),
    )

    endpoint: str = Field(
        default="http://localhost:8088",
        description="Casdoor 服务地址（后端换 token 走这里）。",
    )

    front_endpoint: str = Field(
        default="",
        description="浏览器看到的地址，用于拼授权链接。空 = 同 endpoint。",
    )

    client_id: str = Field(default="")

    client_secret: str = Field(
        default="",
        description=(
            "同时是 OAuth state 的签名密钥派生源（见 component.py）。"
            "因此它**不是**可有可无的：没有它登录流程起不来。"
        ),
    )

    org_name: str = Field(
        default="",
        description="组织名。也用于识别 owner 限定的群组名（`org/group`）。",
    )

    application_name: str = Field(
        default="",
        description=(
            "应用名。**它同时是 JWT 的 audience** —— 填错的表现是"
            "验签全过、audience 校验全挂。"
        ),
    )

    certificate: str = Field(
        default="",
        description="内联的 X.509 证书 PEM。与 certificate_file 二选一，本项优先。",
    )

    certificate_file: str = Field(
        default="",
        description="证书 PEM 的路径。",
    )

    redirect_uri: str = Field(
        default="",
        description=(
            "OAuth 回调地址（前端页面，Casdoor 验完会把浏览器跳到这里"
            "并带上 code 与 state）。\n\n"
            "**由配置给出，不由调用方传**：让请求方指定 redirect_uri 就是"
            "开了一个开放重定向的口子。Casdoor 那边也会校验这个地址是否"
            "登记过，两处都不该放松。"
        ),
    )

    groups_claim: str = Field(
        default="groups",
        description=(
            "装群组的 claim 名。默认 `groups`；若你的 Casdoor 用的不是"
            "这个名字，改这里即可（失败时日志会打出实际有哪些 claim）。"
        ),
    )

    roles_claim: str = Field(
        default="roles",
        description=(
            "装角色的 claim 名。默认 `roles`（Casdoor 的默认叫法）。\n\n"
            "与 groups 一样可配：值的形状（数组 / 裸字符串 / 对象数组）"
            "由同一套归一化处理，见 app/common/auth/component.py 的"
            "_group_names。\n\n"
            "⚠️ 与 groups_claim 的**唯一区别**：角色不做 required_* 校验"
            "（那是 groups 的用途），它只是被带进 AuthenticatedUser，"
            "供工具级的 checkauth 拦截器使用。"
        ),
    )

    org_claim: str = Field(
        default="org",
        description=(
            "装**组织**的 claim 名。默认 `org`；取不到就是空（这一项"
            "**不做** required 校验，与 groups 不同）。\n\n"
            "⚠️ 与 groups_claim / roles_claim 的两点差异：\n"
            "  1. 只认**字符串**。那两项有形状容错（数组 / 裸字符串 / "
            "对象数组），组织没有 —— 别的形状一律当没有，不做 str() "
            "兜底（把一个字典转成字符串写进库，谁也认不出那是什么）。\n"
            "  2. 配错的表现是静默的：`agent_session.org_id` 恒为 NULL。"
            "排查时看服务端日志里打出的 claim 名单（见 groups_claim），"
            "或直接查库。"
        ),
    )

    required_roles: list[str] = Field(
        default_factory=list,
        description="必须命中其中之一才放行（any-of）。",
    )

    required_groups: list[str] = Field(
        default_factory=list,
        description="必须命中其中之一才放行（any-of）。",
    )

    admin_role: str = Field(
        default="Y-Trader-Admin",
        description=(
            "管理端端点要求的角色（目前只有 GET /users/search）。\n\n"
            "⚠️ 与 required_groups 的分工：那个是**登录门槛**，不过就"
            " 403、整棵树都进不来；本项只在显式声明了它的那几个端点上"
            "生效，其余接口不受影响。\n\n"
            "匹配语义与群组同一套（见 component.has_role）：token 的"
            " roles 里出现裸名，或用配置的 org_name 限定的 "
            "`org/role` 形式，都算命中；**别的** owner 下的同名角色不算"
            "（宽松的「取 / 之后那段」会让任何组织的同名角色都通过）。\n\n"
            "配错的表现是那几个端点 403；服务端日志会打出 token 里"
            "实际持有的角色名，照着抄进本项即可。"
        ),
    )

    query_token_param: str = Field(default="access_token")

    query_token_paths: list[str] = Field(
        default_factory=list,
        description=(
            "允许从 query string 取 token 的路径（fnmatch 通配）。\n\n"
            "只该放 EventSource 这类**发不了请求头**的端点，"
            "因为 query 里的 token 会进访问日志与浏览器历史。"
        ),
    )

    state_ttl_seconds: int = Field(default=600, gt=0)

    state_cookie_name: str = Field(default="y_trader_oauth_state")

    verify_ssl: bool | str = Field(
        default=True,
        description=(
            "校验 Casdoor 的 TLS 证书。与 requests 的 verify 同语义："
            "True / CA bundle 路径 / False（不安全，仅测试）。"
        ),
    )

    request_timeout_seconds: float = Field(
        default=10.0,
        gt=0,
        description=(
            "调 Casdoor 的超时。**必须有** —— SDK 用的是 aiohttp 默认"
            "的 300 秒，Casdoor 挂掉时会把请求处理器挂满五分钟。"
        ),
    )

    # --------------------------------------------------------
    # 派生值
    # --------------------------------------------------------

    @property
    def resolved_endpoint(self) -> str:
        """
        去掉末尾斜杠的 endpoint。

        拼路径时是 `f"{endpoint}/login/oauth/authorize"`，配置里多写
        一个 `/` 就会得到 `//login/...`。Casdoor 多半会容忍，但
        "容忍"不是保证，而成本只是这里一行。
        """

        return self.endpoint.rstrip("/")

    @property
    def resolved_front_endpoint(self) -> str:
        """浏览器侧地址。空则回退到 endpoint。"""

        return (self.front_endpoint or self.endpoint).rstrip("/")

    def resolve_certificate(self) -> str:
        """
        取证书 PEM 的正文。

        优先内联的 `certificate`，否则读 `certificate_file`。

        ⚠️ 必须挡住"两个都没配"这种情况：`expand_env` 对未设置的变量
        返回**空串**（app/common/utils/env.py —— 它刻意没有"必填"概念），
        pydantic 也接受空串，于是配置看起来是合法的。不在这里拦，
        就会一直到 `x509.load_pem_x509_certificate` 才炸，而那时它是
        被当成"token 无效"处理的 —— 症状是**每个请求都 401**，
        跟证书的事一点关系都看不出来。
        """

        if self.certificate.strip():
            return self.certificate

        if self.certificate_file.strip():

            path = Path(self.certificate_file)

            if not path.exists():
                raise ValueError(
                    f"casdoor.yaml 里 certificate_file 指向的文件"
                    f"不存在：{path}。"
                )

            return path.read_text(encoding="utf-8")

        raise ValueError(
            "Casdoor 鉴权已启用，但没有配置验签证书。二选一：\n"
            "  1. 把证书 PEM 粘进 app/config/casdoor.yaml 的 "
            "`certificate`\n"
            "  2. 把 PEM 文件路径填进 `certificate_file`\n"
            "  3. 用环境变量覆盖：CASDOOR_CERTIFICATE 或 "
            "CASDOOR_CERTIFICATE_FILE\n\n"
            "证书从 Casdoor 控制台（Certs 页）复制，必须是**完整的 "
            "X.509 证书**（含 BEGIN/END CERTIFICATE），不是裸公钥。"
        )

    def require_ready(self) -> None:
        """
        启用状态下，把"根本用不了"的配置挡在启动期。

        只查**缺了就一定跑不通**的项，不查任何"可能配错"的语义问题
        （比如 application_name 写错 —— 那个只有在真拿 token 来验时
        才发现，见 groups_claim 那条注释）。
        """

        if not self.enabled:
            return

        missing = [
            key for key in (
                "endpoint",
                "client_id",
                "client_secret",
                "org_name",
                "application_name",
                "redirect_uri",
            )
            if not getattr(self, key).strip()
        ]

        if missing:
            raise ValueError(
                f"casdoor.yaml 缺少必填项：{'、'.join(missing)}。"
                "这些值来自 Casdoor 控制台（Applications / "
                "Organizations 页），可用同名环境变量覆盖："
                + "、".join(
                    f"CASDOOR_{key.upper()}" for key in missing
                )
                + "。\n"
                  "（本地开发若无所谓，可在 casdoor.yaml 里设 "
                  "enabled: false 跳过鉴权 —— 但生产环境这样会拒绝启动。）"
            )

        if not self.required_groups:
            raise ValueError(
                "casdoor.yaml 的 required_groups 是空的 —— 那等于"
                "任何通过验签的 token 都放行，群组校验形同虚设。"
                "请写上允许访问的群组名（如 Y-Trader-User），"
                "确实不想校验就该显式关掉整个鉴权（enabled: false）。"
            )

        if not self.admin_role.strip():
            raise ValueError(
                "casdoor.yaml 的 admin_role 是空的 —— 管理端端点"
                "（目前是 GET /users/search）会因此对**所有人**关门，"
                "而且是静默的：调用方只看到 403，看不出是配置留了空。"
                "请写上要求的角色名（如 Y-Trader-Admin）；"
                "不想要这类端点就该把它们从路由上摘掉，"
                "而不是把门槛配空。"
            )

        pem = self.resolve_certificate()

        if "BEGIN CERTIFICATE" not in pem:
            raise ValueError(
                "casdoor.yaml 里的证书看起来不是 X.509 证书 PEM"
                "（没找到 `BEGIN CERTIFICATE`）。\n"
                "注意 `BEGIN PUBLIC KEY` 是**裸公钥**，不是证书 —— "
                "从 Casdoor 控制台的 Certs 页复制完整证书。"
            )


@lru_cache(maxsize=4)
def load_casdoor_config(
        path: str | None = None,
) -> CasdoorConfig:
    """
    读配置并展开环境变量占位符。

    **不做 require_ready 校验** —— 那是组件的构造步骤。这样测试可以
    直接构造一份"故意的坏配置"来验校验逻辑本身，不必先绕过它。

    按路径缓存：同一份配置在一次进程里只解析一次。
    """

    file_path = Path(path) if path else DEFAULT_CONFIG_PATH

    if not file_path.exists():
        raise ValueError(
            f"Casdoor 配置文件不存在：{file_path}。"
            "该文件随代码一起提供，缺失说明部署不完整。"
        )

    raw = yaml.safe_load(
        file_path.read_text(encoding="utf-8")
    ) or {}

    return CasdoorConfig.model_validate(expand_env(raw))


def reset_casdoor_config() -> None:
    """
    清掉配置缓存（测试用）。

    存在的理由与 app/common/factories.py::reset_factories、
    app/infra/graph/thread_lock.py::reset_thread_run_lock 相同：
    进程级缓存会让"改了环境变量再读一次"这个动作静默失效，
    而那个症状（测试之间互相污染）比多写一个函数难查得多。
    """

    load_casdoor_config.cache_clear()
