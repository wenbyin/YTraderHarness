"""
Casdoor 鉴权组件（通用机制，不含用户业务）。

    配置      config.py         app/config/casdoor.yaml 的读取与校验
    模型      models.py         AuthenticatedUser（验签之后拿到的调用者）
    机制      component.py      验签、群组断言、授权链接、换/刷新 token
    接线      dependencies.py   FastAPI 的 current_user

怎么用
------
业务接口只需要声明它要用户：

    from app.common.auth import AuthenticatedUser, current_user

    @router.get("/whatever")
    async def handler(
            user: AuthenticatedUser = Depends(current_user),
    ):
        ...

**但通常什么都不用写**：`app/api/v1/router.py` 把 `Depends(current_user)`
挂在聚合路由上了，所有走它注册的接口都会先校验（401 / 403）。上面那种
写法只在需要**用到用户信息**时才加，而且因为 FastAPI 的依赖缓存按
可调用对象去重，它不会导致验两遍。

**角色门槛**（"只有管理员能调"）比登录门槛高一层，得显式声明：

    async def require_admin(
            user: AuthenticatedUser = Depends(current_user),
    ) -> None:
        require_role(user, get_casdoor_component().config.admin_role)

角色取自 token 的 `roles` claim（`roles_claim` 可配），要求哪个角色
由调用方给出（一般来自配置，见 casdoor.yaml 的 admin_role）。

边界（别越过去）
----------------
本包是 app/common 的一部分，**必须保持纯叶子**：不许 import
`app.services.*` 或 `app.infra.*`（理由见 app/common/exceptions.py
开头）。尤其注意 `AuthenticatedUser` 定义在 models.py，而不是
services/user 的 DTO 里 —— 方向是"服务依赖公共"。所以这里**不要**
为了顺手去 re-export 服务层的 DTO，那会把刚绕开的反向依赖重新接上。
"""

from __future__ import annotations

from app.common.auth.component import (
    CasdoorAuthComponent,
    get_casdoor_component,
    reset_casdoor_component,
)
from app.common.auth.config import (
    DEFAULT_CONFIG_PATH,
    CasdoorConfig,
    load_casdoor_config,
    reset_casdoor_config,
)
from app.common.auth.dependencies import (
    current_user,
    extract_bearer_token,
    require_role,
)
from app.common.auth.models import AuthenticatedUser

__all__ = [
    "AuthenticatedUser",
    "CasdoorAuthComponent",
    "CasdoorConfig",
    "DEFAULT_CONFIG_PATH",
    "current_user",
    "extract_bearer_token",
    "get_casdoor_component",
    "load_casdoor_config",
    "require_role",
    "reset_casdoor_component",
    "reset_casdoor_config",
]
