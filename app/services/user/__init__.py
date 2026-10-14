"""
用户与登录（Casdoor OAuth + 目录检索）。

规格
----
    user.yaml     OpenAPI 契约（POST /auth/login、POST /auth/callback、
                  GET /auth/me、POST /auth/refresh、
                  GET /users/search）

边界
----
**这里没有用户实体，也没有表。** Casdoor 是用户的唯一真相，本模块
只是"把 OAuth 那几步串起来 + 把当前调用者翻译成对外契约 + 按关键词
去 Casdoor 目录里捞一行选项"。因此：

    domain/schemas/   对外 DTO（没有 domain/entities/ —— 没有表）
    application/      契约与实现（实现里没有仓储参数 —— 不落库）
    （没有 repository/ 与 models/）

"这个 token 有效吗、这个人够不够格"属于**机制**，在
app/common/auth/；那些判断与业务无关，任何模块都该能直接复用
（/users/search 的 admin 角色门槛就是这么挂上去的）。
把群组 / 角色校验搬到这里来，别的模块就得反向依赖用户服务，而
app/common 与各服务模块之间的依赖方向是单向的。

依赖方向
--------
app/services/user  ->  app/common/auth          （用机制）
app/common/auth    ->  ✗ app/services/*         （纯叶子，不许反向）
"""

__all__: list[str] = []
