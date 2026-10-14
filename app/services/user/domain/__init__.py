"""
用户模块的领域层。

    schemas/    对外契约（对应 user.yaml 的 components/schemas）

**没有 entities/**：实体是"要落库的领域对象"，而本模块不落库 ——
Casdoor 是用户唯一真相。当前调用者的内部表示是
`app/common/auth/models.py::AuthenticatedUser`，它属于**机制**
（谁都能用），不属于本模块的领域模型。
"""

__all__: list[str] = []
