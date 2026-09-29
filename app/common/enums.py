# from enum import Enum
# from utils import to_snake
#
# _REGISTRY: dict[str, type[Enum]] = {}
#
#
# class BaseEnum(str, Enum):
#     """所有要暴露给前端的枚举继承它。"""
#     __enum_key__ = ""  # dunder，不会被当成 Enum 成员
#
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         key = cls.__enum_key__ or to_snake(cls.__name__)
#         if key in _REGISTRY:
#             raise RuntimeError(f"enum key 冲突: {key}")
#         _REGISTRY[key] = cls
#
#
# def get_enum(key: str) -> type[Enum] | None:
#     return _REGISTRY.get(key)
