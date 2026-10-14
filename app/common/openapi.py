"""
生成的 OpenAPI 文档里，把实现类名换回契约里的组件名。

⚠️ 为什么需要这一步
------------------
FastAPI 拿 **类的 `__name__`** 当 components/schemas 的键
（fastapi/_compat/v2.py 的 get_model_name_map：`normalize_name(model.__name__)`）。
Pydantic 的 `model_config = ConfigDict(title=...)` **不管用** —— 它只改 schema
体内那个 "title" 字段，defs 的键仍然由类的 core ref 推出来，也就是类名。
（实测 fastapi 0.141.1 / pydantic 2.13.5：给 DTO 加 title="Message" 之后，
生成的组件名依旧是 AgentMessage。）

于是"类改名、接口不改名"这件事在 FastAPI 里没有声明式的写法，只能生成之后
改一次 —— 就是本模块干的事。

⚠️ 名字为什么会不一致
------------------
本仓库的表名与 Python 类名跟着 agent_ 前缀走（agent_message / AgentMessage，
与 agent_session / agent_run 同一套），而对外契约的名字来自规格
（messages.yaml 声明的是 Message）。两者刻意不相等：改的只是实现，
对外契约一个字不动，钉住契约那一头的地方就是本模块这张表。

新增一项就等于宣布"这个类的改名不进接口"。反过来，想让接口也跟着改名，
就从这张表里删掉它 —— 那时生成的文档会立刻多出一个新组件名，客户端会
看到，是一次真接口变化。
"""

from __future__ import annotations

from typing import Any

__all__ = ["CONTRACT_COMPONENT_NAMES", "pin_contract_component_names"]


#: 实现类名 -> 契约组件名。
CONTRACT_COMPONENT_NAMES: dict[str, str] = {
    # messages 表与实体改叫 agent_message / AgentMessage，契约仍是 Message
    # （见 services/session/domain/schemas/agent_message.py）。
    "AgentMessage": "Message",
}


def pin_contract_component_names(schema: dict[str, Any]) -> None:
    """
    就地改写 openapi 文档：组件名与全部 `$ref` 一起换。

    **就地**是刻意的：调用发生在 FastAPI 已经把文档生成完、塞进
    `app.openapi_schema` 之后，拿到的就是那份缓存对象，复制一份再返回会
    让缓存与返回值分家。

    组件名有冲突时抛 RuntimeError 而不是覆盖 —— 覆盖会安静地吃掉另一个
    模型的 schema，而那种破坏只在客户端解析响应时才暴露。
    """

    components = schema.get("components", {}).get("schemas", {})

    renames = {
        old: new
        for old, new in CONTRACT_COMPONENT_NAMES.items()
        if old in components
    }

    if not renames:
        # 今天没有需要改名的组件。不报错：本模块的职责是"如果出现了就改"，
        # 而不是"必须出现" —— 将来那个 DTO 被别的实现换掉时不该炸启动。
        return

    for old, new in renames.items():
        if new in components:
            raise RuntimeError(
                f"契约组件名 {new!r} 已经被另一个模型占用了，"
                f"把 {old!r} 改过去会吃掉它。"
            )

    # 整个重建而不是 pop 再赋值：`update` 给新键是**追加**的，那样 Message
    # 会跑到 schemas 末尾，文档 diff 里看着像被删了又加。按原顺序重建才能
    # 让改名只体现为键本身的变化。
    rebuilt = {
        renames.get(name, name): value
        for name, value in components.items()
    }

    components.clear()
    components.update(rebuilt)

    _rewrite_refs(schema, renames)


def _rewrite_refs(node: Any, renames: dict[str, str]) -> None:
    """递归替换文档里指向被改名组件的 `$ref`（paths 与 components 都要走）。"""

    if isinstance(node, dict):
        ref = node.get("$ref")

        if isinstance(ref, str):
            for old, new in renames.items():
                prefix = f"#/components/schemas/{old}"

                # `== prefix` 之外还要认 `prefix + "/"`：指向组件内部子路径的
                # 引用（如 `Message/properties/x`）同样要跟着改。
                if ref == prefix or ref.startswith(f"{prefix}/"):
                    node["$ref"] = f"#/components/schemas/{new}{ref[len(prefix):]}"
                    break

        for value in node.values():
            _rewrite_refs(value, renames)

    elif isinstance(node, list):
        for item in node:
            _rewrite_refs(item, renames)
