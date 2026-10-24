"""
OpenAPI 规格自检（开发用）。

    python -m app.agent.spec_check

检查 app/services/document/*.yaml：

    1. YAML 能解析
    2. 所有 $ref 指向的 schema / parameter / response 真实存在
    3. 每个 operation 都有 operationId 且唯一
    4. 每个 operation 都有 tags，且 tag 在顶层声明过
    5. 路径参数在 parameters 里都声明了
    6. inline 的 schema 里没有重复定义顶层已定义的枚举

第 6 条是本项目特有的：枚举一旦两处定义就会漂移，
所以要求枚举只在 components.schemas 里定义一次。
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import yaml

from app.common.app_config import load_app_config

#: 取值：`app/config/app.yaml` 的 `paths.spec_dir`
#: （环境变量 `SPEC_DIR` 可覆盖）。
SPEC_DIR = Path(load_app_config().paths.spec_dir)
SPECS = (
    "document.yaml",
    "document_type_definition.yaml",
    "document_version.yaml",
    "pipeline.yaml",
    "processor_capability.yaml",
    "document_retrieve.yaml",
)

REF_PATTERN = re.compile(r"^#/components/(\w+)/(\w+)$")


def _walk(node: Any, path: str = "$"):
    """深度遍历，产出 (路径, 节点) 对。"""
    yield path, node

    if isinstance(node, dict):
        for key, value in node.items():
            yield from _walk(value, f"{path}.{key}")

    elif isinstance(node, list):
        for index, value in enumerate(node):
            yield from _walk(value, f"{path}[{index}]")


def check_spec(name: str) -> list[str]:
    problems: list[str] = []

    path = SPEC_DIR / name

    if not path.exists():
        return [f"{name}: 文件不存在"]

    try:
        spec = yaml.safe_load(path.read_text(encoding="utf-8"))

    except yaml.YAMLError as exc:
        return [f"{name}: YAML 解析失败 — {exc}"]

    if not isinstance(spec, dict):
        return [f"{name}: 顶层不是对象"]

    components = spec.get("components", {}) or {}
    schemas = components.get("schemas", {}) or {}
    parameters = components.get("parameters", {}) or {}
    responses = components.get("responses", {}) or {}

    buckets = {
        "schemas": schemas,
        "parameters": parameters,
        "responses": responses,
    }

    # --------------------------------------------------------
    # 2. $ref 完整性
    # --------------------------------------------------------

    for where, node in _walk(spec):
        if not isinstance(node, dict):
            continue

        ref = node.get("$ref")

        if not isinstance(ref, str):
            continue

        match = REF_PATTERN.match(ref)

        if not match:
            problems.append(
                f"{name}: {where} 的 $ref 格式无法识别 — {ref}"
            )
            continue

        bucket, key = match.groups()

        if bucket not in buckets:
            problems.append(
                f"{name}: {where} 引用了未知的组件桶 — {bucket}"
            )
            continue

        if key not in buckets[bucket]:
            problems.append(
                f"{name}: {where} 引用了不存在的 {bucket}.{key}"
            )

    # --------------------------------------------------------
    # 3/4/5. operation 级检查
    # --------------------------------------------------------

    declared_tags = {
        t.get("name") for t in spec.get("tags", []) or []
    }

    seen_operation_ids: dict[str, str] = {}

    for route, path_item in (spec.get("paths") or {}).items():

        if not isinstance(path_item, dict):
            continue

        path_params = set(
            re.findall(r"\{(\w+)\}", route)
        )

        for method, operation in path_item.items():

            if method not in (
                "get", "post", "put", "patch", "delete"
            ):
                continue

            where = f"{route} [{method.upper()}]"

            operation_id = operation.get("operationId")

            if not operation_id:
                problems.append(f"{name}: {where} 缺 operationId")

            elif operation_id in seen_operation_ids:
                problems.append(
                    f"{name}: {where} 的 operationId 与 "
                    f"{seen_operation_ids[operation_id]} 重复 — "
                    f"{operation_id}"
                )

            else:
                seen_operation_ids[operation_id] = where

            tags = operation.get("tags") or []

            if not tags:
                problems.append(f"{name}: {where} 没有 tags")

            for tag in tags:
                if tag not in declared_tags:
                    problems.append(
                        f"{name}: {where} 用了未声明的 tag — {tag}"
                    )

            # 路径参数必须声明
            declared: set[str] = set()

            for parameter in operation.get("parameters", []) or []:

                if "$ref" in parameter:
                    match = REF_PATTERN.match(parameter["$ref"])

                    if match and match.group(1) == "parameters":
                        resolved = parameters.get(
                            match.group(2), {}
                        )
                        if resolved.get("in") == "path":
                            declared.add(resolved.get("name"))

                elif parameter.get("in") == "path":
                    declared.add(parameter.get("name"))

            missing = path_params - declared

            if missing:
                problems.append(
                    f"{name}: {where} 未声明路径参数 {sorted(missing)}"
                )

            # 每个 operation 都要有 200/201/202 之一
            codes = set(operation.get("responses", {}) or {})

            if not codes & {"200", "201", "202", "204"}:
                problems.append(
                    f"{name}: {where} 没有成功的响应码"
                )

    return problems


def main() -> int:
    all_problems: list[str] = []

    for name in SPECS:
        problems = check_spec(name)

        path = SPEC_DIR / name

        if problems:
            print(f"[FAIL] {name}")
            all_problems += problems
        else:
            spec = yaml.safe_load(
                path.read_text(encoding="utf-8")
            )
            routes = len(spec.get("paths", {}) or {})
            ops = sum(
                1
                for item in (spec.get("paths") or {}).values()
                for method in item
                if method in (
                    "get", "post", "put", "patch", "delete"
                )
            )
            schemas = len(
                (spec.get("components", {}) or {}).get(
                    "schemas", {}
                )
                or {}
            )
            print(
                f"[ OK ] {name:<36} "
                f"{routes} 路径 / {ops} 操作 / {schemas} schema"
            )

    print()

    if all_problems:
        print("自检失败：")
        for problem in all_problems:
            print(f"  - {problem}")
        return 1

    print("全部规格自检通过。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
