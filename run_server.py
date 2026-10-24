"""
命令行启动入口。

    python run_server.py                 # 默认 localhost:8080
    python run_server.py --port 8081
    python run_server.py --instances 4   # 4 个独立进程

真正的启动逻辑在 app/serve.py —— 那里解释了为什么不能直接用
uvicorn CLI，以及 Windows 上事件循环的坑。

⚠️ `--instances` 不是 uvicorn `--workers`
----------------------------------------
这些状态都是**进程内**的：

    per-thread run lock
    cancellation registry
    run liveness
    SSE run stream registry

所以不能在一个端口后面跑 N 个 worker。`--instances N` 启动的是 N 个
**完全独立的 backend 进程**，各自有独立端口和独立 mock server：

    API   : base_port + i
    mock  : base_mock_port + i

它们共享 Postgres / Redis，但不共享进程内锁和 SSE 注册表。
压测时每个实例创建自己的 stress run，target 指向自己的端口；
普通前端流量不能直接 round-robin，需要按 session/run 做粘性路由。
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from app.serve import serve


def _child_env(
        *,
        api_port: int,
        mock_port: int,
) -> dict[str, str]:
    """一个子实例的环境变量：端口、mock 地址、Redis 客户端名。"""

    env = os.environ.copy()

    env["MOCK_MODEL_PORT"] = str(mock_port)
    env["MOCK_MODEL_BASE_URL"] = f"http://127.0.0.1:{mock_port}/v1"
    env["REDIS_CLIENT_NAME"] = f"ytrader-{api_port}"

    # 多进程时不要让每个进程的 torch/OpenMP 再各开满线程；否则
    # N 个进程 × N 个线程会严重超订 CPU。
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("TORCH_NUM_THREADS", "1")

    return env


def _run_instances(
        *,
        host: str,
        base_port: int,
        base_mock_port: int,
        count: int,
        mock: bool | None,
) -> int:
    """拉起 N 个独立 backend 进程。"""

    script = str(Path(__file__).resolve())
    cwd = str(Path(__file__).resolve().parent)

    processes: list[tuple[int, int, subprocess.Popen]] = []

    try:
        for index in range(count):
            api_port = base_port + index
            mock_port = base_mock_port + index

            command = [
                sys.executable,
                script,
                "--host",
                host,
                "--port",
                str(api_port),
            ]

            if mock is not None:
                command.append("--mock" if mock else "--no-mock")

            process = subprocess.Popen(
                command,
                env=_child_env(
                    api_port=api_port,
                    mock_port=mock_port,
                ),
                cwd=cwd,
            )

            processes.append((api_port, mock_port, process))

            print(
                f"[instances] {index + 1}/{count} "
                f"API=http://{host}:{api_port} "
                f"mock=http://127.0.0.1:{mock_port}/v1",
                flush=True,
            )

        while processes:
            for item in list(processes):
                api_port, mock_port, process = item
                code = process.poll()

                if code is None:
                    continue

                print(
                    f"[instances] API port {api_port} "
                    f"(mock {mock_port}) 退出，code={code}",
                    flush=True,
                )
                processes.remove(item)

            if processes:
                time.sleep(1)

        return 0

    except KeyboardInterrupt:
        print("\n[instances] 收到中断，正在停止子进程…", flush=True)
        return 130

    finally:
        for _, _, process in processes:
            if process.poll() is None:
                process.terminate()

        for _, _, process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YTrader Agent backend server"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument(
        "--mock",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "启用进程内 mock 模型服务（OpenAI-compatible）。"
            "默认跟随 app/config/mock_model.yaml。"
        ),
    )
    parser.add_argument(
        "--mock-port",
        type=int,
        default=None,
        help=(
            "mock server 基础端口。单实例时直接使用；"
            "--instances N 时为每实例递增。默认 18080。"
        ),
    )
    parser.add_argument(
        "--instances",
        type=int,
        default=1,
        help=(
            "启动 N 个完全独立的 backend 进程（不是 uvicorn workers）。"
            "每个进程有独立 event loop、mock server、dispatcher 和进程内锁，"
            "共享 Postgres / Redis。"
        ),
    )
    args = parser.parse_args()

    if args.instances < 1:
        parser.error("--instances 必须 >= 1")

    if args.instances > 1 and args.reload:
        parser.error("--reload 不能与 --instances > 1 一起使用")

    base_mock_port = args.mock_port or 18080

    if args.instances == 1:
        if args.mock_port is not None:
            os.environ["MOCK_MODEL_PORT"] = str(base_mock_port)
            os.environ["MOCK_MODEL_BASE_URL"] = (
                f"http://127.0.0.1:{base_mock_port}/v1"
            )

        serve(
            host=args.host,
            port=args.port,
            reload=args.reload,
            mock=args.mock,
        )

        return 0

    return _run_instances(
        host=args.host,
        base_port=args.port,
        base_mock_port=base_mock_port,
        count=args.instances,
        mock=args.mock,
    )


if __name__ == "__main__":
    raise SystemExit(main())
