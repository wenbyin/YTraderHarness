"""
命令行启动入口。

    python run_server.py                 # 默认 localhost:8080
    python run_server.py --port 8081

真正的启动逻辑在 app/serve.py —— 那里解释了为什么
不能直接用 uvicorn CLI，以及 Windows 上事件循环的坑。

app/main.py 直接运行时走的是同一份逻辑，两个入口行为一致。
"""

from __future__ import annotations

import argparse

from app.serve import serve


def main() -> int:
    parser = argparse.ArgumentParser(
        description="YTrader Agent backend server"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    serve(
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
