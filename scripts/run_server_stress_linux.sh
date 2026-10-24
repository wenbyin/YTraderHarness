#!/usr/bin/env bash
# ============================================================
# Linux 压测容量档：600 并发 run / SSE
# ============================================================
#
# 用法：
#     ./scripts/run_server_stress_linux.sh
#
# 不要在 Windows 的 SelectorEventLoop 上用这组值：
# Windows select() 约 512 socket 上限，进程会直接
# ValueError: too many file descriptors in select()。
#
# 前提：
#   - backend / mock / Postgres / Redis 在 Linux 上；
#   - PostgreSQL max_connections >= 500（本脚本连接预算约 350）；
#   - 进程 ulimit -n >= 65535；
#   - Redis maxclients 足够（默认 10000）。
#
# 对应压测请求：
#     load.mode = session_rate
#     session_rate = 10
#     max_concurrency = 600
#     total_sessions = 600
#     duration_seconds = 60
#     session.turns.mean = 1
#
# 如果单轮耗时仍接近 23s，600 并发约 26 runs/s，600 轮约 23s 完成；
# 但单进程 CPU 可能先饱和，实际以 metrics 为准。

set -euo pipefail

cd "$(dirname "$0")/.."

export MAX_ACTIVE_RUN_STREAMS="${MAX_ACTIVE_RUN_STREAMS:-700}"
export STREAM_MAX_CONNECTIONS="${STREAM_MAX_CONNECTIONS:-800}"
export REDIS_MAX_CONNECTIONS="${REDIS_MAX_CONNECTIONS:-100}"

export GRAPH_DISPATCHER_MAX_CONCURRENCY="${GRAPH_DISPATCHER_MAX_CONCURRENCY:-700}"
export GRAPH_DISPATCHER_QUEUE_SIZE="${GRAPH_DISPATCHER_QUEUE_SIZE:-4096}"

export CHECKPOINTER_POOL_MIN="${CHECKPOINTER_POOL_MIN:-10}"
export CHECKPOINTER_POOL_MAX="${CHECKPOINTER_POOL_MAX:-200}"
export CHECKPOINTER_POOL_SIZE="${CHECKPOINTER_POOL_SIZE:-200}"
export POSTGRES_POOL_SIZE="${POSTGRES_POOL_SIZE:-50}"
export POSTGRES_MAX_OVERFLOW="${POSTGRES_MAX_OVERFLOW:-100}"

export STRESS_MAX_CONCURRENCY="${STRESS_MAX_CONCURRENCY:-600}"
export STRESS_CONCURRENCY_SAFETY_RATIO="${STRESS_CONCURRENCY_SAFETY_RATIO:-0.9}"

PYTHON="${PYTHON:-.venv/bin/python}"

echo "stress profile:"
echo "  max_active_run_streams=$MAX_ACTIVE_RUN_STREAMS"
echo "  stream_max_connections=$STREAM_MAX_CONNECTIONS"
echo "  dispatcher=$GRAPH_DISPATCHER_MAX_CONCURRENCY"
echo "  checkpointer_pool=$CHECKPOINTER_POOL_MAX"
echo "  driver_max_concurrency=$STRESS_MAX_CONCURRENCY"
echo "  driver_safety_ratio=$STRESS_CONCURRENCY_SAFETY_RATIO"

exec "$PYTHON" run_server.py --mock
