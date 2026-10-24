"""
Stress 模块的 SQLAlchemy 模型。

import 本包 = stress_runs 与 stress_run_metrics 两张表注册完毕。

⚠️ 新增模型包时必须在 app/infra/datasource/initializer.py 的
import_all_models() 里加一行 —— 那里硬编码了要 import 的包列表。
漏加的症状是表建不出来，而且启动不报错，直到第一次查询时炸
「relation stress_runs does not exist」。
"""

from .stress_run import StressRun, StressRunMetrics, StressRunMetricsModel

__all__ = [
    "StressRun",
    "StressRunMetrics",
    "StressRunMetricsModel",
]
