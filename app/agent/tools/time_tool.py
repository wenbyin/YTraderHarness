"""
时间查询工具：给模型一个可信的"现在"。

为什么需要它
----------
模型**不知道"现在"** —— 训练数据有截止时间，而主 agent 的纪律里写着
"数字、日期、代码必须来自上下文或工具结果。缺失就问用户，不要猜"。
"今天几号"正好落在这条上：上下文里通常没有，而猜出来的日期看起来
和真的一模一样。

⚠️ 只给"现在"，不做日期运算
------------------------
"还有几天到 10 月 20 日""这个 ISO 时间在纽约是几点"，模型拿到一个可信的
now 之后自己能算 —— 它算得不错，问题从来只是**不知道基准**。把这些也做成
工具，会把一个无状态的查询变成一串要维护的接口。

唯一的例外是**星期**与**UTC 偏移**：那两样最容易算错，而且算错看不出来
（星期错一天不像数字错一位那么刺眼）。所以它们顺手一起返回。

⚠️ 为什么总是附带 UTC
------------------
默认时区是一个**业务假设**（这个产品的用户在中国）。任何默认值都可能对
某个用户是错的，而错的表现是"回答里的时间差了几个小时"—— 调用方看不出来。
所以返回值里永远同时给两份、各自带明确的名字：指定时区的一份 + UTC 的一份。
模型至少看得见两个数，而不是收到一个不知依据的数。

本包在 TOOL_SOURCE_PACKAGES 里，所以这个工具会被自动扫到 —— 不用改注册表。
`source_ref` 按模块路径推导成 `agent`，与其它 agent 级工具同组；这里**刻意
不写** `TOOL_SOURCE_REF`（写了就等于给它单开一个"来源"，见
spec.derive_source_ref 那段关于分组的说明）。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from datetime import timezone as dt_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from langchain_core.tools import tool

__all__ = ["DEFAULT_TIMEZONE", "current_time"]


#: 默认时区。**业务假设**，不是技术选择 —— 见模块 docstring。
DEFAULT_TIMEZONE = "Asia/Shanghai"


#: 星期几的中文名。下标就是 `date.weekday()` 的取值（周一 = 0）。
_WEEKDAYS_ZH = (
    "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日",
)


@tool
async def current_time(timezone: str = DEFAULT_TIMEZONE) -> dict:
    """
    查询当前的日期与时间。用于回答"今天几号""现在几点""今天星期几"。

    你的训练数据里没有"现在"，所以**不要凭印象回答日期**。返回里同时给出
    指定时区与 UTC 两份时间，以及星期与 UTC 偏移。

    参数 timezone 用 IANA 时区名（如 Asia/Shanghai、America/New_York、
    UTC、Europe/London），省略则用 Asia/Shanghai。
    """

    zone = _resolve_timezone(timezone)

    now = datetime.now(zone)

    return {
        "timezone": timezone,
        "iso8601": now.isoformat(timespec="seconds"),
        "date": now.date().isoformat(),
        "time": now.strftime("%H:%M:%S"),
        "weekday": _WEEKDAYS_ZH[now.weekday()],
        "utc_offset": _format_offset(now.utcoffset()),
        # ⚠️ 总是附上 UTC：默认时区是业务假设，错了调用方看不出来。
        "utc_iso8601": now.astimezone(dt_timezone.utc).isoformat(
            timespec="seconds"
        ),
        "epoch_seconds": int(now.timestamp()),
    }


def _resolve_timezone(name: str) -> ZoneInfo:
    """
    时区名 -> `ZoneInfo`。不认识就**抛**，不退回 UTC。

    退回 UTC 的表现是"回答里的时间错了几个小时"，而调用方（模型）没有任何
    办法看出来 —— 它会把这个数当事实用。报错则让它换一个正确的名字重试，
    所以错误信息里给几个例子。
    """

    try:
        return ZoneInfo(name)

    except (ZoneInfoNotFoundError, ValueError) as exc:

        raise ValueError(
            f"未知的时区名 {name!r}。请用 IANA 时区名，例如 "
            "Asia/Shanghai / America/New_York / Europe/London / UTC。"
        ) from exc


def _format_offset(offset: timedelta | None) -> str:
    """
    偏移 -> `"+08:00"` 形式。

    自己算而不是 `str(offset)`：后者的负偏移与**半小时时区**（印度 +05:30）
    会给出 `"-1 day, 16:00:00"` 和 `"5:30:00"` 这种没法读的形态，而模型要
    从这里面再推一次 —— 那正是这个工具要消灭的事。
    """

    if offset is None:
        return "+00:00"

    total = int(offset.total_seconds())
    sign = "+" if total >= 0 else "-"
    total = abs(total)

    return f"{sign}{total // 3600:02d}:{(total % 3600) // 60:02d}"
