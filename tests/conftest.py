"""
测试的公共引导。

为什么要把项目根插进 sys.path
---------------------------
本仓库的测试放在 `tests/` 下，而代码在 `app/` 包内。pytest 默认
（importmode=prepend）只会把测试文件所在的目录插进 sys.path —— 那是
`tests/agent/<模块>/`，不是项目根，于是 `import app.agent.model` 会
ModuleNotFoundError。

加一个 conftest.py 把项目根插进去，比要求"每次都用
`python -m pytest`（它会把 cwd 放进 sys.path）"可靠：IDE 里点单个
用例、CI 里换工作目录都不会失效。

（不改成安装成包 / src 布局：那要动整个项目结构，而这些测试是给
Model / ContextBuilder 补的，不是一次工程改造。）

为什么还要挂**替身目录**
----------------------
跨模块的用例需要别人写的假实现：图集成测试要用 Model 侧的假 provider
适配器。pytest 只把测试文件自己的目录放进 sys.path，所以那些替身
import 不到。

这里把几个替身目录一起挂上，靠**文件名不同**避免互相遮蔽：

    tests/agent/model/fakes.py           模型侧的假适配器 / 假注册表
    tests/agent/context/context_fakes.py 上下文侧的假源 / 假压缩器
    tests/agent/event/stream_fakes.py    一条内存里的 Redis Stream

最后那个是**真替身**而不是"排好的返回值"：事件流的补发/尾随边界要靠
按 id 范围查询才验得出来（见那个文件开头的说明）。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: 放共享测试替身的目录（文件名各不相同，不会互相遮蔽）。
_SUPPORT_DIRS = (
    ROOT / "tests" / "agent" / "model",
    ROOT / "tests" / "agent" / "context",
    ROOT / "tests" / "agent" / "event",
)

for _directory in reversed((ROOT, *_SUPPORT_DIRS)):

    if str(_directory) not in sys.path:
        sys.path.insert(0, str(_directory))
