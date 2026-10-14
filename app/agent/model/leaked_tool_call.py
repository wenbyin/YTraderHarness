"""
模型把**工具调用写成正文**时的出口过滤。

症状（2026-10-13 实测，run 51b31758）
------------------------------------
deepseek-v4-flash 在**思考模式**下会把工具调用写成它自家的 DSML 标记，当
普通正文吐出来。那次回复长这样（标记里的竖线是**全角**，下面按半角写）：

    开标签：  尖括号 + 两个竖线 + DSML + 两个竖线 + 空格 + 标签名
    实参：    … parameter name="query" … 爬山 大蜀山 … 闭合
    收尾：    </ 两个竖线 DSML 两个竖线  invoke>

langchain 在 `content` 里读不到 `tool_calls`，于是图判定"模型没要求调工具"，
把这段标记当**回答**交给用户 —— 那次运行 149 条 `message.delta` 与
`message.completed` 的全文都是它，而 `finish_reason` 是 `stop`、
`agent_run.status` 是 `completed`：**没有任何一处报错**。

第一道闸在模型侧（`adapters.thinking_off_extra_body`）：在**带工具 /
带结构化输出**的调用上把思考关掉（判据是调用形状）。但那只是**降低概率**，
不是保证 —— 它猜漏过一次（纯文本调用一条判据都不占）。

⚠️ 而**纯文本调用按当前配置是开着思考的**（主循环那个模型 2026-10-13 撤掉了
`disable_thinking`：思维链要留给用户看，见 model.yaml）。也就是说这一类**没有
第一道闸**，只靠下面这道出口闸兜底 —— 它不关心模型为什么这么写，只看产出。

为什么必须是**有状态**的闸，而不是一个 `replace`
---------------------------------------------
实测那次泄漏的增量分片是 1~10 个字，标记**跨分片**：先是那个左尖括号，然后
是两个竖线加 DSML，然后是空格加标签名……逐块做字符串替换时每一块单独看都
不是完整标记，于是**一块都匹配不上**；而"先攒齐再过滤"等于放弃逐 token
推流。所以这里是一个跑在字符流上的闸：只扣住**可能是记号前缀**的那几个
字符，其余立刻放行。

判据
----
    进入区段   出现那个**记号**（两个竖线夹着 DSML）—— 认记号而不是认整
               个标签：标签名是变的（calls / invoke / parameter），而闭合
               标签与开标签一样要吃掉，少认一种就会在结尾漏出一截
    区段内     **一个字都不放**
    离开区段   看见收尾标记（见 `LEAK_MARKER_END`）
    流结束时区段还开着 -> 扣着的整段丢掉（这次泄漏就是这个形状：整条回复
               除了标记没有别的内容）

⚠️ 为什么收尾标记必须精确到 invoke
--------------------------------
区段**内部**的参数闭合长得很像收尾（同样以斜杠起头、同样有那个记号），
拿"任何一个闭合"当结束就会在第一个参数之后恢复放行 —— 剩下的参数名与取值
照样漏出去，而过滤器看起来"工作正常"。所以收尾只认 ` invoke>`。

⚠️ 两种竖线都认（全角与半角）
--------------------------
库里存的是**全角**（U+FF5C，实测），而模型未必每次都用同一种宽度。两者在
等宽字体里几乎一样，肉眼分辨不了 —— 只认一种的症状是"过滤器装好了却什么
都没拦"。这里先把输入折叠成半角再匹配（**一换一**，因此匹配到的下标可以
直接用于原文切片），于是不必为每种宽度各写一份逻辑。

⚠️ 常量刻意写成**纯 ASCII**
-------------------------
折叠之后剩下的就是半角的那些字符，所以下面两个常量里一个全角字符都没有，
全角只以 `｜` 的形式出现在折叠表里。**这条纪律是有来历的**：这份模块的
第一版把标记当成字面量抄了进来，抄错成了半角竖线，而它看起来完全正常 ——
直到拿真实数据比对，才发现一处都匹配不上（`hits=0`）。标记里哪种宽度不是
靠眼睛能分辨的事，所以正文里不许出现它的字面量。

⚠️ 取舍：误伤
------------
用户**故意**问"这段标记是什么"时，回答里引用的标记也会被拦掉。这个取舍是
接受的：那类问题在本项目里几乎不存在，而没有这道闸时**每一次**泄漏都会把
几百字的垃圾当答复发给用户。真遇到时该让模型换个方式引用，而不是把闸拆了。

⚠️ 词表是**实测值**，不是规范
--------------------------
只认 deepseek 这一族 DSML 记号（本仓库主模型就是它，见 model.yaml）。别的
provider 另有写法时不在这里猜 —— 加一条实测记录，再加一条常量。
"""

from __future__ import annotations

from typing import Final

__all__ = [
    "LEAK_MARKER_END",
    "LEAK_MARKER_TOKEN",
    "LeakGate",
    "strip_leaked_tool_calls",
]


#: 标记的**核心记号**（折叠到半角之后的样子）：两个竖线夹着 DSML。
#:
#: 只认它、不认整个标签：标签名是变的（calls / invoke / parameter），而
#: 开标签与闭合标签都要吃掉。
LEAK_MARKER_TOKEN: Final[str] = "||DSML||"

#: 区段的收尾（折叠到半角之后的样子）。精确到 ` invoke>` —— 理由见模块
#: docstring：区段内部的参数闭合与它长得很像。
LEAK_MARKER_END: Final[str] = "</||DSML|| invoke>"

#: 全角竖线（U+FF5C）。**唯一**一处提到它 —— 折叠表用它。
_FULLWIDTH_BAR: Final[str] = "｜"

#: 区段的**开头**：记号本身，**连同它前面的尖括号**。
#:
#: ⚠️ 那个尖括号必须算进来
#: --------------------
#: 只从记号开始丢的话，每个标记前面的 `<`（以及闭合标签那个 `</`）会被当成
#: 正常正文放出去 —— 实测就是这么漏出 6 个字来的（`<\n<\n</`）。逐字符推
#: 流时它们还是**前一个分片**里的字符，所以只能在"发现区段"时回头把已经发
#: 出去的那一个字符取消掉 —— 做不到，那就只能从一开始就把它算进区段里。
_OPEN_TAG: Final[str] = "<" + LEAK_MARKER_TOKEN

#: 闭合标签的开头（`</…DSML…`）。
_CLOSE_TAG: Final[str] = "</" + LEAK_MARKER_TOKEN

#: 块级区段的结束：`</… invoke>` 或 `</… calls>`。
_REGION_ENDS: Final[tuple[str, ...]] = (
    "</" + LEAK_MARKER_TOKEN + " invoke>",
    "</" + LEAK_MARKER_TOKEN + " calls>",
)

#: 孤立闭合标签的结束：它自己的那个 `>` 就是终点。
_STRAY_END: Final[tuple[str, ...]] = (">",)


def _earliest(folded: str, patterns: tuple[str, ...]) -> int:
    """几个模式里最早出现的那个下标（都没有就是 -1）。"""

    found = [index for pattern in patterns if (index := folded.find(pattern)) >= 0]

    return min(found) if found else -1


def _fold(text: str) -> str:
    """
    全角竖线折叠成半角，只用来**匹配**；返回给调用方的永远是原文切片。

    一换一（一个字符换一个字符），所以折叠后的下标与原文一一对应 ——
    这是"两种宽度都认"却不用写两份逻辑的原因。
    """

    if _FULLWIDTH_BAR not in text:
        return text

    return text.replace(_FULLWIDTH_BAR, "|")


def _partial_prefix_length(folded: str) -> int:
    """
    末尾有多少个字符**可能是**区段开头的前缀（0 = 不可能是）。

    这就是"只扣住可能是前缀的那几个字符"的全部实现：把尾巴上这一段先扣
    下来，等下一个分片到齐了再判。多数分片返回 0，于是整块立刻放行 ——
    逐 token 推流的代价因此只有几个字符的延迟。
    """

    longest = 0

    for pattern in (_OPEN_TAG, _CLOSE_TAG):

        for length in range(len(pattern) - 1, 0, -1):

            if length > longest and folded.endswith(pattern[:length]):
                longest = length

    return longest


class LeakGate:
    """
    字符流上的泄漏闸。一个**流**（一次模型调用）一个实例。

    ⚠️ 不要把两路文本喂给同一个实例
    ------------------------------
    增量分片与"整块正文"（`content-block-finish` 那种）是同一段文字的两种
    形状，先后喂进去会让状态机把本该丢的区段当成新文本再判一次。整段文本
    请用 `strip_leaked_tool_calls` —— 它是自带实例的一次性版本。
    """

    __slots__ = ("_buffer", "_region_ends", "dropped", "hits")

    def __init__(self) -> None:
        self._buffer = ""

        #: 区段开着时，用什么模式判它结束（None = 没开着）。
        #:
        #: 两种区段的边界不同，所以存的是"这一次的结束模式"而不是一个布尔：
        #:
        #:     块（`<…DSML…` 开头）      丢到 `</… invoke>` 或 `</… calls>`
        #:     孤立闭合标签（`</…DSML…`） 只丢到它自己的 `>`
        #:
        #: 第二种是那个**收尾**的 `</… calls>`：它前面没有开着的块（块已经在
        #: `</… invoke>` 处结束了），所以它只是一截残渣 —— 按块处理会让区段
        #: 永远闭合不了，把**后面所有正常正文一起吃掉**（实测：泄漏后面的那句
        #: 话被吞了）。
        self._region_ends: tuple[str, ...] | None = None

        #: 命中过几处、一共丢了多少字。交给调用方去打观测 —— 本模块刻意
        #: 不 import logging：它也会被投影层用到，而那一层有自己的纪律
        #: （见 run_event_projector 的模块 docstring）。
        self.hits = 0
        self.dropped = 0

    # ----------------------------------------------------------

    def feed(self, text: str) -> str:
        """喂一段增量，返回**这一段里可以放出去的部分**（可能是空串）。"""

        if not text:
            return ""

        self._buffer += text

        return self._drain(final=False)

    def flush(self) -> str:
        """
        流结束。返回还扣着的那点文本。

        区段没开着时，扣着的只可能是"看着像记号前缀、其实不是"的几个字符
        （比如回复正好以竖线结尾）—— 那是**正常正文**，必须放出去。区段开着
        就整段丢掉：宁可少说几个字，也不要把半截标记发给用户。
        """

        return self._drain(final=True)

    # ----------------------------------------------------------

    def _drain(self, *, final: bool) -> str:

        out: list[str] = []

        while True:

            folded = _fold(self._buffer)

            if self._region_ends is not None:

                index = _earliest(folded, self._region_ends)

                if index < 0:

                    if final:
                        # 区段没闭合就结束了 —— 那次泄漏的真实形状（半截标记
                        # 同样不该发出去）。
                        self.dropped += len(self._buffer)
                        self._buffer = ""

                    return "".join(out)

                # 长度要按**匹配到的那个**模式算：`_earliest` 只返回下标，
                # 而两个模式的长度不一定相同。
                end = index + next(
                    len(pattern)
                    for pattern in self._region_ends
                    if folded.startswith(pattern, index)
                )

                self.dropped += end
                self._buffer = self._buffer[end:]
                self._region_ends = None
                continue

            index = _earliest(folded, (_OPEN_TAG, _CLOSE_TAG))

            if index < 0:

                hold = 0 if final else _partial_prefix_length(folded)

                keep = len(self._buffer) - hold

                out.append(self._buffer[:keep])
                self._buffer = self._buffer[keep:]

                return "".join(out)

            out.append(self._buffer[:index])
            self.hits += 1

            is_block = folded.startswith(_OPEN_TAG, index)

            #: 区段从**尖括号**算起，不是从记号算起（见 _OPEN_TAG 上面那段）。
            skip = len(_OPEN_TAG if is_block else _CLOSE_TAG)

            self.dropped += index + skip
            self._buffer = self._buffer[index + skip:]
            self._region_ends = _REGION_ENDS if is_block else _STRAY_END


def strip_leaked_tool_calls(text: str) -> tuple[str, int]:
    """
    整段文本的版本：返回 `(清理后的正文, 命中处数)`。

    给"手里已经有一整段字"的地方用（整块正文、整条消息）。流式的地方用
    `LeakGate` —— 两者是**同一套规则**，只是实例的生命周期不同。
    """

    if LEAK_MARKER_TOKEN[:2] not in _fold(text):
        # 快路径：绝大多数正文里连一个竖线都没有，不必建状态机。
        return text, 0

    gate = LeakGate()

    return gate.feed(text) + gate.flush(), gate.hits
