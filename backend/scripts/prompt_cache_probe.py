"""prompt 缓存探针 —— `docs/PROMPT-CACHE.md` 的数据来源。

这里量的不是「缓存快不快」,而是**它到底命中没有**。服务端前缀缓存是隐式的:
没有 API 要调,没有开关要开,命中与否只体现在回包里的
`usage.prompt_tokens_details.cached_tokens` 上。这个数**在接进指标之前被整个
丢掉了**,于是「prompt 有多少是缓存供给的」在日志、SSE、指标表里都无从回答。

这个脚本回答三个具体问题:

    1. 重复同一份 prompt,读数是几?
    2. 只换**尾部**(问题),前面那段共用的资料还算命中吗?
    3. 只换**前部**(资料),system 那一段还算命中吗?

**第 3 问是生产里真正发生的情形。** 每次提问都会重新检索,资料块必然变化。
如果命中的是「从头到第一个分歧点」这段前缀,那 system 那一段(实测算下来
244 token)本该能复用;如果前缀缓存要求更长的一致性,那它就一点都省不下来。
两种结论指向完全不同的优化方向,只能量。

**探针必须用唯一前缀起步。** 缓存是**账号级**的,同一份文本之前跑过就直接命中
——上一轮的残留会让「冷启动」测出非零,于是整个实验的前提就没了。
所以下面用 uuid 拼进 system 和资料里,每次运行都是一批全新的前缀。

用法:
    python backend/scripts/prompt_cache_probe.py

**联网、花钱、依赖平台当下的行为。** 缓存策略是平台侧的实现细节,不写在
任何合约里:今天能复现的读数明天可能变了,而变的方式不会有任何报错。
这个脚本的价值在于**能重跑**,不在于某一次的结论。
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from pathlib import Path

# backend/ 加入 sys.path,使 `import app.*` 生效(脚本直接跑时没有包上下文)
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.providers.llm import build_llm  # noqa: E402

# Windows 默认控制台是 GBK(cp936),直接 print 中文会抛 UnicodeEncodeError,
# 而且是在跑到一半时抛,前面的结果全丢。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover
        pass


def _system(tag: str, lines: int = 40) -> str:
    """一段够长的、唯一的 system 提示。

    长度是关键变量之一:前缀缓存按块(实测 64 token 对齐)工作,**分歧点
    落在第一块以内**时一个块都复用不了。system 太短的话,「换资料」那一格
    测出来的永远是 0,而那说明不了任何关于 system 长度的事。
    """
    return "\n".join(
        f"{i}. 探针 {tag} 行为约束第 {i} 条:只能依据【资料】作答,并在句末标注来源编号。"
        for i in range(1, lines + 1)
    )


def _materials(tag: str, flavor: str, lines: int = 25) -> str:
    return "\n".join(
        f"[{i}] 探针{tag} 资料{flavor} 第 {i} 条:类目 显卡,价格 4299 元,"
        f"库存 12 件,三年质保。" for i in range(1, lines + 1)
    )


def _user(materials: str, question: str) -> str:
    return f"【资料】\n{materials}\n\n【问题】\n{question}"


class Probe:
    def __init__(self, llm) -> None:
        self._llm = llm
        self._rows: list[tuple[str, int, int | None]] = []

    async def call(self, label: str, messages: list[dict[str, str]]) -> None:
        result = await self._llm.acomplete(messages)
        usage = result.usage
        self._rows.append((label, usage.prompt_tokens, usage.cached_tokens))

    def report(self) -> None:
        print()
        print(f"  {'情形':<38} {'prompt':>8} {'cache_read':>11} {'占比':>7}")
        print("  " + "-" * 68)
        for label, prompt, cached in self._rows:
            if cached is None:
                shown, pct = "未上报", "—"
            else:
                shown, pct = str(cached), f"{round(cached / prompt * 100)}%" if prompt else "—"
            print(f"  {label:<38} {prompt:>8} {shown:>11} {pct:>7}")


async def main() -> int:
    print()
    print("=" * 72)
    print("  prompt 缓存探针")
    print("=" * 72)
    print(f"LLM:{settings.LLM_MODEL}  backend={settings.LLM_BACKEND}")
    if settings.LLM_BACKEND != "dashscope":
        print("  ⚠ 当前不是 dashscope 后端 —— 这个结果与百炼的缓存策略无关")
        return 2

    tag = uuid.uuid4().hex[:8]
    system = _system(tag)
    first, second = _materials(tag, "甲"), _materials(tag, "乙")
    q1, q2 = "RX9070 多少钱", "RX9070 库存多少"
    print(f"  唯一标记 {tag}  system {len(system)} 字符  资料 {len(first)} 字符")
    print("  所有前缀都是全新的:缓存是账号级的,不换标记就量不到冷启动。")

    llm = build_llm(settings)
    probe = Probe(llm)
    sys_msg = {"role": "system", "content": system}

    await probe.call("① 冷启动(全新前缀)", [sys_msg, {"role": "user", "content": _user(first, q1)}])
    await probe.call("② 原样重复", [sys_msg, {"role": "user", "content": _user(first, q1)}])
    await probe.call("③ 只换问题(尾部分歧)", [sys_msg, {"role": "user", "content": _user(first, q2)}])
    await probe.call("④ 只换资料(前部分歧)", [sys_msg, {"role": "user", "content": _user(second, q1)}])
    await probe.call("⑤ 回到 ② 那条", [sys_msg, {"role": "user", "content": _user(first, q1)}])

    # 多轮:历史原样保留并追加。**这是生产里唯一可能真的省下钱的地方**——
    # 同一会话的相邻两轮之间,前缀 [system, 上一问, 上一答] 是完全一致的。
    history = [
        {"role": "user", "content": _user(first, q1)},
        {"role": "assistant", "content": "RX9070 显卡的价格是 4299 元 [1]。"},
    ]
    await probe.call("⑥ 多轮:首轮(建立历史)", [sys_msg, *history, {"role": "user", "content": _user(first, q2)}])
    await probe.call("⑦ 多轮:再来一轮(前缀更长)", [sys_msg, *history, {"role": "user", "content": _user(first, q2)}])

    probe.report()

    print()
    print("怎么读这张表:")
    print("  * ① 是 0 才算模板成立;不是 0 说明标记没起效,整张表都不作数。")
    print("  * ② 与 ⑤ 说明**原样重复**会不会命中。")
    print("  * ③ 与 ④ 是一对,它们才是生产里的形状:③ 只动尾部、④ 只动前部。")
    print("    ④ 掉回 0 的话,说明 system 那一段**单独不足以被复用**——")
    print("    而每次提问都会重新检索,所以线上每轮都是 ④ 的形状。")
    print("  * ⑥⑦ 是多轮,也是这一层唯一可能稳定省钱的地方。")
    print()
    print("  ⚠ 这个读数**不稳定**:同一个前缀在不同时刻命中的长度会不同")
    print("    (新写入的条目有可见性延迟)。多跑几次比只看一次可靠。")
    print()
    print("    上面的读数已经接进指标:每轮问答的 cached_prompt_tokens,")
    print("    面板上的 cache_read_rate 就是它除以报过读数的 prompt token。")
    print("    所以真流量里的命中率不必靠这个脚本估。")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
