"""本地性能基准 —— `docs/BENCHMARK.md` 的数据来源。

**只测本地确定性部分,不联网。**

云端组件(embedding / rerank / LLM)的延迟随网络、额度、服务端排队波动,
复现出来的数字没有意义。那些数据由 `scripts/check_api.py` 产出,记在
`docs/API-CHECK.md`。这里负责的是**同一台机器上可复现**的那部分:CPU 上的
分词与检索,以及落到本地 SQLite 的缓存读写。

三个测点,对应三个设计决定:
  1. 分词器冷启动 —— 为什么缓存粒度取在**分词器**上,而不是索引上
  2. BM25 关键词路 —— 那一项在该规模下占总耗时多少
  3. 缓存读写    —— 落 SQLite 的读写代价是不是「比不缓存还贵」

用法:
    python backend/scripts/benchmark.py            # 只打印
    python backend/scripts/benchmark.py --write    # 同时写入 docs/BENCHMARK.md

**这些是测量,不是断言,所以不放进测试套件。** 机器一忙阈值就偶发翻车,
把「性能数字」变成一条会自己变红的测试,只会训练人忽略红色。
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import logging
import os
import platform
import statistics
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

# backend/ 加入 sys.path,使 `import app.*` 生效(脚本直接跑时没有包上下文)
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import jieba  # noqa: E402
from app.core.config import PROJECT_ROOT, Settings  # noqa: E402
from app.db.base import Database  # noqa: E402
from app.indexing.bm25 import (  # noqa: E402
    BM25Document,
    JiebaBM25,
    _shared_tokenizer,
    clear_tokenizer_cache,
)
from app.services.cache import CacheStore  # noqa: E402

# Windows 默认控制台编码是 GBK(cp936),直接 print 中文会抛 UnicodeEncodeError,
# 而且是在脚本跑到一半时抛,前面的测量结果全丢。与 check_api.py 同一处理。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover - 极老的 Python 或重定向流
        pass

# jieba 默认把「Building prefix dict … / Loading model cost 0.247 s」按 DEBUG
# 级别打到 stderr。它每次冷启动都刷一遍,既淹掉输出,又**在计时区间内做 IO**,
# 让被测量的东西受测量方式影响。关掉它再测。
jieba.setLogLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# 语料与查询
# ---------------------------------------------------------------------------

# 自定义词表。取非空是刻意的:`_shared_tokenizer` 按词表内容做缓存键,
# 空词表会退化成一个全局单例,测出来的复用效果会比真实情况更乐观。
_USER_WORDS = ("显卡", "降噪耳机", "机械键盘", "扫地机器人", "质保")

_QUERY = "RX9070 显卡支持几年质保"
_DEPTH = 20

_CATEGORIES = ("显卡", "机械键盘", "降噪耳机", "扫地机器人", "显示器")


def _corpus(size: int) -> list[BM25Document]:
    """造一份商品资料语料。

    长度取 ~150 字,比真实分块(500-800 字)短,是为了让 3000 分块这一档
    仍能在可接受时间内跑完多次取中位数。**文本越长,建索引越贵,而词典加载
    那 ~250ms 不变**,所以这里算出的占比是**偏乐观**的上界——换成长分块,
    词典加载的占比只会更低。
    """
    docs: list[BM25Document] = []
    for i in range(size):
        category = _CATEGORIES[i % len(_CATEGORIES)]
        text = (
            f"{category}型号 {i:04d} 的产品说明。"
            f"该型号支持三年质保,售价 {999 + i * 7} 元,发货地为上海仓。"
            "售后政策:七天无理由退货,十五天内出现非人为损坏可换新,"
            "保修期内免费维修。配件清单包含说明书一份、保修卡一张、数据线一根。"
            "安装前请确认电源规格与接口类型匹配。"
        )
        docs.append(BM25Document(chunk_id=i, kb_id=1, text=text))
    return docs


# ---------------------------------------------------------------------------
# 计时
# ---------------------------------------------------------------------------


def _median(samples: list[float]) -> float:
    return statistics.median(samples)


def _p95(samples: list[float]) -> float:
    if len(samples) < 2:
        return samples[0]
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]


def _timed(fn: Callable[[], object]) -> float:
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000


def _fmt(value: float) -> str:
    """毫秒,单位随量级切换(自带单位,调用方不再拼 "ms")。

    `292.00000000000006 ms` 会让人以为测到了亚皮秒,实际重复跑一次的抖动
    就有十几毫秒——报告精度不该超过测量精度。而这一层偏偏要靠**亚毫秒**的
    差别说话:统一按 1 位小数显示时,0.02ms 会变成 `0.0 ms`,读起来像没测到。
    所以小于 1ms 时换微秒。
    """
    if value < 1.0:
        return f"{value * 1000:.1f} µs"
    return f"{value:.1f} ms"


# ---------------------------------------------------------------------------
# 1. 分词器冷启动
# ---------------------------------------------------------------------------


def bench_tokenizer(runs: int = 7) -> list[tuple[str, float, float]]:
    """测 jieba 主词典加载的成本,以及复用之后它是否真的消失。

    返回 (标签, 中位数 ms, p95 ms)。
    """
    text = _QUERY
    fresh_first: list[float] = []
    fresh_second: list[float] = []
    reused: list[float] = []

    for _ in range(runs):
        # 模拟旧实现:每个请求新建一个 Tokenizer
        clear_tokenizer_cache()
        tokenizer = jieba.Tokenizer()
        fresh_first.append(_timed(functools.partial(tokenizer.lcut, text)))
        fresh_second.append(_timed(functools.partial(tokenizer.lcut, text)))

        # 新实现:按词表复用同一个 Tokenizer
        clear_tokenizer_cache()
        warm = _shared_tokenizer(_USER_WORDS)
        warm.lcut(text)  # 首次,付一次词典加载
        reused.append(_timed(functools.partial(warm.lcut, text)))

    return [
        ("新建 Tokenizer 后第一次分词", _median(fresh_first), _p95(fresh_first)),
        ("同一实例第二次分词", _median(fresh_second), _p95(fresh_second)),
        ("复用分词器后每次分词", _median(reused), _p95(reused)),
    ]


# ---------------------------------------------------------------------------
# 2. BM25 关键词路
# ---------------------------------------------------------------------------


def _bm25_pass(docs: list[BM25Document]) -> int:
    """与 `RetrievalService._search_bm25` 的内核一致:建索引 + 检索。"""
    index = JiebaBM25(user_words=_USER_WORDS)
    index.index(docs)
    return len(index.search(_QUERY, top_k=_DEPTH))


@dataclass
class _Bm25Row:
    size: int
    cold: float
    warm: float

    @property
    def load(self) -> float:
        """词典加载 = 冷启动 − 复用。差值为负说明测量噪声压过了信号。"""
        return self.cold - self.warm

    @property
    def share(self) -> float:
        return 0.0 if self.cold <= 0 else 100.0 * self.load / self.cold


def bench_bm25(sizes: tuple[int, ...], runs: int = 3) -> list[_Bm25Row]:
    rows: list[_Bm25Row] = []
    for size in sizes:
        docs = _corpus(size)

        # 冷启动:每次都清掉分词器缓存,强制重付词典加载
        cold: list[float] = []
        for _ in range(runs):
            clear_tokenizer_cache()
            cold.append(_timed(functools.partial(_bm25_pass, docs)))

        # 复用:先热一次,再计时
        _bm25_pass(docs)
        warm = [_timed(functools.partial(_bm25_pass, docs)) for _ in range(runs)]

        rows.append(_Bm25Row(size=size, cold=_median(cold), warm=_median(warm)))
    return rows


# ---------------------------------------------------------------------------
# 3. 缓存读写
# ---------------------------------------------------------------------------

# 三档载荷对应三种真实条目:
#   4096B —— 1024 维 float32 向量(embedding 缓存)
#    240B —— 20 个 chunk id 的 JSON(检索缓存)
#   4096B —— 一份带来源的答案(答案缓存,实测 5 条来源约 4182B)
_PAYLOADS: tuple[tuple[str, str, int], ...] = (
    ("embedding", "向量(1024 维 float32)", 4 * 1024),
    ("retrieval", "检索(20 个 chunk id)", 240),
    ("answer", "答案(约 4KB)", 4 * 1024),
)


@dataclass
class _CacheRow:
    label: str
    size: int
    put: float
    hit: float
    miss: float


async def bench_cache(runs: int = 30) -> list[_CacheRow]:
    """测落 SQLite 的缓存读写代价。

    **每次 `set` / `get` 都会开一个会话并提交**(见 `CacheStore` 的文档),
    所以这里量到的是「一次真实往返」,不是一个内存字典的查找。
    """
    with tempfile.TemporaryDirectory(prefix="rag-bench-") as tmp:
        root = Path(tmp)
        settings = Settings(
            _env_file=None,  # type: ignore[call-arg]
            DATA_DIR=root,
            CHROMA_DIR=root / "chroma",
            UPLOAD_DIR=root / "uploads",
            DATABASE_URL=f"sqlite:///{(root / 'bench.db').as_posix()}",
            DASHSCOPE_API_KEY="sk-bench-not-a-real-key",
            # 不设的话 Settings 每次都会警告「JWT_SECRET 仍是默认值」。
            # 这只是一次性的临时库,警告在这里是噪音。
            JWT_SECRET="bench-not-a-real-secret",
        )
        settings.ensure_dirs()
        database = Database(settings)
        await database.create_all()
        try:
            store = CacheStore(database)
            rows: list[_CacheRow] = []
            for index, (namespace, label, size) in enumerate(_PAYLOADS):
                # 前缀区分三档,否则同名键会互相覆盖,后测的档位只是在做 UPDATE
                ns = f"{namespace}{index}"
                payload = bytes(size)
                await store.set(ns, "warm", payload, ttl_seconds=60)
                await store.get(ns, "warm")  # 探热:首次调用含建连开销

                put: list[float] = []
                hit: list[float] = []
                miss: list[float] = []
                for i in range(runs):
                    key = f"k{i}"
                    start = time.perf_counter()
                    await store.set(ns, key, payload, ttl_seconds=60)
                    put.append((time.perf_counter() - start) * 1000)

                    start = time.perf_counter()
                    await store.get(ns, key)
                    hit.append((time.perf_counter() - start) * 1000)

                    start = time.perf_counter()
                    await store.get(ns, f"absent-{i}")
                    miss.append((time.perf_counter() - start) * 1000)

                rows.append(
                    _CacheRow(
                        label=label,
                        size=size,
                        put=_median(put),
                        hit=_median(hit),
                        miss=_median(miss),
                    )
                )
            return rows
        finally:
            await database.dispose()


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

_HEADER = """\
# 性能基准:分词、检索与缓存

> 由 `backend/scripts/benchmark.py` 于 {timestamp} 自动生成。
> 手工编辑会被下次运行覆盖 —— 要改数字或解释,改脚本里的模板。

**这份文档只覆盖本地确定性部分。** 云端组件(embedding / rerank / LLM)的
延迟随网络与额度波动,复现没有意义;那部分数据由 `scripts/check_api.py`
产出,记在 [`API-CHECK.md`](API-CHECK.md)。

## 环境

| 项 | 值 |
|---|---|
| 平台 | {platform} |
| Python | {python} |
| CPU 逻辑核 | {cpus} |

数字都是**中位数**,重复测量取中间值。同一台机器上换个时刻跑,抖动可以到
±15%;绝对值不必较真,要看的是**各档之间的比例**。
"""


def _render(
    tokenizer: list[tuple[str, float, float]],
    bm25: list[_Bm25Row],
    cache: list[_CacheRow],
) -> str:
    lines: list[str] = [_HEADER.format(
        timestamp=datetime.now(UTC).astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        platform=f"{platform.system()} {platform.release()}",
        python=platform.python_version(),
        cpus=os.cpu_count(),
    )]

    # ---- 分词器 ----
    lines.append("\n## 1. 分词器冷启动\n")
    lines.append("jieba 的主词典是**进程级**的,但每个新建的 `Tokenizer` 实例第一次\n")
    lines.append("分词时都要把它读进内存。与语料大小无关,是一次固定成本。\n\n")
    lines.append("| 场景 | 中位数 | p95 |\n|---|---|---|\n")
    for label, median, p95 in tokenizer:
        lines.append(f"| {label} | {_fmt(median)} | {_fmt(p95)} |\n")

    load = tokenizer[0][1] - tokenizer[1][1]
    lines.append(
        f"\n**读法**:第一行与第二行的差就是词典加载成本,约 **{_fmt(load)}**。\n"
        "第三行是分词器复用之后每次分词的实付代价。旧实现每个请求新建一个\n"
        "`Tokenizer`,于是每个问题都为同一本从不变化的词典重付一次这一项。\n"
    )

    # ---- BM25 ----
    lines.append("\n## 2. BM25 关键词路\n")
    lines.append(
        "一次完整的 `_search_bm25`:分词器 + 全量建索引 + 检索 top20。\n"
        "「冷启动」列每次都清掉分词器缓存(等价于旧实现的每次新建)。\n\n"
    )
    lines.append("| 语料规模 | 冷启动 | 分词器已热 | 差(冷 − 热) | 占比 |\n")
    lines.append("|---|---|---|---|---|\n")
    for row in bm25:
        lines.append(
            f"| {row.size} 分块 | {_fmt(row.cold)} | {_fmt(row.warm)} | "
            f"{_fmt(row.load)} | {row.share:.0f}% |\n"
        )
    lines.append(
        "\n**读法**:「差」列是两次测量的差值,**不是直测的词典加载**——纯词典加载\n"
        "的权威数字在第 1 节。差值在大语料上会**高估**它:两次测量的差别不止词典\n"
        "一项,还含建索引时的内存分配与 GC,而这一项随语料变重。所以 3000 分块\n"
        "那一格比另两格高,是差值法放大了噪声,不是词典变大了。\n\n"
        "真正要看的是一件事:这一项**不随语料规模增长**。200 → 3000 分块,建索引\n"
        "从几十毫秒涨到几百毫秒(十余倍),而词典加载始终在同一个量级。它是**固定\n"
        "成本**。缓存粒度因此取在分词器上:按索引缓存会在每次语料写入后失效、把\n"
        "固定成本重新付一遍;按分词器缓存则跨越语料变更持续有效。\n"
    )

    # ---- 缓存 ----
    lines.append("\n## 3. 缓存读写\n")
    lines.append(
        "`CacheStore` 的每次读写都自开会话并提交,量到的是一次真实往返。\n\n"
        "**量的是自持会话那种模式**(`CacheStore(db)`)。另一种\n"
        "`CacheStore.on(session)`,把写入并进调用方已有的事务,省掉开会话与\n"
        "提交——它存在的理由是别的:摄取时调用方持着 SQLite 写锁,另开会话\n"
        "会死锁(`database is locked`)。那条路径不在这里测,因为它的耗时\n"
        "取决于调用方的事务,**不是一个可以单独量出来的常数**。\n\n"
    )
    lines.append("| 载荷 | 大小 | 写入 | 命中 | 未命中 |\n|---|---|---|---|---|\n")
    for row in cache:
        lines.append(
            f"| {row.label} | {row.size} B | {_fmt(row.put)} | "
            f"{_fmt(row.hit)} | {_fmt(row.miss)} |\n"
        )
    hit_med = statistics.median([r.hit for r in cache])
    miss_med = statistics.median([r.miss for r in cache])
    lines.append(
        f"\n**读法**:三档载荷的耗时几乎一样(240 B 与 4096 B 的差别看不出来),\n"
        "说明这一层的代价**不随条目大小变化**,而是每次操作固定的一次开销:\n"
        f"开一个会话、走一次提交。命中({_fmt(hit_med)})与未命中({_fmt(miss_med)})\n"
        "几乎相同——「真的取回一行」本身几乎不花钱,开销全在开会话与提交上。\n"
        "**注意命中并不比未命中快**,差的符号在两次运行之间会翻,因为它俩的差别\n"
        "已经落进噪声里了。\n\n"
        "这条结论有两个直接后果。一是**别指望靠压缩条目大小提速**——要省就得省\n"
        "操作次数,`CacheStore.get_many` 的批量读就是为此存在的(入库时几百个分块\n"
        "逐个查会把「批量」的意义抵消掉)。二是这层的成本下限由 SQLite 往返决定,\n"
        "与它挡住的那次网络编码或 LLM 往返差着两三个数量级——这是三层缓存值得\n"
        "存在的前提;哪天换成更廉价的本地模型,这个前提需要重新测。\n"
    )

    # ---- 解释 ----
    lines.append(
        """
---

## 这些数字支持了什么

基准本身不产生价值,它只是让三个已经做出的决定变得可核对:

1. **缓存粒度取在分词器而不是索引上。** 词典加载是一次固定成本,按索引缓存
   会在每次语料写入后失效、把固定成本重新付一遍;按分词器缓存则跨越语料变更
   持续有效。第 1、2 节的差值就是这个选择的收益,而且是**可复现的**。

2. **BM25 索引仍然整份重建。** 第 2 节的「复用分词器」列里,建索引的开销随
   语料线性增长,这部分没有被缓存掉,也不该被缓存掉——增量维护 BM25 的 IDF
   要重算全局文档频次,换来的复杂度远大于它省下的时间,还会引入「删了文档
   索引里还在」这类难查的漂移。

3. **缓存值得落 SQLite。** 第 3 节说明一次往返在毫秒级,而它挡住的是网络
   编码与 LLM 生成。这个量级差是三层缓存能成立的全部依据;若哪天上层引入
   了更廉价的本地模型,这个结论需要重新测。

## 不在本文件里的数字

| 指标 | 在哪 |
|---|---|
| embedding 批量延迟、批量上限 | [`API-CHECK.md`](API-CHECK.md) |
| rerank 延迟与区分度 | [`API-CHECK.md`](API-CHECK.md) |
| 首 token 时延(TTFT) | [`API-CHECK.md`](API-CHECK.md) |
| 并发自增:读改写会丢更新 | `backend/tests/test_cache.py::TestCorpusVersion::test_concurrent_bumps_do_not_lose_updates` |

最后一行值得单说。**「并发下读改写会丢更新」不是经验之谈,是有测试盯着的。**
`CorpusVersion.bump` 用 `INSERT ... ON CONFLICT DO UPDATE`,自增在一条 SQL
语句里完成,不存在「读到的值已经过期」这个窗口;而那条测试用五路并发
(`gather` 五个 `bump()`)断言最终值就是 5。

需要说清楚的是:**测试守的是正解,不是复现错误。** 读改写版本「五路只数出 2」
是在探针里实测到的,没有写进测试——因为它依赖能否精确重现那个交错时序,
而守正解的用例不依赖。性能数字会被机器状态影响,那条不会。
"""
    )
    return "".join(lines)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

_SIZES = (200, 1000, 3000)


async def _main(write: bool) -> int:
    print("跑基准中(本地,不联网)…", file=sys.stderr)
    tokenizer = bench_tokenizer()
    bm25 = bench_bm25(_SIZES)
    cache = await bench_cache()
    text = _render(tokenizer, bm25, cache)
    print(text)

    if write:
        target = PROJECT_ROOT / "docs" / "BENCHMARK.md"
        target.write_text(text, encoding="utf-8")
        print(f"\n已写入 {target}", file=sys.stderr)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="本地性能基准")
    parser.add_argument("--write", action="store_true", help="写入 docs/BENCHMARK.md")
    args = parser.parse_args()
    return asyncio.run(_main(args.write))


if __name__ == "__main__":
    raise SystemExit(main())
