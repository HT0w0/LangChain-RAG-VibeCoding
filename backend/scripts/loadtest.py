"""并发压力测试:模拟 100 人在线使用。

**这个脚本和 `benchmark.py` 是两件事,不要合并。**

`benchmark.py` 测的是**单请求的本地确定性开销**(分词器、BM25、缓存),单线程、
不联网、不起服务。它回答「这一步要多久」。

本脚本回答的是另一个问题:**「很多请求一起来的时候,系统会怎样」**。所以它要起
一个真的 uvicorn、发真的 HTTP、开真的 SSE 流。

## 口径(赛前定死,不许看到曲线再改)

用户给的口径是「**100 人在线,其中十几人同时提问**」,不是「100 人同时提问」。
这个应用**没有任何推送通道**,「在线」只体现在「正在发请求」上,所以:

  * 「在线人数」与「并发请求数」是两个数,**两个都要报、各自标清楚**。
    只报一个,两种数法都讲得通。

## 两层 profile,不要混着比

  * `mock` —— `LLM_BACKEND=mock` 等,把全部外呼换成本地实现。它隔离出的是
    **本机骨架**的并发上限(事件循环、线程池、SQLite 写锁、SSE 连接)。
  * `real` —— 真打百炼。它测的是**端到端**的表现。

**两层的绝对延迟不能放进一张表比**:mock 的生成耗时里有 `MockLLM` 每 6 字
`sleep(0.01)` 的**固定合成成分**,而真实那层有网络方差。能比的是**趋势**
(同一层内不同并发档之间),不是跨层的绝对值。

## 同机压测的偏差方向

生成器与被测服务抢同一台机器。绝大多数偏差是把服务器测得**更差**
(CPU 争抢、热降频),那是保守的。**但有一个例外朝着「好看」的方向偏**:

    httpx 默认 `max_keepalive=20`。HTTP/1.1 下每条 SSE 独占一条 TCP,
    keepalive 不够时请求会在池里排队 —— 你以为在压 100 并发,实际同时在飞
    的远少于 100,服务器被压得比你以为的轻。

所以本脚本**显式放大连接池**,并且报告里记的是**实测 in-flight 并发峰值**,
不是「目标并发数」。峰值远低于目标 = 这次压测作废,得重来。

用法:
    python backend/scripts/loadtest.py --scenario browse --users 100
    python backend/scripts/loadtest.py --scenario qa --users 100 --concurrency 12
    python backend/scripts/loadtest.py --profile real --scenario qa --concurrency 5
    python backend/scripts/loadtest.py --base-url http://127.0.0.1:8000 --scenario browse
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import os
import random
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# 控制台编码。Windows 默认 GBK,中文输出会抛 UnicodeEncodeError ——
# 而且是在跑到一半的时候抛,前面测出来的数字全丢。与 `dev.py` 同一个理由。
for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, OSError):
        _stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# 赛前定死的口径
# ---------------------------------------------------------------------------

#: 丢掉预热的前几个请求。事先定死,不许事后调。
WARMUP_REQUESTS = 20

#: 膝盖判据:p95(TTFT) 涨到最低档的多少倍。
#:
#: **用 TTFT 不用总时长** —— 总时长被 LLM 的大常数延迟盖住,会稀释掉本地排队
#: 的信号。声明的是「延迟膝盖」,不是吞吐膝盖。
KNEE_FACTOR = 3.0

#: 每个并发档固定发多少个请求(不用固定时长:那会让高档样本被时间截断)。
#: p95 至少要 100+ 个样本,否则它只是「倒数第二个最差」。
REQUESTS_PER_LEVEL = 200

#: 压测用的提问。**事先固定**,不许随机生成 —— 缓存命中率会随时间漂移,
#: 那是会改变被测系统的隐藏自变量,不是噪声。
QUESTIONS = (
    "RX9070 显卡多少钱",
    "27寸显示器还有货吗",
    "退换货政策是怎样的",
    "发货要多久",
    "有没有便宜一点的显卡",
    "显示器的分辨率是多少",
)


def _pct(samples: list[float], percent: float) -> float:
    """最近秩分位数 —— 与 `app/services/metrics.py` 的 `percentile` 同一口径。

    不用线性插值:最近秩**总是返回真实存在的观测值**,而「p95 = 5.5 毫秒」
    这种东西在一个真实系统里是编出来的。

    ⚠️ **这里写错过一次,症状极其隐蔽**:第一版写成
    `-(-percent // 100 * len(ordered))`,括号放错,乘法被卷进了整除 ——
    于是 p50 和 p95 都算成 `ordered[n-1]`,**两个分位数打印出来一模一样**。
    我在第一次 100 并发跑完时才发现,因为「p50 = p95 = 715.0ms」不可能。
    教训:下面用 `math.ceil` 直写,不用那个 `-(-a // b)` 的小聪明 ——
    它省不下一个字符,却能省掉一次这样的排查。
    """
    if not samples:
        return 0.0
    ordered = sorted(samples)
    rank = max(1, math.ceil(percent / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]


def _fmt(value: float) -> str:
    if value < 1.0:
        return f"{value * 1000:.1f} µs"
    if value >= 1000:
        return f"{value / 1000:.2f} s"
    return f"{value:.1f} ms"


def _selftest() -> int:
    """量测原语的自检。`python backend/scripts/loadtest.py --selftest`

    **为什么必要**:`_pct` 写错过一次(括号位置),p50 和 p95 都算成最大值,
    而脚本一声不吭地报出一个漂亮的数字。这类错误不报错、只是结果不对 ——
    正是本项目最讨厌的那一类。所以口径要有一组固定的用例钉着。
    """
    cases: list[tuple[str, bool]] = []

    data = list(range(1, 101))  # 1..100,分位数就是它自己
    cases.append(("p50(n=100)", _pct(data, 50) == 50))
    cases.append(("p95(n=100)", _pct(data, 95) == 95))
    cases.append(("p99(n=100)", _pct(data, 99) == 99))
    cases.append(("p100(n=100)", _pct(data, 100) == 100))
    cases.append(("空样本", _pct([], 95) == 0.0))
    # 最近秩,不是插值:n=7 的 p99 应当取第 7 个(最大值),而不是 6.93 插值。
    cases.append(("p99(n=7)=最大值", _pct([1.0, 2, 3, 4, 5, 6, 7], 99) == 7))
    # **判别性用例**:p50 与 p95 必须给出不同的数。第一版那个 bug 会让它们相同。
    cases.append(("p50 ≠ p95", _pct(data, 50) != _pct(data, 95)))

    # 判据看的是 **TTFT**,所以夹具必须填 `ttft_ms`。
    # (第一版这里只填了 `total_ms`,于是每档的 TTFT 都是空集、p95 全是 0,
    #  用例红了。**那是夹具的错,不是判据的错** —— 判据本来就该看 TTFT。)
    def level(conc: int, ttft: float, fail: int = 0) -> tuple[int, Outcome]:
        outcome = Outcome(f"并发 {conc}")
        outcome.samples = [
            Sample(ok=True, total_ms=ttft * 2, ttft_ms=ttft) for _ in range(100 - fail)
        ] + [Sample(ok=False, total_ms=30_000.0, error="HTTP 500") for _ in range(fail)]
        return conc, outcome

    # 单调上升:基线 10,3 倍是 30 —— 从第 3 档(50)起越线且此后单调不降。
    cases.append((
        "单调上升时找得到延迟膝盖",
        find_latency_knee(
            [level(1, 10.0), level(10, 12.0), level(50, 40.0), level(100, 90.0)], factor=3.0
        ) == (10, 50),
    ))
    cases.append((
        "平的曲线没有延迟膝盖",
        find_latency_knee(
            [level(1, 10.0), level(10, 10.0), level(100, 10.0)], factor=3.0
        ) is None,
    ))
    # **判别性用例**:越线之后又掉回来的曲线不该算膝盖 —— 那只是单点毛刺,
    # 判据要求「从该档起单调不降」。
    cases.append((
        "越线后回落不算延迟膝盖",
        find_latency_knee([level(1, 10.0), level(10, 40.0), level(50, 12.0)], factor=3.0) is None,
    ))

    # ---- 崩溃点(主判据) ----
    cases.append((
        "崩溃点:第一个失败率超线的档",
        find_breakdown([level(1, 5.0), level(20, 80.0), level(50, 200.0, fail=90)]) == (20, 50),
    ))
    cases.append((
        "全都不失败就没有崩溃点",
        find_breakdown([level(1, 5.0), level(100, 200.0)]) is None,
    ))
    # **判别性用例**:失败的档绝不能被当成延迟数据点 —— 那些样本是
    # 「排队 30 秒后的超时」,混进来会把曲线整个带歪。
    cases.append((
        "崩溃的档不参与延迟膝盖计算",
        find_latency_knee(
            [level(1, 10.0), level(50, 10_000.0, fail=90)], factor=3.0
        ) is None,
    ))

    # ---- 服务端根因提取 ----
    # 它如果**静默地什么都没捞到**,文档就会缺掉「根因」那一节,而我不会知道 ——
    # 所以它也得有自检。(`Target.__init__` 不起进程,可以直接构造。)
    probe = Target(profile="mock", port=0, data_dir=Path("."))
    for raw in (
        "2026-10-10 17:12:33,123 INFO uvicorn running on http://127.0.0.1:8000",
        # traceback 的标题行:夹具里**必须有它**,否则上面那条「不算根因」的
        # 用例是空过的 —— 没有输入就无所谓判据。
        "Traceback (most recent call last):",
        "sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached",
        "sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached",
        "    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^",
        "ValueError: 别的异常",
        # 同一个原因的**两条**,只有 cancel scope 的地址不同 ——
        # 它们必须并成一条、计数 2。不并的话「计数」这个功能就是摆设。
        "asyncio.exceptions.CancelledError: Cancelled via cancel scope 25206a33610",
        "asyncio.exceptions.CancelledError: Cancelled via cancel scope 2527e6e9a90",
    ):
        # 走和 `_drain` 同一条路径 —— 测的是真实链路,不是绕过它。
        clipped = Target._clip_cause(raw)  # noqa: SLF001
        if clipped is not None:
            probe._causes[clipped] += 1  # noqa: SLF001
    causes = probe.root_causes()
    cases.append(("根因:抓到异常行", any("QueuePool" in c for c in causes)))
    cases.append(("根因:按次数归并", any(c.startswith("2 次") for c in causes)))
    cases.append(("根因:不捞普通日志行", not any("uvicorn running" in c for c in causes)))
    cases.append(("根因:三种异常都抓到", len(causes) == 3))
    cases.append(
        (
            "根因:同一原因的不同地址并成一条(计数 2)",
            any(c.startswith("2 次") and "CancelledError" in c for c in causes),
        )
    )
    cases.append(
        ("根因:traceback 的标题行不算根因", not any("most recent call last" in c for c in causes))
    )

    failed = 0
    print("\n量测原语自检\n")
    for name, ok in cases:
        if not ok:
            failed += 1
        print(f"  {'✅' if ok else '❌'} {name}")
    print(f"\n合计 {len(cases)} 项:通过 {len(cases) - failed},失败 {failed}\n")
    return 1 if failed else 0


@dataclass
class Stats:
    """一组样本的分布。**永远报分布,不报均值。**"""

    name: str
    samples: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    def row(self) -> dict[str, float]:
        s = self.samples
        return {
            "n": self.n,
            "p50": _pct(s, 50),
            "p90": _pct(s, 90),
            "p95": _pct(s, 95),
            "p99": _pct(s, 99),
            "max": max(s) if s else 0.0,
        }


@dataclass
class Sample:
    """一次请求的结果。"""

    ok: bool
    total_ms: float
    ttft_ms: float | None = None
    status: int | None = None
    error: str | None = None
    #: `done.stage_timings`。**服务端只给均值,而阻塞体现在尾部** ——
    #: 所以阶段级的 p99 要在客户端这边自己算。
    stages: dict[str, float] | None = None
    #: **重排到底有没有真的跑。取自 `sources` 事件,不是 `done`。**
    #:
    #: 这里连错两回,记下来免得再踩:
    #:   ① 先想用 `stage_timings` 里有没有 `rerank` 这个键 —— **不行**,
    #:      `retrieval.py:207-208` 是无条件记时的,不管后端是不是 Noop 键都在。
    #:   ② 再想用 `done.reranked` —— **也没有**,`DoneEvent` 上没有这个字段;
    #:      它是 `SourcesEvent.reranked`(`rag.py:685`)。
    reranked: bool | None = None


class Outcome:
    """一轮的统计容器,含 in-flight 并发峰值。"""

    def __init__(self, label: str) -> None:
        self.label = label
        self.samples: list[Sample] = []
        self._inflight = 0
        self.peak_inflight = 0

    def enter(self) -> None:
        self._inflight += 1
        self.peak_inflight = max(self.peak_inflight, self._inflight)

    def exit(self) -> None:
        self._inflight -= 1

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def done(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]

    @property
    def failed(self) -> list[Sample]:
        return [s for s in self.samples if not s.ok]

    def errors_by_kind(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.failed:
            key = s.error or f"HTTP {s.status}"
            out[key] = out.get(key, 0) + 1
        return out

    def latency(self, *, ttft: bool = False) -> Stats:
        if ttft:
            vals = [s.ttft_ms for s in self.done if s.ttft_ms is not None]
        else:
            vals = [s.total_ms for s in self.done]
        return Stats(self.label, vals)


# ---------------------------------------------------------------------------
# HTTP 客户端
# ---------------------------------------------------------------------------


class Api:
    """一个复用连接的 httpx 客户端,外加 in-flight 计数。

    **连接池必须显式放大。** 见文件头那条:httpx 的 `DEFAULT_LIMITS` 是
    `max_connections=100, max_keepalive_connections=20`(**实测过这个常量**,
    虽然裸的 `httpx.Limits()` 打印出来是 `None`)—— keepalive 只有 20 时,
    「100 并发」实际只有 20 条连接在飞,服务器被压得比你以为的轻。
    这是唯一一个把结果测得**更好看**的偏差,方向与其他偏差相反,最容易漏。
    """

    def __init__(self, base_url: str, *, pool: int) -> None:
        import httpx

        self._httpx = httpx
        # 上限给到并发数 + 余量:浏览类的短请求与长 SSE 流共用这个池子,
        # 卡在正好等于并发数会让短请求排在长流后面。
        #
        # 参数名是 `max_keepalive_connections`,不是 `max_keepalive` ——
        # 写错会直接 TypeError,这个还算好的;真正难查的是它**默认为 20**
        # 而没人注意。
        limit = pool + 50
        self._client = httpx.AsyncClient(
            base_url=base_url,
            limits=httpx.Limits(
                max_connections=limit,
                max_keepalive_connections=limit,
                # 默认 5 秒。压测的突发之间有停顿,5 秒会把连接池清空、
                # 下一轮全部重新握手 —— 那是生成器自己的开销,不是被测系统的。
                keepalive_expiry=30.0,
            ),
            # SSE 流可能很长,读超时给足;连接超时短一些,连不上要立刻知道。
            timeout=httpx.Timeout(120.0, connect=10.0),
            headers={"User-Agent": "loadtest"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def get(self, url: str, **kw: Any) -> Any:
        return await self._client.get(url, **kw)

    async def post(self, url: str, **kw: Any) -> Any:
        return await self._client.post(url, **kw)

    @contextlib.asynccontextmanager
    async def open_stream(
        self, payload: dict[str, Any], headers: dict[str, str]
    ) -> Any:
        """打开一条 SSE 流,交出响应对象。

        S4 要的是**每个 token 到达的时刻**,而 `stream_chat` 把它们聚合成一个
        `Sample` 就丢了。与其在 `Api` 上开个后门让调用方直接摸 `_client`,
        不如把「开一条流」这件事本身做成公开接口。
        """
        async with self._client.stream(
            "POST", "/api/chat/stream", json=payload, headers=headers
        ) as resp:
            yield resp

    async def stream_chat(
        self,
        payload: dict[str, Any],
        headers: dict[str, str],
        outcome: Outcome,
    ) -> Sample:
        """发一次问答并**边读边测首 token 时延**。

        `smoke_e2e.py::parse_sse` 吃的是完整响应体 —— 那是 TestClient 会把整个
        响应缓冲下来的缘故。压测必须真的边读边算,否则 TTFT 只能等于总时长,
        整个「膝盖用 TTFT 判」的判据就没有意义了。
        """
        httpx = self._httpx
        outcome.enter()
        start = time.perf_counter()
        ttft: float | None = None
        reranked: bool | None = None
        status: int | None = None
        # 增量 SSE 解析:累积 event/data,遇到空行才成帧。
        event_name: str | None = None
        data_lines: list[str] = []
        done_payload: dict[str, Any] | None = None
        err: str | None = None

        def _elapsed_ms() -> float:
            return (time.perf_counter() - start) * 1000

        try:
            async with self._client.stream(
                "POST", "/api/chat/stream", json=payload, headers=headers
            ) as resp:
                status = resp.status_code
                if resp.status_code != 200:
                    body = (await resp.aread())[:200]
                    return Sample(
                        ok=False,
                        total_ms=_elapsed_ms(),
                        status=status,
                        error=f"HTTP {status}: {body.decode('utf-8', 'replace')[:120]}",
                    )
                async for line in resp.aiter_lines():
                    if line == "":
                        if event_name is not None and data_lines:
                            raw = "\n".join(data_lines)
                            if event_name == "token" and ttft is None:
                                ttft = _elapsed_ms()
                            elif event_name == "sources":
                                # `reranked` 在这里,不在 `done` 里。
                                payload = json.loads(raw)
                                raw_flag = payload.get("reranked")
                                reranked = raw_flag if isinstance(raw_flag, bool) else None
                            elif event_name == "done":
                                done_payload = json.loads(raw)
                            elif event_name == "error":
                                err = json.loads(raw).get("message", "error 事件")
                        event_name = None
                        data_lines = []
                        continue
                    if line.startswith(":"):
                        continue
                    key, _, value = line.partition(":")
                    value = value.lstrip(" ")
                    if key == "event":
                        event_name = value
                    elif key == "data":
                        data_lines.append(value)
        except (httpx.HTTPError, OSError) as exc:
            return Sample(
                ok=False,
                total_ms=_elapsed_ms(),
                status=status,
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            outcome.exit()

        total = _elapsed_ms()
        if err is not None:
            return Sample(ok=False, total_ms=total, status=status, error=err)
        if done_payload is None:
            return Sample(ok=False, total_ms=total, status=status, error="流结束了但没有 done")
        raw_stages = done_payload.get("stage_timings")
        stages = raw_stages if isinstance(raw_stages, dict) else None
        return Sample(
            ok=True,
            total_ms=total,
            ttft_ms=ttft,
            status=status,
            stages=stages,
            reranked=reranked,
        )


# ---------------------------------------------------------------------------
# 被测服务
# ---------------------------------------------------------------------------


class ServerUnavailable(Exception):
    """目标服务起不来。消息要说清**怎么补**,不只是缺什么 —— 照 `dev.py` 的风格。"""


def _remove_tree(path: Path, *, attempts: int = 6) -> bool:
    """删一棵目录树,删不掉就等一会儿再试。**返回是否真的删掉了。**

    为什么不能一句 `rmtree(ignore_errors=True)` 了事:那会把失败吞掉。
    实测(2026-10-10)症状就是「跑完目录还在」,而**一个字的原因都没有** ——
    于是我只能靠数目录个数才发现它在漏。删不掉是可以发生的事
    (Windows 上文件还被人握着),但**不能是静默的**。
    """
    for _ in range(attempts):
        if not path.exists():
            return True
        shutil.rmtree(path, ignore_errors=True)
        if not path.exists():
            return True
        time.sleep(0.6)
    return not path.exists()


def _free_port() -> int:
    """让内核挑一个空闲端口。

    拿到之后就关掉,所以理论上有个极小的竞争窗口。可接受:压测脚本崩在
    「端口被占」上时,报错是清楚的,而为此引一个端口管理库不值得。
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


class Target:
    """被测服务:自己起一个,或者打一个已经在跑的。

    **不复用 `dev.py` 的 `Service`。** 它的日志是实时转发到控制台的 ——
    那对启动器是对的,对压测是错的:一百个并发请求会把服务端日志刷满,
    而压测要看的数字被冲得看不见。这里改成**缓冲起来,只在失败时倒出来**。
    需求不同,不是重复。
    """

    def __init__(self, *, profile: str, port: int, data_dir: Path) -> None:
        self.profile = profile
        self.port = port
        self.data_dir = data_dir
        self.base_url = f"http://127.0.0.1:{port}"
        self._proc: subprocess.Popen[str] | None = None
        self._log: list[str] = []
        #: 异常行的**流式计数**。与 `_log` 分开:`_log` 是有上限的环形缓冲
        #: (给 `dump_log` 排障用),而这个不设上限、也不丢 —— 见 `_drain`。
        self._causes: Counter[str] = Counter()
        self._pump: threading.Thread | None = None

    # ---- 环境 ----

    def env(self) -> dict[str, str]:
        """子进程环境。**环境变量优先于 `.env`(已实测)**,所以这覆盖得住。"""
        env = dict(os.environ)
        env.update(
            {
                "DATA_DIR": str(self.data_dir / "data"),
                "CHROMA_DIR": str(self.data_dir / "data" / "chroma"),
                "UPLOAD_DIR": str(self.data_dir / "data" / "uploads"),
                "DATABASE_URL": f"sqlite:///{(self.data_dir / 'data' / 'app.db').as_posix()}",
                # 一次性密钥。**这个脚本的输出会被贴进作品集材料**,
                # 所以绝不在这里签任何能复用的 token。
                "JWT_SECRET": "loadtest-only-not-a-secret",
                "ADMIN_USERNAME": "admin",
                "ADMIN_PASSWORD": "loadtest-admin-pw",
                "PYTHONUTF8": "1",
            }
        )
        if self.profile == "mock":
            # ⚠️ 保险来自这三个**变量名写对**,不来自「没有钥匙」——
            # mock profile 下真 key 仍然在环境里。见 assert_effective_profile。
            env["LLM_BACKEND"] = "mock"
            env["EMBED_BACKEND"] = "mock"
            env["RERANK_BACKEND"] = "none"
        return env

    # ---- 生命周期 ----

    def start(self) -> None:
        (self.data_dir / "data").mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(  # noqa: S603 - 参数是写死的
            [
                sys.executable,
                "-m",
                "uvicorn",
                "app.main:app",
                "--app-dir",
                "backend",
                "--port",
                str(self.port),
                # 单 worker —— 与生产启动方式一致。多开 worker 会掩盖
                # 本脚本要测的那些「单进程内的」瓶颈。
            ],
            cwd=str(PROJECT_ROOT),
            env=self.env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform.startswith("win") else 0,
        )
        self._pump = threading.Thread(target=self._drain, daemon=True)
        self._pump.start()

    def _drain(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            text = line.rstrip()
            self._log.append(text)
            # 只留尾巴:一场压测能刷出几十万行,全留会把内存吃光。
            if len(self._log) > 4000:
                del self._log[:2000]
            # **异常行必须边到边归并,不能等跑完再回看缓冲区。**
            # 踩过一次(2026-10-10):第一版就是从 `_log` 里捞,结果缓冲区
            # 早把**最早、最要紧的那批**滚掉了,捞出来的是最后才出现的
            # 客户端断连噪音 —— 而报告里那个位置写着的正是「根因」。
            # **写错根因比不写更坏,因为它看起来言之凿凿。**
            # 计数是增量式的,多大流量都不丢。
            clipped = self._clip_cause(text)
            if clipped is not None:
                self._causes[clipped] += 1

    @classmethod
    def _clip_cause(cls, line: str) -> str | None:
        """这一行是异常吗?是就返回归一化后的文本,不是就返回 None。"""
        text = line.strip()
        # **`Traceback (most recent call last):` 要单独滤掉。**
        # 它是 traceback 的**标题行**,不是异常本身 —— 留着它的话,
        # 「根因」那一栏的第一条会是一句没有任何信息的话(而且因为它每段
        # traceback 都出现,计数还是最高的,永远排在第一)。读的人得先
        # 越过它才能看到真正的异常。
        if text.startswith("Traceback (most recent call last)"):
            return None
        # **只按 marker 判,不加长度过滤。** 加过一个 `len < 20`,它会丢掉
        # 短的异常行(比如 `ValueError: 别的异常`)—— 那是真信息,不是噪音。
        if not any(m in text for m in cls._EXC_MARKERS):
            return None
        # **把「每次都不同」的部分换成占位符,否则同一种异常永远归并不了。**
        #
        # 实测(2026-10-10):取消异常的文案里带着 cancel scope 的十六进制地址
        #(`Cancelled via cancel scope 25206a33610`),于是**同一个原因**的 7 条
        # 在报告里躺成 7 行,每行 2 次 —— 计数功能等于没做,读的人还得自己
        # 去辨认「这七行其实是一件事」。
        return cls._NORMALIZE.sub("<id>", text)[:200]

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def stop(self) -> None:
        if self._proc is None or not self.alive:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()

    #: 一个活着的服务进程**不可能**小于这个数。
    #: Python + FastAPI + SQLAlchemy + chromadb 的常驻集在百 MB 量级,
    #: 光是解释器加基础库就十几 MB。
    _RSS_FLOOR_MB = 30.0

    def rss_mb(self) -> float | None:
        """服务进程的常驻内存(MB)。**测不准就返回 `None`,绝不返回一个猜测值。**

        为什么量不到也要留着这个函数:把 argon2 挪进线程池是**把阻塞换成了
        内存压力** —— 它是内存硬化算法,每次哈希 64MB,而修法同时加了并发闸门。
        闸门设得对不对,理论上只有这个数能回答。

        ⚠️ **但在这台机器上它测不出来。** 2026-10-10 实测:用 `tasklist` 与
        PowerShell 的 `WorkingSet64` 两条独立通路,对三个对照进程(裸解释器 /
        导入 fastapi+sqlalchemy+chromadb / 只导 json)读到的都是**同一个 4.6MB** ——
        测量本身是坏的,不是进程真的那么小。所以下面加了物理下限守卫:
        读数低于 `_RSS_FLOOR_MB` 一律当「测不到」处理。

        **宁可报告上写着「测不到」,也不要印一个看起来合理但是错的数字** ——
        后者会被当成事实用下去,而没有任何人会发现。
        """
        if self._proc is None:
            return None
        pid = self._proc.pid
        try:
            if sys.platform.startswith("win"):
                out = subprocess.run(  # noqa: S603
                    ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=10, check=False,
                ).stdout
                # 形如: "python.exe","1234","Console","1","123,456 K"
                parts = out.strip().strip('"').split('","')
                if len(parts) < 5:
                    return None
                mb = float(
                    parts[-1].strip().strip('"').replace(",", "")
                    .removesuffix("K").strip()
                ) / 1024
            else:
                out = subprocess.run(  # noqa: S603
                    ["ps", "-o", "rss=", "-p", str(pid)],
                    capture_output=True, text=True, timeout=10, check=False,
                ).stdout.strip()
                if not out:
                    return None
                mb = float(out) / 1024
        except (OSError, ValueError, subprocess.SubprocessError):
            return None
        return mb if mb >= self._RSS_FLOOR_MB else None

    #: 服务端日志里「这是一条异常」的判据。**故意收得很紧**:放宽了会捞到
    #: 一堆缩进堆栈的中间行,反而看不出是哪条异常。
    _EXC_MARKERS = (
        "Error:",
        "Exception:",
        "Traceback (most recent call last)",
        "ERROR:",
    )

    #: 长十六进制串(取消异常的 `cancel scope <地址>` 就是这种)。
    #: 只收 8 位以上,免得把 `WinError 10054` 这种真信息也吃掉。
    _NORMALIZE = re.compile(r"\b[0-9a-f]{8,}\b")

    def root_causes(self, *, limit: int = 8) -> list[str]:
        """服务端出现过哪些异常,按次数排。**计数是流式累计的**(见 `_drain`)。

        为什么非要它:压测客户端看到的只有 `HTTP 500: Internal Server Error` ——
        **那是一句症状,不是原因**。真正的原因(比如「连接池打满、30 秒超时」)
        只写在服务端的 traceback 里。把根因留在日志里让读报告的人自己去翻,
        等于把报告最重要的一半扔掉了。

        返回值形如 `["589 次  sqlalchemy.exc.TimeoutError: QueuePool limit of ..."]`。
        """
        ranked = sorted(self._causes.items(), key=lambda kv: -kv[1])
        return [f"{count} 次  {text}" for text, count in ranked[:limit]]

    def dump_log(self, tail: int = 40) -> None:
        if not self._log:
            return
        print(f"\n  服务端日志最后 {tail} 行:")
        for line in self._log[-tail:]:
            print(f"    {line}")
        print()

    async def wait_ready(self, client: Any, *, timeout: float = 90.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._proc is not None and not self.alive:
                self.dump_log()
                raise ServerUnavailable("后端进程已退出,没能起来 —— 日志在上面。")
            try:
                resp = await client.get("/api/health")
                if resp.status_code == 200:
                    return
            except Exception:  # noqa: BLE001 - 起服务期间任何异常都只是「还没好」
                pass
            await asyncio.sleep(0.4)
        self.dump_log()
        raise ServerUnavailable(f"等了 {timeout:.0f} 秒还没就绪。")

    def seed(self) -> None:
        """灌示例数据。**以子进程跑 `seed_data.py`** —— 它自带 `create_all`
        与 `ensure_admin`,能独立运行,所以直接复用,不在这里重写一遍灌数据。
        """
        result = subprocess.run(  # noqa: S603
            [sys.executable, str(BACKEND_DIR / "scripts" / "seed_data.py")],
            cwd=str(PROJECT_ROOT),
            env=self.env(),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        if result.returncode != 0:
            print(result.stdout[-1500:])
            print(result.stderr[-1500:])
            raise ServerUnavailable("示例数据写入失败 —— 日志在上面。")


# ---------------------------------------------------------------------------
# 有效配置断言:mock profile 必须**证明**自己没有在打真实 API
# ---------------------------------------------------------------------------


#: 每个 profile 期望服务端生效的后端。**这是判据本身,写在明处。**
EXPECTED_BACKENDS = {
    "mock": {"llm_backend": "mock", "embed_backend": "mock", "rerank_backend": "none"},
    "real": {"llm_backend": "dashscope", "embed_backend": "dashscope",
             "rerank_backend": "dashscope"},
}


async def assert_effective_profile(
    api: Api, headers: dict[str, str], profile: str
) -> dict[str, float]:
    """确认服务端**实际生效**的后端就是我们要的那个。返回预热那次的阶段耗时。

    为什么不能只信环境变量:见 `Target.env` 那条 —— mock profile 下真 key
    仍然在环境里,变量名写错一个字就会静默回落成 `dashscope`,**拿真额度压测**。
    这个项目最讨厌的一类失败就是「写错了但不报错、只是结果不对」。

    **判据只有一个:`/api/health/ready` 自报的配置。** 那是服务端读自己的
    `Settings` 报出来的,是权威来源,不用我猜。

    ⚠️ **试过从「实际行为」反推,连错三回,记在这里免得再走一遍:**
      ① 看 `stage_timings` 里有没有 `rerank` 键 —— **不行**。
         `retrieval.py:207-208` 是无条件记时的,后端是 Noop 时那个键也在。
      ② 看 `done.reranked` —— **那个字段不在 `done` 上**,在 `sources` 上。
      ③ 改用 `sources.reranked` —— **语义不对**。它的意思是「重排这个阶段
         执行了」,而 `NoopReranker.rerank` 是个真实现(返回 `1.0 - i*0.001`
         的合成分数),所以关掉重排时它照样是 `True`。

    结论:**这个项目的「关掉重排」是换了个恒等实现,不是不执行那一步** ——
    所以没有任何行为信号能区分它与真重排。配置自报才是对的切面。
    """
    ready = await api.get("/api/health/ready")
    if ready.status_code != 200:
        raise ServerUnavailable(f"/api/health/ready 返回 {ready.status_code},没法确认配置。")
    reported = (ready.json().get("checks") or {}).get("config") or {}
    expected = EXPECTED_BACKENDS[profile]
    wrong = {
        key: (value, reported.get(key))
        for key, value in expected.items()
        if reported.get(key) != value
    }
    if wrong:
        detail = "、".join(f"{k}: 期望 {v[0]} 实际 {v[1]}" for k, v in wrong.items())
        raise ServerUnavailable(
            f"服务端生效的配置不是 {profile} profile({detail})。\n"
            "  环境变量没能覆盖 .env —— 继续跑可能攻击真实平台,已中止。"
        )

    # 再打一问,确认链路真的能跑通,并把阶段耗时打出来。
    #
    # ⚠️ **这些数字不是判据,也不能当性能参考 —— 它是冷调用。**
    # 第一次请求要付:Chroma 打开 collection(实测 ~160ms)、jieba 词典冷启动
    # (`BENCHMARK.md` 实测 249.8ms)、SQLite 建页缓存。所以看到
    # `vector: 162ms` 不代表配置错了,那是一次性成本。
    # (我差点在这里写一条「mock 下 vector 该是个位数毫秒」的判据 ——
    #  照它办会在冷调用上误报。判据只有上面的配置自报。)
    sample = await _one_question(api, headers, "RX9070 显卡多少钱")
    if not sample.ok:
        raise ServerUnavailable(f"预热问题没答上来({sample.error}),没法确认链路可用。")
    stages = sample.stages or {}
    print(
        f"  ✓ 服务端自报配置:llm={reported.get('llm_backend')} "
        f"embed={reported.get('embed_backend')} rerank={reported.get('rerank_backend')}"
    )
    print(f"    冷调用阶段耗时(含一次性成本,仅确认链路跑通):"
          f"{ {k: round(v, 1) for k, v in stages.items()} }")
    return stages


async def _one_question(
    api: Api, headers: dict[str, str], question: str
) -> Sample:
    """打一个问题。`Sample.stages` 里带着服务端实际跑过的阶段。"""
    return await api.stream_chat({"question": question}, headers, Outcome("warmup"))


# ---------------------------------------------------------------------------
# 准备:用户与会话
# ---------------------------------------------------------------------------


def username_of(index: int) -> str:
    """第 `index` 个压测用户的名字。

    **登录风暴那一臂要用同一套命名**,所以抽成一个函数 —— 在别处再拼一次
    字符串的话,两边迟早漂开,而症状是那一整臂拿 401,看起来像「服务端
    拒绝并发登录了」。
    """
    return f"loaduser{index:03d}"


async def _login(api: Api, username: str, password: str) -> str | None:
    resp = await api.post("/api/auth/login", json={"username": username, "password": password})
    if resp.status_code != 200:
        return None
    return str(resp.json()["access_token"])


async def prepare_users(
    api: Api, *, count: int, password: str, concurrency: int = 8
) -> list[dict[str, str]]:
    """注册 `count` 个普通用户并登录,返回它们的请求头。

    **这是准备阶段,不是被测负载**,所以用有界并发(默认 8)跑 —— 串行的话
    100 个用户 × 每次 argon2 约 100ms ≈ 10 秒纯等待。但它仍然会真实地占用
    事件循环,所以并发度压得很低,不把准备阶段混进测量。

    ⚠️ **不打印任何 token。** 这个脚本的输出会被贴进作品集材料。
    """
    sem = asyncio.Semaphore(concurrency)
    headers: list[dict[str, str]] = []
    failures: list[str] = []

    async def one(index: int) -> None:
        username = username_of(index)
        async with sem:
            resp = await api.post(
                "/api/auth/register", json={"username": username, "password": password}
            )
            # 409 = 上一轮已经建过。幂等,继续登录即可。
            if resp.status_code not in (201, 409):
                failures.append(f"{username}: HTTP {resp.status_code}")
                return
            token = await _login(api, username, password)
            if token is None:
                failures.append(f"{username}: 登录失败")
                return
            headers.append({"Authorization": f"Bearer {token}"})

    await asyncio.gather(*(one(i) for i in range(1, count + 1)))
    if failures:
        raise ServerUnavailable(f"{len(failures)} 个用户准备失败:{failures[:3]}")
    return headers


async def prepare_sessions(
    api: Api, headers_list: list[dict[str, str]], *, kb_ids: list[int] | None
) -> list[int]:
    """给每个用户建一个会话,返回会话 id(失败的位置用会话 id 0 占位)。"""
    ids: list[int] = []
    for headers in headers_list:
        resp = await api.post("/api/sessions", json={"kb_ids": kb_ids}, headers=headers)
        ids.append(resp.json()["id"] if resp.status_code == 201 else 0)
    return ids


async def _kb_ids(api: Api, admin_headers: dict[str, str]) -> list[int]:
    resp = await api.get("/api/kb/options", headers=admin_headers)
    if resp.status_code != 200:
        return []
    return [int(item["id"]) for item in resp.json()["items"]]


# ---------------------------------------------------------------------------
# 场景 S1:浏览(100 人同时打开页面)
# ---------------------------------------------------------------------------


async def scenario_browse(api: Api, headers_list: list[dict[str, str]], session_ids: list[int]) -> Outcome:
    """100 人同时打开页面:拉会话列表 / 知识库选项 / 历史消息。

    这是**最轻**的一类负载(纯 IO,不碰模型也不碰向量库),所以它先跑:
    它验证「100 个并发连接」这条链路本身是通的。这一关过不了,后面
    任何数字都不用看。
    """
    outcome = Outcome("浏览")

    async def one(index: int, headers: dict[str, str]) -> None:
        outcome.enter()
        start = time.perf_counter()
        ok, err, status = True, None, None
        try:
            for path in ("/api/sessions", "/api/kb/options"):
                resp = await api.get(path, headers=headers)
                if resp.status_code != 200:
                    ok, err, status = False, f"HTTP {resp.status_code}", resp.status_code
                    break
            if ok:
                sid = session_ids[index] if index < len(session_ids) else 0
                if sid:
                    resp = await api.get(f"/api/sessions/{sid}/messages", headers=headers)
                    if resp.status_code != 200:
                        ok, err, status = False, f"HTTP {resp.status_code}", resp.status_code
        except Exception as exc:  # noqa: BLE001 - 客户端侧任何异常都要算进去
            ok, err = False, f"{type(exc).__name__}: {exc}"
        finally:
            outcome.exit()
        outcome.samples.append(
            Sample(ok=ok, total_ms=(time.perf_counter() - start) * 1000, status=status, error=err)
        )

    await asyncio.gather(*(one(i, h) for i, h in enumerate(headers_list)))
    return outcome


# ---------------------------------------------------------------------------
# 场景 S2:稳态(100 在线,其中十几人同时提问)
# ---------------------------------------------------------------------------


async def scenario_steady(
    api: Api,
    headers_list: list[dict[str, str]],
    *,
    qa_concurrency: int,
    questions_per_user: int,
    seed: int,
) -> tuple[Outcome, Outcome]:
    """100 个虚拟用户各自循环:读列表 → 随机思考 → 其中一部分提问。

    **必须是受控的到达过程**:固定随机种子、固定的浏览:问答比。
    不固定的话,结果不可复现,也无法归因到任何子系统 —— 只能给「感觉」。

    返回 (浏览, 问答) 两个 Outcome。**两个数都要报**:「100 在线」与
    「十几并发问答」是两个不同的事实,只报一个会让人以为另一个也是那个数。
    """
    rng = random.Random(seed)
    browse = Outcome("浏览")
    qa = Outcome("问答")
    qa_sem = asyncio.Semaphore(qa_concurrency)

    async def browse_step(headers: dict[str, str]) -> None:
        browse.enter()
        start = time.perf_counter()
        ok, err, status = True, None, None
        try:
            resp = await api.get("/api/sessions", headers=headers)
            if resp.status_code != 200:
                ok, err, status = False, f"HTTP {resp.status_code}", resp.status_code
        except Exception as exc:  # noqa: BLE001
            ok, err = False, f"{type(exc).__name__}: {exc}"
        finally:
            browse.exit()
        browse.samples.append(
            Sample(ok=ok, total_ms=(time.perf_counter() - start) * 1000, status=status, error=err)
        )

    async def qa_step(headers: dict[str, str]) -> None:
        # 信号量在这里限的是「同时在提问的人数」。没有它,100 个用户会
        # 一起提问 —— 那是 S3 在测的东西,不是「100 在线」的语义。
        question = rng.choice(QUESTIONS)
        async with qa_sem:
            qa.samples.append(await api.stream_chat({"question": question}, headers, qa))

    async def user(index: int, headers: dict[str, str]) -> None:
        for turn in range(questions_per_user):
            # 绝大多数轮次是「翻页看历史」,少数才是提问 —— 真人不会每秒问一次。
            if index % 8 == 0:
                await qa_step(headers)
            else:
                await browse_step(headers)
            # 随机思考时间。**这是「在线」与「并发」的区别所在**。
            await asyncio.sleep(rng.uniform(0.05, 0.3) * (1 + turn))

    await asyncio.gather(*(user(i, h) for i, h in enumerate(headers_list)))
    return browse, qa


# ---------------------------------------------------------------------------
# 场景 S3:阶梯上探,找膝盖
# ---------------------------------------------------------------------------


async def warmup(api: Api, headers: dict[str, str], *, count: int) -> Outcome:
    """预热。**轮数事先定死**,不许事后调。

    为什么单独做一遍而不是每档丢前 N 个:分词器词典、Chroma 的 collection、
    SQLite 的页缓存都要一次冷启动。把这些成本摊进第一档,会让第一档看起来
    特别慢 —— 而第一档正是膝盖判据里的**基线**。基线被污染,整个判据就歪了。
    """
    outcome = Outcome("预热")
    for i in range(count):
        question = QUESTIONS[i % len(QUESTIONS)]
        outcome.samples.append(await api.stream_chat({"question": question}, headers, outcome))
    return outcome


async def _ramp_level(
    api: Api,
    headers_list: list[dict[str, str]],
    *,
    concurrency: int,
    requests: int,
    seed: int,
) -> Outcome:
    """一个并发档:固定发 `requests` 个请求,`concurrency` 个 worker 一起拉。

    **固定请求数,不是固定时长。** 固定时长的话,高并发档会在同样的时间里
    发出更多请求,样本数随档位漂移;而低档样本太少时 p95 就不是 p95 了,
    它只是「倒数第二个最差」。
    """
    rng = random.Random(seed + concurrency)
    outcome = Outcome(f"并发 {concurrency}")
    queue: asyncio.Queue[int] = asyncio.Queue()
    for i in range(requests):
        queue.put_nowait(i)

    async def worker() -> None:
        while True:
            try:
                queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            question = rng.choice(QUESTIONS)
            headers = headers_list[rng.randrange(len(headers_list))]
            outcome.samples.append(
                await api.stream_chat({"question": question}, headers, outcome)
            )

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return outcome


async def scenario_ramp(
    api: Api,
    headers_list: list[dict[str, str]],
    *,
    levels: list[int],
    per_level: int,
    seed: int,
    order: str,
) -> list[tuple[int, Outcome]]:
    """逐级上探。

    `order="desc"` 会倒着来一遍。**这是为了对付非平稳性**:单调上探时,
    WAL 在增长、CPU 在热降频、缓存在冷热之间漂移,这些都是随档位漂移的
    自变量,和「负载」混在一起。膝盖可能是漂移的假象。两个方向都跑,
    **不一致就报不一致,不硬选一个**。
    """
    seq = list(levels) if order == "asc" else list(reversed(levels))
    results: list[tuple[int, Outcome]] = []
    for conc in seq:
        outcome = await _ramp_level(
            api, headers_list, concurrency=conc, requests=per_level, seed=seed
        )
        results.append((conc, outcome))
        lat = outcome.latency(ttft=True)
        print(
            f"    并发 {conc:>3}: p50={_fmt(lat.row()['p50'])} "
            f"p95={_fmt(lat.row()['p95'])} 失败={len(outcome.failed)}/{outcome.n} "
            f"峰值并发={outcome.peak_inflight}"
        )
    if order == "desc":
        results.reverse()
    return results


#: 失败率超过这个数,就算「这一档已经崩了」,不再当它是延迟数据点。
BREAKDOWN_FAILURE_RATE = 0.05


def find_breakdown(
    levels: list[tuple[int, Outcome]], *, threshold: float = BREAKDOWN_FAILURE_RATE
) -> tuple[int, int] | None:
    """**主判据:系统从哪一档开始崩。** 返回 `(上一档, 该档)`。

    **为什么这才是主判据,而不是延迟的某个倍数。**
    2026-10-10 实测踩到:原先只按「p95 涨到最低档的 3 倍」判,结果是
    「膝盖 ∈ (1, 2]」—— 因为最低档的 p95 只有 6.2ms,3 倍是 18.6ms,
    而第 2 档就是 23ms。**判据没错,但它回答的是一个没人关心的问题**:
    在 6ms 和 23ms 之间画界线,对「能撑多少人」毫无意义。

    而真正的信号在那张表里明摆着:**并发 50 档失败 176/200**。
    失败是**无歧义**的 —— 它不需要我挑一个倍数。

    所以:失败率是主判据,延迟曲线是被解释的对象,不是判据。
    """
    for index, (conc, outcome) in enumerate(levels):
        if outcome.n == 0:
            continue
        rate = len(outcome.failed) / outcome.n
        if rate > threshold:
            previous = levels[index - 1][0] if index > 0 else conc
            return previous, conc
    return None


def find_latency_knee(levels: list[tuple[int, Outcome]], *, factor: float) -> tuple[int, int] | None:
    """**副判据**:延迟在崩溃之前是怎么爬的。

    ⚠️ 只在**没有崩溃**的那几档里找。把已经崩掉的档混进来算,等于拿
    「排队 30 秒后的超时」当延迟样本 —— 那会把曲线整个带歪。
    """
    healthy = [(c, o) for c, o in levels if o.n > 0 and len(o.failed) / o.n <= BREAKDOWN_FAILURE_RATE]
    p95s = [(conc, outcome.latency(ttft=True).row()["p95"]) for conc, outcome in healthy]
    if not p95s:
        return None
    baseline = p95s[0][1]
    if baseline <= 0:
        return None
    for index, (conc, value) in enumerate(p95s):
        if value < factor * baseline:
            continue
        rest = [v for _, v in p95s[index:]]
        if all(b >= a for a, b in zip(rest, rest[1:], strict=False)):
            previous = p95s[index - 1][0] if index > 0 else conc
            return previous, conc
    return None


# ---------------------------------------------------------------------------
# 场景 S4:登录风暴 —— argon2 会不会把事件循环堵住
# ---------------------------------------------------------------------------

#: S4 的判据。**赛前定死**:
#:   风暴期间 SSE 相邻 token 的 p99 间隙 ≥ 基线的多少倍。
GAP_FACTOR = 5.0

#: 绝对下限,用来跨过 Windows ~15.6ms 的**计时器量化地板**。
#:
#: ⚠️ **这一条第一版写成了 100ms,是错的,而且错的方式值得记下来。**
#: 我按「argon2 约 100ms」估的 —— 那是个**假设,不是测量**。实测这台机器上
#: 单次 argon2 约 50ms,于是登录臂的 p99 停在 48ms,**判据说「不成立」,
#: 而四条臂的对照明明已经把成因指出来了**。计划里写着「判据赛前定死,
#: 不许拍脑袋」——那张 100ms 的表就是拍脑袋拍的。
#:
#: 改法:绝对下限只留一个**跨过量化地板**的最小值(20ms),剩下的交给
#: 对照臂自校准(见 `judge_storm`)。
GAP_FLOOR_MS = 20.0

#: 同时挂几条流。**不挂 1 条** —— 单条流会被单核调度的偶然性骗到。
STORM_STREAMS = 6

#: 几个并发的**事件循环探针**(见 `_poll_gaps`)。这是 S4 的**主判据**。
PROBE_CLIENTS = 4


async def _stream_gaps(
    api: Api, headers: dict[str, str], *, duration: float
) -> tuple[list[float], int]:
    """**连续重连地**挂 SSE 流,收集 token 间隙,直到时间用完。

    ⚠️ 这条路在 mock profile 下**不太靠得住**,原因是实测出来的:
    一次问答的答案会进**答案缓存**,而缓存的答案是「整段一次发出」的
    (`rag.py`:命中时只发一个 token 事件)—— 于是第二遍问同一句话时**根本没有
    token 流可看**。2026-10-10 冒烟跑出来的症状就是:基线臂收到 114 个间隙,
    而它后面三臂各只有 12 个 —— 不是没停顿,是**没东西可量**。

    所以判据移到 `_poll_gaps` 那边。这一路留着,因为它是**用户看得见的症状**
    (SSE 卡顿),值得作为旁证;但报告里会标明它在缓存命中时会失效。
    """
    gaps: list[float] = []
    requests = 0
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        stamps: list[float] = []
        requests += 1
        try:
            async with api.open_stream(
                {"question": "介绍一下 RX9070 显卡的规格参数和价格,尽量详细"}, headers
            ) as resp:
                if resp.status_code != 200:
                    await asyncio.sleep(0.05)
                    continue
                async for line in resp.aiter_lines():
                    if line.startswith("data:"):
                        stamps.append(time.perf_counter())
                    if time.perf_counter() >= deadline:
                        break
        except Exception:  # noqa: BLE001 - 这一臂的失败本身不是观测量
            await asyncio.sleep(0.05)
            continue
        gaps.extend((b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False))
    return gaps, requests


async def _poll_gaps(api: Api, headers: dict[str, str], *, duration: float) -> list[float]:
    """**事件循环探针(主判据)。** 反复打 `/api/health`,量两次响应之间的间隔。

    为什么用它而不是只看 SSE:
      * **对答案缓存免疫** —— 它不做问答,不存在「命中缓存所以没流可看」。
      * 它测的**正是同一个因**:事件循环被同步工作占住时,一个已经在等的请求
        也会跟着等,于是这里的间隔直接跳起来。
      * `/api/health` 是写死的轻量返回,它自己几乎不花时间 —— 量到的几乎全是
        「排队等待」,而不是「这个接口慢」。

    健康的循环下这个间隔是亚毫秒到几毫秒;**被占住时会跳一个量级**。
    """
    stamps: list[float] = []
    deadline = time.perf_counter() + duration
    while time.perf_counter() < deadline:
        try:
            resp = await api.get("/api/health", headers=headers)
            if resp.status_code == 200:
                stamps.append(time.perf_counter())
        except Exception:  # noqa: BLE001
            pass
    return [(b - a) * 1000 for a, b in zip(stamps, stamps[1:], strict=False)]


async def _cpu_burners(seconds: float) -> list[subprocess.Popen[bytes]]:
    """起几个纯计算进程烧 CPU,**不碰被测服务的那个事件循环**。

    ⚠️ **这是个粗对照,不是配对对照。** 它制造的是「机器上有别的重负载」,
    而 argon2 那个臂制造的是「事件循环本身被占用」。两者消耗的 CPU 总量
    也不相等。**真正配对的是 A/B 那一臂**(同样的 argon2 工作、只挪个位置),
    所以归因要看 A/B,这一臂只回答「是不是随便什么 CPU 压力都会让流停顿」。
    """
    procs = []
    burners = max(2, (os.cpu_count() or 4) // 2)
    code = f"import time\nt=time.time()+{seconds}\nwhile time.time()<t:\n    pass"
    for _ in range(burners):
        procs.append(
            subprocess.Popen(  # noqa: S603
                [sys.executable, "-c", code],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    return procs


def _stop_burners(procs: list[subprocess.Popen[bytes]]) -> None:
    for proc in procs:
        with contextlib.suppress(OSError):
            proc.kill()


def judge_storm(
    storm: dict[str, dict[str, list[float]]], *, login_label: str
) -> tuple[bool, str]:
    """按判据裁决,并把裁决过程**写成一句话**。

    判据(事先定死):
      * 登录臂的探针 p99 ≥ 基线的 `GAP_FACTOR` 倍,且 ≥ `GAP_FLOOR_MS`;
      * **并且** ≥ `GAP_FACTOR` 倍**所有对照臂里最高的那个 p99**。

    第二条是**自校准**:对照臂存在的意义就是给出「这个噪声底有多高」。
    只用基线乘以一个固定倍数的话,那个倍数就成了另一个拍脑袋的数。
    """
    probe = {label: _pct(data["probe"], 99) for label, data in storm.items()}
    if not probe:
        return False, "没有数据"
    base = probe.get("空载基线", 0.0)
    hit = probe.get(login_label, 0.0)
    controls = {k: v for k, v in probe.items() if k not in ("空载基线", login_label)}
    control_max = max(controls.values()) if controls else 0.0

    relative = hit >= GAP_FACTOR * base if base > 0 else False
    above_floor = hit >= GAP_FLOOR_MS
    beyond_controls = hit >= GAP_FACTOR * control_max if control_max > 0 else relative
    ok = relative and above_floor and beyond_controls

    detail = (
        f"登录 {_fmt(hit)} / 基线 {_fmt(base)} = "
        f"{hit / base:.1f}× (要 ≥{GAP_FACTOR:g}×: {'✓' if relative else '✗'})、"
        f"对照臂最高 {_fmt(control_max)}"
        f" (要 ≥{GAP_FACTOR:g}×: {'✓' if beyond_controls else '✗'})、"
        f"≥{GAP_FLOOR_MS:g}ms: {'✓' if above_floor else '✗'}"
    )
    return ok, detail


async def _storm_arm(
    api: Api,
    user_headers: list[dict[str, str]],
    *,
    label: str,
    duration: float,
    logins: int,
    users: int,
    password: str,
    arm: str,
    rss_log: dict[str, float | None] | None = None,
    rss_probe: Any = None,
) -> tuple[list[float], list[float], int]:
    """跑一臂:先挂 `STORM_STREAMS` 条流,再按 `arm` 制造压力。

    | arm | 做什么 | 排除什么 |
    |---|---|---|
    | `none` | 什么都不做 | 给出**零分布**。没有它无法判断任何间隙是否异常 |
    | `login` | 30 个并发登录 | 实验臂本身 |
    | `cheap` | 30 个并发廉价 GET | 排除「请求数本身」这个原因 |
    | `cpu` | 若干纯计算进程 | 粗对照:随便什么 CPU 压力会不会也让流停顿 |
    """
    probes = [
        asyncio.create_task(_poll_gaps(api, user_headers[i], duration=duration + 0.4))
        for i in range(PROBE_CLIENTS)
    ]
    streams = [
        asyncio.create_task(_stream_gaps(api, user_headers[i], duration=duration))
        for i in range(STORM_STREAMS)
    ]
    # 等探针与流都真的跑起来再制造压力 —— 否则测到的是一段空转。
    await asyncio.sleep(1.0)

    procs: list[subprocess.Popen[bytes]] = []
    try:
        if arm == "login":
            # 用**真实存在**的用户:argon2 对不存在的用户也会走一次假哈希
            # (`auth.py:83`),耗时相当,但拿 401 掩盖真实行为不好 ——
            # 我们要看的是「登录成功这条路径」。
            await asyncio.gather(
                *(
                    api.post(
                        "/api/auth/login",
                        json={"username": username_of(i % users + 1), "password": password},
                    )
                    for i in range(logins)
                ),
                return_exceptions=True,
            )
        elif arm == "cheap":
            await asyncio.gather(
                *(api.get("/api/health") for _ in range(logins)), return_exceptions=True
            )
        elif arm == "cpu":
            procs = await _cpu_burners(duration * 0.8)
    finally:
        # 压力刚过的那一刻采样 —— 此时线程池里的 argon2 还没散,是峰值附近。
        if rss_log is not None and rss_probe is not None:
            rss_log[label] = rss_probe()
        probe_gaps: list[float] = []
        stream_gaps: list[float] = []
        stream_requests = 0
        for task in probes:
            with contextlib.suppress(Exception):
                probe_gaps.extend(await task)
        for task in streams:
            with contextlib.suppress(Exception):
                # 这些也是**真实的问答请求**,会对账 —— 漏掉它们会让
                # 「服务端记录到的比客户端发出的还多」,然后被误读成异常。
                gaps, count = await task
                stream_gaps.extend(gaps)
                stream_requests += count
        _stop_burners(procs)
    return probe_gaps, stream_gaps, stream_requests


async def scenario_login_storm(
    api: Api,
    user_headers: list[dict[str, str]],
    *,
    logins: int,
    users: int,
    password: str,
    duration: float,
    rss_log: dict[str, float | None] | None = None,
    rss_probe: Any = None,
) -> tuple[dict[str, dict[str, list[float]]], int]:
    """四臂对照。**缺一不可。**

    「真没阻塞」和「阻塞了但被 CPU 争抢掩盖成弥散噪声」,在只看间隙时
    是分不开的 —— 两者都会让间隙变宽,只是形状不同。少了对照臂,会得出一个
    **错误但可信**的因果结论,而那种结论比没有结论更坏。

    每臂返回两个观测量:`probe`(事件循环探针,**判据用它**)与
    `stream`(SSE 的 token 间隙,旁证)。
    """
    arms = [
        ("none", "空载基线"),
        ("cheap", f"{logins} 并发廉价请求"),
        ("cpu", "CPU 饱和"),
        ("login", f"{logins} 并发登录"),
    ]
    out: dict[str, dict[str, list[float]]] = {}
    total_requests = 0
    for arm, label in arms:
        probe, stream, requests = await _storm_arm(
            api,
            user_headers,
            label=label,
            duration=duration,
            logins=logins,
            users=users,
            password=password,
            arm=arm,
            rss_log=rss_log,
            rss_probe=rss_probe,
        )
        out[label] = {"probe": probe, "stream": stream}
        total_requests += requests
        # 测不到就写「测不到」,不写一个猜测值。见 Target.rss_mb 的说明。
        rss = ""
        if rss_log is not None and label in rss_log:
            value = rss_log[label]
            rss = f"  RSS={value:.0f}MB" if value is not None else "  RSS=测不到"
        print(
            f"    {label:<22} 探针 p50={_fmt(_pct(probe, 50))} p99={_fmt(_pct(probe, 99))} "
            f"n={len(probe)}   | 流间隙 n={len(stream)}{rss}"
        )
    # 返回请求数是为了**对账**:风暴里的那些流也是真实的问答请求,
    # 漏掉它们会让「服务端记录到的比客户端发出的还多」,然后被误读成异常。
    return out, total_requests


# ---------------------------------------------------------------------------
# 服务端读数(交叉核对用)
# ---------------------------------------------------------------------------


async def fetch_metrics(api: Api, admin: dict[str, str], *, minutes: int) -> dict[str, Any] | None:
    """抓一次窗口内的服务端聚合。

    ⚠️ **它是交叉核对用的,不是权威读数。** 三个已知偏差,报告里必须写明:
      * 只统计**已完成**的请求 —— 挂住的那些(往往最慢)不在里面,所以偏乐观
      * 指标写入失败**被吞**(`metrics.py:114-121`),于是错误率被低估、
        **而且最慢的那批被删掉,p95 反而更好看**
      * `window_minutes` 是时间尾窗,起止会切进压测前后的空闲
    所以判据是:客户端请求数 vs 服务端 `requests`,**对不上就要查明差额**。
    """
    resp = await api.get(f"/api/admin/metrics?window_minutes={minutes}", headers=admin)
    if resp.status_code != 200:
        return None
    return dict(resp.json())


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

_HEADER = """# 并发压力测试

**测的是「很多人一起用」,不是「单次请求多快」** —— 后者在 `BENCHMARK.md`。

- 时间:{timestamp}
- 机器:{platform} / Python {python} / {cpus} 逻辑核
- profile:**{profile}**
- 重复:阶梯每档观测**两次**(升序一遍、降序一遍),不是三次。
  这是**对消顺序**的做法而不是取中位数:单调上探时 WAL 在增长、
  CPU 在热降频、缓存在冷热之间漂移,这些自变量与「负载」混在一起;
  反过来再跑一遍能把「顺序」这个因素对消掉。**两个方向给出的膝盖不一致时,
  报告两个,不挑一个。**

> ⚠️ **压测生成器与被测服务跑在同一台机器上**,两者抢同一批核。
> 多数偏差(CPU 争抢、热降频)会把服务器测得**更差** —— 那是保守的。
> 但**客户端**连接池没放开会让实际并发低于目标,把服务器测得**更好**,
> 所以每张表都报**实测的 in-flight 并发峰值**,而不是「我打算压多少」。
> (注意分清楚:**服务端**自己也有一个连接池,那个是本轮真正打到的那一个 ——
>  见第三节的根因。)
"""


def _table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    head = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join("---" for _ in columns) + "|"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |" for row in rows
    ]
    return "\n".join([head, sep, *body])


def report_errors(outcome: Outcome, *, limit: int = 3) -> str:
    """把失败的种类原样打出来。

    **不打印的话,一串「失败 30/30」等于什么都没说** —— 而失败原因往往是
    脚本自己的解析出了问题,不是被测系统的问题。这个项目里已经吃过一次
    这种亏(注释统计脚本读错捕获组,静默报出 100%)。
    """
    kinds = outcome.errors_by_kind()
    if not kinds:
        return ""
    lines = [f"      ⚠ {n} 条失败:{k}" for k, n in list(kinds.items())[:limit]]
    if len(kinds) > limit:
        lines.append(f"      …还有 {len(kinds) - limit} 种")
    return "\n".join(lines)


def _outcome_row(label: str, outcome: Outcome) -> dict[str, Any]:
    lat = outcome.latency()
    ttft = outcome.latency(ttft=True)
    return {
        "场景": label,
        "请求数": outcome.n,
        "失败": len(outcome.failed),
        "峰值并发": outcome.peak_inflight,
        "总时长 p50": _fmt(lat.row()["p50"]),
        "总时长 p95": _fmt(lat.row()["p95"]),
        "首token p50": _fmt(ttft.row()["p50"]) if ttft.n else "—",
        "首token p95": _fmt(ttft.row()["p95"]) if ttft.n else "—",
    }


def _render(
    *,
    profile: str,
    browse: Outcome | None,
    qa: Outcome | None,
    ramp_asc: list[tuple[int, Outcome]],
    ramp_desc: list[tuple[int, Outcome]],
    storm: dict[str, dict[str, list[float]]],
    storm_seconds: float,
    rss_log: dict[str, float | None],
    root_causes: list[str],
    client_requests: int,
    server_metrics: dict[str, Any] | None,
    notes: list[str],
) -> str:
    import platform as _platform

    out = [_HEADER.format(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        platform=f"{_platform.system()} {_platform.release()}",
        python=_platform.python_version(),
        cpus=os.cpu_count(),
        profile=profile,
    )]

    # ---- 场景总表 ----
    out.append("\n## 一、场景\n")
    cols = ["场景", "请求数", "失败", "峰值并发", "总时长 p50", "总时长 p95",
            "首token p50", "首token p95"]
    rows = []
    if browse is not None:
        rows.append(_outcome_row("S1 浏览(100 并发)", browse))
    if qa is not None:
        rows.append(_outcome_row("S2 稳态问答", qa))
    out.append(_table(rows, cols))
    out.append(
        "\n**读法**:**「在线人数」与「并发请求数」是两个数。** "
        "这个应用没有推送通道,\"在线\"只体现在\"正在发请求\"上,"
        "所以 100 在线不等于 100 并发请求。两个数都在上表里,各自看各的列。\n"
    )

    # ---- 阶梯 ----
    out.append("\n## 二、阶梯:系统能撑到哪一档\n")
    out.append(
        "\n**主判据是失败率,不是延迟的某个倍数。** 这一条是改过的:"
        "原先只按「p95 涨到最低档的 3 倍」判,而最低档的 p95 只有几毫秒 ——"
        "「3 倍」于是只有十几毫秒,**在 6ms 和 23ms 之间画条线,对"
        "「能撑多少人」毫无意义**。判据没算错,但它回答的是个没人关心的问题。\n\n"
        "所以:**失败率是主判据**(无歧义,不需要挑倍数),"
        "延迟曲线是被解释的对象(用首 token 时延 —— 总时长被模型的大常数延迟"
        "盖住,会把本地排队的信号稀释掉)。\n"
    )
    for order, levels in (("升序", ramp_asc), ("降序", ramp_desc)):
        if not levels:
            continue
        out.append(f"\n### {order}\n")
        rows = []
        for conc, outcome in levels:
            lat = outcome.latency(ttft=True).row()
            rows.append({
                "并发": conc,
                "样本": outcome.n,
                "失败": len(outcome.failed),
                "峰值并发": outcome.peak_inflight,
                "首token p50": _fmt(lat["p50"]),
                "首token p95": _fmt(lat["p95"]),
                "首token p99": _fmt(lat["p99"]),
            })
        out.append(_table(rows, ["并发", "样本", "失败", "峰值并发",
                                 "首token p50", "首token p95", "首token p99"]))
        breakdown = find_breakdown(levels)
        if breakdown is None:
            out.append(
                f"\n**未出现崩溃**(每档失败率都不超过 {BREAKDOWN_FAILURE_RATE:.0%})。\n"
            )
        else:
            out.append(
                f"\n**🔴 崩溃点 ∈ ({breakdown[0]}, {breakdown[1]}]** —— "
                f"从这一档起失败率超过 {BREAKDOWN_FAILURE_RATE:.0%}。\n"
                "  档位是离散的,所以给的是区间,不是「能撑 37 个」这种假精度。\n"
            )
        knee = find_latency_knee(levels, factor=KNEE_FACTOR)
        if knee is not None:
            out.append(
                f"\n(副判据)**崩溃之前**首 token p95 涨到最低档 {KNEE_FACTOR:g} 倍的位置:"
                f"∈ ({knee[0]}, {knee[1]}]。\n"
                "  ⚠️ 这个副判据只有在基线不接近 0 时才说明问题:基线是 6ms 时"
                "「3 倍」是 18ms,那种界线对「能撑多少人」没有意义。**崩溃点才是答案。**\n"
            )
    # 这一段**必须跟着结果走,不能硬编码。**
    # 第一版这里写死了上一轮的数字(「并发 50 档失败 176/200」云云),
    # 扩容修掉连接池之后重跑,系统不崩了,而那几句话还留在报告里 ——
    # 读起来仍然像事实。**过期的数字看起来仍然像事实,比没有数字更坏。**
    asc = find_breakdown(ramp_asc) if ramp_asc else None
    desc = find_breakdown(ramp_desc) if ramp_desc else None

    if ramp_asc and ramp_desc and asc != desc:
        out.append(
            "\n> ⚠️ **两个方向给出的崩溃点不一致**,说明结论不稳 —— "
            "单调上探时 WAL 在增长、CPU 在热降频,这些是随档位漂移的自变量,"
            "与「负载」混在一起。**两个都如实报出来,不硬选一个。**\n"
        )
    elif asc is not None:
        out.append(
            f"\n两个方向都指向同一个崩溃点(都是 ({asc[0]}, {asc[1]}])),"
            "说明它**不是顺序漂移造成的**。\n"
            "\n⚠️ **但崩溃点是单次观测,别当锐利阈值用。** 两个方向一致只能说明"
            "**本轮之内**稳;跨运行的抖动它测不出来。要界定这个边界需要**多次运行**,"
            "本脚本一轮只给一个点。\n"
        )
    elif ramp_asc:
        highest = max(c for c, _ in ramp_asc)
        out.append(
            f"\n⚠️ **「未出现崩溃」不等于「能撑 {highest} 个」。** 它只说明"
            f"**测到的那几档里**没崩,而最高那档就是 {highest} —— "
            "再往上是什么样,这一轮没有数据。**这不是一个容量上限,是一个「还没测到边」。**\n"
            "\n  还有一条更要紧的限定:这是 **mock profile**(本机骨架,无真实外呼)。"
            "真实情况下每轮检索**要多两次网络往返** —— 一次 embedding"
            "(`docs/API-CHECK.md` 实测单条 157ms)和一次 rerank(实测 146ms)。"
            "**连接正是在检索阶段被持有的**,而持有得越久,同样大的池子能同时容纳的"
            "请求就越少 —— 所以**真实情况下天花板只会更低。别把这一行当作对外承诺。**\n"
        )

    # ---- 失败都是什么 ----
    #
    # **这一节不能省。** 只报「失败 176/200」等于什么都没说 —— 而失败原因
    # 恰恰是压测最有价值的产出:它指向具体的机制,不是一个数字。
    all_outcomes = [o for o in (browse, qa) if o is not None]
    all_outcomes += [o for _, o in ramp_asc + ramp_desc]
    kinds: dict[str, int] = {}
    for outcome in all_outcomes:
        for kind, count in outcome.errors_by_kind().items():
            kinds[kind] = kinds.get(kind, 0) + count
    if kinds or root_causes:
        out.append("\n## 三、服务端出过什么异常\n")
        # **这一节的措辞必须跟着数据走。**
        # 踩过两次:先是硬编码了「客户端只看得见 HTTP 500」——而零失败的那一轮
        # 根本没有 500;接着又说「**这两栏**要对着看」——而那一轮客户端一栏
        # 压根没渲染出来(块是 `if kinds:` 控制的)。
        # **引用了不存在的东西,比不写更坏**:读的人会以为自己漏看了什么。
        if kinds:
            out.append("\n**客户端看到的失败**(按出现次数排,原文照抄):\n\n")
            for kind, count in sorted(kinds.items(), key=lambda kv: -kv[1]):
                out.append(f"- **{count} 次** — `{kind[:300]}`\n")
            if root_causes:
                out.append("\n**服务端的根因**(从服务端日志里捞的,按出现次数排):\n\n")
                for cause in root_causes:
                    out.append(f"- `{cause}`\n")
                out.append(
                    "\n⚠️ **这两栏要对着看。** 客户端那一栏是**症状**(请求没拿到结果);"
                    "真正的原因(哪一层打满了、超时多少秒)只在服务端这一栏。"
                    "只报症状不报根因,读报告的人拿不到任何可行动的信息。\n"
                )
            else:
                out.append(
                    "\n(服务端的根因没捞到 —— 可能没起服务,或者日志窗口里没有异常。)\n"
                )
            out.append(
                "\n**读法**:失败原因指向机制,失败数字不指向任何东西 —— "
                "看它是 500 还是 429/超时:**500 指向服务端自己,429/超时指向外部平台**。\n"
            )
        else:
            # 零失败的那一轮:只有服务端日志这一栏。
            out.append(
                "\n**这一轮客户端一个请求都没失败。** 下面列的是**服务端日志里出现过"
                "的异常** —— 它们**没有变成用户看得见的失败**,而且绝大多数是客户端"
                "主动断连留下的取消异常(压测里很常见:一条流挂到时间上限就被掐掉)。\n"
                "\n所以「第二节说每档都是 0 失败、第三节却列出若干异常」**并不矛盾**:"
                "前者是**用户看到的结果**,后者是**服务端记下的事件**。\n"
                "\n**那为什么还要列?** 因为「没失败」和「没出过事」不是一回事。"
                "这些异常现在无害,但**数量突然变大就是个信号** —— 它是这条链路"
                "最灵敏的一根温度计。\n"
            )
            if root_causes:
                out.append("\n**服务端日志里出现过的异常**(按出现次数排):\n\n")
                for cause in root_causes:
                    out.append(f"- `{cause}`\n")
            else:
                out.append("\n(服务端日志里一条异常都没有。)\n")

    # ---- 登录风暴 ----
    if storm:
        out.append("\n## 四、登录风暴:argon2 会不会堵住事件循环\n")
        out.append(
            "**主判据是事件循环探针**:挂着 "
            f"{PROBE_CLIENTS} 个客户端反复打 `/api/health`,量**两次响应之间的间隔**。\n\n"
            "为什么不用 SSE 流当判据(实测踩到过):一次问答的答案会进**答案缓存**,"
            "而缓存的答案是「整段一次发出」的 —— 第二遍问同一句话时**根本没有 token 流可看**。"
            "冒烟跑出来的症状是基线臂收到 114 个间隙、后面三臂各只有 12 个:"
            "不是没停顿,是**没东西可量**。探针不做问答,对缓存免疫,"
            "而它测的是**同一个因** —— 循环被占住时,一个已经在等的请求会跟着等。\n\n"
            f"**判据赛前定死**:风暴臂的探针间隔 p99 ≥ 基线的 {GAP_FACTOR:g} 倍,"
            f"且 ≥ {GAP_FLOOR_MS:g}ms(后者用来跨过 Windows ~15.6ms 的计时器量化地板)。\n"
        )
        base = _pct(storm.get("空载基线", {}).get("probe", []), 99)
        base_avail = len(storm.get("空载基线", {}).get("probe", [])) / storm_seconds
        rows = []
        for label, data in storm.items():
            probe = data["probe"]
            p99 = _pct(probe, 99) if probe else 0.0
            ratio = f"{p99 / base:.1f}×" if base > 0 else "—"
            avail = len(probe) / storm_seconds if storm_seconds else 0.0
            rows.append({
                "臂": label,
                "探针 p50": _fmt(_pct(probe, 50)),
                "探针 p99": _fmt(p99),
                "相对基线 p99": ratio,
                "循环可用性(次/秒)": f"{avail:.0f}",
                "相对基线可用性": f"{avail / base_avail * 100:.0f}%" if base_avail else "—",
                "流间隙 n": len(data["stream"]),
            })
        out.append(
            "\n"
            + _table(rows, ["臂", "探针 p50", "探针 p99", "相对基线 p99",
                            "循环可用性(次/秒)", "相对基线可用性", "流间隙 n"])
        )
        out.append(
            "\n**读法**:判别因果的**不是**登录臂自己涨没涨,而是**四条臂的对比**。\n\n"
            "  * **「循环可用性」那一列比 p99 更好读** —— 它直接回答「事件循环"
            "还有多少空档留给别人」:100% 表示没受影响,掉到 60% 表示有四成时间"
            "被占住了。\n"
            "  * 「CPU 饱和」臂也涨 → 停顿是 **CPU 被抢**,不是事件循环被占。\n"
            "    这两种原因症状很像,却指向完全不同的修法。\n"
            "  * 「廉价请求」臂也涨 → 是**请求数本身**造成的,与 argon2 无关。\n"
            "  * 只有「登录」臂涨 → 才归因到 argon2 占住了事件循环。\n\n"
            "⚠️ **「CPU 饱和」臂是粗对照,不是配对对照。** 它制造的是「机器上有别的重负载」,"
            "消耗的 CPU 总量也与 30 次 argon2 不相等。**真正配对的是「修与不修」的 A/B**"
            "(同样的 argon2 工作,只挪出事件循环),归因以那一组为准。\n"
        )

    # ---- 交叉核对 ----
    out.append("\n## 五、客户端与服务端对账\n")
    out.append(
        "\n**只对账「问答」请求。** `request_metrics` 是问答链路独有的(在 `chat.py` 里写),"
        "浏览类请求不落那张表 —— 把浏览也算进来会凭空造出一个「差额」,"
        "然后被误读成「指标写入被吞了」。\n"
    )
    if server_metrics is None:
        out.append("\n拿不到服务端读数(没抓到 / 没权限),这一节缺失。\n")
    else:
        server_requests = server_metrics.get("requests", 0)
        delta = server_requests - client_requests
        out.append(
            f"\n| | 值 |\n|---|---|\n| 客户端发出的问答请求 | {client_requests} |\n"
            f"| 服务端记录到的 | {server_requests} |\n"
            f"| 差额(服务端 − 客户端) | {delta:+d} |\n"
        )
        if delta == 0:
            out.append("\n✅ 两边**相等** —— 服务端一条不漏地记下了每个问答请求。\n")
        elif delta < 0:
            sent_but_lost = kinds.get("ReadError: ", 0) + sum(
                v for k, v in kinds.items() if k.startswith("ConnectError")
            )
            out.append(
                f"\n⚠️ 服务端**少记了 {-delta} 条**。这个差额里混着两种**完全不同**的东西,"
                "要分开说,不能笼统当成「网络丢包」:\n\n"
                f"  1. **客户端根本没发出去的**(连接层失败:ReadError / ConnectError):"
                f"本轮 {sent_but_lost} 条。这些请求压根没到服务端,不记是**对的**。\n"
                f"  2. **发出去了、服务端却没记上的**:剩下的 "
                f"{max(0, -delta - sent_but_lost)} 条。**这一部分才是问题。**\n\n"
                "  第 2 类为什么危险:指标写入失败在本项目里是**被吞掉的**"
                "(`metrics.py:114-121`)—— 而在连接池打满的情况下,"
                "**写指标那一次也需要一条连接,它同样拿不到**。于是:"
                "请求因为拿不到连接而失败 → 记这次失败又要一条连接 → 也失败 → 被吞。"
                "症状是**错误率被低估,而且最慢的那批请求从分布里消失,p95 反而更好看**。\n\n"
                "  ⚠️ 本轮**无法把第 2 类精确归因** —— 要坐实得在服务端数"
                "「进到路由的请求数」,而现在的指标表本身就不完整。这条记为待查。\n"
            )
        else:
            out.append(
                f"\n⚠️ 服务端**多记了 {delta} 条**。多半是**客户端的计数漏了**,"
                "而不是服务端记重了 —— 先怀疑这条对账本身算错了哪个来源"
                "(每条问答路径都要计入:预热、稳态、阶梯、登录风暴里的流式请求)。\n"
            )

    # ---- 没测到的 ----
    out.append("\n## 六、这份测试**没有**覆盖的\n")
    out.append(
        "\n- **不改服务端指标结构**(用户已定):所以服务端那一栏区分不出"
        "超时 / 429 / 客户端断连,分类在客户端侧做。\n"
        "- **不做分布式压测**:只有一台机器,生成器与服务同机争抢,"
        "所以绝对延迟是**上界偏保守**,而实际并发峰值需要逐表核对。\n"
        "- **不测文档摄取**:解析走 `BackgroundTasks`,与请求在同一个事件循环里"
        "却**没有包线程池** —— 那是另一条独立的问题线,本脚本不覆盖。\n"
        "- **不测真实平台的并发上限**:mock profile 假设外部服务可无限扩展,"
        "**若真瓶颈在供应商,它会高估可达容量**。real profile 只做小剂量校准。\n"
        "- **测不到内存峰值。** 本环境里读别的进程的常驻内存是坏的(实测:裸解释器、"
        "导入 fastapi+chromadb、只导 json 三个对照进程读到的都是同一个 4.6MB),"
        "所以 S4 那几臂跑的时候会打印「RSS=测不到」,而不是一个数字"
        "(那一行只在终端上,不进这张表)。\n"
        "  这一条对本轮结论有实际影响:argon2 是**内存硬化**算法(每次哈希 64MB),"
        "把它挪进线程池的代价正是内存压力,而**那个代价本轮没有实测数据** —— "
        "只知道代码里给它加了并发闸门(预算 512MB ÷ 64MB = 8 个并发)。\n"
    )
    if notes:
        out.append("\n### 本次运行的备注\n\n")
        for note in notes:
            out.append(f"- {note}\n")
    return "".join(out)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

USER_PASSWORD = "loadtest-pw-123"
STORM_USERS = 30


async def _run(args: argparse.Namespace) -> int:
    profile = args.profile
    target: Target | None = None
    notes: list[str] = list(args.note)

    print("\n" + "=" * 68)
    print("  并发压力测试")
    print("=" * 68)

    if args.base_url:
        base_url = args.base_url
        # **这不是客套话。** 压测会往目标库灌几百上千行 `request_metrics`
        # 与会话记录,而 `purge_older_than()` 至今没接定时任务 ——
        # 打在开发库上就是永久污染。
        print(f"\n  ⚠️  打的是已经在跑的服务:{base_url}")
        print("     它用的**不是**临时库。这次压测会往它的数据库里灌数据。")
        print("     要干净的测量请去掉 --base-url,让脚本自己起一个临时环境。\n")
        notes.append(f"打的是外部服务 {base_url},不是脚本自起的临时环境。")
    else:
        tmp = Path(tempfile.mkdtemp(prefix="loadtest-"))
        target = Target(profile=profile, port=_free_port(), data_dir=tmp)
        # ⚠️ **先灌数据,再起服务。顺序反了会炸,而且是很难查的那种炸。**
        #
        # 踩过一次(2026-10-10):那一刻的写法是「起服务 → 断言配置 → 灌数据」,
        # 结果每个问答都 500:
        #     chromadb.errors.InternalError: Error creating hnsw segment reader:
        #     Nothing found on disk
        # 原因:断言那次请求让**服务进程**缓存了一个空的 collection 句柄,
        # 随后 `seed_data.py` 作为**另一个进程**重建了磁盘上的索引 ——
        # 服务进程手里的句柄就指向了不存在的东西。
        #
        # 应用自己的启动器 `dev.py` 顺序本来就是对的(先 `run_seed()` 再起后端),
        # 这里照它办。**把顺序收进代码里,而不是写成一句注释提醒后来人。**
        print(f"\n  临时环境 {tmp}")
        print("  灌示例数据……")
        target.seed()
        target.start()
        base_url = target.base_url
        print(f"  已起后端 {base_url}(profile={profile})")

    api = Api(base_url, pool=max(args.users, args.concurrency) + STORM_STREAMS)
    started = time.time()
    try:
        # 先等就绪再登录。反过来会在服务还没起来时拿一堆 ECONNREFUSED 去
        # 撞那个登录接口,报出来是「管理员登录失败」—— 指不到真正的原因。
        if target is not None:
            await target.wait_ready(api)

        admin_token = await _login(api, "admin", "loadtest-admin-pw")
        if admin_token is None:
            if target is not None:
                target.dump_log()
            raise ServerUnavailable(
                "管理员登录失败。\n"
                "  自己起服务时用的凭据是 admin / loadtest-admin-pw(见 Target.env)。\n"
                "  --base-url 模式下则要用那个服务自己的管理员口令。"
            )
        admin = {"Authorization": f"Bearer {admin_token}"}

        # ---- 有效配置断言:mock profile 必须自证没在打真实 API ----
        await assert_effective_profile(api, admin, profile)

        # ---- 准备 ----
        kb_ids = await _kb_ids(api, admin)
        print(f"  知识库:{kb_ids}")

        # 建够**所有场景**要用的用户数:在线用户与登录风暴取大的那个。
        # (第一版把 `prepare_users` 调了两次,第二次纯属多余 —— 而每次注册
        #  都要付一次 argon2,那是白等的十几秒。)
        user_count = max(args.users, STORM_USERS)
        print(f"  准备 {user_count} 个用户……")
        headers_list = await prepare_users(api, count=user_count, password=USER_PASSWORD)
        session_ids = await prepare_sessions(api, headers_list, kb_ids=kb_ids or None)
        print(f"  就绪:{len(headers_list)} 个用户 / {sum(1 for s in session_ids if s)} 个会话\n")

        browse = qa = None
        ramp_asc: list[tuple[int, Outcome]] = []
        ramp_desc: list[tuple[int, Outcome]] = []
        storm: dict[str, dict[str, list[float]]] = {}
        storm_requests = 0
        rss_log: dict[str, float | None] = {}

        # 上面的用户数是按「所有场景要用的最大值」建的,具体场景只取自己要的那部分。
        online = headers_list[: args.users]
        online_sessions = session_ids[: args.users]

        if args.scenario in ("browse", "all"):
            print(f"  ▶ S1 浏览({args.users} 并发)")
            browse = await scenario_browse(api, online, online_sessions)
            lat = browse.latency()
            print(f"    p50={_fmt(lat.row()['p50'])} p95={_fmt(lat.row()['p95'])} "
                  f"失败={len(browse.failed)}/{browse.n} 峰值并发={browse.peak_inflight}\n")

        if args.scenario in ("qa", "all"):
            print(f"  ▶ S2 稳态({args.users} 在线 / {args.concurrency} 并发问答)")
            browse2, qa = await scenario_steady(
                api,
                online,
                qa_concurrency=args.concurrency,
                questions_per_user=args.turns,
                seed=args.seed,
            )
            if browse is None:
                browse = browse2
            lat = qa.latency(ttft=True)
            print(f"    问答 p50={_fmt(lat.row()['p50'])} p95={_fmt(lat.row()['p95'])} "
                  f"失败={len(qa.failed)}/{qa.n} 峰值并发={qa.peak_inflight}")
            print(report_errors(qa))
            print(f"    浏览 p50={_fmt(browse2.latency().row()['p50'])} "
                  f"峰值并发={browse2.peak_inflight}\n")

        if args.scenario in ("ramp", "all"):
            print(f"  ▶ 预热 {WARMUP_REQUESTS} 个请求")
            warm = await warmup(api, admin, count=WARMUP_REQUESTS)
            print(f"    预热失败 {len(warm.failed)}/{warm.n}(不进统计)")
            print(report_errors(warm))
            print()
            levels = [int(x) for x in args.levels.split(",")]
            print(f"  ▶ S3 阶梯(升序,每档 {args.per_level} 个请求)")
            ramp_asc = await scenario_ramp(
                api, headers_list, levels=levels,
                per_level=args.per_level, seed=args.seed, order="asc",
            )
            print("  ▶ S3 阶梯(降序,用来暴露非平稳性)")
            ramp_desc = await scenario_ramp(
                api, headers_list, levels=levels,
                per_level=args.per_level, seed=args.seed, order="desc",
            )
            print()

        if args.scenario in ("login-storm", "all"):
            print(f"  ▶ S4 登录风暴({STORM_STREAMS} 条流 / {STORM_USERS} 个并发登录)")
            storm, storm_requests = await scenario_login_storm(
                api, headers_list,
                logins=STORM_USERS, users=user_count,
                password=USER_PASSWORD, duration=args.storm_seconds,
                rss_log=rss_log,
                rss_probe=target.rss_mb if target is not None else None,
            )
            login_label = f"{STORM_USERS} 并发登录"
            ok, detail = judge_storm(storm, login_label=login_label)
            print(f"    判据(探针):{'成立' if ok else '**不成立**'} —— {detail}")
            avail = {
                label: len(data["probe"]) / args.storm_seconds
                for label, data in storm.items()
            }
            base_avail = avail.get("空载基线", 0.0)
            drop = 1 - avail.get(login_label, 0.0) / base_avail if base_avail else 0.0
            print(f"    事件循环可用性:基线 {base_avail:.0f} 次/秒 → "
                  f"登录臂 {avail.get(login_label, 0.0):.0f} 次/秒(掉了 {drop * 100:.0f}%)")
            print(f"    对照:CPU 饱和臂 {avail.get('CPU 饱和', 0.0):.0f} 次/秒"
                  f"(它也掉 → 成因是 CPU 被抢而非循环被占)\n")

        # 有失败就把服务端日志倒出来。**不然「失败 30/30」等于什么都没说**,
        # 而失败往往是客户端或夹具的问题 —— 那正是最该看清堆栈的时候。
        failed_outcomes = [o for o in (browse, qa) if o is not None and o.failed]
        failed_outcomes += [o for _, o in ramp_asc + ramp_desc if o.failed]
        if failed_outcomes and target is not None:
            for outcome in failed_outcomes[:1]:
                print(report_errors(outcome, limit=5))
            target.dump_log(tail=50)

        # ---- 服务端读数 ----
        #
        # ⚠️ 对账只能拿**问答**请求去比。`request_metrics` 是问答链路独有的
        # (在 `chat.py` 里写),浏览类的请求根本不落那张表 —— 把浏览也算进来
        # 的话,差额会被读成「指标写入被吞了」,而那是另一回事。
        minutes = max(1, int((time.time() - started) / 60) + 2)
        metrics = await fetch_metrics(api, admin, minutes=minutes)
        chat_requests = (
            1  # 有效配置断言的那一次
            + (WARMUP_REQUESTS if ramp_asc else 0)
            + (qa.n if qa is not None else 0)
            + sum(o.n for _, o in ramp_asc)
            + sum(o.n for _, o in ramp_desc)
            + storm_requests
        )

        text = _render(
            profile=profile,
            browse=browse,
            qa=qa,
            ramp_asc=ramp_asc,
            ramp_desc=ramp_desc,
            storm=storm,
            storm_seconds=args.storm_seconds,
            rss_log=rss_log,
            root_causes=target.root_causes() if target is not None else [],
            client_requests=chat_requests,
            server_metrics=metrics,
            notes=notes,
        )
        print(text)

        if args.write:
            # **文件名要带 profile,否则跑一次 real 就把 mock 那份静默覆盖了。**
            # 两份报告回答的是不同的问题(本机骨架 vs 端到端),而且口径本来
            # 就不许混着比 —— 覆盖掉其中一份是不可接受的。
            # mock 是默认的那份,用原名;其余带后缀。
            name = "LOAD-TEST.md" if profile == "mock" else f"LOAD-TEST-{profile}.md"
            target_path = PROJECT_ROOT / "docs" / name
            target_path.write_text(text, encoding="utf-8")
            print(f"\n已写入 {target_path}", file=sys.stderr)
        return 0
    finally:
        await api.aclose()
        if target is not None:
            target.stop()
            # **先停服务,再删目录。** 顺序反过来的话,服务进程还句柄着
            # Chroma 的文件,Windows 会拒绝删除(而且不报错,只是删不掉)。
            if args.keep_temp:
                print(f"\n--keep-temp:临时环境留在 {target.data_dir}", file=sys.stderr)
            else:
                # 第一版漏了这一步:跑一次漏一个临时目录,26 次下来攒了 26 个 ——
                # **而本脚本的文档字符串里写着「拆干净」。** 说了就要做到。
                #
                # 而且加了删除之后**还是删不掉**:Chroma 的文件在服务进程刚死时
                # 还被 Windows 锁着,`rmtree` 只删掉能删的、把 chroma 那几个留下。
                # 更糟的是 `ignore_errors=True` 把这件事**完全掩盖了** ——
                # 现象只是「目录还在」,看不到任何原因。
                # 所以这里改成:重试几次,而且**删不掉要说出来**。
                if not _remove_tree(target.data_dir):
                    print(
                        f"\n⚠️ 临时环境没删干净:{target.data_dir}\n"
                        "   (多半是 Chroma 的文件还被系统锁着 —— 手动删即可)",
                        file=sys.stderr,
                    )


def main() -> int:
    parser = argparse.ArgumentParser(description="并发压力测试")
    parser.add_argument(
        "--scenario",
        default="all",
        choices=["browse", "qa", "ramp", "login-storm", "all"],
    )
    parser.add_argument("--profile", default="mock", choices=["mock", "real"],
                        help="mock=本地实现,不打真实 API(默认)")
    parser.add_argument("--users", type=int, default=100, help="在线用户数")
    parser.add_argument("--concurrency", type=int, default=12, help="同时提问的人数")
    parser.add_argument("--turns", type=int, default=3, help="每个用户循环几轮")
    parser.add_argument("--levels", default="1,2,5,10,20,50,100", help="阶梯的并发档位")
    parser.add_argument("--per-level", type=int, default=REQUESTS_PER_LEVEL,
                        help=f"阶梯每档的请求数(默认 {REQUESTS_PER_LEVEL};"
                             "p95 至少要 100+ 样本,调小只该用于冒烟)")
    parser.add_argument("--seed", type=int, default=20261010, help="随机种子(可复现)")
    parser.add_argument("--storm-seconds", type=float, default=6.0,
                        help="登录风暴每一臂持续多久")
    parser.add_argument("--base-url", default=None,
                        help="打一个已经在跑的服务(会污染它的库)")
    parser.add_argument("--write", action="store_true", help="写入 docs/LOAD-TEST.md")
    parser.add_argument("--keep-temp", action="store_true",
                        help="跑完保留临时环境(排障用;默认会删掉)")
    parser.add_argument("--note", action="append", default=[],
                        help="写进报告「本次运行的备注」的一段背景,可多次给。"
                             "用来记「这一轮的数是在什么改动之后测的」——"
                             "否则文档里的数字会变成一堆没有来历的事实。")
    parser.add_argument("--selftest", action="store_true", help="自检量测原语(分位数/膝盖判据)")
    args = parser.parse_args()

    if args.selftest:
        return _selftest()

    try:
        return asyncio.run(_run(args))
    except ServerUnavailable as exc:
        print(f"\n✗ {exc}\n")
        return 1
    except KeyboardInterrupt:
        print("\n中断。\n")
        return 130


if __name__ == "__main__":
    sys.exit(main())
