"""用户级限流。

## 为什么是「滑动窗口」而不是「整分钟清零」

整分钟清零的实现会在窗口边界上放出 **两倍** 的流量:第 59 秒用满额度,
第 60 秒窗口一清零又能用满 —— 两秒之内两倍额度。限流器一旦能在边界上
被绕过,它挡的就不是它声称挡的东西。

滑动窗口每次只看「过去 `window_seconds` 秒内用了几次」,没有边界。

## 已知代价(必须说清楚,不能只写「已实现限流」)

**状态在进程内存里。** 这个应用目前是**单进程**跑的(见 `scripts/dev.py`,
uvicorn 不带 `--workers`),所以内存计数是准确的。但一旦哪天上多进程:

* 每个 worker 各有一份计数 → **实际额度 = 配置值 × worker 数**;
* 而且进程重启即清零。

真要上多进程,这里得换成 Redis 之类的共享存储 —— 那时改的是这一个模块,
调用方不用动。**别把它当成一个「已经万无一失」的限流。**

## 被拒的请求不占用额度

记了的话,一个持续重试的客户端会把自己的窗口一直往后推,于是**永远解禁不了** ——
越急越出不来。所以只有放行的那次才记一笔。
"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RateDecision:
    """一次限流判定的结果。"""

    allowed: bool
    #: 被拒时还要等几秒才可能放行;放行时为 0。
    retry_after_seconds: float = 0.0


class SlidingWindowLimiter:
    """按 key(这里是用户)计的滑动窗口限流。

    `clock` 可以注入 —— 测试靠它把窗口「拨」过去,不用真的 sleep 60 秒。
    """

    #: 每多少次检查清扫一遍过期条目。
    #:
    #: 不清扫的话,每个**来过一次就再也不来**的用户都会永久留一个字典条目 ——
    #: 内存随「历史上出现过的用户数」线性增长。单机演示量级下无所谓,
    #: 但这是个只增不减的容器,而这类东西的坏处总是迟到很久才显形。
    SWEEP_EVERY = 128

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if limit < 1:
            # **当场报错,不要静默接受。** 配置写成 0 的时候,一半人想的是
            # 「谁都别想用」,另一半人想的是「不限流」—— 这种歧义留到线上
            # 就是一次事故。挡在构造处,让它在这里响。
            raise ValueError(f"limit 必须 >= 1(收到 {limit})。要关闭限流请别构造它。")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds 必须 > 0(收到 {window_seconds})。")
        self._limit = limit
        self._window = window_seconds
        self._clock = clock
        self._hits: dict[str, deque[float]] = {}
        self._since_sweep = 0

    @property
    def limit(self) -> int:
        return self._limit

    @property
    def window_seconds(self) -> float:
        return self._window

    @property
    def tracked_keys(self) -> int:
        """当前记着多少个 key。给测试与运维看的。"""
        return len(self._hits)

    def check(self, key: str) -> RateDecision:
        """记一笔(若放行)并返回判定。"""
        now = self._clock()
        cutoff = now - self._window

        hits = self._hits.get(key)
        if hits is None:
            hits = deque()
            self._hits[key] = hits

        # 先把滚出窗口的丢掉,再判 —— 顺序反了会把已经过期的算进额度。
        while hits and hits[0] <= cutoff:
            hits.popleft()

        if len(hits) >= self._limit:
            # 最早那一笔滚出去的时候,就腾出一个名额。
            decision = RateDecision(False, max(0.0, hits[0] + self._window - now))
        else:
            hits.append(now)
            decision = RateDecision(True)

        # **清扫放在两条路径都会走到的地方。**
        # 第一版只挂在「放行」那一支上,于是**被大量拒绝的时候反而永远不清扫** ——
        # 而那正是 key 最多、最需要清的时候(测试当场把它抓出来了:
        # 300 个过期 key 一个都没清掉)。
        self._sweep_if_due(now)
        return decision

    def _sweep_if_due(self, now: float) -> None:
        self._since_sweep += 1
        if self._since_sweep < self.SWEEP_EVERY:
            return
        self._since_sweep = 0
        cutoff = now - self._window
        stale = [key for key, hits in self._hits.items() if not hits or hits[-1] <= cutoff]
        for key in stale:
            del self._hits[key]
