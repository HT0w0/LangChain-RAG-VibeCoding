"""请求指标面板的出参模型。

分位数单独抽成一个模型(而不是在汇总里摊平成 `p50_ms` / `p95_ms`),是因为
**同一组数在不同定义下会给出不同的值**:最近秩和线性插值在 n=10 时分别是
5 和 5.5。摊平之后没人看得出面板用的是哪一种,而这决定了「p95 = 1.2s」
这句话到底能不能和别的系统对比。
"""

from __future__ import annotations

from app.schemas.common import UTCModel


class LatencyBucket(UTCModel):
    """一组时延的分位数与均值,单位毫秒。"""

    p50: float
    p95: float
    max: float
    avg: float


class MetricsSummary(UTCModel):
    """一个时间窗口内的聚合结果。

    字段全部是**窗口内**的,不是开服以来的累计值——累计值在系统变好时几乎
    不动,看起来像是优化没生效。
    """

    window_minutes: int
    requests: int

    # 按 outcome 分组计数(如 `{"done": 120, "error": 3}`)而不是只给一个错误数:
    # 加一个新的终态(超时、取消)时,这里不用改就能看见它。
    outcomes: dict[str, int]
    # 错误率 = 非 done 的占比。**分母是全部请求**,失败也要算进去——
    # 只看成功请求的延迟分布会把慢到超时的那部分整个漏掉。
    error_rate: float

    total_ms: LatencyBucket
    # 没有任何一条记录报过首 token 时为 None。**不是 0**——0 会被读成
    # 「首 token 是瞬间出来的」,而事实是这一轮一个字都没流出来。
    first_token_ms: LatencyBucket | None

    # 各阶段平均耗时。分母是「记过这个阶段」的请求数,不是总请求数,
    # 否则关掉重排之后它的平均值会被拉低,看起来像是重排变快了。
    stages: dict[str, float]

    cache_hit_rate_retrieval: float
    cache_hit_rate_answer: float
    grounded_rate: float

    prompt_tokens: int
    completion_tokens: int

    #: 窗口内从 prompt 缓存读出来的 token 总数(所有**报过读数**的请求之和)。
    cached_prompt_tokens: int
    #: 命中率 = 缓存读数 / 报过读数的那些请求的 prompt token 总数。
    #:
    #: **分母不是全部请求。** 把没报读数的请求算进去,命中率会被系统性
    #: 拉低——而拉低的方向恰好是「缓存看起来更没用」,与真实情况相反。
    #:
    #: **一条都没报过时是 `None`,不是 0.0。** `0.0` 会被读成「缓存一条都
    #: 没命中」,而事实是平台没说。与 `first_token_ms` 同一条理由。
    cache_read_rate: float | None
