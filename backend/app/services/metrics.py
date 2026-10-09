"""请求指标:埋点的写入与面板要的聚合。

**这一层不参与回答**,所以它的失败容忍度比缓存还高——用户已经看到答案了,
此刻抛异常只会把一条正常的流变成「连接莫名断掉」。但它又必须**记得住失败**:
只记成功请求的指标表会把超时和重试全部漏掉,而它们恰恰是最该被看见的部分。
面板上「p95 很健康」和「有一半请求根本没进来」可以同时成立。

写入用**调用方的事务**(`on(session)`),聚合自己开会话。理由见 `_scope`。
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import AsyncIterator, Sequence
from datetime import timedelta

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import log_extra
from app.db.base import Database
from app.db.models import RequestMetric, naive_utc, utcnow
from app.schemas.metrics import LatencyBucket, MetricsSummary

logger = logging.getLogger(__name__)

#: 明细行的保留上限——见 `purge_older_than`。
DEFAULT_RETENTION_DAYS = 30


def percentile(values: Sequence[float], percent: int) -> float:
    """**最近秩**分位数:排序后取第 `ceil(percent/100 × n)` 个(从 1 数)。

    这个定义必须钉死,因为它是个会漂的数:n=10、取值 1..10 时,最近秩给出
    p50=5、p95=10,而线性插值给出 5.5 和 9.55——两个都「对」,所以面板上
    那句「p95 = 1.2s」在没有定义时无法与别的系统对比。

    选最近秩而不是插值,是因为它**总是返回集合里真实存在的值**。延迟这种
    分布,报一个谁都没经历过的插值数字,对排查没帮助。
    """
    ordered = sorted(values)
    if not ordered:
        raise ValueError("空序列没有分位数")
    rank = math.ceil(percent / 100 * len(ordered))
    return ordered[max(rank, 1) - 1]


def _bucket(values: list[float]) -> LatencyBucket:
    return LatencyBucket(
        p50=percentile(values, 50),
        p95=percentile(values, 95),
        max=max(values),
        avg=sum(values) / len(values),
    )


class MetricsService:
    """埋点读写。

    两种数据源都支持,和 `CacheStore` / `CorpusVersion` 同一个形状:

    - `MetricsService(database)` —— 自己开会话、自己提交。面板查询走这个。
    - `MetricsService.on(session)` —— 借用调用方已有的事务。**埋点走这个。**
    """

    def __init__(self, source: Database | AsyncSession) -> None:
        if isinstance(source, AsyncSession):
            self._db: Database | None = None
            self._borrowed: AsyncSession | None = source
        else:
            self._db = source
            self._borrowed = None

    @classmethod
    def on(cls, session: AsyncSession) -> MetricsService:
        """绑定到调用方已有的会话。

        问答的埋点**必须**走这里,不能图省事让它各开各的会话:请求进行到
        埋点这一步时,那个会话里往往还压着未提交的助手消息。另开一个连接
        去写指标,拿不到 SQLite 的写锁,会一直等到 `busy_timeout` 才失败——
        而失败是被吞掉的,所以症状是「指标表偶尔少几行」,不是报错。
        """
        return cls(session)

    @contextlib.asynccontextmanager
    async def _scope(self) -> AsyncIterator[AsyncSession]:
        """借来的会话还给调用方,自己开的自己收。"""
        if self._borrowed is not None:
            yield self._borrowed
            return
        async with self._db.session() as session:
            yield session

    async def _commit(self, session: AsyncSession) -> None:
        """自己的会话才提交。**借来的提交会把调用方半个事务也提交掉。**"""
        if self._borrowed is None:
            await session.commit()

    # -- 写 ------------------------------------------------------------------

    async def record(self, metric: RequestMetric) -> None:
        """落一条指标。**任何异常都不往外抛。**

        调用点在一轮问答的尾巴上(用户已经拿到完整答案了)和失败处理里
        (那里正拿着一件更重要的事在抛)。两处都不该因为「指标写不进去」
        而改变自己的行为,所以这里吞掉全部异常,只在日志里留痕。

        借用事务时这条路径尤其干净:不提交、不回滚,失败就只是这一行的
        丢失——**回滚借来的会话会把调用方没提交的助手消息一起丢掉**,
        那才是真的因小失大。
        """
        try:
            async with self._scope() as session:
                session.add(metric)
                await self._commit(session)
        except Exception:
            logger.exception(
                "指标写入失败", extra=log_extra(trace_id=metric.trace_id, outcome=metric.outcome)
            )

    # -- 读 ------------------------------------------------------------------

    async def summary(self, *, window_minutes: int) -> MetricsSummary:
        """窗口内的聚合。

        **窗口之外的行不参与**,而不是全表:算成全表的话面板显示的是「开服
        以来的平均延迟」,它在系统越来越好时几乎不动,看起来像「优化没生效」。
        """
        rows = await self._window(window_minutes)
        return self._summarize(rows, window_minutes=window_minutes)

    async def _window(self, window_minutes: int) -> list:
        """窗口内的行,**只取用得上的列**。

        不取整个 ORM 实体:一个忙时段的窗口能有几千行,而聚合只要九个字段。
        也不在 SQL 里算分位数——SQLite 没有 `percentile`,要么为每个分位数
        各扫一次带 `OFFSET` 的查询,要么上窗口函数;而这里要的是一次扫描、
        在 Python 里算完。

        **行数是窗口本身封住的**(路由把 `window_minutes` 卡在 24 小时以内),
        所以内存不是问题——但这个理由只对「一次查询要拉多少行」成立。
        这张表**会一直长**:明细条数由问答量决定,没人删。清理是
        `purge_older_than` 的事,而它需要一个定时任务来调用,**目前还没接上**
        ——单机演示的量级下不影响正确性,但那是一个已知的欠账。

        时间比较的右侧必须先 `naive_utc` 去时区 —— 存储层存的是 naive UTC,
        拿 aware 的值去比会得到一种介于「能跑」和「比错」之间的结果。
        """
        since = naive_utc(utcnow() - timedelta(minutes=window_minutes))
        async with self._scope() as session:
            result = await session.execute(
                select(
                    RequestMetric.outcome,
                    RequestMetric.total_ms,
                    RequestMetric.first_token_ms,
                    RequestMetric.stage_timings,
                    RequestMetric.retrieval_cache_hit,
                    RequestMetric.answer_cache_hit,
                    RequestMetric.grounded,
                    RequestMetric.prompt_tokens,
                    RequestMetric.completion_tokens,
                    RequestMetric.cached_prompt_tokens,
                ).where(RequestMetric.created_at >= since)
            )
            return list(result)

    @staticmethod
    def _summarize(rows: Sequence, *, window_minutes: int) -> MetricsSummary:
        total = len(rows)
        if total == 0:
            # 空窗口是**最常见的状态**——刚部署、刚清库、或者只是这一小时
            # 没人提问。它在除零崩溃和显示 0 之间,只有后者是对的。
            return MetricsSummary(
                window_minutes=window_minutes,
                requests=0,
                outcomes={},
                error_rate=0.0,
                total_ms=LatencyBucket(p50=0.0, p95=0.0, max=0.0, avg=0.0),
                first_token_ms=None,
                stages={},
                cache_hit_rate_retrieval=0.0,
                cache_hit_rate_answer=0.0,
                grounded_rate=0.0,
                prompt_tokens=0,
                completion_tokens=0,
                cached_prompt_tokens=0,
                cache_read_rate=None,
            )

        outcomes: dict[str, int] = {}
        stage_totals: dict[str, float] = {}
        stage_counts: dict[str, int] = {}
        latencies: list[float] = []
        first_tokens: list[float] = []
        #: 只累计**报过缓存读数**的请求的 prompt token——它们才是分母。
        reported_prompt_tokens = 0
        cached_total = 0

        for row in rows:
            outcomes[row.outcome] = outcomes.get(row.outcome, 0) + 1
            latencies.append(row.total_ms)

            # `None` 表示这一轮一个字都没流出来,它**不参与分母**;当成 0
            # 会让首 token 时延系统性偏低,而这是个不会随机抵消的偏差。
            if row.first_token_ms is not None:
                first_tokens.append(row.first_token_ms)

            # `0` 与 `None` 在这里必须分开:报了 0 的请求**算报过**,要进分母
            # (它确实花了这些 prompt token,只是没有一个来自缓存);没报的
            # 不进。写成 `if row.cached_prompt_tokens:` 会把报了 0 的那些
            # 从分母里剔掉——于是「一直不命中」反而让命中率的分母越来越小。
            if row.cached_prompt_tokens is not None:
                reported_prompt_tokens += row.prompt_tokens or 0
                cached_total += row.cached_prompt_tokens

            for stage, elapsed in (row.stage_timings or {}).items():
                # 单个坏值只跳过它自己。JSON 列里混进一个字符串就能让整个
                # 面板 500,而面板挂了会连带把「这次故障看得见」这件事也带走。
                if not isinstance(elapsed, (int, float)):
                    continue
                stage_totals[stage] = stage_totals.get(stage, 0.0) + elapsed
                stage_counts[stage] = stage_counts.get(stage, 0) + 1

        return MetricsSummary(
            window_minutes=window_minutes,
            requests=total,
            outcomes=outcomes,
            # 错误率的分母是**全部**请求:只看成功请求的延迟分布,会把慢到
            # 超时的那部分整个漏掉,而那正是最该被看见的。
            error_rate=sum(n for outcome, n in outcomes.items() if outcome != "done") / total,
            total_ms=_bucket(latencies),
            first_token_ms=_bucket(first_tokens) if first_tokens else None,
            # 每个阶段除以**记过这个阶段**的请求数。用总请求数当分母的话,
            # 关掉重排之后它的平均值会被那些没跑过它的请求拉低,看起来像是
            # 「重排变快了」。
            stages={s: stage_totals[s] / stage_counts[s] for s in stage_totals},
            cache_hit_rate_retrieval=sum(r.retrieval_cache_hit for r in rows) / total,
            cache_hit_rate_answer=sum(r.answer_cache_hit for r in rows) / total,
            grounded_rate=sum(r.grounded for r in rows) / total,
            prompt_tokens=sum(r.prompt_tokens or 0 for r in rows),
            completion_tokens=sum(r.completion_tokens or 0 for r in rows),
            cached_prompt_tokens=cached_total,
            # 分母为零 = 一条都没报过读数 → `None`(平台没说),不是 0.0
            # (`缓存一条都没命中`)。
            cache_read_rate=(cached_total / reported_prompt_tokens)
            if reported_prompt_tokens
            else None,
        )

    async def purge_older_than(self, days: int = DEFAULT_RETENTION_DAYS) -> int:
        """删掉 `days` 天**之前**的明细,返回删除条数。**尚未接上调用点。**

        判据是「早于」而不是「不晚于」:边界上那一条留着。两者在真机上只
        差一行,而写反之后没有任何症状——面板少一行,没人看得出来。

        **这张表是唯一会无限增长的**:每次问答写一行,而它答不了任何业务
        问题,只服务于面板。留 30 天是因为「上周三下午那阵慢」是个会被问到
        的问题,而「上季度」不会——真要看季度趋势也该先落成快照,而不是
        留着一亿行原始明细现算。

        清理需要一个定时任务来调,**目前没接上**,这是一笔已知的欠账。
        「加一个不跑的后台任务」比欠着这笔账更容易让人误以为已经收口了,
        所以这里宁可不加。
        """
        cutoff = naive_utc(utcnow() - timedelta(days=days))
        async with self._db.session() as session:
            result = await session.execute(
                delete(RequestMetric).where(RequestMetric.created_at < cutoff)
            )
            await session.commit()
            return int(result.rowcount or 0)

