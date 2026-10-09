"""检索编排:向量路 + 关键词路 → RRF 融合 → 重排。

    查询 ─┬─ 向量路:编码 → Chroma top-k(带 kb/类目/价格 预过滤) ─┐
          └─ 关键词路:jieba 分词 → BM25 top-k ────────────────────┤
                                                                  ▼
                                                          RRF 融合(k=60)
                                                                  ▼
                                                     回库取正文与引用信息
                                                                  ▼
                                                    重排 → 阈值截断 → top-n

**为什么两路并发**:向量路要发一次 embedding 网络请求(几十毫秒),
关键词路要在 CPU 上建索引并打分。串行就是两者相加,并发取较慢的那个。
两者之间没有数据依赖,串行没有任何理由。

**为什么融合用「排名」而不是「分数」**:余弦相似度落在 [-1,1],BM25 是无上界
的正数,量纲不可比。归一化再相加需要知道各自的分布,而分布随查询变化;
只取排名则天然免除了这个问题。详见 `indexing/fusion.py`。

**元数据过滤必须两路都做**。这条容易漏:向量路天然支持 Chroma 的服务端
预过滤,而 BM25 索引的只有正文、没有类目字段。若只顾向量路,「只看显示器」
的查询会把显卡分块也从关键词路带进来,融合后照样出现在答案里——过滤条件
变成了摆设。所以关键词路是**先按元数据筛语料、再建索引**。
"""

from __future__ import annotations

import asyncio
import functools
import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

import anyio

from app.core.config import Settings
from app.core.logging import log_extra
from app.db.repositories.document import ChunkRecord
from app.indexing.bm25 import BM25Document, BM25Hit, JiebaBM25
from app.indexing.fusion import reciprocal_rank_fusion
from app.indexing.vectorstore import build_where
from app.providers.base import Embedder, Reranker
from app.providers.reranker import RerankUnavailableError
from app.services.context import RetrievedChunk

logger = logging.getLogger(__name__)


class ChunkSource(Protocol):
    """检索层需要的分块读取能力。

    只声明这两个方法,而不是依赖整个 `ChunkRepository`——检索层不该知道
    分块是怎么存的,测试里也就不必搭一个真数据库。
    `ChunkRepository` 结构上已经满足它。
    """

    async def list_records_by_kb(self, kb_ids: Sequence[int] | None) -> list[ChunkRecord]: ...

    async def get_records_by_ids(self, chunk_ids: Sequence[int]) -> dict[int, ChunkRecord]: ...


@dataclass(frozen=True, slots=True)
class RetrievalFilters:
    """检索期的元数据预过滤(电商场景的类目 / 价格筛选)。"""

    category: str | None = None
    price_min: float | None = None
    price_max: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return bool(
            self.category is not None
            or self.price_min is not None
            or self.price_max is not None
            or self.extra
        )


def matches_filters(meta: Mapping[str, Any], filters: RetrievalFilters) -> bool:
    """分块元数据是否满足过滤条件。

    **字段缺失一律判为「不匹配」。** 若缺失时默认放行,过滤条件对
    「没解析出元数据的文档」就形同虚设——而那类文档在实际语料里是最多的
    (纯文本、PDF 大多提取不出类目和价格)。默认放行等于过滤只对少数文档生效。
    """
    if filters.category is not None and str(meta.get("category") or "") != filters.category:
        return False

    if filters.price_min is not None or filters.price_max is not None:
        price = meta.get("price")
        # bool 是 int 的子类,显式排除,免得 True 被当成 1 元。
        if isinstance(price, bool) or not isinstance(price, (int, float)):
            return False
        if filters.price_min is not None and price < filters.price_min:
            return False
        if filters.price_max is not None and price > filters.price_max:
            return False

    return all(meta.get(key) == want for key, want in filters.extra.items())


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    chunks: list[RetrievedChunk]
    route_sizes: dict[str, int]
    stage_timings: dict[str, float]
    reranked: bool = False
    degraded: bool = False
    # 这次结果是不是直接从缓存拿的。**排障的第一个问题**就是「这是不是
    # 缓存给的」,而它只能由生产这个结果的那一层回答——上层靠比耗时去猜
    # 是不可靠的,快和慢本来就有正常波动。
    cache_hit: bool = False
    """`True` 表示重排这一环**本该跑却没跑成**(通常是云端不可用),
    已经退回复合排序。上层据此决定是否在答案里标注、是否计入指标——
    降级必须可观测,不能悄悄发生。"""


@dataclass(slots=True)
class _Route:
    """一路召回的原始结果。"""

    ids: list[int]
    scores: dict[int, float]
    elapsed_ms: float


class RetrievalService:
    def __init__(
        self,
        *,
        settings: Settings,
        embedder: Embedder,
        vector_store: Any,
        chunk_source: ChunkSource,
        reranker: Reranker | None = None,
        user_words: Sequence[str] | None = None,
    ) -> None:
        self._settings = settings
        self._embedder = embedder
        self._vector_store = vector_store
        self._chunks = chunk_source
        self._reranker = reranker
        self._user_words = list(user_words or ())

    # -- 对外入口 ------------------------------------------------------------

    async def retrieve(
        self,
        query: str,
        *,
        kb_ids: Sequence[int] | None,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult:
        """跑完整条检索链路。

        `kb_ids` 三种取值的含义**互不相同**,与 `build_where` 保持一致:
        `None` = 不限制(管理员预览)、非空序列 = 限定这些库、
        **空序列 = 一个都不许**。
        """
        started = time.perf_counter()
        timings: dict[str, float] = {}
        filters = filters or RetrievalFilters()
        depth = top_k if top_k is not None else self._settings.RETRIEVE_TOP_K
        scope = None if kb_ids is None else list(kb_ids)

        # **空范围必须在进查询之前短路。** 传给 build_where 会抛错(那是故意的),
        # 更要紧的是绝不能退化成「不过滤」——那会让没有任何知识库权限的用户
        # 检索到全部内容,而且不报错、不留异常日志。
        if scope is not None and not scope:
            return RetrievalResult(
                query=query,
                chunks=[],
                route_sizes={"vector": 0, "keyword": 0, "fused": 0, "final": 0},
                stage_timings={"total": _ms(started)},
            )

        # 两路无数据依赖,并发跑。注意两边都**不碰同一个数据库会话**:
        # 向量路只走 Chroma,关键词路才读库——`AsyncSession` 不是并发安全的,
        # 两条协程同时用它会出现难以复现的错乱。
        vector, keyword = await asyncio.gather(
            self._vector_route(query, scope, depth, filters),
            self._keyword_route(query, scope, depth, filters),
        )
        timings["vector"] = vector.elapsed_ms
        timings["keyword"] = keyword.elapsed_ms

        mark = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {"vector": vector.ids, "keyword": keyword.ids},
            k=self._settings.RRF_K,
            top_k=depth,
        )
        timings["fuse"] = _ms(mark)

        mark = time.perf_counter()
        records = await self._chunks.get_records_by_ids([h.chunk_id for h in fused])
        candidates = self._hydrate(fused, records, vector, keyword)
        timings["hydrate"] = _ms(mark)

        mark = time.perf_counter()
        candidates, reranked, degraded = await self._apply_rerank(query, candidates)
        timings["rerank"] = _ms(mark)

        timings["total"] = _ms(started)
        return RetrievalResult(
            query=query,
            chunks=candidates,
            route_sizes={
                "vector": len(vector.ids),
                "keyword": len(keyword.ids),
                "fused": len(fused),
                "final": len(candidates),
            },
            stage_timings=timings,
            reranked=reranked,
            degraded=degraded,
        )

    # -- 两路召回 ------------------------------------------------------------

    async def _vector_route(
        self,
        query: str,
        scope: list[int] | None,
        depth: int,
        filters: RetrievalFilters,
    ) -> _Route:
        started = time.perf_counter()
        embedding = await self._embedder.embed_query(query)
        where = build_where(
            scope,
            category=filters.category,
            price_min=filters.price_min,
            price_max=filters.price_max,
            extra=filters.extra or None,
        )
        hits = await self._vector_store.query(embedding=embedding, top_k=depth, where=where)

        ids: list[int] = []
        scores: dict[int, float] = {}
        for hit in hits:
            # chunk_id 为 None 说明向量主键不是 "c{id}" 形式(历史遗留或手工写入),
            # 拿不回来就跳过——引用溯源没有 id 可指。
            if hit.chunk_id is None:
                continue
            ids.append(hit.chunk_id)
            scores[hit.chunk_id] = hit.similarity
        return _Route(ids=ids, scores=scores, elapsed_ms=_ms(started))

    async def _keyword_route(
        self,
        query: str,
        scope: list[int] | None,
        depth: int,
        filters: RetrievalFilters,
    ) -> _Route:
        started = time.perf_counter()
        records = await self._chunks.list_records_by_kb(scope)

        # 先筛语料再建索引 = 真正的前置过滤。若改成「检索完再丢掉不匹配的」,
        # 结果数会凭空变少(前 top-k 条被丢掉后就补不进来了)。
        if filters.active:
            records = [r for r in records if matches_filters(r.meta, filters)]

        hits = await anyio.to_thread.run_sync(
            functools.partial(self._search_bm25, records, query, depth)
        )
        return _Route(
            ids=[h.chunk_id for h in hits],
            scores={h.chunk_id: h.score for h in hits},
            elapsed_ms=_ms(started),
        )

    def _search_bm25(
        self, records: Sequence[ChunkRecord], query: str, depth: int
    ) -> list[BM25Hit]:
        """同步内核,只在线程池里跑。

        jieba 分词 + BM25 建索引都是 CPU 密集的阻塞调用,直接在 `async def`
        里跑会卡住整个事件循环——所有并发请求一起等,这正是「企业级」和
        「能跑通」的分界线。

        **每次查询整份重建索引**,这一条是有意保留的:增量维护 BM25 需要
        重算全局文档频次,换不来可感知的收益,还会引入「删了文档索引里还在」
        这类难查的漂移。索引因此永远是当下语料算出来的,不存在过期问题。

        **但「重建索引」不等于「重建分词器」。** 这里曾经每次新建一个
        `JiebaBM25`,而它的第一次分词会触发 jieba 加载主词典——实测 250ms
        上下,与语料大小无关。也就是说每个问题都为同一本从不变化的词典重付
        一次冷启动:1000 分块时这一路,**约六成的耗时是这一项**,而它与该走
        多少语料毫无关系。分词器改由 `_shared_tokenizer` 按词表复用之后,
        这份固定成本只在进程内付一次。数据见 `docs/BENCHMARK.md`。
        """
        index = JiebaBM25(user_words=self._user_words)
        index.index(
            [
                BM25Document(chunk_id=r.chunk_id, kb_id=r.kb_id, text=r.content)
                for r in records
            ]
        )
        return index.search(query, top_k=depth)

    # -- 回填与重排 ----------------------------------------------------------

    def _hydrate(
        self,
        fused,
        records: Mapping[int, ChunkRecord],
        vector: _Route,
        keyword: _Route,
    ) -> list[RetrievedChunk]:
        """把融合后的 id 还原成带正文与引用信息的分块,并附上打分明细。

        库里查不到的 id 直接跳过:那是「向量库还留着、SQLite 已删除」的
        不一致残留(摄取与删除分两步写,中途失败就会留下)。不跳过的话,
        引用卡片点开是空的。
        """
        out: list[RetrievedChunk] = []
        for hit in fused:
            record = records.get(hit.chunk_id)
            if record is None:
                continue
            out.append(
                RetrievedChunk(
                    chunk_id=record.chunk_id,
                    content=record.content,
                    kb_id=record.kb_id,
                    doc_id=record.doc_id,
                    doc_name=record.doc_name,
                    kb_name=record.kb_name,
                    chunk_index=record.chunk_index,
                    meta=dict(record.meta),
                    vector_rank=hit.ranks.get("vector"),
                    keyword_rank=hit.ranks.get("keyword"),
                    vector_similarity=vector.scores.get(hit.chunk_id),
                    bm25_score=keyword.scores.get(hit.chunk_id),
                    fusion_score=hit.score,
                )
            )
        return out

    async def _apply_rerank(
        self, query: str, candidates: list[RetrievedChunk]
    ) -> tuple[list[RetrievedChunk], bool, bool]:
        """重排 + 阈值截断。返回 (结果, 是否重排过, 是否降级)。

        **重排是增强,不是必需。** 云端不可用时退回复合排序,而不是把整个
        问答接口带崩——用户宁可拿到一份稍差的答案,也不想看到 500。
        但降级要上报(`degraded`),不能悄悄发生。
        """
        if self._reranker is None or not self._settings.RERANK_ENABLED or not candidates:
            return candidates, False, False

        try:
            hits = await self._reranker.rerank(
                query,
                [c.content for c in candidates],
                top_n=self._settings.RERANK_TOP_N,
            )
        except RerankUnavailableError as exc:
            logger.warning("重排不可用,退回复合排序", extra=log_extra(error=str(exc)))
            return candidates, False, True

        reordered: list[RetrievedChunk] = []
        for hit in hits:
            # 重排服务只认位置,越界下标说明它返回了不在输入里的东西——
            # 静默跳过,别让一个第三方响应把检索打成 500。
            if not 0 <= hit.index < len(candidates):
                logger.warning("重排返回越界下标", extra=log_extra(index=hit.index))
                continue
            reordered.append(replace(candidates[hit.index], rerank_score=hit.score))

        # 低于阈值的一律丢掉:宁可让模型说「资料中没有找到相关内容」,
        # 也不要把明显不相关的资料塞进上下文诱导它编一个像模像样的答案。
        threshold = self._settings.RERANK_SCORE_THRESHOLD
        reordered = [c for c in reordered if (c.rerank_score or 0.0) >= threshold]
        return reordered, True, False


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)
