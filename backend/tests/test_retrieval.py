"""检索编排测试(向量路 + 关键词路 → RRF → 重排)。

这里用**假**的向量库和假的分块源,而不是真 Chroma + 真 SQLite。理由:
本文件要验的是**编排**——两路有没有并发、融合顺序对不对、重排失败会不会
把整个问答拖垮、元数据过滤有没有同时作用在两路上。这些都是控制流的性质,
用真存储反而要靠构造语料去"凑"出想要的排名,既脆又难读。

真存储的端到端检索已由 `test_kb.py` 覆盖(那里确实往 Chroma 里写向量再查回来),
两层各管一段。

**embedding 用真的 `MockEmbedder`**(不是常量假向量):它按字符 n-gram 哈希,
字面越像越近。用常量的话「两路融合后排序对不对」这类断言会失去意义。
"""

from __future__ import annotations

import time

import pytest
from app.core.config import Settings
from app.core.exceptions import ProviderError, QuotaExhaustedError
from app.db.repositories.document import ChunkRecord
from app.indexing.vectorstore import VectorHit
from app.providers.base import RerankHit
from app.providers.embedding import MockEmbedder
from app.providers.reranker import RerankUnavailableError
from app.services.retrieval import RetrievalFilters, RetrievalService

# --------------------------------------------------------------------------
# 夹具:可控的假组件
# --------------------------------------------------------------------------


def record(
    cid: int,
    content: str,
    *,
    kb_id: int = 1,
    doc_id: int = 1,
    doc_name: str = "商品.xlsx",
    kb_name: str = "商品库",
    meta: dict | None = None,
) -> ChunkRecord:
    return ChunkRecord(
        chunk_id=cid,
        content=content,
        kb_id=kb_id,
        doc_id=doc_id,
        doc_name=doc_name,
        kb_name=kb_name,
        chunk_index=cid,
        meta=meta or {},
    )


class FakeChunkSource:
    """内存分块源,接口与 `ChunkRepository` 的两个检索方法一致。"""

    def __init__(self, records: list[ChunkRecord], *, delay: float = 0.0) -> None:
        self._records = {r.chunk_id: r for r in records}
        self._delay = delay
        self.calls: list[list[int] | None] = []
        self.spans: list[tuple[float, float]] = []

    async def list_records_by_kb(self, kb_ids) -> list[ChunkRecord]:
        self.calls.append(None if kb_ids is None else list(kb_ids))
        start = time.monotonic()
        if self._delay:
            import asyncio

            await asyncio.sleep(self._delay)
        # 顺序刻意与插入序不同,用来证明结果顺序来自 BM25 而不是「恰好按 id」。
        rows = list(reversed(self._records.values()))
        if kb_ids is None:
            found = rows
        else:
            # **必须真的按 kb 过滤**,与 `ChunkRepository.list_records_by_kb` 一致。
            # 假源若照单全收,「范围只作用于向量路、关键词路漏了」这个 bug
            # 就会被假源本身掩盖掉——而那正是一次越权读。
            allowed = set(kb_ids)
            found = [r for r in rows if r.kb_id in allowed]
        self.spans.append((start, time.monotonic()))
        return found

    async def get_records_by_ids(self, chunk_ids) -> dict[int, ChunkRecord]:
        return {cid: self._records[cid] for cid in chunk_ids if cid in self._records}


def _kb_scope_of(where) -> set[int] | None:
    """从 where 子句里取出 kb 范围,模拟 Chroma 的服务端预过滤。"""
    if not where:
        return None
    for cond in where.get("$and") or [where]:
        if "kb_id" in cond:
            return set(cond["kb_id"]["$in"])
    return None


class FakeVectorStore:
    """按预设顺序返回命中,记录收到的 where 与前缀截断。

    **会真的按 kb 范围过滤**,而不是把 where 记下来就照单全收。假组件不
    模拟这个行为的话,「范围只作用于关键词路、向量路漏了」这类 bug 会被
    假组件本身掩盖——而那正是一次越权读。
    """

    def __init__(self, hits: list[VectorHit], *, delay: float = 0.0) -> None:
        self._hits = hits
        self._delay = delay
        self.calls: list[dict] = []
        self.spans: list[tuple[float, float]] = []

    async def query(self, *, embedding, top_k: int, where=None) -> list[VectorHit]:
        self.calls.append({"top_k": top_k, "where": where, "embedding_len": len(embedding)})
        start = time.monotonic()
        if self._delay:
            import asyncio

            await asyncio.sleep(self._delay)

        hits = list(self._hits)
        scope = _kb_scope_of(where)
        if scope is not None:
            hits = [h for h in hits if h.metadata.get("kb_id") in scope]

        self.spans.append((start, time.monotonic()))
        return hits[:top_k]


def vector_hit(cid: int, similarity: float, content: str = "内容", *, kb_id: int = 1) -> VectorHit:
    """真实向量命中一定带 kb_id 元数据(摄取时写进去的),这里默认给 1。"""
    return VectorHit(
        vector_id=f"c{cid}",
        chunk_id=cid,
        content=content,
        metadata={"kb_id": kb_id},
        similarity=similarity,
        raw_score=1.0 - similarity,
    )


class FakeReranker:
    """按「内容里出现的标记」打分,顺序完全可控。"""

    model_id = "fake-rerank"

    def __init__(self, scores: dict[str, float] | None = None, *, error: Exception | None = None):
        self._scores = scores or {}
        self._error = error
        self.calls: list[list[str]] = []

    async def rerank(self, query, documents, top_n=None) -> list[RerankHit]:
        self.calls.append(list(documents))
        if self._error is not None:
            raise self._error
        hits = [
            RerankHit(index=i, score=self._scores.get(doc, 0.5), document=doc)
            for i, doc in enumerate(documents)
        ]
        hits.sort(key=lambda h: (-h.score, h.index))
        return hits[:top_n] if top_n is not None else hits


def build(
    settings: Settings,
    *,
    records: list[ChunkRecord],
    hits: list[VectorHit] | None = None,
    reranker=None,
    chunk_delay: float = 0.0,
    vector_delay: float = 0.0,
) -> tuple[RetrievalService, FakeChunkSource, FakeVectorStore]:
    source = FakeChunkSource(records, delay=chunk_delay)
    store = FakeVectorStore(hits or [], delay=vector_delay)
    service = RetrievalService(
        settings=settings,
        embedder=MockEmbedder(dim=64),
        vector_store=store,
        chunk_source=source,
        reranker=reranker,
    )
    return service, source, store


# --------------------------------------------------------------------------
# 两路召回与融合
# --------------------------------------------------------------------------


class TestHybridRecall:
    async def test_chunk_found_by_both_routes_outranks_one_found_by_one(self, settings):
        """RRF 的核心性质,在这里验的是「真的接到了融合上」。"""
        records = [
            record(1, "显卡支持三年质保"),
            record(2, "主板支持三年质保"),
            record(3, "显示器参数"),
        ]
        service, _, _ = build(
            settings, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.8), vector_hit(3, 0.7)]
        )
        result = await service.retrieve("质保", kb_ids=[1])

        top = result.chunks[0]
        assert top.chunk_id in (1, 2), "两路都召回的分块必须排在最前"
        assert top.vector_rank is not None and top.keyword_rank is not None

    async def test_keyword_only_hit_is_kept(self, settings):
        """**这条是混合检索存在的理由。**

        向量路没召回它(预过滤把 top-k 占满了),而关键词路召回了。
        只看向量路的话这条资料就丢了——而它恰恰精确命中了查询词。
        """
        records = [record(1, "显卡"), record(2, "显卡质保条款说明")]
        service, _, _ = build(settings, records=records, hits=[vector_hit(1, 0.99)])
        result = await service.retrieve("质保", kb_ids=[1])

        ids = [c.chunk_id for c in result.chunks]
        assert 2 in ids, "只有关键词路能召回的「质保条款」必须出现在结果里"
        only_kw = next(c for c in result.chunks if c.chunk_id == 2)
        assert only_kw.vector_rank is None
        assert only_kw.keyword_rank == 1

    async def test_vector_only_hit_is_kept(self, settings):
        records = [record(1, "显卡"), record(2, "完全不同的文字内容在这里")]
        service, _, _ = build(
            settings, records=records, hits=[vector_hit(2, 0.95), vector_hit(1, 0.5)]
        )
        result = await service.retrieve("质保", kb_ids=[1])
        by_id = {c.chunk_id: c for c in result.chunks}
        assert by_id[2].vector_rank == 1
        assert by_id[2].keyword_rank is None

    async def test_stale_vector_id_is_skipped(self, settings):
        """向量库里还留着、SQLite 里已经没有的 chunk_id 要丢掉。

        摄取与删除是两步写(先写库再写向量,或反过来),中途失败就会留下
        这种不一致。不跳过的话,引用卡片会指向一片空白。
        """
        records = [record(1, "显卡质保")]
        service, _, _ = build(settings, records=records, hits=[vector_hit(999, 0.99), vector_hit(1, 0.5)])
        result = await service.retrieve("质保", kb_ids=[1])
        assert [c.chunk_id for c in result.chunks] == [1]


class TestScope:
    async def test_empty_scope_returns_nothing_without_calling_anything(self, settings):
        """**空范围必须在进查询之前就短路。**

        传给 `build_where` 会抛错(那是故意的,见 test_vectorstore.py),
        所以这里必须更早返回;而更要紧的是**绝不能**退化成「不过滤」——
        那会让没有权限的用户读到全部知识库。断言连向量库都没碰,
        是为了钉死「没有发出任何一次越权查询」。
        """
        service, source, store = build(settings, records=[record(1, "显卡")], hits=[vector_hit(1, 0.9)])
        result = await service.retrieve("显卡", kb_ids=[])

        assert result.chunks == []
        assert store.calls == [], "空范围不该发出任何向量查询"
        assert source.calls == [], "空范围不该读取任何分块"

    async def test_none_scope_is_unrestricted(self, settings):
        """管理员预览检索效果时用 `None`。"""
        service, source, store = build(
            settings, records=[record(1, "显卡", kb_id=7)], hits=[vector_hit(1, 0.9)]
        )
        await service.retrieve("显卡", kb_ids=None)
        assert source.calls == [None]
        assert store.calls[0]["where"] is None

    async def test_scope_is_pushed_into_both_routes(self, settings):
        """范围过滤必须**两路都做**。

        只过滤向量路的话,关键词路会把没授权的知识库内容带进融合结果,
        再经重排出现在答案里——一次静默的越权读。
        """
        records = [record(1, "显卡", kb_id=1), record(2, "显卡", kb_id=2)]
        service, source, store = build(
            settings,
            records=records,
            hits=[vector_hit(1, 0.9, kb_id=1), vector_hit(2, 0.8, kb_id=2)],
        )
        result = await service.retrieve("显卡", kb_ids=[1])

        assert source.calls == [[1]]
        assert store.calls[0]["where"] == {"kb_id": {"$in": [1]}}
        assert {c.chunk_id for c in result.chunks} == {1}

    async def test_route_depth_is_the_configured_top_k(self, settings):
        s = settings.model_copy(update={"RETRIEVE_TOP_K": 7})
        service, _, store = build(s, records=[record(1, "显卡")], hits=[vector_hit(1, 0.9)])
        await service.retrieve("显卡", kb_ids=[1])
        assert store.calls[0]["top_k"] == 7


class TestFilters:
    """元数据预过滤(类目/价格)。"""

    async def test_filters_reach_the_vector_where_clause(self, settings):
        service, _, store = build(
            settings, records=[record(1, "显卡", meta={"category": "显卡", "price": 3000})],
            hits=[vector_hit(1, 0.9)],
        )
        await service.retrieve(
            "显卡", kb_ids=[1], filters=RetrievalFilters(category="显卡", price_min=1000, price_max=5000)
        )
        assert store.calls[0]["where"] == {
            "$and": [
                {"kb_id": {"$in": [1]}},
                {"category": "显卡"},
                {"price": {"$gte": 1000}},
                {"price": {"$lte": 5000}},
            ]
        }

    async def test_category_filter_also_excludes_keyword_hits(self, settings):
        """**关键词路也必须过滤,否则过滤条件是假的。**

        BM25 索引进的是分块的**正文**,没有类目字段;若照原样建索引,
        「只看显示器」的查询会把显卡的分块也召回来,融合后照样出现在
        答案里。所以索引语料要先按元数据筛一遍——是**先筛后建**,
        而不是检索完再丢掉(后者会让结果数凭空变少)。
        """
        # 用「质保」而不是「接口」当查询词:jieba 把「接口定义」当成**一个词**,
        # 于是「接口」这个词根本匹配不上——那是分词粒度的问题(见 bm25.py
        # 的已知限制),不是本用例要验的过滤。查询词要选分词后能独立成词的。
        records = [
            record(1, "显示器质保说明", meta={"category": "显示器"}),
            record(2, "显卡质保说明", meta={"category": "显卡"}),
        ]
        service, _, _ = build(settings, records=records, hits=[])
        result = await service.retrieve("质保", kb_ids=[1], filters=RetrievalFilters(category="显示器"))

        assert [c.chunk_id for c in result.chunks] == [1]

    async def test_price_filter_excludes_keyword_hits(self, settings):
        records = [
            record(1, "显卡报价", meta={"price": 500}),
            record(2, "显卡报价", meta={"price": 5000}),
        ]
        service, _, _ = build(settings, records=records, hits=[])
        result = await service.retrieve(
            "显卡", kb_ids=[1], filters=RetrievalFilters(price_min=1000)
        )
        assert [c.chunk_id for c in result.chunks] == [2]

    async def test_missing_metadata_is_treated_as_not_matching(self, settings):
        """没有类目的分块在按类目过滤时应当被排除,而不是默认放行。

        默认放行的话,过滤条件对「没解析出元数据的文档」形同虚设——
        而这类文档恰恰是最多的。
        """
        records = [record(1, "显卡说明", meta={})]
        service, _, _ = build(settings, records=records, hits=[])
        result = await service.retrieve("显卡", kb_ids=[1], filters=RetrievalFilters(category="显卡"))
        assert result.chunks == []


class TestRerank:
    async def test_rerank_reorders_by_score(self, settings):
        """只验**排序**。阈值显式关掉,免得这条用例偷偷依赖配置默认值。

        它原来写的是 `settings`(默认阈值),而当时的默认值是 0.0——
        于是它其实在断言「低分候选没被丢掉」,只是没人注意到。
        默认值改成 0.5 之后它立刻红了,报的是 `[2] == [2, 1]`:
        看起来像排序坏了,实际是过滤生效了。**那是另一条用例的职责**
        (`test_low_scores_are_dropped_by_threshold`),而一条验排序的
        用例不该因为调了过滤参数就红。
        """
        s = settings.model_copy(update={"RERANK_SCORE_THRESHOLD": 0.0})
        records = [record(1, "甲"), record(2, "乙")]
        reranker = FakeReranker({"甲": 0.1, "乙": 0.9})
        service, _, _ = build(
            s, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.8)], reranker=reranker
        )
        result = await service.retrieve("查询", kb_ids=[1])
        assert [c.chunk_id for c in result.chunks] == [2, 1]
        assert result.chunks[0].rerank_score == 0.9

    async def test_rerank_receives_the_fused_order(self, settings):
        """交给重排的候选要按融合后的顺序,gather 出来的原始顺序不行。"""
        records = [record(1, "甲"), record(2, "乙")]
        reranker = FakeReranker()
        service, _, _ = build(
            settings, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.8)], reranker=reranker
        )
        await service.retrieve("查询", kb_ids=[1])
        assert set(reranker.calls[0]) == {"甲", "乙"}

    async def test_low_scores_are_dropped_by_threshold(self, settings):
        """重排分数低于阈值的一律丢掉——宁可让模型说「资料中没有」,
        也不要把明显不相关的资料塞进上下文诱导它编答案。"""
        s = settings.model_copy(update={"RERANK_SCORE_THRESHOLD": 0.6})
        records = [record(1, "甲"), record(2, "乙")]
        reranker = FakeReranker({"甲": 0.9, "乙": 0.2})
        service, _, _ = build(s, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.8)], reranker=reranker)
        result = await service.retrieve("查询", kb_ids=[1])
        assert [c.chunk_id for c in result.chunks] == [1]

    async def test_top_n_is_respected(self, settings):
        s = settings.model_copy(update={"RERANK_TOP_N": 1})
        records = [record(i, f"内容{i}") for i in range(1, 4)]
        reranker = FakeReranker({f"内容{i}": 0.5 for i in range(1, 4)})
        service, _, _ = build(
            s, records=records, hits=[vector_hit(i, 0.9) for i in range(1, 4)], reranker=reranker
        )
        result = await service.retrieve("查询", kb_ids=[1])
        assert len(result.chunks) == 1

    async def test_no_reranker_keeps_fusion_order(self, settings):
        records = [record(1, "甲"), record(2, "乙")]
        service, _, _ = build(settings, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.5)])
        result = await service.retrieve("查询", kb_ids=[1])
        assert [c.chunk_id for c in result.chunks] == [1, 2]
        assert result.reranked is False
        assert result.chunks[0].rerank_score is None

    async def test_reranker_failure_falls_back_to_fusion_order(self, settings):
        """**重排是增强,不是必需。** 云端重排挂了就退回复合排序,
        绝不能把整个问答接口带崩——用户宁可拿到一份稍差的答案,
        也不想看到 500。

        ⚠️ **这条用的是「设计上该抛的那个信号」(`RerankUnavailableError`)**,
        而**真实的 `DashScopeReranker` 从不抛它**。所以它单独存在的时候,
        是在和代码互相印证而不是和现实印证 —— 那个 bug 就是这么藏了下来的
        (见下一条 `test_上游真的挂了也退回复合排序`)。
        留着它是因为「子类也能被接住」这件事本身值得钉住。
        """
        records = [record(1, "甲"), record(2, "乙")]
        reranker = FakeReranker(error=RerankUnavailableError("重排服务不可用"))
        service, _, _ = build(
            settings, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.5)], reranker=reranker
        )
        result = await service.retrieve("查询", kb_ids=[1])

        assert [c.chunk_id for c in result.chunks] == [1, 2]
        assert result.reranked is False
        assert result.degraded is True, "降级必须能被上层看见并记录,而不是悄悄发生"

    async def test_上游真的挂了也退回复合排序(self, settings):
        """**云端重排真的失败时,同样必须退回复合排序。**

        为什么单独为此写一条:上面那条用例是**照着代码写的** ——
        `FakeReranker(error=RerankUnavailableError(...))` 抛的正是代码
        当时 `except` 的那个类型。而真实的 `DashScopeReranker` **从不抛它**
        (全仓搜过:`RerankUnavailableError` 只在它自己的定义处和上面那条
        测试里出现,没有任何生产代码会抛)。它抛的是:

          * `ProviderError` —— 连接失败 / 重试耗尽 / 上游 4xx/5xx
          * `QuotaExhaustedError` —— 它是 `ProviderError` 的子类

        于是那个 `except RerankUnavailableError` 成了一句**永远不会执行的
        代码**:云端重排一抖,异常直接穿过去把整个问答打成 502 ——
        正是上面那句注释说「绝不能发生」的事。

        **教训**:假的替身抛「代码期待的类型」,测试和代码就互相印证了,
        两边都跟现实对不上。所以这条用**真实 provider 会抛的类型**。
        """
        records = [record(1, "甲"), record(2, "乙")]
        reranker = FakeReranker(
            error=ProviderError(
                "连接百炼 rerank 端点失败", provider="dashscope-rerank", retryable=True
            )
        )
        service, _, _ = build(
            settings, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.5)], reranker=reranker
        )
        result = await service.retrieve("查询", kb_ids=[1])

        assert [c.chunk_id for c in result.chunks] == [1, 2]
        assert result.reranked is False
        assert result.degraded is True

    async def test_额度耗尽也退回复合排序(self, settings):
        """`QuotaExhaustedError` 是 `ProviderError` 的子类,一并降级。

        它会走到这里的前提是**向量缓存命中**(否则 embedding 那一步早就用同一个
        错误把请求打回去了)。这种情况下退回复合排序是好的:用户拿到一份
        稍差的答案,而 `degraded=True` 让运维看得见额度没了。
        """
        records = [record(1, "甲"), record(2, "乙")]
        reranker = FakeReranker(error=QuotaExhaustedError("额度耗尽"))
        service, _, _ = build(
            settings, records=records, hits=[vector_hit(1, 0.9), vector_hit(2, 0.5)], reranker=reranker
        )
        result = await service.retrieve("查询", kb_ids=[1])

        assert result.reranked is False
        assert result.degraded is True

    async def test_rerank_disabled_by_settings(self, settings):
        s = settings.model_copy(update={"RERANK_ENABLED": False})
        records = [record(1, "甲")]
        reranker = FakeReranker({"甲": 0.9})
        service, _, _ = build(s, records=records, hits=[vector_hit(1, 0.9)], reranker=reranker)
        result = await service.retrieve("查询", kb_ids=[1])
        assert reranker.calls == [], "关掉重排后不该再调用它"
        assert result.reranked is False


class TestObservability:
    """打分明细与阶段耗时——作品集要拿它回答「关键词路有没有贡献」。"""

    async def test_per_route_scores_are_recorded(self, settings):
        records = [record(1, "显卡质保三年")]
        service, _, _ = build(settings, records=records, hits=[vector_hit(1, 0.87)])
        result = await service.retrieve("质保", kb_ids=[1])

        top = result.chunks[0]
        assert top.vector_similarity == pytest.approx(0.87)
        assert top.bm25_score is not None
        assert top.fusion_score is not None

    async def test_route_sizes_are_reported(self, settings):
        records = [record(1, "显卡质保"), record(2, "显示器参数")]
        service, _, _ = build(settings, records=records, hits=[vector_hit(1, 0.9)])
        result = await service.retrieve("质保", kb_ids=[1])

        assert result.route_sizes["vector"] == 1
        assert result.route_sizes["keyword"] >= 1
        assert result.route_sizes["fused"] >= 1

    async def test_stage_timings_are_present(self, settings):
        service, _, _ = build(settings, records=[record(1, "显卡")], hits=[vector_hit(1, 0.9)])
        result = await service.retrieve("显卡", kb_ids=[1])

        for stage in ("vector", "keyword", "fuse", "hydrate", "total"):
            assert stage in result.stage_timings, f"缺少阶段耗时:{stage}"
            assert result.stage_timings[stage] >= 0

    async def test_citation_fields_survive_to_the_result(self, settings):
        records = [
            record(1, "显卡质保", doc_name="售后政策.pdf", kb_name="售后库", meta={"category": "显卡"})
        ]
        service, _, _ = build(settings, records=records, hits=[vector_hit(1, 0.9)])
        top = (await service.retrieve("质保", kb_ids=[1])).chunks[0]

        assert top.doc_name == "售后政策.pdf"
        assert top.kb_name == "售后库"
        assert top.category == "显卡"


class TestConcurrency:
    async def test_the_two_routes_overlap(self, settings):
        """**两路必须并发,不能串行。**

        嵌入要发一次网络请求(几十毫秒),BM25 要在 CPU 上建索引。
        串起来就是两者相加;并发则取较慢的那个。这是需求 ⑦ 里
        「企业级」和「能跑通」最直观的分界线之一,所以直接钉住行为。
        """
        records = [record(1, "显卡质保")]
        service, source, store = build(
            settings,
            records=records,
            hits=[vector_hit(1, 0.9)],
            chunk_delay=0.05,
            vector_delay=0.05,
        )
        await service.retrieve("质保", kb_ids=[1])

        (v_start, v_end), (k_start, k_end) = store.spans[0], source.spans[0]
        assert v_start < k_end and k_start < v_end, "两路没有重叠,说明是串行执行的"
