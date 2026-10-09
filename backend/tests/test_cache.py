"""缓存层。

三件事在这里被钉住,它们各自对应一个「不报错、只变坏」的失效模式:

1. **键必须包含全部影响结果的因素。** 少一个(模型名、query/document 角色、
   知识库范围、过滤条件),缓存就会把 A 的结果发给 B——而 A 和 B 单独看都
   是好结果,只有对比才发现张冠李戴。
2. **知识库写入必须让内容相关的缓存全失效。** 这是需求 7 里唯一会
   **给出错误答案**的缓存故障:文档已删除/已更新,缓存却还在拿旧的 chunk
   回答,而且回答里引用的来源编号指向一个已经不存在的片段。
3. **过期条目必须读作 miss,而不是读作空值。** 两者的区别在于前者会回源,
   后者会让检索拿到一个空结果并把它当成「知识库里没有」。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from app.db.base import Database

# ---------------------------------------------------------------------------
# 向量编解码
# ---------------------------------------------------------------------------


class TestVectorCodec:
    def test_round_trip_preserves_values(self) -> None:
        from app.services.cache import pack_vector, unpack_vector

        vec = [0.5, -1.25, 0.0, 3.75]
        assert unpack_vector(pack_vector(vec)) == pytest.approx(vec, abs=1e-6)

    def test_float32_costs_four_bytes_per_dimension(self) -> None:
        """体积是选 float32 而不是 JSON 的全部理由,所以钉住它。

        1024 维的 JSON 数组大约 12KB(每个浮点约 12 个字符),float32 是 4KB。
        这个差别在「每个分块都要存一条」的规模上才显现,而那时换编码格式
        意味着所有已有缓存失效——所以从一开始就用紧凑格式。
        """
        from app.services.cache import pack_vector

        assert len(pack_vector([0.0] * 1024)) == 4096

    def test_empty_vector_is_rejected(self) -> None:
        """空向量缓存下来,余弦相似度会变成除以零。宁可当场炸。"""
        from app.services.cache import pack_vector

        with pytest.raises(ValueError):
            pack_vector([])

    def test_corrupted_blob_raises_instead_of_returning_garbage(self) -> None:
        """截断的 blob 若被 `frombytes` 静默接受,会得到一个短向量。

        短向量的后果是余弦相似度算出 0 或抛 IndexError,取决于两条长度谁更长
        ——一个「有时崩、有时只是不命中」的故障。所以长度不对必须当场报错。
        """
        from app.services.cache import pack_vector, unpack_vector

        blob = pack_vector([0.1, 0.2, 0.3, 0.4])
        with pytest.raises(ValueError):
            unpack_vector(blob[:-3])

    def test_expected_dim_is_enforced_when_given(self) -> None:
        """缓存里存着 768 维、模型现在出 1024 维:这种情况必须被发现。

        换 embedding 模型但没清缓存时,检索会拿两种维度的向量做相似度比较。
        """
        from app.services.cache import pack_vector, unpack_vector

        blob = pack_vector([0.1] * 768)
        with pytest.raises(ValueError):
            unpack_vector(blob, expected_dim=1024)


# ---------------------------------------------------------------------------
# 通用 KV 存储
# ---------------------------------------------------------------------------


@pytest.fixture
async def db(settings) -> AsyncIterator[Database]:
    """独立引擎 + 建好表的临时库。

    不共用 `client` 夹具:那一个会把整个 FastAPI 应用也带起来,而这一层
    要测的是缓存本体,不该依赖路由、认证、lifespan。
    """
    settings.ensure_dirs()
    database = Database(settings)
    await database.create_all()
    try:
        yield database
    finally:
        await database.dispose()


class TestCacheStore:
    async def test_round_trip(self, db: Database) -> None:
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "k", b"v", ttl_seconds=60)
        assert await store.get("demo", "k") == b"v"

    async def test_miss_returns_none(self, db: Database) -> None:
        from app.services.cache import CacheStore

        assert await CacheStore(db).get("demo", "nope") is None

    async def test_namespaces_are_isolated(self, db: Database) -> None:
        """同名不同域必须互不可见。

        检索缓存和答案缓存的键长得可能一模一样(都是问题文本的哈希),
        少一层 namespace 就会互相覆盖。
        """
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("retrieval", "same-key", b"retrieval-value", ttl_seconds=60)
        await store.set("answer", "same-key", b"answer-value", ttl_seconds=60)
        assert await store.get("retrieval", "same-key") == b"retrieval-value"
        assert await store.get("answer", "same-key") == b"answer-value"

    async def test_overwrite_replaces_the_value(self, db: Database) -> None:
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "k", b"old", ttl_seconds=60)
        await store.set("demo", "k", b"new", ttl_seconds=60)
        assert await store.get("demo", "k") == b"new"

    async def test_expired_entry_reads_as_a_miss(self, db: Database) -> None:
        """过期的条目必须等同于不存在。

        返回 None 而不是空字节串:调用方对「没有缓存」和「缓存了一个空值」
        的处理完全不同,后者会被当成一次真实的空结果。
        """
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "k", b"v", ttl_seconds=-1)  # 写入即过期
        assert await store.get("demo", "k") is None

    async def test_none_ttl_never_expires(self, db: Database) -> None:
        """`None` = 永久。向量缓存必须永久——同一个文本同一个模型,
        向量永远一样,给它设 TTL 只会白白丢掉重算代价最高的一层。"""
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "k", b"v", ttl_seconds=None)
        assert await store.get("demo", "k") == b"v"

    async def test_purge_expired_removes_only_expired_rows(self, db: Database) -> None:
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "dead", b"v", ttl_seconds=-1)
        await store.set("demo", "alive", b"v", ttl_seconds=600)

        assert await store.purge_expired() == 1
        assert await store.get("demo", "alive") == b"v"

    async def test_clear_leaves_other_namespaces_alone(self, db: Database) -> None:
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("retrieval", "k", b"v", ttl_seconds=60)
        await store.set("embedding", "k", b"v", ttl_seconds=None)

        assert await store.clear("retrieval") == 1
        assert await store.get("embedding", "k") == b"v"

    async def test_get_many_returns_only_the_hits(self, db: Database) -> None:
        """批量读是给 embedding 用的:一次入库几百个分块,逐个建会话去查
        会把「批量」这件事的意义抵消掉。"""
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "a", b"1", ttl_seconds=60)
        await store.set("demo", "c", b"3", ttl_seconds=60)

        got = await store.get_many("demo", ["a", "b", "c"])
        assert got == {"a": b"1", "c": b"3"}

    async def test_get_many_skips_expired_entries(self, db: Database) -> None:
        """批量读必须和单条读一样对待过期条目。

        这条单独钉住 `get_many` 里的过期过滤:单条读的过滤有它自己的用例,
        而批量读是**另一条代码路径**——它走 SQL 把整批捞回来再筛,少一个
        判断就会把过期向量当成有效向量发给检索。表现是「偶尔召回到一批
        语义上完全无关的分块」,而不是报错。
        """
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "fresh", b"1", ttl_seconds=60)
        await store.set("demo", "stale", b"2", ttl_seconds=-1)

        assert await store.get_many("demo", ["fresh", "stale"]) == {"fresh": b"1"}


# ---------------------------------------------------------------------------
# 语料代次
# ---------------------------------------------------------------------------


class TestCorpusVersion:
    """代次是失效机制的地基。它错了,上层所有缓存都会悄悄给出旧答案。"""

    async def test_starts_at_zero(self, db: Database) -> None:
        from app.services.cache import CorpusVersion

        assert await CorpusVersion(db).current() == 0

    async def test_bump_increments_and_is_visible_to_readers(self, db: Database) -> None:
        from app.services.cache import CorpusVersion

        version = CorpusVersion(db)
        assert await version.bump() == 1
        assert await version.current() == 1
        assert await version.bump() == 2
        assert await version.current() == 2

    async def test_concurrent_bumps_do_not_lose_updates(self, db: Database) -> None:
        """并发写入不能丢代次。

        **这条用例存在的唯一理由,是否掉「读-改-写」实现。**

        最自然的写法是 `value = read(); write(value + 1)`。它在单请求下完全
        正确,测试也全绿;但在 `gather` 下各路会**先各自读完再各自写回**,
        于是五路只数出了 2(实测,五次重跑稳定复现;生产实现稳定是 5)。
        丢了代次就意味着:两次知识库写入之后代次只前进了一格,而中间那次
        缓存条目会一直命中——**正是本模块文档里说的那个「给出错误答案」的
        故障**,而且它只在并发下出现,压测之外的任何时候都看不出来。

        实现因此用 SQL 侧的 `INSERT ... ON CONFLICT DO UPDATE`,
        自增发生在数据库里,不存在「读到的值已经过期」这个窗口。
        """
        from app.services.cache import CorpusVersion

        version = CorpusVersion(db)
        await asyncio.gather(*(version.bump() for _ in range(5)))

        assert await version.current() == 5


# ---------------------------------------------------------------------------
# 检索缓存
# ---------------------------------------------------------------------------


def _chunk(cid: int, content: str = "RX9070 显卡售价 4299 元") -> Any:
    from app.services.context import RetrievedChunk

    return RetrievedChunk(
        chunk_id=cid,
        content=content,
        kb_id=1,
        doc_id=7,
        doc_name="显卡手册.pdf",
        kb_name="商品库",
        chunk_index=2,
        meta={"category": "显卡", "price": 4299.0},
        vector_rank=1,
        keyword_rank=3,
        vector_similarity=0.83,
        bm25_score=11.2,
        fusion_score=0.032,
        rerank_score=0.71,
    )


class _CountingRetriever:
    """记录每次调用参数的假检索器。**不 mock 检索逻辑,只数调用次数**——
    这一层要验证的是「有没有回源」,而不是检索本身对不对(那是
    `test_retrieval*.py` 的事)。"""

    def __init__(self, chunks: list[Any] | None = None, *, degraded: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self._chunks = chunks if chunks is not None else [_chunk(1), _chunk(2, "库存 12 件")]
        self._degraded = degraded

    async def retrieve(
        self,
        query: str,
        *,
        kb_ids: list[int] | None,
        filters: Any = None,
        top_k: int | None = None,
    ) -> Any:
        from app.services.retrieval import RetrievalResult

        self.calls.append({"query": query, "kb_ids": kb_ids, "filters": filters, "top_k": top_k})
        return RetrievalResult(
            query=query,
            chunks=list(self._chunks),
            route_sizes={"vector": 2, "keyword": 1},
            stage_timings={"total": 512.0, "vector": 400.0},
            reranked=not self._degraded,
            degraded=self._degraded,
        )


def _cached(inner: Any, store: Any, version: Any, settings: Any, **kw: Any) -> Any:
    from app.services.cache import CachedRetriever

    return CachedRetriever(inner=inner, store=store, version=version, settings=settings, **kw)


@pytest.fixture
async def wired(db: Database, settings) -> tuple[Any, Any, Any, Any]:
    """一套接好的 (retriever, inner, store, version)。"""
    from app.services.cache import CacheStore, CorpusVersion

    store = CacheStore(db)
    version = CorpusVersion(db)
    inner = _CountingRetriever()
    return _cached(inner, store, version, settings), inner, store, version


class TestRetrievalCache:
    async def test_second_identical_query_never_reaches_the_inner_retriever(
        self, wired
    ) -> None:
        """缓存存在的**全部意义**。检索是这条链路上最贵的一段
        (编码 + 两路召回 + 重排),命中一次就省一次。"""
        retriever, inner, _store, _version = wired

        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])

        assert len(inner.calls) == 1

    async def test_a_hit_is_observable(self, wired) -> None:
        """命中必须能被看见。

        这不只是为了打指标:**排障时第一个要回答的问题是「这个结果是不是
        缓存给的」**。看不见它,就只能靠「重启一下试试」来二分,而那既不
        可复现,也不能写进事故报告。
        """
        retriever, _inner, _store, _version = wired

        first = await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        second = await retriever.retrieve("RX9070 多少钱", kb_ids=[1])

        assert first.cache_hit is False
        assert second.cache_hit is True

    async def test_a_hit_restores_the_full_chunk_payload(self, wired) -> None:
        """反序列化必须是无损的。

        丢掉 `rerank_score` 或 `vector_rank` 不会报错——来源卡片照常渲染,
        只是「这条是关键词路召回的」这类信息没了,而它正是调检索参数的
        唯一依据。所以这里逐字段比,不只比 chunk_id。
        """
        retriever, _inner, _store, _version = wired

        first = await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        second = await retriever.retrieve("RX9070 多少钱", kb_ids=[1])

        assert second.chunks == first.chunks
        assert second.route_sizes == first.route_sizes
        assert second.reranked is True

    async def test_different_kb_scope_is_a_different_entry(self, wired) -> None:
        """**跨库串味是这一层最严重的故障**:用户只勾了 A 库,却拿到 B 库
        的结果,而且答案读起来毫无异样。"""
        retriever, inner, _store, _version = wired

        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        await retriever.retrieve("RX9070 多少钱", kb_ids=[2])

        assert len(inner.calls) == 2

    async def test_unscoped_and_empty_scope_are_different_entries(self, wired) -> None:
        """`None`(不限)与 `[]`(什么都不看)语义不同,键也必须不同。

        把两者归一成同一个键,等于让「不勾任何知识库」的提问命中一次
        「全库检索」的结果——一次越权读,而且不报错。"""
        retriever, inner, _store, _version = wired

        await retriever.retrieve("RX9070 多少钱", kb_ids=None)
        await retriever.retrieve("RX9070 多少钱", kb_ids=[])

        assert len(inner.calls) == 2

    async def test_kb_scope_order_does_not_matter(self, wired) -> None:
        """`[1, 2]` 与 `[2, 1]` 是同一次检索,不该占两条缓存。

        不归一化的话命中率会随用户勾选顺序抖动,而这是纯粹的白丢。"""
        retriever, inner, _store, _version = wired

        await retriever.retrieve("RX9070 多少钱", kb_ids=[1, 2])
        await retriever.retrieve("RX9070 多少钱", kb_ids=[2, 1])

        assert len(inner.calls) == 1

    async def test_filters_are_part_of_the_key(self, wired) -> None:
        """类目/价格预过滤改变的是**召回集合本身**,不是一个后处理。"""
        from app.services.retrieval import RetrievalFilters

        retriever, inner, _store, _version = wired

        await retriever.retrieve("显卡", kb_ids=[1])
        await retriever.retrieve("显卡", kb_ids=[1], filters=RetrievalFilters(category="显卡"))

        assert len(inner.calls) == 2

    async def test_top_k_none_means_the_configured_default(self, wired, settings) -> None:
        """`top_k=None` 与显式传默认值是同一件事,不该占两条缓存。"""
        retriever, inner, _store, _version = wired

        await retriever.retrieve("RX9070", kb_ids=[1])
        await retriever.retrieve("RX9070", kb_ids=[1], top_k=settings.RETRIEVE_TOP_K)

        assert len(inner.calls) == 1

    async def test_a_corpus_write_invalidates_the_entry(self, wired) -> None:
        """**需求 7 里唯一会给出错误答案的缓存故障。**

        文档已删除而缓存还在,答案会引用一个不存在的来源编号——用户看不
        出来,系统也不报错。代次是这里唯一的防线。
        """
        retriever, inner, _store, version = wired

        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        await version.bump()
        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])

        assert len(inner.calls) == 2

    async def test_a_corrupted_entry_falls_back_instead_of_failing(
        self, wired, db: Database
    ) -> None:
        """缓存损坏**绝不能**让提问失败。

        缓存是纯优化:它坏了,代价应该是「这次慢一点」,而不是「用户问不
        了问题」。所以反序列化失败一律退化成 miss 并回源,同时留一条
        warning——静默吞掉会让「缓存一直在坏」这件事永远没人发现。
        """
        from sqlalchemy import text

        retriever, inner, _store, _version = wired
        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])

        async with db.session() as session:
            await session.execute(
                text("UPDATE cache_entries SET value = X'DEADBEEF' WHERE namespace = 'retrieval'")
            )
            await session.commit()

        result = await retriever.retrieve("RX9070 多少钱", kb_ids=[1])

        assert len(inner.calls) == 2
        assert result.cache_hit is False
        assert result.chunks

    async def test_the_cache_can_be_switched_off(self, db: Database, settings) -> None:
        """每一层都必须能单独关掉。

        排障时第一步就是确认「这是不是缓存带来的」;关不掉的话,就只能靠
        读代码推断,而推断出来的结论没法当作证据。
        """
        from app.services.cache import CacheStore, CorpusVersion

        inner = _CountingRetriever()
        retriever = _cached(
            inner, CacheStore(db), CorpusVersion(db), settings, enabled=False
        )

        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        await retriever.retrieve("RX9070 多少钱", kb_ids=[1])

        assert len(inner.calls) == 2

    async def test_a_degraded_result_is_never_cached(self, db: Database, settings) -> None:
        """重排没跑成(云端不可用)的结果不许进缓存。

        `degraded` 的含义是「本该重排却退回复合排序」,也就是一次**已知的
        质量让步**,它对应的是「重排服务此刻不可用」这个**瞬时**状态。
        把它写进缓存,等于把一次几分钟的抖动固化成一小时的默认行为:
        云端早就恢复了,用户还在拿没重排的结果,而且系统看上去一切正常
        ——降级本身是有标记的,但没人会去查,因为「缓存命中」让它显得
        像一条正常记录。

        同一个理由也适用于「向量路为空」这类瞬时失败。判断标准就一条:
        **只有当结果是「按当前的语料和配置正常算出来的」时才缓存它。**
        """
        from app.services.cache import CacheStore, CorpusVersion

        store, version = CacheStore(db), CorpusVersion(db)
        inner = _CountingRetriever(degraded=True)
        retriever = _cached(inner, store, version, settings)

        first = await retriever.retrieve("RX9070", kb_ids=[1])
        second = await retriever.retrieve("RX9070", kb_ids=[1])

        assert first.degraded is True
        assert second.degraded is True, "降级标记不能因为缓存而被抹掉"
        assert len(inner.calls) == 2, "降级结果被缓存了"

    async def test_a_config_change_invalidates_the_entry(self, db: Database, settings) -> None:
        """换 embedding/rerank 模型后旧缓存必须失效。

        只按问题文本做键的话,换模型当天所有人都在拿旧模型的召回结果回答,
        而「检索质量好像没变」这件事无从察觉——因为它确实没变,根本没生效。
        """
        from app.services.cache import CacheStore, CorpusVersion

        store, version = CacheStore(db), CorpusVersion(db)
        inner = _CountingRetriever()

        await _cached(inner, store, version, settings).retrieve("RX9070", kb_ids=[1])

        changed = settings.model_copy(update={"EMBED_MODEL": "另一个模型"})
        await _cached(inner, store, version, changed).retrieve("RX9070", kb_ids=[1])

        assert len(inner.calls) == 2


# ---------------------------------------------------------------------------
# 知识库写入必须让缓存失效
#
# 上面 `test_a_corpus_write_invalidates_the_entry` 证明的是「代次一变,缓存就
# 读不到」。**代次本身会不会变,是另一回事**——它取决于每一处写入是否都记得
# `bump()`。下面这些用例钉的是那个「记得」。
# ---------------------------------------------------------------------------


class _StubVectorStore:
    """只记调用的向量库替身。这一组用例要验的是代次,不是向量。"""

    def __init__(self) -> None:
        self.upserts = 0
        self.deletes: list[dict[str, Any]] = []

    async def upsert(self, **_kw: Any) -> None:
        self.upserts += 1

    async def delete_where(self, where: dict[str, Any]) -> None:
        self.deletes.append(where)


async def _make_document(
    db: Database, settings: Any, *, content: str = "RX9070 显卡售价 4299 元。库存 12 件。"
) -> tuple[int, int]:
    """建一个知识库 + 一篇已落盘的文档,返回 (kb_id, doc_id)。"""
    import uuid
    from pathlib import Path

    from app.db.models import Document, DocumentStatus, KnowledgeBase

    settings.ensure_dirs()
    path = Path(settings.UPLOAD_DIR) / f"{uuid.uuid4().hex}.txt"
    path.write_text(content, encoding="utf-8")

    async with db.session() as session:
        kb = KnowledgeBase(name=f"库-{uuid.uuid4().hex[:8]}")
        session.add(kb)
        await session.flush()
        doc = Document(
            kb_id=kb.id,
            filename="商品手册.txt",
            stored_path=str(path),
            file_hash=uuid.uuid4().hex,
            mime="text/plain",
            size=len(content.encode("utf-8")),
            status=str(DocumentStatus.PENDING),
        )
        session.add(doc)
        await session.commit()
        return kb.id, doc.id


async def _ingest(db: Database, settings: Any, doc_id: int, store: _StubVectorStore) -> int:
    """跑一次真实摄取。**不 mock 摄取本身**——这一组用例要验的正是
    「真实的写入路径有没有记得换代次」,把摄取换成假的就等于没验。"""
    from app.db.models import Document
    from app.providers.embedding import build_embedder
    from app.services.ingest import IngestService

    async with db.session() as session:
        doc = await session.get(Document, doc_id)
        service = IngestService(
            session, settings, embedder=build_embedder(settings), vector_store=store
        )
        return await service.ingest_document(doc)


class TestKnowledgeBaseWritesInvalidateTheCache:
    async def test_ingesting_a_document_bumps_the_version(
        self, db: Database, settings: Any
    ) -> None:
        from app.services.cache import CorpusVersion

        version = CorpusVersion(db)
        _kb_id, doc_id = await _make_document(db, settings)

        chunks = await _ingest(db, settings, doc_id, _StubVectorStore())

        assert chunks > 0, "摄取本身失败了,这条用例就什么都没验到"
        assert await version.current() == 1

    async def test_deleting_a_document_bumps_the_version(
        self, db: Database, settings: Any
    ) -> None:
        from app.db.models import Document
        from app.providers.embedding import build_embedder
        from app.services.cache import CorpusVersion
        from app.services.ingest import IngestService

        version = CorpusVersion(db)
        _kb_id, doc_id = await _make_document(db, settings)
        await _ingest(db, settings, doc_id, _StubVectorStore())
        assert await version.current() == 1

        async with db.session() as session:
            doc = await session.get(Document, doc_id)
            await IngestService(
                session,
                settings,
                embedder=build_embedder(settings),
                vector_store=_StubVectorStore(),
            ).delete_document(doc)

        assert await version.current() == 2

    async def test_deleting_a_knowledge_base_bumps_the_version(
        self, db: Database, settings: Any
    ) -> None:
        """删库走的是外键级联,分块不是被显式删掉的。

        级联删除**绕过了 `IngestService` 里所有显式的分块操作**,所以它特别
        容易漏——「我们只是在删知识库,又没碰分块」。但它确实把分块删光了,
        缓存里那些 chunk 现在指向不存在的行。
        """
        from app.db.models import KnowledgeBase
        from app.providers.embedding import build_embedder
        from app.services.cache import CorpusVersion
        from app.services.ingest import IngestService

        version = CorpusVersion(db)
        kb_id, doc_id = await _make_document(db, settings)
        await _ingest(db, settings, doc_id, _StubVectorStore())
        assert await version.current() == 1

        async with db.session() as session:
            kb = await session.get(KnowledgeBase, kb_id)
            await IngestService(
                session, settings, embedder=build_embedder(settings), vector_store=_StubVectorStore()
            ).delete_knowledge_base(kb)

        assert await version.current() == 2

    async def test_a_failed_ingest_does_not_bump(self, db: Database, settings: Any) -> None:
        """**反向对照:失败的摄取不该换代次。**

        这条钉的是「代次在事务里」,而不是「代次被调用了」。若把 `bump()`
        放在写入**之前**(或放在回滚不会撤销的地方),失败路径也会推进代次
        ——那本身只是白白清空缓存,不算错。真正错的是反过来的那种布局:
        分块写进去了、代次没跟上。用一条「失败也涨」的实现,两条都会被
        这条用例挡住,迫使人去写对位置。

        做法:让向量库在 upsert 时炸,分块写入随之回滚。
        """
        from app.services.cache import CorpusVersion

        class _ExplodingStore(_StubVectorStore):
            async def upsert(self, **_kw: Any) -> None:
                raise RuntimeError("向量库炸了")

        version = CorpusVersion(db)
        _kb_id, doc_id = await _make_document(db, settings)

        chunks = await _ingest(db, settings, doc_id, _ExplodingStore())

        assert chunks == 0
        assert await version.current() == 0, "失败的摄取推进了代次,说明它不在写入的事务里"

    async def test_a_real_ingest_invalidates_a_cached_retrieval(
        self, db: Database, settings: Any
    ) -> None:
        """**这一条是整个失效机制的端到端验收。**

        前面几条各自只验了一半:缓存那条验「代次变了就读不到」,摄取那几条
        验「写入会换代次」。这里把两半接起来——**真实摄取之后,之前缓存的
        检索结果必须读不到了**。做作品集时,「缓存失效」这件事要么能用一条
        这样的用例说清楚,要么就只是一句自我感觉良好的注释。
        """
        from app.services.cache import CacheStore, CorpusVersion

        retriever = _cached(_CountingRetriever(), CacheStore(db), CorpusVersion(db), settings)

        first = await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        assert first.cache_hit is False
        assert (await retriever.retrieve("RX9070 多少钱", kb_ids=[1])).cache_hit is True

        _kb_id, doc_id = await _make_document(db, settings)
        await _ingest(db, settings, doc_id, _StubVectorStore())

        after = await retriever.retrieve("RX9070 多少钱", kb_ids=[1])
        assert after.cache_hit is False, "知识库写完之后还在用旧缓存"


# ---------------------------------------------------------------------------
# 答案缓存
#
# 这一层比检索缓存贵得多(省下的是一次 LLM 调用,而且用户不用等),
# 也比它危险得多:**它缓存的是最终答案**。检索缓存给错了,顶多是引用的
# 片段偏了;答案缓存给错了,用户看到的就是一段说得头头是道、来源编号
# 还齐全的错误回答——所有表面信号都正常。
#
# 因此这里多出一条检索缓存没有的约束:**带历史的提问一律不进这一层**。
# 理由见 `test_a_question_with_history_is_never_served_from_the_cache`。
# ---------------------------------------------------------------------------


def _source(index: int = 1, *, chunk_id: int = 1) -> Any:
    from app.services.rag import SourceRef

    return SourceRef(
        index=index,
        chunk_id=chunk_id,
        doc_id=7,
        doc_name="显卡手册.pdf",
        kb_id=1,
        kb_name="商品库",
        chunk_index=2,
        content="RX9070 支持三年质保。",
        category="显卡",
        price=4299.0,
        rerank_score=0.71,
        fusion_score=0.032,
        vector_rank=1,
        keyword_rank=3,
    )


def _answer(text: str = "RX9070 支持三年质保 [1]。") -> Any:
    from app.services.cache import CachedAnswer

    return CachedAnswer(answer=text, sources=[_source()])


@pytest.fixture
async def answers(db: Database, settings) -> tuple[Any, Any, Any]:
    """一套接好的 (cache, store, version)。"""
    from app.services.cache import AnswerCache, CacheStore, CorpusVersion

    store, version = CacheStore(db), CorpusVersion(db)
    return AnswerCache(store=store, version=version, settings=settings), store, version


class TestAnswerCache:
    async def test_a_stored_answer_comes_back(self, answers) -> None:
        """缓存存在的全部意义:命中一次就省一次 LLM 调用。"""
        cache, _store, _version = answers

        await cache.put("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        assert await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None) == _answer()

    async def test_a_miss_returns_none(self, answers) -> None:
        """未命中必须是 `None`,不能是「空答案」。

        两者在调用方那里都会走回源,但一个空字符串会一路流到 `done` 事件里,
        用户看到的是一个**空白的回答**,而不是模型重新生成的答案。
        """
        cache, _store, _version = answers

        assert await cache.get("从没问过的问题", history_messages=0, kb_ids=[1], filters=None) is None

    async def test_sources_survive_the_round_trip(self, answers) -> None:
        """来源必须无损还原。

        来源卡片是这个产品相对通用大模型的价值所在,而它的每一个字段都
        会被前端用到:`index` 是正文里 `[1]` 的锚点、`content` 是点开的原文、
        `vector_rank`/`keyword_rank` 是调检索参数的唯一依据。少一个字段不会
        报错,只是那个功能静默失效。
        """
        cache, _store, _version = answers
        stored = _answer()

        await cache.put("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=stored)
        got = await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None)

        assert got is not None
        assert got.sources == stored.sources

    async def test_a_corpus_write_invalidates_the_entry(self, answers) -> None:
        """知识库一变更,旧答案立刻不可用。

        这是这一层最不能出的错:文档已经删了/改了,用户还拿到按旧文档写的
        答案,而且**引用编号指向一个不存在的片段**——点开是空的,系统不报错。
        """
        cache, _store, version = answers
        await cache.put("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        await version.bump()

        assert await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None) is None

    async def test_a_prompt_change_invalidates_the_entry(
        self, answers, monkeypatch
    ) -> None:
        """改了系统提示词,旧答案必须失效。

        **提示词本身就是「答案长什么样」的一部分。** 往系统提示里加一条
        「价格一律带单位」之后,缓存里那些不带单位的答案会继续发给用户,
        而部署方完全看不出哪里错了——代码、模型、语料都没动过。

        所以这一条同时钉住实现方式:提示词必须在**调用时**读取。
        `from ... import SYSTEM_PROMPT` 会在模块导入时把它绑死,之后
        怎么改都读不到,而这条用例会照常通过——所以断言的是「改了就读不到」,
        不是「有个哈希字段」。
        """
        import app.services.prompts as prompts_module

        cache, _store, _version = answers
        await cache.put("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        monkeypatch.setattr(prompts_module, "SYSTEM_PROMPT", "完全不同的提示词")

        assert await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None) is None

    async def test_a_model_change_invalidates_the_entry(self, db: Database, settings) -> None:
        """换模型后旧答案必须失效——两个模型写出来的不是同一份东西。"""
        from app.services.cache import AnswerCache, CacheStore, CorpusVersion

        store, version = CacheStore(db), CorpusVersion(db)
        await AnswerCache(store=store, version=version, settings=settings).put(
            "RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer()
        )

        changed = settings.model_copy(update={"LLM_MODEL": "另一个模型"})
        other = AnswerCache(store=store, version=version, settings=changed)

        assert await other.get("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None) is None

    async def test_kb_scope_is_part_of_the_key(self, answers) -> None:
        """问同一个问题的两个人,勾选的知识库不同,答案可能完全不同。

        范围不进键的话,先问的那个人决定了后问的那个人看到什么——而后者
        看到的答案引用了**他没勾选的知识库**里的资料。
        """
        cache, _store, _version = answers
        await cache.put("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        assert await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[2], filters=None) is None
        assert await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[1, 2], filters=None) is None

    async def test_filters_are_part_of_the_key(self, answers) -> None:
        """「5000 以下的显卡」和「不限价位的显卡」不是同一个问题。"""
        from app.services.retrieval import RetrievalFilters

        cache, _store, _version = answers
        await cache.put("显卡有哪些", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        priced = RetrievalFilters(price_max=5000.0)
        assert await cache.get("显卡有哪些", history_messages=0, kb_ids=[1], filters=priced) is None

    async def test_a_question_with_history_is_never_served_from_the_cache(
        self, answers
    ) -> None:
        """**带历史的提问一律不走答案缓存——哪怕问题一字不差。**

        「它多少钱?」在「RX9070 的会话」里和在「键盘的会话里」是**两个不同
        的问题**,但它们的**键完全一样**:答案缓存拿不到历史正文(那要整段
        塞进键),只能看到「问题 + 范围 + 配置」。于是第二轮起,缓存会把
        上一个会话的答案原样发出去——措辞、来源、编号全都对得上,因为它
        本来就是一个真答案,只是回答的是另一个问题。

        这是「缓存给出错误答案」在信息检索里最纯粹的形态,而它没有任何
        表面迹象。所以约束必须做进结构里:**不是在键里加一个标记**,而是
        这条路径根本不启用。
        """
        cache, _store, _version = answers
        await cache.put("它多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        assert await cache.get("它多少钱", history_messages=2, kb_ids=[1], filters=None) is None

    async def test_an_answer_derived_from_history_is_never_written(self, answers) -> None:
        """反过来那一半:**带历史生成的答案也不许写进去**。

        只挡读不挡写是不够的,而且不够得很隐蔽:追问「它多少钱」生成出的
        答案是接着上下文答的,它的键却和一个**没有历史**的「它多少钱」完
        全相同。写进去之后,一个刚打开页面的用户问出同样三个字,拿到的
        是另一个人的追问答案。

        读挡的是「别拿错的」,写挡的是「别放错的」——两半都要有。
        """
        cache, _store, _version = answers

        await cache.put("它多少钱", history_messages=2, kb_ids=[1], filters=None, answer=_answer())

        assert await cache.get("它多少钱", history_messages=0, kb_ids=[1], filters=None) is None

    async def test_a_corrupted_entry_falls_back_to_a_miss(
        self, answers, db: Database
    ) -> None:
        """缓存损坏绝不能让提问失败,但也不能静默——留一条 warning。

        这一层尤其如此:抛出去就是「用户问不了问题」,而缓存坏了本该只是
        「这次慢一点」。
        """
        from sqlalchemy import text

        cache, _store, _version = answers
        await cache.put("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        async with db.session() as session:
            await session.execute(
                text("UPDATE cache_entries SET value = X'DEADBEEF' WHERE namespace = 'answer'")
            )
            await session.commit()

        assert await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None) is None

    async def test_the_cache_can_be_switched_off(self, db: Database, settings) -> None:
        """每一层都必须能单独关掉,排障第一步就是关掉可疑的那一层。"""
        from app.services.cache import AnswerCache, CacheStore, CorpusVersion

        cache = AnswerCache(
            store=CacheStore(db),
            version=CorpusVersion(db),
            settings=settings,
            enabled=False,
        )

        await cache.put("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None, answer=_answer())

        assert await cache.get("RX9070 多少钱", history_messages=0, kb_ids=[1], filters=None) is None


# ---------------------------------------------------------------------------
# 批量写入
# ---------------------------------------------------------------------------


class TestBatchWrites:
    async def test_many_entries_land_in_one_go(self, db: Database) -> None:
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set_many(
            "demo",
            {"a": b"1", "b": b"2", "c": b"3"},
            ttl_seconds=60,
        )

        assert await store.get("demo", "a") == b"1"
        assert await store.get("demo", "b") == b"2"
        assert await store.get("demo", "c") == b"3"

    async def test_an_empty_batch_writes_nothing(self, db: Database) -> None:
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set_many("demo", {}, ttl_seconds=60)

        assert await store.stats() == {}

    async def test_a_batch_does_not_disturb_other_namespaces(self, db: Database) -> None:
        """批量写必须仍按 namespace 分区——写 embedding 不能碰到 retrieval。

        逐个 `set` 时这一条是**自动成立**的,换成一条批量语句之后就成了
        要自己保证的事:`ON CONFLICT` 的冲突判据若只写了 key 而漏了
        namespace,两个域会互相覆盖,而且是静默的。
        """
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("retrieval", "k", b"original", ttl_seconds=60)
        await store.set_many("embedding", {"k": b"batch"}, ttl_seconds=60)

        assert await store.get("retrieval", "k") == b"original"
        assert await store.get("embedding", "k") == b"batch"

    async def test_a_batch_overwrites_existing_keys(self, db: Database) -> None:
        from app.services.cache import CacheStore

        store = CacheStore(db)
        await store.set("demo", "k", b"old", ttl_seconds=60)
        await store.set_many("demo", {"k": b"new"}, ttl_seconds=60)

        assert await store.get("demo", "k") == b"new"

    async def test_an_overlong_key_in_a_batch_is_rejected(self, db: Database) -> None:
        """单个 `set` 有键长校验,批量也必须走同一道校验。

        漏掉的话,超长键会在**入库跑到一半时**才炸,而那时前面几百条
        已经写进去了——一个「部分写入」的缓存,比全没写更难排查。
        """
        from app.services.cache import MAX_KEY_LENGTH, CacheStore

        store = CacheStore(db)
        with pytest.raises(ValueError):
            await store.set_many("demo", {"x" * (MAX_KEY_LENGTH + 1): b"v"}, ttl_seconds=60)


# ---------------------------------------------------------------------------
# 借用调用方事务的缓存
# ---------------------------------------------------------------------------


class TestBorrowedStore:
    """`CacheStore.on(session)` —— 让缓存写进**调用方那个事务**。

    **这不是风格选择,是 SQLite 的写锁决定的。** 默认的 `CacheStore(db)`
    每次都另开一个会话、自己提交;如果调用方此刻正持有一个未提交的写事务
    (文档摄取就是:它一边往 `chunks` 里写,一边要编码),那个新会话拿不到
    写锁。`busy_timeout` 帮不上忙——它不是「抢不到稍后再试」,而是**真死锁**:
    持锁的那个事务要等编码结束才提交,而编码正在等这把锁。实测等待满
    `busy_timeout=5000` 毫秒后以 `database is locked` 失败。

    `CorpusVersion` 早就是这个形状了(见其 `on()` 与 `_scope()`),这里把
    同一个解法搬到缓存上。**两条语义必须都对**:借来的不自己提交,自己的
    照常提交。
    """

    async def test_borrowed_writes_are_not_committed_by_the_store(self, db: Database) -> None:
        """**借来的事务由调用方决定生死。** 缓存自己提交的话,调用方随后
        `rollback()` 就只回滚了一半——文档没了而缓存在,或者反过来。

        这里断言的是「回滚之后缓存里也没有」,而那正是「没有自作主张提交」
        的可观测结果。
        """
        from app.services.cache import CacheStore

        async with db.session() as session:
            await CacheStore.on(session).set("demo", "k", b"v", ttl_seconds=60)
            await session.rollback()

        assert await CacheStore(db).get("demo", "k") is None

    async def test_borrowed_writes_survive_a_caller_commit(self, db: Database) -> None:
        """另一半:调用方提交了,缓存就得留下。"""
        from app.services.cache import CacheStore

        async with db.session() as session:
            await CacheStore.on(session).set("demo", "k", b"v", ttl_seconds=60)
            await session.commit()

        assert await CacheStore(db).get("demo", "k") == b"v"

    async def test_a_write_succeeds_while_the_caller_holds_the_write_lock(
        self, db: Database
    ) -> None:
        """**回归测试:调用方持写锁时,缓存写入不能失败。**

        先让调用方发出一个未提交的写(SQLite 在这一刻拿到写锁),再写缓存。
        用另开会话的实现,这里会等满 `busy_timeout` 然后抛
        `OperationalError: database is locked`——而那个错误出现在**文档摄取
        途中**,表现是「上传的文档莫名其妙变成 failed」,和缓存看不出关系。
        """
        from app.db.models import CacheEntry
        from app.services.cache import CacheStore

        async with db.session() as session:
            session.add(CacheEntry(namespace="held", key="held", value=b"x"))
            await session.flush()  # 写锁到手,且不提交

            await CacheStore.on(session).set("demo", "k", b"v", ttl_seconds=60)
            await session.commit()

        assert await CacheStore(db).get("demo", "k") == b"v", "持锁期间的缓存写入丢了"

    async def test_a_borrowed_write_is_invisible_until_the_caller_flushes(
        self, db: Database
    ) -> None:
        """**同一事务里读不到自己刚写的缓存,直到 flush。**

        会话关掉了 `autoflush`(见 `Database.session_factory`),这是全项目
        统一的约定,不是缓存引入的。写在这里是因为借用模式最容易踩:
        `self.set(); await self.get()` 在自持模式下成立(每次 `set` 自己提交),
        借来的模式不成立——而它**不报错**,只是返回 `None`,看起来像缓存没生效。

        生产路径不受影响:摄取只写不读(读发生在编码之前),所以这条只钉住
        「别指望借来的写入立刻可见」。
        """
        from app.services.cache import CacheStore

        async with db.session() as session:
            store = CacheStore.on(session)
            await store.set("demo", "k", b"v", ttl_seconds=60)

            assert await store.get("demo", "k") is None

            await session.flush()
            assert await store.get("demo", "k") == b"v"

    async def test_a_borrowed_store_still_reads_other_sessions_writes(
        self, db: Database
    ) -> None:
        """借用只是改写入的归属,读取的可见性不该跟着变窄。"""
        from app.services.cache import CacheStore

        await CacheStore(db).set("demo", "k", b"v", ttl_seconds=60)

        async with db.session() as session:
            assert await CacheStore.on(session).get("demo", "k") == b"v"


# ---------------------------------------------------------------------------
# Embedding 缓存
# ---------------------------------------------------------------------------


class _CountingEmbedder:
    """记下每一次被要求编码了哪些文本。

    **不 mock 编码本身。** 向量由文本哈希派生,于是「同一段文本两次编码
    得到同一个向量」是真的,测试可以直接比较向量相等,而不是去断言一个
    假的调用次数——后者在实现换成「每次都编码但结果对」时依然全绿。
    """

    def __init__(self, *, dim: int = 8, model_id: str = "fake-embed") -> None:
        from app.services.cache import text_hash

        self._dim = dim
        self._model_id = model_id
        self._hash = text_hash
        self.document_batches: list[list[str]] = []
        self.query_calls: list[str] = []

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def model_id(self) -> str:
        return self._model_id

    def vector(self, text: str) -> list[float]:
        seed = int(self._hash(text)[:8], 16) % 10_000
        return [float(seed)] * self._dim

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(list(texts))
        return [self.vector(t) for t in texts]

    async def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return self.vector(text)


@pytest.fixture
def embedder() -> _CountingEmbedder:
    return _CountingEmbedder()


def _cached_embedder(db: Database, settings, inner, **kwargs):  # type: ignore[no-untyped-def]
    from app.services.cache import CachedEmbedder, CacheStore

    return CachedEmbedder(inner=inner, store=CacheStore(db), settings=settings, **kwargs)


def _cached_over(store, settings, inner, **kwargs):  # type: ignore[no-untyped-def]
    """同一个 store 上再套一个包装器。用来测「两个不同配置共用一份缓存」。"""
    from app.services.cache import CachedEmbedder

    return CachedEmbedder(inner=inner, store=store, settings=settings, **kwargs)


class TestEmbeddingCache:
    async def test_a_repeated_query_is_served_from_the_cache(self, db, settings, embedder) -> None:
        cached = _cached_embedder(db, settings, embedder)

        first = await cached.embed_query("RX9070 显卡")
        second = await cached.embed_query("RX9070 显卡")

        assert first == second
        assert embedder.query_calls == ["RX9070 显卡"]

    async def test_the_dimension_and_model_are_delegated(self, db, settings, embedder) -> None:
        cached = _cached_embedder(db, settings, embedder)

        assert cached.dim == 8
        assert cached.model_id == "fake-embed"

    async def test_a_query_and_a_document_are_not_the_same_entry(self, db, settings, embedder) -> None:
        """同一段正文,一次当查询、一次当文档,必须各编码一次。

        部分模型的 query 侧要加指令前缀(bge-zh 系列),两条向量本就不同;
        就算某个模型不需要,**把两者当同一条缓存也是把「现在成立」当成
        「永远成立」**,而换模型时不会有任何信号提示这里要改。
        """
        cached = _cached_embedder(db, settings, embedder)

        await cached.embed_query("RX9070 显卡")
        await cached.embed_documents(["RX9070 显卡"])

        assert embedder.query_calls == ["RX9070 显卡"]
        assert embedder.document_batches == [["RX9070 显卡"]]

    async def test_a_different_model_does_not_reuse_entries(self, db, settings) -> None:
        """换 embedding 模型必须让缓存自然失效——两个模型的向量空间不可比。"""
        from app.services.cache import CacheStore

        store = CacheStore(db)
        first = _CountingEmbedder(model_id="model-a")
        second = _CountingEmbedder(model_id="model-b")

        await _cached_embedder(db, settings, first).embed_query("显卡")
        await _cached_over(store, settings, second).embed_query("显卡")

        assert first.query_calls == ["显卡"]
        assert second.query_calls == ["显卡"]

    async def test_a_dimension_change_is_a_miss_not_garbage(self, db, settings) -> None:
        """维度变了要重新编码,而不是把旧向量按新维度解出错值。

        解错值的后果不是崩溃而是**静默的**:向量还在、长度也对不上时才会
        抛,对得上时算出一个毫无意义的余弦相似度,检索照样返回结果。
        """
        from app.services.cache import CacheStore

        store = CacheStore(db)
        small = _CountingEmbedder(dim=8)
        big = _CountingEmbedder(dim=4)

        await _cached_embedder(db, settings, small).embed_query("显卡")
        await _cached_over(store, settings, big).embed_query("显卡")

        assert big.query_calls == ["显卡"]

    async def test_a_batch_only_encodes_the_misses(self, db, settings, embedder) -> None:
        """批量编码只发缺失的那些。**这是向量缓存省下最多的一处**——
        入库时几百个分块,重复措辞的商品文案非常多。"""
        cached = _cached_embedder(db, settings, embedder)

        await cached.embed_documents(["文案甲", "文案乙", "文案丙"])
        embedder.document_batches.clear()

        out = await cached.embed_documents(["文案甲", "文案乙", "文案丁"])

        assert embedder.document_batches == [["文案丁"]]
        assert out == [embedder.vector(t) for t in ("文案甲", "文案乙", "文案丁")]

    async def test_duplicates_inside_one_batch_are_encoded_once(self, db, settings, embedder) -> None:
        cached = _cached_embedder(db, settings, embedder)

        out = await cached.embed_documents(["文案甲", "文案甲", "文案乙"])

        assert embedder.document_batches == [["文案甲", "文案乙"]]
        assert len(out) == 3
        assert out[0] == out[1]

    async def test_the_batch_order_always_matches_the_input(self, db, settings, embedder) -> None:
        """返回顺序必须与入参一一对应,否则向量会被安到别的分块上——
        一个不报错、只让检索结果整体串位的故障。"""
        cached = _cached_embedder(db, settings, embedder)

        texts = ["甲", "乙", "丙", "丁", "戊"]
        await cached.embed_documents(texts[:2])  # 先垫一部分进缓存
        out = await cached.embed_documents(texts)

        assert out == [embedder.vector(t) for t in texts]

    async def test_a_corpus_write_does_not_invalidate_embeddings(self, db, settings, embedder) -> None:
        """**向量缓存不受语料代次影响,这正是它与检索缓存的区别。**

        一段文本的向量只是 (文本, 模型) 的函数,与知识库里有没有这段文本
        无关。检索缓存与答案缓存必须随知识库写入失效,而这一层**不该**——
        让它跟着失效,等于每次入库都把最贵的那部分编码全部重付一遍。
        """
        from app.services.cache import CorpusVersion

        cached = _cached_embedder(db, settings, embedder)
        version = CorpusVersion(db)

        await cached.embed_query("RX9070 显卡")
        await version.bump()
        await cached.embed_query("RX9070 显卡")

        assert embedder.query_calls == ["RX9070 显卡"]

    async def test_an_empty_batch_encodes_nothing(self, db, settings, embedder) -> None:
        cached = _cached_embedder(db, settings, embedder)

        assert await cached.embed_documents([]) == []
        assert embedder.document_batches == []

    async def test_the_cache_can_be_switched_off(self, db, settings, embedder) -> None:
        cached = _cached_embedder(db, settings, embedder, enabled=False)

        await cached.embed_query("显卡")
        await cached.embed_query("显卡")

        assert embedder.query_calls == ["显卡", "显卡"]

    async def test_a_corrupted_entry_falls_back_to_encoding(self, db, settings, embedder) -> None:
        """缓存坏掉只能退化成「这次慢一点」,不能让用户问不了问题。"""
        from app.services.cache import CacheStore
        from sqlalchemy import text

        cached = _cached_embedder(db, settings, embedder)
        await cached.embed_query("显卡")

        async with db.session() as session:
            await session.execute(
                text("UPDATE cache_entries SET value = X'DEADBEEF' WHERE namespace = 'embedding'")
            )
            await session.commit()

        await cached.embed_query("显卡")

        assert embedder.query_calls == ["显卡", "显卡"]
        assert await CacheStore(db).get("embedding", "x") is None  # 命名空间没被写坏


# ---------------------------------------------------------------------------
# 语义缓存
# ---------------------------------------------------------------------------


class _ScriptedEmbedder:
    """向量由用例指定,而不是由文本算出来。

    **这是语义缓存唯一可测的方式。** 这一层的行为完全取决于「两条问题向量
    有多近」,而那必须由用例说了算。改用 mock embedder(字符 n-gram 哈希)
    会让用例测的是**字符重叠**:它可能认为「RX9070 显卡质保多久」和
    「RX9070 显卡多少钱」比「RX9070 显卡保修几年」更近——于是用例会对一个
    完全正确的实现报错,而失败信息指向的方向是错的。

    没登记的文本直接抛:`KeyError` 比「悄悄返回一个零向量」有用得多,后者
    会让所有未登记的文本彼此相似度为 1。
    """

    def __init__(self, vectors: dict[str, list[float]], *, model_id: str = "scripted") -> None:
        self._vectors = vectors
        self._model_id = model_id
        self.calls: list[str] = []

    @property
    def dim(self) -> int:
        return len(next(iter(self._vectors.values())))

    @property
    def model_id(self) -> str:
        return self._model_id

    async def embed_query(self, text: str) -> list[float]:
        self.calls.append(text)
        return list(self._vectors[text])

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[t]) for t in texts]


def _unit(angle_cos: float, *, dim: int = 2) -> list[float]:
    """构造一个与 `[1, 0, …]` 夹角余弦恰为 `angle_cos` 的单位向量。

    用几何构造而不是随手写两个数:阈值附近那几条用例(0.9499 / 0.9501)
    必须**精确**落在两侧,靠手写数字凑不出来。
    """
    import math

    return [angle_cos, math.sqrt(max(0.0, 1.0 - angle_cos**2))] + [0.0] * (dim - 2)


def _semantic(db: Database, settings, *, vectors: dict[str, list[float]], **kwargs: Any):  # type: ignore[no-untyped-def]
    from app.services.cache import CorpusVersion, SemanticCache

    embedder = kwargs.pop("embedder", None) or _ScriptedEmbedder(vectors)
    return SemanticCache(
        db=db,
        version=CorpusVersion(db),
        settings=settings,
        embedder=embedder,
        **kwargs,
    )


@pytest.fixture
def semantic_settings(settings):
    """把开关打开的一份配置。

    默认是关的(见 `Settings.SEMANTIC_CACHE_ENABLED`),所以这一组用例必须
    显式打开——**顺带也就证明了那个开关真的在读**,而不是一个没人看的字段。
    """
    settings.SEMANTIC_CACHE_ENABLED = True
    return settings


class TestSemanticCache:
    async def test_a_near_duplicate_question_reuses_the_answer(
        self, db, semantic_settings
    ) -> None:
        """这一层存在的全部理由:换一种问法也能命中。

        阈值取 0.95(见配置),这里用 0.99 —— 一个明确高于它的近义改写。
        """
        cache = _semantic(
            db,
            semantic_settings,
            vectors={"RX9070 显卡质保多久": _unit(1.0), "RX9070 显卡保修几年": _unit(0.99)},
        )
        await cache.put(
            "RX9070 显卡质保多久", history_messages=0, kb_ids=[1], answer=_answer()
        )

        hit = await cache.get("RX9070 显卡保修几年", history_messages=0, kb_ids=[1])

        assert hit is not None
        assert hit.answer == _answer().answer

    async def test_a_dissimilar_question_is_a_miss(self, db, semantic_settings) -> None:
        """**这一层唯一真正的风险**:把另一个问题的答案发出去。

        0.5 远低于阈值,必须未命中。放在这里是因为「命中率」这个指标看不出
        假阳性——它只会显示缓存工作得很好。
        """
        cache = _semantic(
            db,
            semantic_settings,
            vectors={"RX9070 显卡质保多久": _unit(1.0), "RX9070 显卡多少钱": _unit(0.5)},
        )
        await cache.put(
            "RX9070 显卡质保多久", history_messages=0, kb_ids=[1], answer=_answer()
        )

        assert await cache.get("RX9070 显卡多少钱", history_messages=0, kb_ids=[1]) is None

    @pytest.mark.parametrize(
        ("similarity", "expected"),
        [(0.9, False), (0.9499, False), (0.9501, True), (0.999, True)],
    )
    async def test_the_threshold_is_the_boundary(
        self, db, semantic_settings, similarity: float, expected: bool
    ) -> None:
        """**阈值判定必须严格按配置来。**

        0.9499 与 0.9501 这对是刻意的:它们只差 0.0002,却必须落在两侧。
        放宽比对方向(写成 `> 0.9`,或者反过来判成「不相似则命中」)在
        「相似的问题命中」那条用例上是看不出来的。
        """
        cache = _semantic(
            db,
            semantic_settings,
            vectors={"原来": _unit(1.0), "换一种问法": _unit(similarity)},
        )
        await cache.put("原来", history_messages=0, kb_ids=[1], answer=_answer())

        hit = await cache.get("换一种问法", history_messages=0, kb_ids=[1])
        assert (hit is not None) is expected

    async def test_the_closest_candidate_wins(self, db, semantic_settings) -> None:
        """多个候选时取最近的那个,而不是「第一个够阈值的」。

        取第一个的话,答案取决于行的物理顺序——同样的数据换个插入顺序就
        给出不同的答案,而两次都在阈值之上,看不出任何异常。

        **查询向量取 `_unit(1.0)`(参照轴本身),于是每条候选与它的余弦恰好
        就是那条候选构造时给的那个数**——省掉手算夹角的步骤,顺带让两侧
        的差距由数值本身保证,而不是靠 0.0002 这种量级去卡。

        更近的那条(0.99)故意**放在后面插入**:「取第一个够阈值的」实现在
        这里会返回 0.97 那份,只有真的取 argmax 才会返回 0.99 那份。
        """
        cache = _semantic(
            db,
            semantic_settings,
            vectors={"远的": _unit(0.97), "近的": _unit(0.99), "查询": _unit(1.0)},
        )
        await cache.put("远的", history_messages=0, kb_ids=[1], answer=_answer("远的"))
        await cache.put("近的", history_messages=0, kb_ids=[1], answer=_answer("近的"))

        hit = await cache.get("查询", history_messages=0, kb_ids=[1])

        assert hit is not None
        assert hit.answer == "近的"

    async def test_a_different_scope_never_matches(self, db, semantic_settings) -> None:
        """**范围不同就是不同的问题。**

        向量一模一样也必须分开:同一个问题在「只看显卡库」和「全部知识库」
        两种范围下,检索到的资料不同,答案因此不同。这一条失守的后果是
        **一次不报错的越权读**——拿另一个库的资料回答。
        """
        cache = _semantic(db, semantic_settings, vectors={"显卡质保": _unit(1.0)})
        await cache.put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())
        await cache.put("显卡质保", history_messages=0, kb_ids=None, answer=_answer("全库"))

        assert await cache.get("显卡质保", history_messages=0, kb_ids=[2]) is None
        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is not None
        # `None`(不限)与 `[1]`(只看 1 号库)必须是两条互不可见的缓存。
        all_kb = await cache.get("显卡质保", history_messages=0, kb_ids=None)
        assert all_kb is not None
        assert all_kb.answer == "全库"

    async def test_a_corpus_write_invalidates_it(self, db, semantic_settings) -> None:
        """知识库一写入,语义缓存必须整体失效。

        和另外两层同一个机制(scope 里带语料代次),但这一层更要紧:文档
        被删掉之后,缓存里那条答案的来源编号指向一个不存在的片段,而它
        照常显示成一张可展开的来源卡片。
        """
        from app.services.cache import CorpusVersion

        cache = _semantic(db, semantic_settings, vectors={"显卡质保": _unit(1.0)})
        await cache.put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())
        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is not None

        await CorpusVersion(db).bump()

        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is None

    async def test_a_different_model_never_matches(self, db, settings) -> None:
        """换 embedding 模型后,旧向量算出来的余弦数值合法、含义为零。"""
        from app.services.cache import CorpusVersion, SemanticCache

        settings.SEMANTIC_CACHE_ENABLED = True
        vectors = {"显卡质保": _unit(1.0)}
        await SemanticCache(
            db=db,
            version=CorpusVersion(db),
            settings=settings,
            embedder=_ScriptedEmbedder(vectors, model_id="model-a"),
        ).put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())

        other = SemanticCache(
            db=db,
            version=CorpusVersion(db),
            settings=settings,
            embedder=_ScriptedEmbedder(vectors, model_id="model-b"),
        )

        assert await other.get("显卡质保", history_messages=0, kb_ids=[1]) is None

    async def test_a_question_with_history_is_never_served(self, db, semantic_settings) -> None:
        """带历史的提问一律不走这一层,和答案缓存同一条规矩。

        在语义缓存上它更危险:「它多少钱?」和另一个会话里的「它多少钱?」
        是两个问题,而它们的向量**完全相同**——比措辞不同还要更容易命中。
        """
        cache = _semantic(db, semantic_settings, vectors={"它多少钱": _unit(1.0)})
        await cache.put("它多少钱", history_messages=0, kb_ids=[1], answer=_answer())

        assert await cache.get("它多少钱", history_messages=2, kb_ids=[1]) is None

    async def test_an_answer_derived_from_history_is_never_written(
        self, db, semantic_settings
    ) -> None:
        """另一半:带历史生成出来的答案也不许写进去。"""
        cache = _semantic(db, semantic_settings, vectors={"它多少钱": _unit(1.0)})

        await cache.put("它多少钱", history_messages=2, kb_ids=[1], answer=_answer())

        assert await cache.get("它多少钱", history_messages=0, kb_ids=[1]) is None

    async def test_it_is_off_by_default(self, db, settings) -> None:
        """**默认关闭是这一层的产品决定,不是漏配。**

        另外两层给错了只是快慢;这一层给错了,用户拿到的是另一个问题的
        答案——来源、编号、措辞全都齐整。所以它必须显式打开。
        """
        cache = _semantic(db, settings, vectors={"显卡质保": _unit(1.0)})
        await cache.put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())

        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is None

    async def test_the_cache_can_be_switched_off_explicitly(self, db, semantic_settings) -> None:
        cache = _semantic(db, semantic_settings, vectors={"显卡质保": _unit(1.0)}, enabled=False)
        await cache.put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())

        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is None

    async def test_the_lookup_embeds_the_question(self, db, semantic_settings) -> None:
        """查一次语义缓存需要一次问题向量。

        **这正是它必须排在向量缓存之后才能启用的原因**:没有向量缓存时,
        每次未命中都要白搭一次网络编码(实测约 157ms),把最常见的路径
        拖慢——而缓存该做的是让常见路径更快。
        """
        embedder = _ScriptedEmbedder({"显卡质保": _unit(1.0)})
        cache = _semantic(db, semantic_settings, vectors={}, embedder=embedder)

        await cache.get("显卡质保", history_messages=0, kb_ids=[1])

        assert embedder.calls == ["显卡质保"]

    async def test_a_question_is_not_embedded_when_the_cache_is_off(self, db, settings) -> None:
        """关掉时必须**一步都不走**——包括那次编码。

        只断言「拿不到答案」是不够的:一个每次都编码、每次都不命中的实现
        同样满足它,而那种实现比没有缓存更慢。
        """
        embedder = _ScriptedEmbedder({"显卡质保": _unit(1.0)})
        cache = _semantic(db, settings, vectors={}, embedder=embedder)

        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is None
        assert embedder.calls == []

    async def test_a_corrupted_entry_falls_back_to_a_miss(self, db, semantic_settings) -> None:
        """缓存坏掉只能退化成「这次慢一点」,不能让用户问不了问题。"""
        from sqlalchemy import text

        cache = _semantic(db, semantic_settings, vectors={"显卡质保": _unit(1.0)})
        await cache.put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())

        async with db.session() as session:
            await session.execute(text("UPDATE cache_semantic SET value = X'DEADBEEF'"))
            await session.commit()

        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is None

    async def test_expired_entries_are_not_candidates(self, db, semantic_settings) -> None:
        """过期条目必须读作 miss,而不是「读作空值」。

        区别在于前者会回源、后者会让这一层返回一个空答案。
        """
        from datetime import timedelta

        from app.db.models import CacheSemanticEntry, utcnow

        cache = _semantic(db, semantic_settings, vectors={"显卡质保": _unit(1.0)})
        await cache.put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())

        async with db.session() as session:
            await session.execute(
                CacheSemanticEntry.__table__.update().values(
                    expires_at=utcnow() - timedelta(seconds=1)
                )
            )
            await session.commit()

        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is None

    async def test_purging_keeps_entries_that_are_still_alive(
        self, db, semantic_settings
    ) -> None:
        cache = _semantic(db, semantic_settings, vectors={"显卡质保": _unit(1.0)})
        await cache.put("显卡质保", history_messages=0, kb_ids=[1], answer=_answer())

        assert await cache.purge_expired() == 0
        assert await cache.get("显卡质保", history_messages=0, kb_ids=[1]) is not None
