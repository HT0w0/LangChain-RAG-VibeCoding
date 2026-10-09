"""检索链路的**真实**端到端测试:真 SQLite + 真 Chroma + 真仓储 JOIN。

`test_retrieval.py` 用的是假向量库和假分块源,验的是编排的控制流。但假组件
有个致命的盲区:**它们返回的形状是我自己写的**,而不是真实管线产出的形状。

这个坑在本项目已经踩过一次——分块器的单元测试喂的是「已经归一化好的字典,
价格是 int」,而真实解析出来的键是中文表头、价格是字符串,于是「按类目和
价格预过滤」这个功能在真实链路上**根本没有数据可用**,而所有单元测试全绿。

所以这里专门验三件假测试验不了的事:
  1. `ChunkRepository` 的 JOIN 真的能把 `documents.filename` 和
     `knowledge_bases.name` 带出来(引用卡片的来源名全靠它);
  2. 摄取写进 Chroma 的 `kb_id` 元数据真的能被 `build_where` 过滤到;
  3. 删除文档后,检索真的再也拿不到它的分块。

embedding 用 `MockEmbedder`(见 conftest),确定性、不联网。
"""

from __future__ import annotations

import io

from fastapi.testclient import TestClient

from tests.helpers import admin_token, auth_header

KB = "/api/kb"

CSV = (
    "商品名称,类目,价格,库存\n"
    "RX9070 显卡,显卡,4299,12\n"
    "星曜显卡,显卡,3599,5\n"
    "27寸显示器,显示器,1299,30\n"
).encode()


def _headers(client: TestClient) -> dict[str, str]:
    return auth_header(admin_token(client))


def make_kb(client: TestClient, name: str = "商品库") -> dict:
    r = client.post(KB, json={"name": name, "description": "检索集成测试"}, headers=_headers(client))
    assert r.status_code == 201, r.text
    return r.json()


def upload_csv(client: TestClient, kb_id: int, content: bytes = CSV, filename: str = "报价.csv") -> dict:
    r = client.post(
        f"{KB}/{kb_id}/documents",
        files={"files": (filename, io.BytesIO(content), "text/csv")},
        headers=_headers(client),
    )
    assert r.status_code == 202, r.text
    doc = r.json()[0]
    # 上传接口返回的是「刚建好」的记录(pending)——摄取是**后台任务**。
    # 在 TestClient 里后台任务会在响应之后同步跑完,所以再查一次才是终态。
    detail = client.get(f"{KB}/{kb_id}/documents/{doc['id']}", headers=_headers(client)).json()
    assert detail["status"] == "ready", detail
    return detail


def retrieve(client: TestClient, query: str, *, kb_ids, **kwargs):
    """在应用自己的事件循环里跑一次真实检索。

    数据库引擎与 Chroma 客户端都绑定在 TestClient 启动的那个循环上,
    另起 `asyncio.run` 会因为跨循环使用而炸——`client.portal` 是通往那个
    循环的入口。
    """
    import app.db.base as db_base
    import app.main as main_module
    from app.db.repositories.document import ChunkRepository
    from app.indexing.vectorstore import build_vector_store
    from app.providers.embedding import MockEmbedder
    from app.services.retrieval import RetrievalService

    settings = main_module.settings

    async def _run():
        async with db_base.database.session() as session:
            service = RetrievalService(
                settings=settings,
                embedder=MockEmbedder(dim=settings.EMBED_DIM),
                vector_store=build_vector_store(settings),
                chunk_source=ChunkRepository(session),
            )
            return await service.retrieve(query, kb_ids=kb_ids, **kwargs)

    return client.portal.call(_run)  # type: ignore[attr-defined]


def contents(client: TestClient, query: str, *, kb_ids, **kwargs) -> list[str]:
    return [c.content for c in retrieve(client, query, kb_ids=kb_ids, **kwargs).chunks]


class TestRealPipeline:
    def test_uploaded_product_is_retrievable_with_citation_info(self, client: TestClient) -> None:
        """**引用卡片的三个字段必须真的能取到。**

        前端要显示「来源:报价.csv(商品库)」,靠的就是 JOIN 出来的
        `documents.filename` 与 `knowledge_bases.name`。假分块源直接返回
        我拼好的字段,所以 JOIN 写错时假测试照样全绿——只有这条能发现。
        """
        kb = make_kb(client)
        upload_csv(client, kb["id"])

        result = retrieve(client, "显卡", kb_ids=[kb["id"]])
        assert result.chunks, "上传后应当检索得到"

        top = result.chunks[0]
        assert top.doc_name == "报价.csv"
        assert top.kb_name == "商品库"
        assert top.kb_id == kb["id"]
        assert "显卡" in top.content

    def test_exact_model_code_is_found_by_the_keyword_route(self, client: TestClient) -> None:
        """型号这种「精确匹配才管用」的查询,是关键词路存在的理由。"""
        kb = make_kb(client)
        upload_csv(client, kb["id"])

        result = retrieve(client, "RX9070", kb_ids=[kb["id"]])
        assert result.route_sizes["keyword"] >= 1, "关键词路应当召回型号"
        top = result.chunks[0]
        assert top.keyword_rank is not None, "这条必须有关键词路的贡献"
        assert "RX9070" in top.content

    def test_structured_metadata_survives_ingestion(self, client: TestClient) -> None:
        """**回归:结构化的类目/价格必须真的落进元数据。**

        曾经的 bug 是分块器没有调用 `normalize_columns`,于是元数据里只有
        `source`/`row_start`,类目和价格**根本没有值**——而过滤功能就建立在
        它们上面,等于形同虚设。单元测试当时喂的是手工归一化好的字典,
        所以全绿。这条走真实 CSV 解析,能把它钉死。
        """
        kb = make_kb(client)
        upload_csv(client, kb["id"])

        top = retrieve(client, "RX9070", kb_ids=[kb["id"]]).chunks[0]
        assert top.meta.get("category") == "显卡", f"类目没落进元数据:{top.meta}"
        assert top.meta.get("price") == 4299, f"价格没落进元数据:{top.meta}"

    def test_each_row_is_its_own_chunk_with_its_own_metadata(self, client: TestClient) -> None:
        """**回归:一份混合类目的商品表,按类目过滤必须精确到行。**

        这里钉的是一个真实的、静默的缺陷:三行合块时,块的 category 只取
        块内**第一行**的值。于是一份「显卡 / 显卡 / 显示器」的表里,
        按「只看显示器」过滤会**漏掉**显示器那一行——过滤不报错,只是
        少返回一条,而少返回的东西没人会发现。

        类目/价格预过滤是本项目点名的卖点,所以默认改成了一行一个分块
        (见 `rows_to_sentences`)。这条测试走**真实** CSV 解析 + 真实 Chroma
        预过滤,是那个默认值唯一有意义的验证方式。
        """
        kb = make_kb(client)
        upload_csv(client, kb["id"])  # 两行显卡 + 一行显示器

        from app.services.retrieval import RetrievalFilters

        everything = retrieve(client, "显示器", kb_ids=[kb["id"]])
        assert len(everything.chunks) == 3, f"三行应当是三个分块:{[c.content for c in everything.chunks]}"

        by_category = retrieve(
            client, "显示器", kb_ids=[kb["id"]], filters=RetrievalFilters(category="显示器")
        )
        assert [c.meta["category"] for c in by_category.chunks] == ["显示器"], (
            "按显示器过滤却混进了别的类目,或者漏掉了显示器那一行"
        )

        by_price = retrieve(
            client, "显卡", kb_ids=[kb["id"]], filters=RetrievalFilters(price_min=4000)
        )
        assert by_price.chunks, "4299 那行应当被召回"
        assert all(c.meta["price"] >= 4000 for c in by_price.chunks), (
            f"价格过滤没挡住:{[c.meta.get('price') for c in by_price.chunks]}"
        )

    def test_category_prefilter_works_end_to_end(self, client: TestClient) -> None:
        """带类目过滤的检索要能真的把别的类目挡在外面——向量路走 Chroma 的
        服务端预过滤,关键词路走建索引前的语料筛选,两边都得生效。"""
        kb = make_kb(client)
        upload_csv(client, kb["id"], content="商品名称,类目,价格,库存\n27寸显示器,显示器,1299,30\n".encode())

        from app.services.retrieval import RetrievalFilters

        result = retrieve(
            client, "显示器", kb_ids=[kb["id"]], filters=RetrievalFilters(category="显示器")
        )
        assert result.chunks, "显示器那条应当被召回"
        assert all(c.category == "显示器" for c in result.chunks)

        # 同类目过滤到别的类目上应当一条都不剩——证明过滤真的在起作用,
        # 而不是「反正查什么都返回同一批」。
        none = retrieve(
            client, "显示器", kb_ids=[kb["id"]], filters=RetrievalFilters(category="显卡")
        )
        assert none.chunks == []

    def test_knowledge_bases_are_isolated(self, client: TestClient) -> None:
        """两个库内容相似时,限定范围必须只返回该库的分块。

        这是权限的最后一道验证:向量路靠 Chroma 的 `where`,关键词路靠
        `list_records_by_kb`——两边的 id 都来自真实的摄取,不是我编的。
        """
        kb_a = make_kb(client, "显卡库")
        kb_b = make_kb(client, "显示器库")
        upload_csv(client, kb_a["id"], filename="显卡报价.csv")
        upload_csv(
            client,
            kb_b["id"],
            content="商品名称,类目,价格,库存\n显示器支架,显示器,199,50\n".encode(),
            filename="显示器报价.csv",
        )

        scoped = retrieve(client, "显卡", kb_ids=[kb_a["id"]])
        assert scoped.chunks
        assert {c.kb_id for c in scoped.chunks} == {kb_a["id"]}
        assert {c.doc_name for c in scoped.chunks} == {"显卡报价.csv"}

    def test_deleted_document_disappears_from_retrieval(self, client: TestClient) -> None:
        """删掉文档后必须检索不到——否则引用会指向已经不存在的内容。

        这条同时覆盖了「向量库清干净没有」:若向量还在、SQLite 记录已删,
        融合会把一个查不到的 chunk_id 带进来;检索层应当跳过它
        (见 `_hydrate`),而不是抛异常或返回空白引用。
        """
        kb = make_kb(client)
        doc = upload_csv(client, kb["id"])
        assert contents(client, "RX9070", kb_ids=[kb["id"]])

        r = client.delete(f"{KB}/{kb['id']}/documents/{doc['id']}", headers=_headers(client))
        assert r.status_code in (200, 204), r.text

        assert contents(client, "RX9070", kb_ids=[kb["id"]]) == []

    def test_empty_knowledge_base_returns_nothing(self, client: TestClient) -> None:
        """空知识库是**正常的初始状态**,不该让检索炸掉。

        Chroma 在 collection 为空时 `query` 会抛异常,而用户完全可能先建库
        再上传。所以空库必须安静地返回空结果。
        """
        kb = make_kb(client, "空库")
        result = retrieve(client, "显卡", kb_ids=[kb["id"]])
        assert result.chunks == []
        assert result.route_sizes["vector"] == 0


class TestScopeIsolation:
    def test_empty_scope_never_reads_anything(self, client: TestClient) -> None:
        """空范围在真实组件上也要短路——**绝不能**退化成「查全部」。"""
        kb = make_kb(client, "有内容的库")
        upload_csv(client, kb["id"])

        result = retrieve(client, "显卡", kb_ids=[])
        assert result.chunks == []
        assert result.route_sizes["vector"] == 0
