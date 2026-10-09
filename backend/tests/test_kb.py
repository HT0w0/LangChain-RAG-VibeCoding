"""知识库管理与文档摄取的接口测试。

这些测试覆盖的是**整条摄取链路**:HTTP 上传 → 落盘 → 解析 → 分块 →
向量化 → SQLite + Chroma 双写 → 可被检索。任何一环断了,最后一组
「上传完真的检索得到吗」的断言就会挂——这正是它存在的意义。

embedding 由 `MockEmbedder` 提供(见 conftest),不联网、确定性。
"""

from __future__ import annotations

import io

import httpx
from fastapi.testclient import TestClient

from tests.helpers import admin_token, auth_header, embedding_entries, in_app_loop, register

KB = "/api/kb"


def upload(
    client: TestClient,
    kb_id: int,
    *,
    filename: str,
    content: bytes,
    content_type: str = "text/csv",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return client.post(
        f"{KB}/{kb_id}/documents",
        files={"files": (filename, io.BytesIO(content), content_type)},
        headers=headers if headers is not None else auth_header(admin_token(client)),
    )


def make_kb(client: TestClient, name: str = "商品库") -> dict:
    r = client.post(KB, json={"name": name, "description": "测试用"}, headers=auth_header(admin_token(client)))
    assert r.status_code == 201, r.text
    return r.json()


def vector_count(client: TestClient) -> int:
    """整个 collection 里的向量条数。"""
    from app.indexing.vectorstore import build_vector_store
    from app.main import settings as app_settings

    async def _count() -> int:
        return await build_vector_store(app_settings).count()

    return in_app_loop(client, _count)


def search_vectors(
    client: TestClient,
    query: str,
    *,
    kb_id: int | None = None,
    top_k: int = 5,
    where: dict | None = None,
) -> list:
    """直接在向量库里检索。用于验证「上传的东西真的能被检索到」。

    `where` 显式传入时覆盖默认的 `kb_id` 过滤,用于测试类目/价格预过滤。
    """
    from app.indexing.vectorstore import build_vector_store
    from app.main import settings as app_settings
    from app.providers.embedding import build_embedder

    if where is None and kb_id is not None:
        where = {"kb_id": kb_id}

    async def _search() -> list:
        embedder = build_embedder(app_settings)
        query_vec = await embedder.embed_query(query)
        return await build_vector_store(app_settings).query(
            embedding=query_vec, top_k=top_k, where=where
        )

    return in_app_loop(client, _search)


CSV_BYTES = "商品名称,类目,价格,库存\nRX9070,显卡,4299,12\nB650M,主板,899,5\n".encode()


# ---------------------------------------------------------------------------
# 权限:这是需求 6 的回归测试
# ---------------------------------------------------------------------------


class TestKBPermissions:
    def test_regular_user_cannot_list_kbs(self, client: TestClient) -> None:
        token = register(client, "kb-user-1")["access_token"]
        r = client.get(KB, headers=auth_header(token))
        assert r.status_code == 403
        assert r.json()["code"] == "forbidden"

    def test_regular_user_cannot_create_kb(self, client: TestClient) -> None:
        token = register(client, "kb-user-2")["access_token"]
        r = client.post(KB, json={"name": "偷偷建的"}, headers=auth_header(token))
        assert r.status_code == 403

    def test_regular_user_cannot_upload(self, client: TestClient) -> None:
        """普通用户直接 curl 上传也必须拿不到。前端藏起按钮不算数。"""
        kb = make_kb(client)
        token = register(client, "kb-user-3")["access_token"]
        r = upload(client, kb["id"], filename="x.csv", content=CSV_BYTES,
                   headers=auth_header(token))
        assert r.status_code == 403

    def test_regular_user_cannot_delete_kb(self, client: TestClient) -> None:
        kb = make_kb(client, "别删我")
        token = register(client, "kb-user-4")["access_token"]
        r = client.delete(f"{KB}/{kb['id']}", headers=auth_header(token))
        assert r.status_code == 403

    def test_anonymous_rejected_everywhere(self, client: TestClient) -> None:
        kb = make_kb(client, "匿名别碰")
        assert client.get(KB).status_code == 401
        assert client.post(KB, json={"name": "x"}).status_code == 401
        assert client.get(f"{KB}/{kb['id']}/documents").status_code == 401

    def test_options_are_visible_to_regular_users(self, client: TestClient) -> None:
        """问答页要选知识库范围,所以这个接口不能是管理员专属。"""
        kb = make_kb(client, "问答可选")
        token = register(client, "kb-user-5")["access_token"]
        r = client.get(f"{KB}/options", headers=auth_header(token))
        assert r.status_code == 200
        assert {"id": kb["id"], "name": "问答可选"} in r.json()["items"]

    def test_options_do_not_leak_admin_fields(self, client: TestClient) -> None:
        """普通用户不需要知道每个库有几篇文档、谁建的。"""
        make_kb(client, "字段检查")
        token = register(client, "kb-user-6")["access_token"]
        item = client.get(f"{KB}/options", headers=auth_header(token)).json()["items"][0]
        assert set(item) == {"id", "name"}

    def test_anonymous_cannot_read_options(self, client: TestClient) -> None:
        assert client.get(f"{KB}/options").status_code == 401


# ---------------------------------------------------------------------------
# 知识库 CRUD
# ---------------------------------------------------------------------------


class TestKBCrud:
    def test_create_and_get(self, client: TestClient) -> None:
        kb = make_kb(client, "笔记本")
        assert kb["doc_count"] == 0
        assert kb["chunk_count"] == 0
        assert kb["created_at"].endswith("+00:00")

        r = client.get(f"{KB}/{kb['id']}", headers=auth_header(admin_token(client)))
        assert r.status_code == 200
        assert r.json()["name"] == "笔记本"

    def test_duplicate_name_conflicts(self, client: TestClient) -> None:
        make_kb(client, "重名库")
        r = client.post(KB, json={"name": "重名库"}, headers=auth_header(admin_token(client)))
        assert r.status_code == 409
        assert r.json()["code"] == "conflict"

    def test_blank_name_rejected(self, client: TestClient) -> None:
        r = client.post(KB, json={"name": "   "}, headers=auth_header(admin_token(client)))
        assert r.status_code == 422

    def test_rename(self, client: TestClient) -> None:
        kb = make_kb(client, "旧名")
        headers = auth_header(admin_token(client))
        r = client.patch(f"{KB}/{kb['id']}", json={"name": "新名"}, headers=headers)
        assert r.status_code == 200
        assert r.json()["name"] == "新名"

        # 只改描述时不该把名字清空
        r = client.patch(f"{KB}/{kb['id']}", json={"description": "改了描述"}, headers=headers)
        assert r.json()["name"] == "新名"
        assert r.json()["description"] == "改了描述"

    def test_unknown_kb_404(self, client: TestClient) -> None:
        headers = auth_header(admin_token(client))
        assert client.get(f"{KB}/999999", headers=headers).status_code == 404
        assert client.patch(f"{KB}/999999", json={"name": "x"}, headers=headers).status_code == 404
        assert client.delete(f"{KB}/999999", headers=headers).status_code == 404


# ---------------------------------------------------------------------------
# 上传与摄取
# ---------------------------------------------------------------------------


class TestUpload:
    def test_csv_upload_is_ingested_and_chunked(self, client: TestClient) -> None:
        kb = make_kb(client, "显卡库")
        r = upload(client, kb["id"], filename="报价.csv", content=CSV_BYTES)
        assert r.status_code == 202, r.text
        docs = r.json()
        assert len(docs) == 1

        # 后台任务在 TestClient 里是同步跑完的,所以这里能看到终态
        detail = client.get(
            f"{KB}/{kb['id']}/documents/{docs[0]['id']}",
            headers=auth_header(admin_token(client)),
        ).json()
        assert detail["status"] == "ready", detail
        assert detail["chunk_count"] > 0
        assert detail["error"] is None

    def test_ingesting_fills_the_embedding_cache(self, client: TestClient) -> None:
        """**摄取路径也走向量缓存——这一侧省下的比问答侧多得多。**

        `get_rag_service` 装了 `CachedEmbedder` 而 `_build_ingest` 忘了装的话,
        问答侧一切正常、`test_chat_api` 那组也全绿,只有入库在白白重付编码。
        两条装配路径各有各的接线,所以各要有一条只可能由它点亮的用例。

        **断言的是「每个分块一条」而不是「大于零」。** 后者在「分块文本被
        当成一个整体去编码」或「键退化成一个常量」时照样成立,而那两种
        情况下向量和正文是错配的——检索会返回结果,只是返回错的。
        """
        kb = make_kb(client, "向量缓存库")
        r = upload(client, kb["id"], filename="报价.csv", content=CSV_BYTES)
        doc = r.json()[0]
        detail = client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}",
            headers=auth_header(admin_token(client)),
        ).json()

        # 先确认摄取成功。少了这一步,失败会表现成「缓存条数对不上」,
        # 而真正的原因(摄取报错)藏在 detail 里,得回头再查一遍。
        assert detail["status"] == "ready", detail

        # fixture 的两行商品正文互不相同,所以每个分块应当各占一条。
        assert detail["chunk_count"] == 2, f"用例的前提变了:分块数 {detail['chunk_count']}"
        assert embedding_entries(client) == detail["chunk_count"]

    def test_kb_counts_are_updated(self, client: TestClient) -> None:
        """冗余计数必须真的被维护——它漂移了列表页就会一直显示错数字。"""
        kb = make_kb(client, "计数库")
        upload(client, kb["id"], filename="a.csv", content=CSV_BYTES)

        after = client.get(f"{KB}/{kb['id']}", headers=auth_header(admin_token(client))).json()
        assert after["doc_count"] == 1
        assert after["chunk_count"] > 0

    def test_chunks_are_retrievable_from_vector_store(self, client: TestClient) -> None:
        """**端到端断言**:上传的内容真的进向量库了,而且检索得到。

        前面的断言都只证明「数据库里有记录」。只有这一条能证明
        「用户提问时能命中这份资料」——而后者才是产品的全部意义。
        """
        kb = make_kb(client, "可检索库")
        upload(client, kb["id"], filename="报价.csv", content=CSV_BYTES)

        hits = search_vectors(client, "显卡 价格", kb_id=kb["id"])
        assert hits, "上传后向量库里应该有可检索的内容"
        assert any("显卡" in h.content for h in hits)
        # 元数据必须带上过滤主键,否则多知识库隔离就是空话
        assert all(h.metadata.get("kb_id") == kb["id"] for h in hits)

    def test_metadata_supports_category_and_price_prefiltering(self, client: TestClient) -> None:
        """**类目/价格预过滤是产品承诺的功能,必须有数据喂给它。**

        `build_where` 和 ChromaStore 的过滤实现都是完整的,但整条摄取链路上
        曾经没有任何地方做表头归一化——于是元数据里只有 `source`/`row_start`,
        过滤条件**静默筛出空集**:接口不报错,只是永远没有结果。

        这里的 CSV 两行商品会被并进同一个分块,而分块元数据取的是块内
        **第一行**的类目,所以「显卡」筛得到、「主板」筛不到。这是
        `rows_per_chunk` 的已知取舍(见 `rows_to_sentences` 的说明),
        要按类目精确过滤就得把它设成 1。
        """
        from app.indexing.vectorstore import build_where

        kb = make_kb(client, "过滤元数据库")
        upload(client, kb["id"], filename="报价.csv", content=CSV_BYTES)

        metas = [h.metadata for h in search_vectors(client, "显卡", kb_id=kb["id"])]
        assert any(m.get("category") == "显卡" for m in metas), metas
        assert any(isinstance(m.get("price"), int) for m in metas), (
            "价格必须是数字:存成字符串的话 Chroma 的 $gte/$lte 筛不出东西"
        )

        filtered = search_vectors(
            client,
            "产品",
            where=build_where([kb["id"]], category="显卡", price_min=1000, price_max=5000),
        )
        assert filtered, "按类目 + 价格区间预过滤应当能筛出商品"

        none_match = search_vectors(
            client,
            "产品",
            where=build_where([kb["id"]], category="显示器"),
        )
        assert none_match == [], "过滤条件是服务端预过滤,不存在的类目就该返回空"

    def test_duplicate_content_is_skipped(self, client: TestClient) -> None:
        """同一知识库里重复上传同一份资料不该产生两份分块。

        重复入库会让同一条资料在检索结果里占两个 top-k 名额,
        引用卡片也会显示成两个不同来源。
        """
        kb = make_kb(client, "去重库")
        first = upload(client, kb["id"], filename="报价.csv", content=CSV_BYTES).json()
        assert len(first) == 1

        second = upload(client, kb["id"], filename="报价改个名.csv", content=CSV_BYTES).json()
        assert second == [], "内容相同应被跳过"

        detail = client.get(f"{KB}/{kb['id']}", headers=auth_header(admin_token(client))).json()
        assert detail["doc_count"] == 1

    def test_response_length_tells_you_how_many_were_skipped(self, client: TestClient) -> None:
        """返回值是「真正新建的文档数」,前端靠它算跳过了几个。

        全部重复时返回空列表是**正常结果**。如果这里改成抛错,管理员
        重传一次文件就会看到红色报错,而其实什么都没出错。
        """
        kb = make_kb(client, "混合批次库")
        upload(client, kb["id"], filename="已有.csv", content=CSV_BYTES)

        fresh = "# 新文档\n\n这是另一份资料。".encode()
        batch = client.post(
            f"{KB}/{kb['id']}/documents",
            files=[
                ("files", ("已有.csv", io.BytesIO(CSV_BYTES), "text/csv")),
                ("files", ("新的.md", io.BytesIO(fresh), "text/markdown")),
            ],
            headers=auth_header(admin_token(client)),
        )
        assert batch.status_code == 202
        assert len(batch.json()) == 1, "提交 2 个、其中 1 个重复 → 只应新建 1 个"
        assert batch.json()[0]["filename"] == "新的.md"

    def test_duplicate_does_not_leave_an_orphan_file_on_disk(self, client: TestClient) -> None:
        """被跳过的文件已经落盘过一次才做哈希,必须把那份临时文件删掉。

        不删的话,重复上传 100 次就在 uploads 里留 100 个没人引用的副本,
        而且永远不会有任何清理逻辑知道它们是垃圾。
        """
        kb = make_kb(client, "孤儿文件库")
        upload(client, kb["id"], filename="a.csv", content=CSV_BYTES)
        upload(client, kb["id"], filename="a.csv", content=CSV_BYTES)

        from app.main import settings as app_settings

        files = list(app_settings.resolved_upload_dir.rglob("*"))
        assert len([f for f in files if f.is_file()]) == 1

    def test_same_content_in_different_kbs_is_allowed(self, client: TestClient) -> None:
        """两个知识库各自引用同一份商品手册是合理的业务场景。"""
        kb1 = make_kb(client, "库一")
        kb2 = make_kb(client, "库二")
        assert len(upload(client, kb1["id"], filename="手册.csv", content=CSV_BYTES).json()) == 1
        assert len(upload(client, kb2["id"], filename="手册.csv", content=CSV_BYTES).json()) == 1

    def test_multiple_files_in_one_request(self, client: TestClient) -> None:
        kb = make_kb(client, "批量库")
        r = client.post(
            f"{KB}/{kb['id']}/documents",
            files=[
                ("files", ("a.csv", io.BytesIO(CSV_BYTES), "text/csv")),
                ("files", ("b.md", io.BytesIO("# 手册\n\n这是正文内容。".encode()), "text/markdown")),
            ],
            headers=auth_header(admin_token(client)),
        )
        assert r.status_code == 202
        assert len(r.json()) == 2

    def test_empty_file_rejected(self, client: TestClient) -> None:
        kb = make_kb(client, "空文件库")
        r = upload(client, kb["id"], filename="空.csv", content=b"")
        assert r.status_code == 400
        assert r.json()["code"] == "ingest_error"

    def test_unsupported_type_rejected_before_storing(self, client: TestClient) -> None:
        kb = make_kb(client, "格式库")
        r = upload(client, kb["id"], filename="木马.zip", content=b"PK\x03\x04")
        assert r.status_code == 400
        assert "不支持的文件类型" in r.json()["message"]

    def test_legacy_doc_gives_actionable_message(self, client: TestClient) -> None:
        """「不支持 .doc」是没用的提示;要告诉用户该做什么。"""
        kb = make_kb(client, "旧格式库")
        r = upload(client, kb["id"], filename="说明书.doc", content=b"\xd0\xcf\x11\xe0")
        assert r.status_code == 400
        assert "docx" in r.json()["message"]

    def test_scanned_pdf_fails_with_explanation(self, client: TestClient) -> None:
        """扫描件提取不出文字时必须**明确报错**,不能静默入库 0 个分块。"""
        from pypdf import PdfWriter

        buf = io.BytesIO()
        writer = PdfWriter()
        writer.add_blank_page(width=200, height=200)
        writer.write(buf)

        kb = make_kb(client, "扫描件库")
        docs = upload(client, kb["id"], filename="扫描.pdf", content=buf.getvalue(),
                      content_type="application/pdf").json()
        detail = client.get(
            f"{KB}/{kb['id']}/documents/{docs[0]['id']}",
            headers=auth_header(admin_token(client)),
        ).json()
        assert detail["status"] == "failed"
        assert "扫描" in detail["error"]

    def test_upload_to_unknown_kb_404(self, client: TestClient) -> None:
        r = upload(client, 999999, filename="a.csv", content=CSV_BYTES)
        assert r.status_code == 404


class TestDocumentLifecycle:
    def _ready_doc(self, client: TestClient, kb_name: str = "生命周期库") -> tuple[dict, dict]:
        kb = make_kb(client, kb_name)
        doc = upload(client, kb["id"], filename="报价.csv", content=CSV_BYTES).json()[0]
        return kb, doc

    def test_list_documents(self, client: TestClient) -> None:
        kb, doc = self._ready_doc(client)
        r = client.get(f"{KB}/{kb['id']}/documents", headers=auth_header(admin_token(client)))
        assert r.status_code == 200
        assert r.json()["total"] == 1
        assert r.json()["items"][0]["id"] == doc["id"]

    def test_list_documents_filtered_by_status(self, client: TestClient) -> None:
        kb, _ = self._ready_doc(client, "筛选库")
        headers = auth_header(admin_token(client))
        assert client.get(f"{KB}/{kb['id']}/documents?status=ready", headers=headers).json()["total"] == 1
        assert client.get(f"{KB}/{kb['id']}/documents?status=failed", headers=headers).json()["total"] == 0

    def test_chunk_preview_shows_split_result(self, client: TestClient) -> None:
        """管理员要能**看见**分块长什么样,否则没法判断切得好不好。"""
        kb, doc = self._ready_doc(client, "预览库")
        r = client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}/chunks",
            headers=auth_header(admin_token(client)),
        )
        assert r.status_code == 200
        body = r.json()
        assert body["total"] > 0
        assert body["items"][0]["content"], "分块内容不能为空"
        assert body["items"][0]["chunk_index"] == 0

    def test_chunk_preview_pagination(self, client: TestClient) -> None:
        kb = make_kb(client, "分页库")
        big = "商品名称,类目,价格\n" + "\n".join(
            f"型号{i},显卡,{1000 + i}" for i in range(40)
        )
        doc = upload(client, kb["id"], filename="大表.csv", content=big.encode()).json()[0]
        headers = auth_header(admin_token(client))

        page = client.get(f"{KB}/{kb['id']}/documents/{doc['id']}/chunks?limit=2&offset=0",
                          headers=headers).json()
        assert len(page["items"]) == 2
        assert page["total"] > 2

        page2 = client.get(f"{KB}/{kb['id']}/documents/{doc['id']}/chunks?limit=2&offset=2",
                           headers=headers).json()
        assert page2["items"][0]["chunk_index"] != page["items"][0]["chunk_index"]

    def test_delete_document_removes_chunks_and_vectors(self, client: TestClient) -> None:
        kb, doc = self._ready_doc(client, "删除库")
        headers = auth_header(admin_token(client))

        assert client.delete(f"{KB}/{kb['id']}/documents/{doc['id']}", headers=headers).status_code == 204
        assert client.get(f"{KB}/{kb['id']}/documents/{doc['id']}", headers=headers).status_code == 404

        kb_after = client.get(f"{KB}/{kb['id']}", headers=headers).json()
        assert kb_after["doc_count"] == 0
        assert kb_after["chunk_count"] == 0

        # 向量也必须清掉,否则检索会命中一个数据库里不存在的 chunk
        assert vector_count(client) == 0

    def test_delete_knowledge_base_cascades(self, client: TestClient) -> None:
        kb, doc = self._ready_doc(client, "级联库")
        headers = auth_header(admin_token(client))

        assert client.delete(f"{KB}/{kb['id']}", headers=headers).status_code == 204
        assert client.get(f"{KB}/{kb['id']}", headers=headers).status_code == 404
        # 文档随知识库一起没了
        assert client.get(f"{KB}/{kb['id']}/documents/{doc['id']}", headers=headers).status_code == 404
        assert vector_count(client) == 0

    def test_reindex_rebuilds_chunks(self, client: TestClient) -> None:
        kb, doc = self._ready_doc(client, "重建库")
        headers = auth_header(admin_token(client))

        before = client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}/chunks", headers=headers
        ).json()["total"]

        r = client.post(f"{KB}/{kb['id']}/documents/{doc['id']}/reindex", headers=headers)
        assert r.status_code == 202

        detail = client.get(f"{KB}/{kb['id']}/documents/{doc['id']}", headers=headers).json()
        assert detail["status"] == "ready"

        after = client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}/chunks", headers=headers
        ).json()["total"]
        assert after == before, "重建不该产生重复分块"

        # 计数也不该翻倍——这是冗余计数最容易漂移的地方
        kb_after = client.get(f"{KB}/{kb['id']}", headers=headers).json()
        assert kb_after["doc_count"] == 1

    def test_reindex_retries_a_failed_document(self, client: TestClient) -> None:
        kb = make_kb(client, "重试库")
        headers = auth_header(admin_token(client))
        bad = upload(client, kb["id"], filename="坏.csv", content=b"")  # 会被拒
        assert bad.status_code == 400

        # 换一个能成功的文档,确认 reindex 会把状态从 failed 拉回 ready
        from pypdf import PdfWriter

        buf = io.BytesIO()
        w = PdfWriter()
        w.add_blank_page(width=100, height=100)
        w.write(buf)
        doc = upload(client, kb["id"], filename="扫描.pdf", content=buf.getvalue(),
                     content_type="application/pdf").json()[0]
        assert client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}", headers=headers
        ).json()["status"] == "failed"

        # 覆盖磁盘上的文件内容 + 换掉文件名,模拟「用户改好了源文件」。
        # 解析器按扩展名分派,所以文件名必须一起改——只换内容的话,
        # 它还是会按 .pdf 去解析一份 CSV。
        #
        # `session()` 退出时只 close,**不会提交**。少写这一行 commit,
        # 改动就只活在这个 session 里,reindex 会重新读到旧的 .pdf 文件名。
        import pathlib

        async def _fix_file() -> None:
            from app.core.deps import database as deps_db
            from app.db.repositories.document import DocumentRepository

            async with deps_db.session() as s:
                fresh = await DocumentRepository(s).get_by_id(doc["id"])
                assert fresh is not None
                pathlib.Path(fresh.stored_path).write_text("商品名称,价格\n甲,10\n", encoding="utf-8")
                fresh.filename = "修好了.csv"
                await s.commit()

        client.portal.call(_fix_file)  # type: ignore[attr-defined]

        assert client.post(
            f"{KB}/{kb['id']}/documents/{doc['id']}/reindex", headers=headers
        ).status_code == 202
        assert client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}", headers=headers
        ).json()["status"] == "ready"


class TestMarkdownAndPdfIngestion:
    def test_markdown_breadcrumb_reaches_the_index(self, client: TestClient) -> None:
        """面包屑要一路走到入库内容里,不能在中途丢掉。"""
        kb = make_kb(client, "Markdown 库")
        md = (
            "# 商品手册\n\n## 售后政策\n\n### 退换货\n\n"
            "自签收之日起 7 天内可无理由退货,商品需保持完好。\n"
        ).encode()
        doc = upload(client, kb["id"], filename="手册.md", content=md,
                     content_type="text/markdown").json()[0]

        chunks = client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}/chunks",
            headers=auth_header(admin_token(client)),
        ).json()["items"]
        assert any("商品手册 › 售后政策 › 退换货" in c["content"] for c in chunks)

    def test_gbk_csv_is_decoded(self, client: TestClient) -> None:
        """中文 Excel 导出的 CSV 是 GBK 编码,按 UTF-8 读会得到乱码/报错。"""
        kb = make_kb(client, "GBK 库")
        content = "商品名称,类目,价格\n显卡甲,显卡,4299\n".encode("gbk")
        doc = upload(client, kb["id"], filename="gbk.csv", content=content).json()[0]

        chunks = client.get(
            f"{KB}/{kb['id']}/documents/{doc['id']}/chunks",
            headers=auth_header(admin_token(client)),
        ).json()["items"]
        assert any("显卡甲" in c["content"] for c in chunks), chunks
