"""问答接口(SSE 流式)的端到端测试:真 SQLite + 真 Chroma + 真仓储。

`test_rag.py` 验的是编排逻辑(假检索、假模型)。这里验的是**接上去之后**
还成不成立的三件事:

1. 协议真的能被客户端解析——`event:` / `data:` 的分帧、事件顺序、
   `done` 里的答案与流出去的 token 拼得回来;
2. 引用卡片要的字段走完真实管线后**仍然在**——JOIN 出来的文件名、
   知识库名、正文原文。假检索直接返回我拼好的字段,这条它验不了;
3. **检索期的异常必须变成带正确状态码的 HTTP 响应,而不是一个 200 开头的
   断流。** 这取决于「检索在响应头发出之前跑完」这个约束有没有被破坏,
   而单元测试完全看不见它。

模型用 `MockLLM`(见 conftest 的 `LLM_BACKEND=mock`),不联网、确定性。
"""

from __future__ import annotations

import io
import json

import pytest
from app.core.exceptions import ProviderError
from app.services.rag import RAGService
from fastapi.testclient import TestClient

from tests.helpers import (
    admin_token,
    auth_header,
    embedding_entries,
    in_app_loop,
    latest_metric,
    register,
    semantic_entries,
)

CHAT = "/api/chat/stream"
KB = "/api/kb"

CSV = (
    "商品名称,类目,价格,库存\n"
    "RX9070 显卡,显卡,4299,12\n"
    "27寸显示器,显示器,1299,30\n"
).encode()


# ---------------------------------------------------------------------------
# SSE 解析
# ---------------------------------------------------------------------------


def parse_sse(body: str) -> list[tuple[str, dict]]:
    """把原始 SSE 报文解析成 [(event_name, payload)]。

    刻意写得宽容(冒号后可有可无空格、忽略 `:` 开头的注释行如心跳),
    因为它模拟的是**浏览器**要面对的东西,而不是我们自己的编码器。
    """
    events: list[tuple[str, dict]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name, data = None, None
        for line in block.split("\n"):
            if line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            value = value.lstrip(" ")
            if key == "event":
                name = value
            elif key == "data":
                data = value
        if name and data is not None:
            events.append((name, json.loads(data)))
    return events


def names_of(events: list[tuple[str, dict]]) -> list[str]:
    return [n for n, _ in events]


def tokens_of(events: list[tuple[str, dict]]) -> str:
    return "".join(p["text"] for n, p in events if n == "token")


def first(events: list[tuple[str, dict]], name: str) -> dict:
    return next(p for n, p in events if n == name)


# ---------------------------------------------------------------------------
# 夹具
# ---------------------------------------------------------------------------


def headers(client: TestClient) -> dict[str, str]:
    return auth_header(admin_token(client))


def make_kb(client: TestClient, name: str = "商品库") -> dict:
    r = client.post(KB, json={"name": name, "description": "问答接口测试"}, headers=headers(client))
    assert r.status_code == 201, r.text
    return r.json()


def upload_csv(client: TestClient, kb_id: int, content: bytes = CSV, filename: str = "报价.csv") -> dict:
    r = client.post(
        f"{KB}/{kb_id}/documents",
        files={"files": (filename, io.BytesIO(content), "text/csv")},
        headers=headers(client),
    )
    assert r.status_code == 202, r.text
    created = r.json()
    # 接口返回的是「本次真正新建的文档」。传一份内容重复的文件是合法操作,
    # 它会得到 202 + 空列表——这个辅助函数假设了「上传必有产出」,重复上传
    # 正是该假设的反例。用断言说明白,比 `r.json()[0]` 抛 IndexError 强:
    # 后者看不出是「接口行为符合预期」还是「接口挂了」。
    assert created, "上传未被去重?接口返回了空列表,见 kb.upload_documents 的文档"
    doc = created[0]
    detail = client.get(f"{KB}/{kb_id}/documents/{doc['id']}", headers=headers(client)).json()
    assert detail["status"] == "ready", detail
    return detail


def ask(client: TestClient, question: str, **payload) -> list[tuple[str, dict]]:
    r = client.post(CHAT, json={"question": question, **payload}, headers=headers(client))
    assert r.status_code == 200, r.text
    return parse_sse(r.text)


@pytest.fixture
def kb_with_products(client: TestClient) -> dict:
    kb = make_kb(client)
    upload_csv(client, kb["id"])
    return kb


@pytest.fixture
def empty_kb(client: TestClient) -> dict:
    """一个**没有任何文档**的知识库。

    它是「检索结果确实为空」的唯一可靠构造:假 embedding 会给任何问题都
    算出一点相似度,替身重排又不做阈值截断,所以只要有文档在,任何提问都
    召得回东西。见 `TestRetrievalCacheIsActuallyWired` 的说明。
    """
    return make_kb(client, name="空库")


# ---------------------------------------------------------------------------
# 协议形状
# ---------------------------------------------------------------------------


class TestProtocol:
    def test_content_type_is_sse(self, client: TestClient) -> None:
        r = client.post(CHAT, json={"question": "你好"}, headers=headers(client))
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")

    def test_event_sequence(self, client: TestClient, kb_with_products: dict) -> None:
        events = ask(client, "RX9070 多少钱", kb_ids=[kb_with_products["id"]])
        assert names_of(events)[0] == "sources"
        assert names_of(events)[-1] == "done"
        assert set(names_of(events)) <= {"sources", "token", "done"}

    def test_sources_event_arrives_before_any_token(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        events = ask(client, "RX9070 多少钱", kb_ids=[kb_with_products["id"]])
        assert names_of(events).index("sources") < names_of(events).index("token")

    def test_tokens_reassemble_into_the_done_answer(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        """断流时前端要靠 done.answer 把状态拼回一致,两者必须一模一样。"""
        events = ask(client, "RX9070 多少钱", kb_ids=[kb_with_products["id"]])
        assert tokens_of(events)
        assert tokens_of(events) == first(events, "done")["answer"]

    def test_done_reports_stage_timings_and_grounded(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        done = first(ask(client, "RX9070 多少钱", kb_ids=[kb_with_products["id"]]), "done")
        assert done["grounded"] is True
        assert "first_token" in done["stage_timings"]
        assert "total" in done["stage_timings"]


# ---------------------------------------------------------------------------
# 引用溯源(走真实管线)
# ---------------------------------------------------------------------------


class TestCitations:
    def test_sources_carry_everything_the_source_card_needs(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        """**引用卡片能不能渲染,全看这四个字段。**

        `doc_name` 和 `kb_name` 来自 `chunks JOIN documents JOIN knowledge_bases`,
        假检索直接返回我拼好的字段,所以 JOIN 写错时它照样全绿。
        `content` 是卡片展开后的原文——只给摘要的话,用户看不到支撑结论的
        那半句话。
        """
        sources = first(
            ask(client, "RX9070", kb_ids=[kb_with_products["id"]]), "sources"
        )["sources"]
        assert sources

        top = sources[0]
        assert top["index"] == 1
        assert top["doc_name"] == "报价.csv"
        assert top["kb_name"] == "商品库"
        assert "RX9070" in top["content"]
        assert isinstance(top["chunk_id"], int)

    def test_carries_structured_metadata_for_prefiltering(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        sources = first(
            ask(client, "RX9070", kb_ids=[kb_with_products["id"]]), "sources"
        )["sources"]
        top = next(s for s in sources if s["category"] == "显卡")
        assert top["price"] == 4299

    def test_answer_cites_real_source_indices(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        """端到端验一次「正文里的 [1] 指向 sources[0]」。"""
        events = ask(client, "RX9070 多少钱", kb_ids=[kb_with_products["id"]])
        sources = first(events, "sources")["sources"]
        done = first(events, "done")
        assert done["cited"], "答案里应当出现引用编号"
        assert all(1 <= i <= len(sources) for i in done["cited"])

    def test_duplicate_documents_do_not_shift_citation_numbers(
        self, client: TestClient
    ) -> None:
        """同一份 CSV 传两次会被去重(库内唯一约束),不得出现两遍来源。"""
        kb = make_kb(client)
        upload_csv(client, kb["id"])

        # 第二次**故意**不走 `upload_csv`:要断言的正是它的返回值。
        # 「重复内容被跳过」是接口契约的一部分(见 `kb.upload_documents`),
        # 直接钉住它,而不是等下一句从「来源里没有重复」反推——反推的话,
        # 哪天去重坏了但检索恰好没把两份都召回,这条测试照样绿。
        dup = client.post(
            f"{KB}/{kb['id']}/documents",
            files={"files": ("报价.csv", io.BytesIO(CSV), "text/csv")},
            headers=headers(client),
        )
        assert dup.status_code == 202, dup.text
        assert dup.json() == [], "内容重复的文件应当被静默跳过并返回空列表"

        sources = first(ask(client, "RX9070", kb_ids=[kb["id"]]), "sources")["sources"]
        contents = [s["content"] for s in sources]
        assert len(contents) == len(set(contents)), "同一段正文出现了两次"
        assert [s["index"] for s in sources] == list(range(1, len(sources) + 1))


# ---------------------------------------------------------------------------
# 范围与过滤
# ---------------------------------------------------------------------------


class TestScope:
    def test_knowledge_base_isolation(self, client: TestClient) -> None:
        kb_a = make_kb(client, "显卡库")
        kb_b = make_kb(client, "显示器库")
        upload_csv(client, kb_a["id"], filename="显卡报价.csv")
        upload_csv(
            client,
            kb_b["id"],
            content="商品名称,类目,价格,库存\n显示器支架,显示器,199,50\n".encode(),
            filename="显示器报价.csv",
        )

        sources = first(ask(client, "显卡", kb_ids=[kb_a["id"]]), "sources")["sources"]
        assert {s["doc_name"] for s in sources} == {"显卡报价.csv"}

    def test_empty_scope_never_reads_anything(self, client: TestClient, kb_with_products: dict) -> None:
        """**空范围必须退化成「没有资料」,绝不能退化成「查全部」。**

        这条是越权读的最后一道验证:`kb_ids=[]` 若被当成 falsy 跳过过滤,
        一个没有任何知识库权限的用户会拿到全部内容——不报错、不留日志。
        """
        events = ask(client, "RX9070 多少钱", kb_ids=[])
        assert first(events, "sources")["sources"] == []
        assert first(events, "done")["grounded"] is False

    def test_category_filter_reaches_the_pipeline(self, client: TestClient, kb_with_products: dict) -> None:
        events = ask(
            client,
            "多少钱",
            kb_ids=[kb_with_products["id"]],
            filters={"category": "显示器"},
        )
        sources = first(events, "sources")["sources"]
        assert sources
        assert {s["category"] for s in sources} == {"显示器"}

    def test_no_hits_still_produces_a_well_formed_stream(self, client: TestClient) -> None:
        """空知识库是**正常的初始状态**:事件形状要和有结果时完全一致,
        前端不该为它写特殊分支。"""
        kb = make_kb(client, "空库")
        events = ask(client, "显卡", kb_ids=[kb["id"]])
        assert names_of(events) == ["sources", "token", "done"]
        assert first(events, "sources")["sources"] == []
        assert first(events, "done")["grounded"] is False
        assert "没有找到" in first(events, "done")["answer"]


# ---------------------------------------------------------------------------
# 鉴权与入参
# ---------------------------------------------------------------------------


class TestAuth:
    def test_anonymous_is_rejected(self, client: TestClient) -> None:
        r = client.post(CHAT, json={"question": "你好"})
        assert r.status_code == 401

    def test_regular_user_may_ask(self, client: TestClient, kb_with_products: dict) -> None:
        """需求 6 只限制**知识库管理**。普通用户必须能问答,否则整个产品
        对非管理员就没有任何用处。"""
        token = register(client, "asker")["access_token"]
        r = client.post(
            CHAT,
            json={"question": "RX9070 多少钱", "kb_ids": [kb_with_products["id"]]},
            headers=auth_header(token),
        )
        assert r.status_code == 200, r.text
        assert first(parse_sse(r.text), "done")["grounded"] is True

    def test_regular_user_cannot_manage_knowledge_bases(self, client: TestClient) -> None:
        """与上一条对照:同一个普通用户撞管理接口必须 403。"""
        token = register(client, "asker2")["access_token"]
        r = client.post(KB, json={"name": "偷偷建的库"}, headers=auth_header(token))
        assert r.status_code == 403


class TestValidation:
    @pytest.mark.parametrize("bad", ["", "   ", "\n\t "])
    def test_blank_question_is_rejected(self, client: TestClient, bad: str) -> None:
        """空问题会白白走完整条检索链路,然后让模型对着一段空话作答。"""
        r = client.post(CHAT, json={"question": bad}, headers=headers(client))
        assert r.status_code == 422

    def test_overlong_question_is_rejected(self, client: TestClient) -> None:
        r = client.post(CHAT, json={"question": "问" * 3000}, headers=headers(client))
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 故障路径
# ---------------------------------------------------------------------------


class TestFailureBeforeStreaming:
    """**检索期的异常必须变成带状态码的 HTTP 响应,而不是 200 + 断流。**

    这取决于「检索在 `EventSourceResponse` 构造之前跑完」这条约束。一旦
    有人把检索挪进生成器里(看起来更「流式」),响应头就会先发出去——
    之后无论抛什么都只能是 200,客户端拿到一个空连接,永远不知道
    自己少了什么。
    """

    def test_retrieval_failure_returns_a_real_http_error(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class ExplodingRetriever:
            async def retrieve(self, query, **kwargs):
                raise ProviderError("向量服务不可用", provider="fake", retryable=True)

        monkeypatch.setitem(
            client.app.dependency_overrides,  # type: ignore[attr-defined]
            _rag_dependency(),
            lambda: RAGService(
                settings=_settings_of(client), retrieval=ExplodingRetriever(), llm=_llm_of(client)
            ),
        )

        r = client.post(CHAT, json={"question": "随便问问"}, headers=headers(client))
        assert r.status_code == 502, r.text
        assert r.json()["code"] == "provider_error"

    def test_retrieval_failure_does_not_start_an_sse_stream(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """不只是状态码——正文里绝不能出现半截事件流。"""

        class ExplodingRetriever:
            async def retrieve(self, query, **kwargs):
                raise ProviderError("向量服务不可用", provider="fake")

        monkeypatch.setitem(
            client.app.dependency_overrides,  # type: ignore[attr-defined]
            _rag_dependency(),
            lambda: RAGService(
                settings=_settings_of(client), retrieval=ExplodingRetriever(), llm=_llm_of(client)
            ),
        )

        r = client.post(CHAT, json={"question": "随便问问"}, headers=headers(client))
        assert not r.headers["content-type"].startswith("text/event-stream")
        assert "event:" not in r.text


class _OneChunkRetriever:
    """只回放一个固定分块。

    「检索成功、生成失败」这类用例里,检索本身的质量是无关变量——真去接
    一整套 `RetrievalService` 只会让失败原因变模糊(到底是编排炸了,还是
    向量库炸了)。检索的质量由 `test_retrieval*.py` 负责。
    """

    async def retrieve(self, query, **kwargs):
        from app.services.context import RetrievedChunk
        from app.services.retrieval import RetrievalResult

        return RetrievalResult(
            query=query,
            chunks=[
                RetrievedChunk(
                    chunk_id=1,
                    content="RX9070 显卡质保三年",
                    kb_id=1,
                    doc_id=1,
                    doc_name="报价.csv",
                    kb_name="商品库",
                    chunk_index=0,
                    meta={},
                )
            ],
            route_sizes={"vector": 1},
            stage_timings={"total": 1.0},
        )


def _rag_dependency():
    from app.api.routers.chat import get_rag_service

    return get_rag_service


def _settings_of(client: TestClient):
    from app.core.config import Settings

    settings = Settings(_env_file=None, DASHSCOPE_API_KEY="sk-test")
    # 依赖覆盖后路由不再自己构造设置,这里给一份能用的即可
    return settings


def _llm_of(client: TestClient):
    from app.providers.llm import MockLLM

    return MockLLM(_settings_of(client), delay=0)


class TestRetrievalCacheIsActuallyWired:
    """**「缓存写好了」和「缓存真的在用」是两件事。**

    `test_cache.py` 里那一整套用例验的是缓存层自身的行为,它们全绿也完全
    可以同时发生「路由根本没把 `CachedRetriever` 装上去」——那种情况下
    缓存一行都不会跑到,而所有单元测试照样通过。

    **注意这里问的是「知识库答不上来」的问题。** 有答案的问题问第二遍会被
    上面那层答案缓存整个截走,检索一步都不跑——那样测出来的是答案缓存,
    而 `CachedRetriever` 拆掉了照样绿。检索为空时答案缓存不参与(见
    `RAGService._remember`),于是第二次提问必然走到检索,`cache_hit` 就
    只可能来自检索缓存。**每一层都要有一条只可能由它点亮的路。**
    """

    def test_an_unanswerable_question_hits_the_retrieval_cache(
        self, client: TestClient, empty_kb: dict
    ) -> None:
        """**空知识库是拿到「检索确实为空」的唯一可靠办法。**

        起初这里用的是「知识库里没有的品类」,结果不成立:mock embedding 是
        字符 n-gram 哈希,任何问题都能和现有分块算出一点相似度,而
        `RERANK_BACKEND=none` 的替身重排**不做阈值截断**——于是无论问什么
        都有召回,`grounded` 恒为真。这种用例不会红,它只是**测不到**它宣称
        要测的东西。
        """
        payload = {"kb_ids": [empty_kb["id"]]}

        first_done = first(ask(client, "洗衣机烘干要多久", **payload), "done")
        second_done = first(ask(client, "洗衣机烘干要多久", **payload), "done")

        assert first_done["grounded"] is False, "空库居然召回了内容,用例的前提不成立"
        assert first_done["cache_hit"] is False
        assert second_done["cache_hit"] is True, "第二次同样的问题没有命中检索缓存"
        assert second_done["answer_cache_hit"] is False, (
            "检索为空的答案不该进答案缓存,否则这条用例就测不到检索缓存了"
        )

    def test_a_cached_retrieval_is_still_honest_about_being_empty(
        self, client: TestClient, empty_kb: dict
    ) -> None:
        """**缓存一次「什么都没找到」,不能变成「找到过什么」。**

        空结果也要能被缓存(否则知识库答不上来的问题每次都去锤向量库),
        但命中之后必须是**同样的空**:来源列表为空、`grounded` 为假、答案
        还是那句固定答复。缓存把一个空结果还原成非空,是最难被发现的一类
        故障——答案读起来完全正常,只是它凭空多出了资料。
        """
        payload = {"kb_ids": [empty_kb["id"]]}

        first_events = ask(client, "洗衣机烘干要多久", **payload)
        second_events = ask(client, "洗衣机烘干要多久", **payload)

        first_done = first(first_events, "done")
        second_done = first(second_events, "done")

        assert second_done["cache_hit"] is True
        assert second_done["grounded"] is False
        assert second_done["answer"] == first_done["answer"]
        assert first(second_events, "sources")["sources"] == []


class TestAnswerCacheIsActuallyWired:
    """答案缓存走**真实 HTTP 接口**的验收。

    `test_rag.py` 里那一组用的是假 LLM 和假检索器,它们能证明「编排对了」,
    证不了「路由把它装上了」——`get_rag_service` 里漏传 `answers=` 的话,
    那一组会全绿,而线上缓存一次都不会命中。

    **证据是 `usage`。** 命中时这一轮根本没有调用模型,`usage` 是 `None`
    (`test_a_hit_reports_no_usage` 钉的就是这个);而没命中时 MockLLM 会
    给出确定的用量。这比断言「快了多少」可靠得多——耗时在 CI 上会飘,
    而「用没用量」是二值的。
    """

    def test_the_second_identical_question_never_calls_the_model(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        payload = {"kb_ids": [kb_with_products["id"]]}

        first_done = first(ask(client, "RX9070 多少钱", **payload), "done")
        second_done = first(ask(client, "RX9070 多少钱", **payload), "done")

        assert first_done["usage"] is not None, "第一次就没拿到用量,用例的前提不成立"
        assert first_done["answer_cache_hit"] is False
        assert second_done["answer_cache_hit"] is True, "第二次同样的提问没有命中答案缓存"
        assert second_done["usage"] is None, "命中了缓存却报了用量,说明模型又被调了一次"

    def test_a_cached_answer_is_still_grounded_and_cited(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        """**命中缓存不能改变答案,只能改变它有多快。**

        缓存最坏的失效方式不是慢,而是「命中之后少了点东西」:来源列表空了、
        引用编号对不上了、`grounded` 变成 False 于是走了固定答复。这些在
        「缓存命中率 100%」的指标上完全看不出来——它只会显示一切正常。
        """
        payload = {"kb_ids": [kb_with_products["id"]]}

        first_events = ask(client, "RX9070 多少钱", **payload)
        second_events = ask(client, "RX9070 多少钱", **payload)

        first_done = first(first_events, "done")
        second_done = first(second_events, "done")

        assert second_done["answer_cache_hit"] is True
        assert second_done["grounded"] == first_done["grounded"] is True
        assert second_done["answer"] == first_done["answer"]
        assert second_done["cited"] == first_done["cited"]
        assert first(second_events, "sources")["sources"] == first(first_events, "sources")["sources"]

    def test_a_question_with_history_is_not_served_from_the_answer_cache(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        """带上会话之后,一字不差的同一个问题也必须重新生成。

        这正是答案缓存唯一挡不住的那类错:它的键里没有历史,所以「带上下文
        的同一句话」和「孤零零的同一句话」在它眼里完全一样。唯一的正确做法
        是让它在有历史时整个不参与——这条用例从 HTTP 这一侧确认那个开关
        真的生效。
        """
        payload = {"kb_ids": [kb_with_products["id"]]}
        session_id = client.post(
            "/api/sessions", json={"title": "追问"}, headers=headers(client)
        ).json()["id"]

        # 先建一条历史,再问那句已经进了缓存的、一字不差的问题。
        ask(client, "RX9070 多少钱", session_id=session_id, **payload)
        done = first(
            ask(client, "RX9070 多少钱", session_id=session_id, **payload), "done"
        )

        assert done["history_messages"] > 0, "会话里没有历史,用例的前提不成立"
        assert done["answer_cache_hit"] is False, "带历史的提问被答案缓存截走了"
        assert done["usage"] is not None, "带历史的提问没有重新生成"

    def test_switching_the_cache_off_stops_it_from_being_used(
        self, client: TestClient, kb_with_products: dict, settings, monkeypatch
    ) -> None:
        """缓存必须能**整体关掉**。

        排障第一步就是确认「这个问题是不是缓存带来的」;关不掉的话,只能靠
        读代码推断,而推断出来的结论没法当证据。这里钉的是那个开关真的接在
        生产装配的那条路上(`get_rag_service` 里透传 `CACHE_ENABLED`),
        而不是一个没人读的配置项。

        **先让它留住一条缓存、再关掉**,而不是一上来就关:后者可能只是
        「什么都没写进去」,和被正确禁用长得一模一样。
        """
        payload = {"kb_ids": [kb_with_products["id"]]}
        assert first(ask(client, "RX9070 多少钱", **payload), "done")["answer_cache_hit"] is False
        assert first(ask(client, "RX9070 多少钱", **payload), "done")["answer_cache_hit"] is True

        monkeypatch.setattr(settings, "CACHE_ENABLED", False)

        done = first(ask(client, "RX9070 多少钱", **payload), "done")
        assert done["answer_cache_hit"] is False, "关掉开关之后缓存还在被读"
        assert done["usage"] is not None, "关掉开关之后模型没被重新调用"


# ---------------------------------------------------------------------------
# 向量缓存的接线
# ---------------------------------------------------------------------------


class TestEmbeddingCacheIsActuallyWired:
    """**「向量缓存写好了」和「路由把它装上了」是两件事。**

    `test_cache.py::TestEmbeddingCache` 那一组验的是包装器自身的行为。它们
    全绿的同时,`get_rag_service` 里漏掉 `CachedEmbedder` 是完全可能的——
    那时每次提问都重新编码一遍,而单元测试一条都不会红。

    **证据是缓存表里的行数。** `embedding` 这个域只有这一层会写,所以
    「一条都没有」是二值且无歧义的:没接上。它也不像耗时那样在 CI 上会飘。

    **问的是空知识库。** 有文档时第二次提问会被答案缓存整个截走,而这里
    要确认的恰恰是「编码真的发生过」,得让提问每次都走到检索。
    """

    def test_asking_encodes_the_query_into_the_embedding_cache(
        self, client: TestClient, empty_kb: dict
    ) -> None:
        assert embedding_entries(client) == 0, "用例开始前缓存就不是空的"

        ask(client, "洗衣机烘干要多久", kb_ids=[empty_kb["id"]])

        # 恰好一条:问答路径只编码查询,文档侧由摄取路径负责。
        # 写成 `>= 1` 会漏掉「每次提问都新写一条」——那说明读路径没生效,
        # 而它每次仍然会产出至少一行。
        assert embedding_entries(client) == 1, "查询向量没有进向量缓存"

    def test_every_distinct_question_gets_its_own_entry(
        self, client: TestClient, empty_kb: dict
    ) -> None:
        """键必须是**逐条文本**算出来的,不能退化成一个常量。

        退化成常量时,行数恒为 1,上面的用例照样绿——而每条查询都会命中
        另一条查询的向量,检索结果整体变成另一句话的。这是「缓存给错答案」
        在向量层的形态:不报错,只是召回的片段和问题无关。

        **这里测的仍然只是写入侧。** 读路径要证「确实省下了一次编码」,需要
        数得清内层被调用了几次,而那只能靠 `test_cache.py` 里那个计数替身
        (`TestEmbeddingCache`)——从 HTTP 这一侧看不见,因为命中与否写进去
        的都是同一个键、同一个值。与其编一条看起来在测读路径、实际测不到的
        用例,不如把这条边界写在这里。
        """
        payload = {"kb_ids": [empty_kb["id"]]}
        ask(client, "洗衣机烘干要多久", **payload)
        ask(client, "洗碗机怎么排水", **payload)

        assert embedding_entries(client) == 2


# ---------------------------------------------------------------------------
# 语义缓存的接线
# ---------------------------------------------------------------------------


def _point_the_semantic_cache_at(client: TestClient, question: str) -> None:
    """把语义缓存里那条记录的向量换成**另一个问题**的向量。

    **这是从 HTTP 这一侧唯一能观察到读路径的办法。** 直接调
    `SemanticCache.get` 只能证明那个类是对的,证明不了 `get_rag_service`
    把它传给了 `RAGService`——而「类写好了但没装上」恰恰是最容易发生的
    那种失败。

    做法成立的依据是 mock 编码器**确定且无状态**(字符 n-gram 哈希,见
    `providers/embedding.py`):这里算出来的向量和请求路径上算出来的完全
    一致,不需要去猜。换向量而不是换问题,是因为问题本身不在 scope 里
    ——scope 只由知识库范围、语料代次、配置三样决定。
    """
    from app.core.config import settings
    from app.db.base import database
    from app.db.models import CacheSemanticEntry
    from app.providers.embedding import build_embedder
    from app.services.cache import pack_vector
    from sqlalchemy import select

    async def _run() -> None:
        vector = await build_embedder(settings).embed_query(question)
        async with database.session() as session:
            row = (await session.execute(select(CacheSemanticEntry))).scalars().first()
            assert row is not None, "语义缓存里没有条目——写入侧就没接上"
            row.question_vec = pack_vector(vector)
            await session.commit()

    in_app_loop(client, _run)


class TestSemanticCacheIsActuallyWired:
    """**两层都要验:写进去,以及读出来。**

    这一层默认关闭(见 `Settings.SEMANTIC_CACHE_ENABLED`),所以每条用例
    都得先显式打开——顺带也就证明那个开关真的在读,而不是一个没人看的
    字段。打开之后它**必须排在向量缓存之后**才有意义:每次查询都要一次
    问题向量,没有向量缓存时,未命中反而比不走缓存更慢。
    """

    def test_a_grounded_answer_is_written_into_the_semantic_cache(
        self, client: TestClient, settings, kb_with_products: dict
    ) -> None:
        settings.SEMANTIC_CACHE_ENABLED = True
        assert semantic_entries(client) == 0, "用例开始前缓存就不是空的"

        ask(client, "RX9070 显卡多少钱", kb_ids=[kb_with_products["id"]])

        # 用有文档的知识库,不用 `empty_kb`:检索为空时那条固定答复**不进
        # 缓存**(见 `RAGService._remember`),行数恒为 0,断言就成了
        # 「缓存没接上」和「本来就不该写」两种情况都能通过的废测试。
        assert semantic_entries(client) == 1, "答案没有进语义缓存——这一层没接上"

    def test_a_lookup_serves_an_answer_the_model_never_generated(
        self, client: TestClient, settings, kb_with_products: dict
    ) -> None:
        """**读路径的证明**:第二个问题从来没被问过,却拿到了缓存里的答案。

        它自己的精确缓存必然是空的(那是另一个问题),所以
        `answer_cache_hit` 为真只有一种可能:这一层真的在 `prepare()` 里
        被查了。
        """
        settings.SEMANTIC_CACHE_ENABLED = True
        kb_id = kb_with_products["id"]
        ask(client, "RX9070 显卡多少钱", kb_ids=[kb_id])

        _point_the_semantic_cache_at(client, "27寸显示器多少钱")
        done = first(ask(client, "27寸显示器多少钱", kb_ids=[kb_id]), "done")

        assert done["answer_cache_hit"] is True, "语义缓存没有被查询"
        assert done["cache_hit"] is True
        # 「来源是缓存给的」这条不变量必须一起报出来,否则前端与埋点会
        # 以为检索跑过一遍。
        assert done["grounded"] is True

    def test_the_layer_stays_out_of_the_way_when_the_switch_is_off(
        self, client: TestClient, settings, kb_with_products: dict
    ) -> None:
        """**默认关闭是产品决定。** 关着时一行都不该写。

        只断言「没命中」是不够的:一个每次都查、每次都写的实现同样满足它,
        而它把一个会给错答案的层白白打开了。
        """
        assert settings.SEMANTIC_CACHE_ENABLED is False, "默认值被改了?"
        ask(client, "RX9070 显卡多少钱", kb_ids=[kb_with_products["id"]])

        assert semantic_entries(client) == 0, "开关是关的,却还是写了缓存"


# ---------------------------------------------------------------------------
# 请求指标
# ---------------------------------------------------------------------------


class TestRequestMetricsAreActuallyWired:
    """**一个只记成功请求的指标表,比没有指标表更危险。**

    它会把超时、重试、检索期报错全部漏掉,于是面板上「p95 很健康」和
    「有一半请求根本没走到这里」可以同时成立,而看图的人只会看到前半句。
    所以这组用例的重点不是「记下来了」,而是**「失败也记下来了」**。
    """

    def test_a_completed_ask_is_recorded(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        r = client.post(
            CHAT,
            json={"question": "RX9070 显卡多少钱", "kb_ids": [kb_with_products["id"]]},
            headers={**headers(client), "X-Trace-Id": "e2e-trace-0001"},
        )
        assert r.status_code == 200

        row = latest_metric(client)
        assert row is not None, "一轮完整的问答没有留下指标"
        # **trace_id 取自请求头**,所以日志里那条慢请求能直接落到这张表的
        # 这一行上。少了这个关联,排查就变成「翻日志猜是哪一次」。
        assert row["trace_id"] == "e2e-trace-0001"
        assert row["outcome"] == "done"
        assert row["total_ms"] > 0
        assert row["grounded"] is True
        assert row["stage_timings"], "阶段耗时是空的——埋点没有接上真实的那份"

    def test_an_ask_without_a_session_is_recorded_too(
        self, client: TestClient, kb_with_products: dict
    ) -> None:
        """**「随口一问」也要记。**

        没有会话时走的是另一条分支(不落消息、不进历史)。指标如果挂在
        「落库」那一步上,这条路径就整条消失在面板之外——而它恰恰是没有
        登录态压力、最容易被脚本刷的那条。
        """
        ask(client, "RX9070 显卡多少钱", kb_ids=[kb_with_products["id"]])

        row = latest_metric(client)
        assert row is not None, "无会话的提问没有留下指标"
        assert row["session_id"] is None

    def test_a_generation_failure_is_recorded(
        self, client: TestClient, kb_with_products: dict, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """生成期报错是**最该被看见**的那一类,不能只活在日志里。"""

        class HalfBrokenLLM:
            """先吐一个字再炸——模拟真实上游:错误总是发生在流的中途。"""

            model_id = "half-broken"

            async def astream(self, messages, **kwargs):
                from app.providers.base import ChatChunk

                yield ChatChunk(text="质保")
                raise ProviderError("上游模型炸了", provider="fake")

        monkeypatch.setitem(
            client.app.dependency_overrides,  # type: ignore[attr-defined]
            _rag_dependency(),
            lambda: RAGService(
                settings=_settings_of(client),
                retrieval=_OneChunkRetriever(),
                llm=HalfBrokenLLM(),
            ),
        )

        events = ask(client, "RX9070 显卡质保多久", kb_ids=[kb_with_products["id"]])
        assert "error" in names_of(events), "用例的前提不成立:这一轮没有报错"

        row = latest_metric(client)
        assert row is not None, "失败的这一轮没有留下指标"
        assert row["outcome"] == "error"
        assert row["total_ms"] > 0, "报错的那一轮没有耗时——面板上它会显示成瞬间返回"

    def test_a_retrieval_failure_is_recorded(
        self, client: TestClient, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """**检索期报错发生在流开始之前**,走的是 `AppError` → 502 那条路,
        连 `_persist_after` 都碰不到。

        这正是漏报的重灾区:它是最慢、最贵、最可能因额度耗尽而出现的一类
        失败,而按「在流尾记一条」来埋点的话,它一次都不会出现。
        """

        class ExplodingRetriever:
            async def retrieve(self, query, **kwargs):
                raise ProviderError("向量服务不可用", provider="fake")

        monkeypatch.setitem(
            client.app.dependency_overrides,  # type: ignore[attr-defined]
            _rag_dependency(),
            lambda: RAGService(
                settings=_settings_of(client), retrieval=ExplodingRetriever(), llm=_llm_of(client)
            ),
        )

        r = client.post(CHAT, json={"question": "随便问问"}, headers=headers(client))
        assert r.status_code == 502

        row = latest_metric(client)
        assert row is not None, "检索失败的请求没有留下指标"
        assert row["outcome"] == "error"
