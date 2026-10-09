"""多用户多会话 + 历史持久化(需求 ③④)。

这个文件的重心不是「增删改查能跑」,而是**越权读**。会话是用户自己的私有内容,
比知识库更敏感——知识库是共享资产,会话不是。所以:

* 每条跨用户访问都断言 **404 而不是 403**。403 等于确认「这个 id 存在,只是
  不给你看」,把别人有多少会话、id 排到几号都泄出去了。对不存在的资源和
  别人的资源回同一个响应,才不留侧信道。
* **管理员也读不到别人的会话。** 需求 6 给管理员的是知识库管理权,不是
  读私信权。这条单独测,因为「admin 能看一切」是个很容易被顺手加上的
  「方便」设定,而它会静默地把产品变成另一个东西。
"""

from __future__ import annotations

import io

import pytest
from app.providers.base import ChatChunk
from app.services.session import SessionService
from fastapi.testclient import TestClient

from tests.helpers import auth_header, register

SESSIONS = "/api/sessions"
CHAT = "/api/chat/stream"
KB = "/api/kb"

CSV = (
    "商品名称,类目,价格,库存\n"
    "RX9070 显卡,显卡,4299,12\n"
    "27寸显示器,显示器,1299,30\n"
).encode()


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def token_of(client: TestClient, username: str) -> str:
    return register(client, username)["access_token"]


def create_session(client: TestClient, token: str, **payload) -> dict:
    r = client.post(SESSIONS, json=payload or {}, headers=auth_header(token))
    assert r.status_code == 201, r.text
    return r.json()


def seed_kb(
    client: TestClient, admin: str, name: str = "会话测试库", content: bytes = CSV
) -> int:
    r = client.post(KB, json={"name": name}, headers=auth_header(admin))
    assert r.status_code == 201, r.text
    kb_id = r.json()["id"]
    up = client.post(
        f"{KB}/{kb_id}/documents",
        files={"files": ("报价.csv", io.BytesIO(content), "text/csv")},
        headers=auth_header(admin),
    )
    assert up.status_code == 202, up.text
    doc = client.get(f"{KB}/{kb_id}/documents/{up.json()[0]['id']}", headers=auth_header(admin))
    assert doc.json()["status"] == "ready", doc.text
    return kb_id


def ask(client: TestClient, token: str, question: str, **payload) -> tuple[list, dict]:
    """问一句,返回 (事件列表, done 事件体)。"""
    from tests.test_chat_api import parse_sse

    r = client.post(
        CHAT, json={"question": question, **payload}, headers=auth_header(token)
    )
    assert r.status_code == 200, r.text
    events = parse_sse(r.text)
    done = next((p for n, p in events if n == "done"), None)
    assert done is not None, r.text
    return events, done


@pytest.fixture
def admin(client: TestClient) -> str:
    from tests.helpers import admin_token

    return admin_token(client)


# ---------------------------------------------------------------------------
# 基本 CRUD
# ---------------------------------------------------------------------------


class TestSessionCrud:
    def test_create_returns_the_new_session(self, client: TestClient) -> None:
        token = token_of(client, "alice")
        session = create_session(client, token, title="显卡咨询")
        assert session["title"] == "显卡咨询"
        assert session["message_count"] == 0
        assert session["id"] > 0

    def test_create_without_title_gets_a_placeholder(self, client: TestClient) -> None:
        """标题不能为空——列表页上一条没有标题的会话是没法点的。"""
        session = create_session(client, token_of(client, "alice"))
        assert session["title"].strip()

    def test_list_returns_own_sessions(self, client: TestClient) -> None:
        token = token_of(client, "alice")
        create_session(client, token, title="甲")
        create_session(client, token, title="乙")
        r = client.get(SESSIONS, headers=auth_header(token))
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["total"] == 2
        assert {s["title"] for s in body["items"]} == {"甲", "乙"}

    def test_get_one_includes_its_kb_scope(self, client: TestClient) -> None:
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[3, 4])
        r = client.get(f"{SESSIONS}/{session['id']}", headers=auth_header(token))
        assert r.json()["kb_ids"] == [3, 4]

    def test_rename(self, client: TestClient) -> None:
        token = token_of(client, "alice")
        session = create_session(client, token, title="旧标题")
        r = client.patch(
            f"{SESSIONS}/{session['id']}", json={"title": "新标题"}, headers=auth_header(token)
        )
        assert r.status_code == 200, r.text
        assert r.json()["title"] == "新标题"

    def test_delete_then_gone(self, client: TestClient) -> None:
        token = token_of(client, "alice")
        session = create_session(client, token)
        assert client.delete(f"{SESSIONS}/{session['id']}", headers=auth_header(token)).status_code == 204
        assert client.get(f"{SESSIONS}/{session['id']}", headers=auth_header(token)).status_code == 404

    def test_anonymous_is_rejected(self, client: TestClient) -> None:
        assert client.get(SESSIONS).status_code == 401
        assert client.post(SESSIONS, json={}).status_code == 401

    def test_blank_title_is_rejected(self, client: TestClient) -> None:
        token = token_of(client, "alice")
        r = client.patch(
            f"{SESSIONS}/1", json={"title": "   "}, headers=auth_header(token)
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# 越权隔离 —— 这个文件的重点
# ---------------------------------------------------------------------------


class TestIsolation:
    def test_list_only_returns_your_own(self, client: TestClient) -> None:
        alice, bob = token_of(client, "alice"), token_of(client, "bob")
        create_session(client, alice, title="爱丽丝的")
        create_session(client, bob, title="鲍勃的")

        r = client.get(SESSIONS, headers=auth_header(alice))
        assert [s["title"] for s in r.json()["items"]] == ["爱丽丝的"]

    def test_reading_another_users_session_is_404_not_403(self, client: TestClient) -> None:
        """403 会确认「这个 id 存在」。用 404 让越权和不存在无法区分。"""
        alice, bob = token_of(client, "alice"), token_of(client, "bob")
        victim = create_session(client, bob, title="鲍勃的秘密")

        r = client.get(f"{SESSIONS}/{victim['id']}", headers=auth_header(alice))
        assert r.status_code == 404, r.text
        assert r.json()["code"] == "not_found"

    def test_renaming_another_users_session_is_404(self, client: TestClient) -> None:
        alice, bob = token_of(client, "alice"), token_of(client, "bob")
        victim = create_session(client, bob, title="鲍勃的秘密")

        r = client.patch(
            f"{SESSIONS}/{victim['id']}", json={"title": "被改了"}, headers=auth_header(alice)
        )
        assert r.status_code == 404
        # 确认真的没被改动,而不是只看了状态码
        after = client.get(f"{SESSIONS}/{victim['id']}", headers=auth_header(bob))
        assert after.json()["title"] == "鲍勃的秘密"

    def test_deleting_another_users_session_is_404(self, client: TestClient) -> None:
        alice, bob = token_of(client, "alice"), token_of(client, "bob")
        victim = create_session(client, bob)

        assert client.delete(
            f"{SESSIONS}/{victim['id']}", headers=auth_header(alice)
        ).status_code == 404
        assert client.get(
            f"{SESSIONS}/{victim['id']}", headers=auth_header(bob)
        ).status_code == 200

    def test_reading_another_users_messages_is_404(self, client: TestClient, admin: str) -> None:
        kb_id = seed_kb(client, admin)
        bob = token_of(client, "bob")
        session = create_session(client, bob, kb_ids=[kb_id])
        ask(client, bob, "RX9070 多少钱", session_id=session["id"])

        alice = token_of(client, "alice")
        r = client.get(f"{SESSIONS}/{session['id']}/messages", headers=auth_header(alice))
        assert r.status_code == 404

    def test_admin_cannot_read_other_users_sessions(self, client: TestClient, admin: str) -> None:
        """**管理员不是全能的。** 需求 6 给的是知识库管理权,不是读私信权。

        把它写成显式测试,是因为「admin 能看全部」常常以「方便排查」为由
        被顺手加上,而那会让「每用户隔离」这句话在产品层面不再成立。
        """
        bob = token_of(client, "bob")
        victim = create_session(client, bob, title="鲍勃的秘密")

        r = client.get(f"{SESSIONS}/{victim['id']}", headers=auth_header(admin))
        assert r.status_code == 404, r.text


# ---------------------------------------------------------------------------
# 历史消息
# ---------------------------------------------------------------------------


class TestHistory:
    def test_messages_start_empty(self, client: TestClient) -> None:
        session = create_session(client, token_of(client, "alice"))
        r = client.get(f"{SESSIONS}/{session['id']}/messages", headers=auth_header(token_of(client, "alice2")))
        assert r.status_code == 404  # 别人的会话

    def test_own_new_session_has_no_messages(self, client: TestClient) -> None:
        token = token_of(client, "alice")
        session = create_session(client, token)
        r = client.get(f"{SESSIONS}/{session['id']}/messages", headers=auth_header(token))
        assert r.status_code == 200, r.text
        assert r.json() == {"total": 0, "items": []}

    def test_asking_persists_both_sides_of_the_turn(
        self, client: TestClient, admin: str
    ) -> None:
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])

        _, done = ask(client, token, "RX9070 多少钱", session_id=session["id"])

        r = client.get(f"{SESSIONS}/{session['id']}/messages", headers=auth_header(token))
        items = r.json()["items"]
        assert [m["role"] for m in items] == ["user", "assistant"]
        assert items[0]["content"] == "RX9070 多少钱"
        assert items[1]["content"] == done["answer"]

    def test_history_keeps_the_source_snapshot(self, client: TestClient, admin: str) -> None:
        """**历史里的来源卡片要能原样重放。**

        只存编号的话,重看旧会话时卡片是空的——而卡片正是这个产品相对
        通用大模型的卖点。所以回答消息里存的是完整的来源快照。
        """
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])
        _, done = ask(client, token, "RX9070 多少钱", session_id=session["id"])

        items = client.get(
            f"{SESSIONS}/{session['id']}/messages", headers=auth_header(token)
        ).json()["items"]
        answer = items[1]

        assert answer["sources"], "回答消息没有存下来源快照"
        assert answer["cited"] == done["cited"]
        top = answer["sources"][0]
        # 卡片所需的字段一个都不能少
        assert top["doc_name"] == "报价.csv"
        assert top["kb_name"] == "会话测试库"
        assert "RX9070" in top["content"]

    def test_message_count_tracks_appended_turns(self, client: TestClient, admin: str) -> None:
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])
        ask(client, token, "RX9070 多少钱", session_id=session["id"])

        r = client.get(f"{SESSIONS}/{session['id']}", headers=auth_header(token))
        assert r.json()["message_count"] == 2

    def test_untitled_session_is_renamed_from_the_first_question(
        self, client: TestClient, admin: str
    ) -> None:
        """列表页上三条「新会话」是没法区分的。首问自动成题,不用多一次往返。"""
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])
        assert session["title"] == "新会话"

        ask(client, token, "RX9070 显卡的库存还有多少", session_id=session["id"])

        after = client.get(f"{SESSIONS}/{session['id']}", headers=auth_header(token)).json()
        assert after["title"] == "RX9070 显卡的库存还有多少"

    def test_a_title_the_user_set_is_never_overwritten(
        self, client: TestClient, admin: str
    ) -> None:
        """自动命名只在**用户还没取过名**时生效。

        否则用户手动改的标题会在下一句提问时被悄悄改回去——那种「我明明改过」
        的体验比不给自动命名更糟。
        """
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, title="我自己取的", kb_ids=[kb_id])

        ask(client, token, "RX9070 显卡的库存还有多少", session_id=session["id"])

        after = client.get(f"{SESSIONS}/{session['id']}", headers=auth_header(token)).json()
        assert after["title"] == "我自己取的"

    def test_history_survives_a_new_login(self, client: TestClient, admin: str) -> None:
        """需求 4 的原话是「重新登录后能找回」。换个 token 再查。"""
        kb_id = seed_kb(client, admin)
        register(client, "alice", password="alice-pw-123")
        from tests.helpers import login

        first = login(client, "alice", "alice-pw-123")["access_token"]
        session = create_session(client, first, kb_ids=[kb_id])
        ask(client, first, "RX9070 多少钱", session_id=session["id"])

        second = login(client, "alice", "alice-pw-123")["access_token"]
        assert second != first
        r = client.get(f"{SESSIONS}/{session['id']}/messages", headers=auth_header(second))
        assert r.json()["total"] == 2, "重新登录后历史丢了"


# ---------------------------------------------------------------------------
# 与问答接口的衔接
# ---------------------------------------------------------------------------


class TestChatIntegration:
    def test_asking_without_a_session_persists_nothing(
        self, client: TestClient, admin: str
    ) -> None:
        """不带 session_id 就是「随口问一句」,不该在库里的任何地方留痕。"""
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        before = client.get(SESSIONS, headers=auth_header(token)).json()["total"]

        _, done = ask(client, token, "RX9070 多少钱", kb_ids=[kb_id])

        assert done.get("session_id") is None
        assert done.get("message_id") is None
        after = client.get(SESSIONS, headers=auth_header(token)).json()["total"]
        assert after == before, "无会话的提问不该凭空建出一个会话"

    def test_done_event_reports_session_and_message_id(
        self, client: TestClient, admin: str
    ) -> None:
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])

        _, done = ask(client, token, "RX9070 多少钱", session_id=session["id"])
        assert done["session_id"] == session["id"]
        assert isinstance(done["message_id"], int)

    def test_done_reports_the_query_that_was_actually_searched(
        self, client: TestClient, admin: str
    ) -> None:
        """**这是前端唯一能看到「追问被改写成了什么」的地方。**

        断言的是「有历史时它变了」而不是它变成了什么:具体措辞是模型的
        输出,不该在单元测试里钉死(那会把测试变成对 mock 的回声)。但
        「变了」本身是编排的契约——不变就说明改写那一步根本没生效,
        而症状只是「追问的检索质量变差」,不看这个字段发现不了。

        改写的内容对不对,由 `smoke_e2e.py` 对着真模型验。
        """
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])

        _, first = ask(client, token, "RX9070 显卡多少钱", session_id=session["id"])
        assert first["search_query"] == "RX9070 显卡多少钱", "第一问没有历史,不该改写"
        assert first["history_messages"] == 0

        _, second = ask(client, token, "它的库存还有多少", session_id=session["id"])
        assert second["search_query"] != "它的库存还有多少", "第二问有历史,必须改写后再检索"
        assert second["history_messages"] == 2, "上一轮的一问一答都该进窗口"

    def test_asking_in_someone_elses_session_is_404(self, client: TestClient, admin: str) -> None:
        """**越权写。** 往别人的会话里塞消息比读更严重——它会污染对方的历史。"""
        kb_id = seed_kb(client, admin)
        bob = token_of(client, "bob")
        session = create_session(client, bob, kb_ids=[kb_id])

        r = client.post(
            CHAT,
            json={"question": "我是爱丽丝", "session_id": session["id"]},
            headers=auth_header(token_of(client, "alice")),
        )
        assert r.status_code == 404, r.text

    def test_session_scope_is_the_default_when_kb_ids_omitted(
        self, client: TestClient, admin: str
    ) -> None:
        """会话记住的知识库范围要生效,否则「继续上次的对话」每次都得重勾。

        **必须有两个库才验得出来。** 只建一个库时,即使 kb_ids 完全没被用上、
        检索退化成「搜全部」,也一样能召回到那唯一的库——测试会绿,而它
        什么也没证明。所以这里建两个库,再问**只存在于另一个库**里的商品:
        范围生效就召不到 kb_b 的内容,范围失效才召得到。

        **不能拿 `grounded is False` 当判据。** 它只说明「检索有没有结果」,
        而范围生效时检索**照样有结果**:kb_a 在范围内,哪怕它跟问题毫不相干,
        向量路也会把它作为唯一候选返回(测试环境是 MockEmbedder + NoopReranker,
        不做相关性截断)。所以 grounded 在这里恒为 True,那条断言测的是
        一个不存在的契约。真正要钉的是**哪几号库的内容进了上下文**。
        """
        kb_a = seed_kb(
            client,
            admin,
            name="显卡库",
            content="商品名称,类目,价格,库存\nRX9070 显卡,显卡,4299,12\n".encode(),
        )
        kb_b = seed_kb(
            client,
            admin,
            name="显示器库",
            content="商品名称,类目,价格,库存\n27寸显示器,显示器,1299,30\n".encode(),
        )
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_a])

        question = "27寸显示器多少钱"
        events, _ = ask(client, token, question, session_id=session["id"])
        sources = next((p["sources"] for n, p in events if n == "sources"), [])

        assert all(s["kb_id"] == kb_a for s in sources), (
            f"会话范围之外的库被检索到了:{[s['kb_name'] for s in sources]}"
        )

        # **正对照。** 上面那条断言对空列表恒为真,所以它单独站着时可能是
        # 空转绿:如果检索因为任何别的原因(分块没入库、向量库没写进去)
        # 什么都没返回,测试照样过,而它什么也没证明。换成 kb_b 再问一遍,
        # 证明这句话在这个环境里**本来就召得回东西**——那么它在会话范围下
        # 被挡住,才是「范围」起的作用。
        control_events, _ = ask(client, token, question, kb_ids=[kb_b])
        control = next((p["sources"] for n, p in control_events if n == "sources"), [])
        assert control, "对照组本身就没召回到东西,这条用例证明不了范围生效"
        assert all(s["kb_id"] == kb_b for s in control), (
            f"显式指定 kb_b 却召回了别的库:{[s['kb_name'] for s in control]}"
        )

    def test_explicit_kb_ids_override_the_session_scope(
        self, client: TestClient, admin: str
    ) -> None:
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])

        # 显式传空范围 = 这一问什么都不看,压过会话自带的 [kb_id]
        events, done = ask(client, token, "RX9070 多少钱", session_id=session["id"], kb_ids=[])
        sources = next((p["sources"] for n, p in events if n == "sources"), [])
        assert sources == []
        assert done["grounded"] is False


# ---------------------------------------------------------------------------
# 生成失败 / 落库失败时,历史里留下什么
#
# 这两条路径都发生在**响应头提交之后**。它们改不了状态码(状态码早发出去了,
# 而且 200 是对的:连接确实是正常建立的),只能在「库里写什么」上做取舍。
# 而写错了不会报错——用户下次翻开历史才会发现「我记错了」或「多了一条空的」。
# ---------------------------------------------------------------------------


class _LLMThatDies:
    """「先流出 N 片,然后断线」的假模型。

    真实的模型连接中断就是这个形状:HTTP 200 已返回、前几个 token 已经
    到浏览器了,然后连接断掉。**MockLLM 复现不了它**——那个替身永远成功,
    所以在此之前,`_persist_after` 里处理失败的两条分支从来没有被测过。
    """

    def __init__(self, chunks_before_dying: int) -> None:
        self._chunks = chunks_before_dying

    @property
    def model_id(self) -> str:
        return "llm-that-dies"

    async def acomplete(self, messages, **kwargs):
        # 只在有历史时被调用(指代消解),而下面每个用例都是新会话。
        # 真被调到说明用例的前提变了,炸掉比静默返回一个假改写好。
        raise AssertionError("新会话不该触发查询改写")

    def astream(self, messages, **kwargs):
        # 普通 def 返回 AsyncIterator,与 `LLMProvider.astream` 的约定一致:
        # 异常要在**迭代期**抛出,而不是构造期。
        return self._stream()

    async def _stream(self):
        for _ in range(self._chunks):
            yield ChatChunk(text="半截答案")
        raise RuntimeError("模型连接中断")


def _use_dying_llm(monkeypatch: pytest.MonkeyPatch, *, chunks: int) -> None:
    from app.api.routers import chat as chat_module

    monkeypatch.setattr(chat_module, "build_llm", lambda settings: _LLMThatDies(chunks))


def ask_raw(client: TestClient, token: str, question: str, **payload) -> list:
    """问一句并返回原始事件列表。

    **不能用 `ask`**:它断言 `done` 事件必须存在,而本节测的正是「没有 done」
    的那条路——用它会先死在辅助函数的断言上,失败信息指向辅助函数而不是被测行为。
    """
    from tests.test_chat_api import parse_sse

    r = client.post(CHAT, json={"question": question, **payload}, headers=auth_header(token))
    assert r.status_code == 200, r.text
    return parse_sse(r.text)


def messages_of(client: TestClient, token: str, session_id: int) -> dict:
    r = client.get(f"{SESSIONS}/{session_id}/messages", headers=auth_header(token))
    assert r.status_code == 200, r.text
    return r.json()


class TestPersistenceWhenGenerationFails:
    def test_partial_answer_is_persisted(self, client: TestClient, admin: str, monkeypatch) -> None:
        """**半截答案也要落库。** 用户已经在屏幕上看到那半句了。

        库里不存的话,他下次翻开历史会觉得「我记错了」;而出错这件事本身
        只存在于当次那条 error 事件里,关掉页面就没了。所以落库时把错误
        一并写进 `error` 字段——历史里能看出这一轮是坏的,而不是一条
        看起来正常、其实被截断的回答。
        """
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])
        _use_dying_llm(monkeypatch, chunks=1)

        events = ask_raw(client, token, "RX9070 多少钱", session_id=session["id"])

        assert events[-1][0] == "error", "生成失败应当以 error 事件收尾,之后不再有事件"
        assert any(n == "token" for n, _ in events), "第一片确实流出去过"

        stored = messages_of(client, token, session["id"])
        assert stored["total"] == 2, "问题 + 半截答案"
        answer = stored["items"][1]
        assert answer["role"] == "assistant"
        assert answer["content"] == "半截答案"
        assert answer["error"], "失败的那一轮必须在库里留下标记"

    def test_nothing_is_persisted_when_it_dies_before_the_first_token(
        self, client: TestClient, admin: str, monkeypatch
    ) -> None:
        """一个字都没流出来时**不落库**。

        存一条空的助手消息看着像是「记录更完整」,实际有两处害处:
        ① 历史列表里多一条点开什么都没有的空行;
        ② 更糟的是它会经 `build_history` 进入后续几轮的 prompt——模型看到
           自己「说过」一句空话,那是纯噪音,还会挤占本就不多的历史预算。

        用户看到的是「提问在、回答没有」,而这**就是**事实:这一轮确实
        什么都没答出来。
        """
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])
        _use_dying_llm(monkeypatch, chunks=0)

        events = ask_raw(client, token, "RX9070 多少钱", session_id=session["id"])

        assert events[-1][0] == "error"
        assert not any(n == "token" for n, _ in events), "本用例的前提是一片都没流出来"
        assert messages_of(client, token, session["id"])["total"] == 1, (
            "一个字都没生成,不该在历史里留下空的助手消息"
        )

    def test_a_persist_failure_does_not_break_the_stream(
        self, client: TestClient, admin: str, monkeypatch
    ) -> None:
        """落库失败不能影响已经发出去的内容。

        响应头早就提交了,此刻再抛异常只会让客户端看到连接莫名断掉——而它
        其实收到了一条完整的回答。所以只能降级:**答案照给,历史少一条**,
        并把这一点如实告诉前端(done 事件里 `message_id` 为 None,前端据此
        知道这轮没能入历史)。

        这里钉的是「降级发生了」,不是「吞异常」——区别在于 done 事件是否
        仍然带着完整答案到达。
        """
        kb_id = seed_kb(client, admin)
        token = token_of(client, "alice")
        session = create_session(client, token, kb_ids=[kb_id])

        async def boom(*args, **kwargs):
            raise RuntimeError("数据库写不进去")

        monkeypatch.setattr(SessionService, "append_answer", boom)

        _, done = ask(client, token, "RX9070 多少钱", session_id=session["id"])

        assert done["answer"], "落库失败不该吃掉已经生成好的答案"
        assert done["message_id"] is None, "没能入历史时,message_id 不能瞎编一个"
        assert done["session_id"] == session["id"]
        assert messages_of(client, token, session["id"])["total"] == 1, "问题应当还在库里"
