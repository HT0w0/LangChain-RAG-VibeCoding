"""RAG 编排:检索 → 组装 → 提示词 → 流式生成 → 引用溯源。

这里是「引用指得准不准」的最后一环,也是整条链路上**最容易静默出错**的一环:

- 编号 `[1]` 与 `sources[0]` 错位 → 答案读起来完全正常,只是来源卡片指错了
- 少了「先发 sources 再发 token」的顺序 → 功能都还在,但用户要等答案写完
  才看得到引用了什么,而「先看到依据」正是这个产品想要的体验
- 没有资料时照样调模型 → 它会用自己的知识答,而用户无从分辨

假 Retriever 与假 LLM 都是刻意的:检索的质量由 `test_retrieval*.py` 负责,
模型的措辞由提示词负责。这里要钉死的是**编排**,混杂真实组件只会让失败
原因变模糊。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field, replace

import pytest
from app.providers.base import ChatChunk, ChatResult, Usage
from app.services.context import RetrievedChunk
from app.services.rag import (
    NO_CONTEXT_ANSWER,
    DoneEvent,
    ErrorEvent,
    RAGService,
    SourcesEvent,
    TokenEvent,
)
from app.services.retrieval import RetrievalFilters, RetrievalResult

# ---------------------------------------------------------------------------
# 假组件
# ---------------------------------------------------------------------------


def chunk(
    chunk_id: int,
    content: str,
    *,
    doc_name: str = "报价.csv",
    kb_name: str = "商品库",
    kb_id: int = 1,
    doc_id: int = 1,
    chunk_index: int = 0,
    meta: dict | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        content=content,
        kb_id=kb_id,
        doc_id=doc_id,
        doc_name=doc_name,
        kb_name=kb_name,
        chunk_index=chunk_index,
        meta=meta or {},
    )


def _history() -> list:
    """一问一答的历史。`Message` 直接构造即可——`build_history` 只用 role/content。"""
    from app.db.models import Message, MessageRole

    return [
        Message(session_id=1, role=str(MessageRole.USER), content="RX9070 显卡多少钱"),
        Message(session_id=1, role=str(MessageRole.ASSISTANT), content="4299 元 [1]"),
    ]


@dataclass
class FakeRetriever:
    """只回放预设结果。`retrieve` 收到的实参记在 `calls` 里,供断言语义。"""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    route_sizes: dict[str, int] = field(default_factory=lambda: {"vector": 1, "keyword": 1})
    stage_timings: dict[str, float] = field(default_factory=lambda: {"total": 12.0})
    reranked: bool = True
    degraded: bool = False
    calls: list[dict] = field(default_factory=list)

    async def retrieve(
        self, query: str, *, kb_ids, filters: RetrievalFilters | None = None, top_k=None
    ) -> RetrievalResult:
        self.calls.append({"query": query, "kb_ids": kb_ids, "filters": filters})
        return RetrievalResult(
            query=query,
            chunks=list(self.chunks),
            route_sizes=dict(self.route_sizes),
            stage_timings=dict(self.stage_timings),
            reranked=self.reranked,
            degraded=self.degraded,
        )


class FakeLLM:
    """按片回放,可指定在第几片抛异常。

    `reasoning_pieces` 模拟**思考模型的真实流式形状**:推理内容先以
    `text=""` 的独立分片流出来,答案分片跟在后面。把它们合并进同一个
    分片(那样也能构造出来)会漏掉真正的泄露路径——推理是**单独**到的,
    按「有 text 就转发」的写法照样会把推理吞进去。
    """

    def __init__(
        self,
        pieces: Sequence[str] = ("这是", "回答", "[1]。"),
        *,
        usage: Usage | None = None,
        fail_at: int | None = None,
        reasoning_pieces: Sequence[str] = (),
        inline_reasoning: str = "",
        rewritten: str | None = None,
        rewrite_delay: float = 0.0,
    ) -> None:
        self._pieces = list(pieces)
        self._usage = usage
        self._fail_at = fail_at
        self._reasoning_pieces = list(reasoning_pieces)
        self._inline_reasoning = inline_reasoning
        self._rewritten = rewritten
        # 改写在真实平台上是一次完整的 LLM 往返(实测 1.28 秒)。要验
        # 「耗时统计有没有把它算进去」就得让假模型也真的花掉一段时间,
        # 否则两边的期望值都是 0,断言恒真。
        self._rewrite_delay = rewrite_delay
        self.calls: list[list[dict[str, str]]] = []
        # 与 `calls`(生成用的消息)分开记:这是**改写**那一次调用。
        # 合在一起的话,「改写被调了几次」就得靠数消息条数去猜。
        self.rewrite_calls: list[list[dict[str, str]]] = []

    model_id = "fake-llm"

    async def acomplete(self, messages, **kwargs):
        """指代消解走这条路(多轮追问)。

        没设 `rewritten` 就抛——**不要静默返回原句**。`rewrite_question`
        本身会把异常吞掉并退化成原问题(那是它的正常降级路径,由
        `test_history.py` 负责),所以这里抛不会让整次提问挂掉;但若返回
        原句,一个「本该有历史、改写却根本没被调用」的用例会静默通过,
        而它恰恰是要验改写生效的那条。
        """
        self.rewrite_calls.append(list(messages))
        if self._rewritten is None:
            raise NotImplementedError("本用例不该触发查询改写")
        if self._rewrite_delay:
            await asyncio.sleep(self._rewrite_delay)
        return ChatResult(content=self._rewritten)

    async def astream(
        self, messages, *, temperature=None, max_tokens=None, **kwargs
    ) -> AsyncIterator[ChatChunk]:
        self.calls.append(list(messages))
        for piece in self._reasoning_pieces:
            yield ChatChunk(text="", reasoning=piece)
        for i, piece in enumerate(self._pieces):
            if self._fail_at is not None and i == self._fail_at:
                raise RuntimeError("上游模型炸了")
            last = i == len(self._pieces) - 1
            yield ChatChunk(
                text=piece,
                reasoning=self._inline_reasoning,
                usage=self._usage if last else None,
            )


@pytest.fixture
def settings():
    from app.core.config import Settings

    return Settings(
        _env_file=None,
        DASHSCOPE_API_KEY="sk-test",
        CONTEXT_TOKEN_BUDGET=3000,
    )


def make_service(
    settings, retriever: FakeRetriever, llm, *, answers=None, semantics=None
) -> RAGService:
    return RAGService(
        settings=settings,
        retrieval=retriever,
        llm=llm,
        answers=answers,
        semantics=semantics,
    )


async def collect(service: RAGService, prepared) -> list:
    return [event async for event in service.stream(prepared)]


# ---------------------------------------------------------------------------
# 组装阶段
# ---------------------------------------------------------------------------


class TestPrepare:
    async def test_sources_are_numbered_from_one(self, settings) -> None:
        retriever = FakeRetriever(
            chunks=[chunk(11, "甲资料"), chunk(22, "乙资料"), chunk(33, "丙资料")]
        )
        prepared = await make_service(settings, retriever, FakeLLM()).prepare(
            "问题", kb_ids=[1]
        )
        assert [s.index for s in prepared.sources] == [1, 2, 3]
        assert [s.chunk_id for s in prepared.sources] == [11, 22, 33]

    async def test_prompt_body_matches_sources_one_to_one(self, settings) -> None:
        """**引用溯源的根本不变式。**

        正文里的 `[n]` 必须指向 `sources[n-1]`。这里直接验:第 n 条资料的
        **正文原文**出现在 prompt 里其编号标记之后,且编号顺序与 sources 一致。
        只断言「数量相等」是不够的——顺序错位时数量照样相等。
        """
        retriever = FakeRetriever(
            chunks=[chunk(11, "第一条正文"), chunk(22, "第二条正文")]
        )
        prepared = await make_service(settings, retriever, FakeLLM()).prepare(
            "问题", kb_ids=[1]
        )
        body = prepared.messages[1]["content"]

        assert "[1]" in body and "[2]" in body
        assert body.index("[1]") < body.index("第一条正文") < body.index("[2]")
        assert body.index("[2]") < body.index("第二条正文")
        # 渲染出来的正文与 sources 携带的正文必须是同一份
        for source in prepared.sources:
            assert source.content in body

    async def test_duplicate_chunks_never_get_a_number(self, settings) -> None:
        """**回归:编号必须建立在 `assemble_context` 的 used 之上,而不是检索结果之上。**

        `assemble_context` 会丢掉重复正文和超预算的分块。若编号照着原始
        检索结果发,被丢掉的那条仍占一个号,于是后面每条 `[n]` 全部指向
        前一格——而正文里的编号看上去完全正常。
        """
        retriever = FakeRetriever(
            chunks=[chunk(11, "重复的正文"), chunk(22, "重复的正文"), chunk(33, "独有正文")]
        )
        prepared = await make_service(settings, retriever, FakeLLM()).prepare(
            "问题", kb_ids=[1]
        )
        assert [s.chunk_id for s in prepared.sources] == [11, 33]
        assert [s.index for s in prepared.sources] == [1, 2]

    async def test_question_reaches_the_prompt(self, settings) -> None:
        prepared = await make_service(
            settings, FakeRetriever(chunks=[chunk(1, "甲")]), FakeLLM()
        ).prepare("RX9070 多少钱", kb_ids=[1])
        assert "RX9070 多少钱" in prepared.messages[1]["content"]

    async def test_scope_and_filters_are_forwarded(self, settings) -> None:
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        filters = RetrievalFilters(category="显卡", price_min=1000)
        await make_service(settings, retriever, FakeLLM()).prepare(
            "问题", kb_ids=[7, 8], filters=filters
        )
        assert retriever.calls[0]["kb_ids"] == [7, 8]
        assert retriever.calls[0]["filters"] is filters

    async def test_sources_carry_citation_metadata(self, settings) -> None:
        retriever = FakeRetriever(
            chunks=[
                chunk(
                    11,
                    "商品 RX9070,类目 显卡,价格 4299元。",
                    doc_name="报价.csv",
                    kb_name="商品库",
                    chunk_index=3,
                    meta={"category": "显卡", "price": 4299},
                )
            ]
        )
        source = (
            await make_service(settings, retriever, FakeLLM()).prepare("问题", kb_ids=[1])
        ).sources[0]
        assert source.doc_name == "报价.csv"
        assert source.kb_name == "商品库"
        assert source.chunk_index == 3
        assert source.category == "显卡"
        assert source.price == 4299

    async def test_empty_retrieval_is_ungrounded(self, settings) -> None:
        prepared = await make_service(settings, FakeRetriever(chunks=[]), FakeLLM()).prepare(
            "问题", kb_ids=[1]
        )
        assert prepared.grounded is False
        assert prepared.sources == []

    async def test_over_budget_chunks_are_not_numbered(self, settings) -> None:
        """超预算被丢掉的分块同样不能占号。

        **不必手算 token 数。** 这里刻意断言「中间那条大的没进、两边的进了」
        以及「最终上下文没超预算」这两条性质,而不是把每条资料的估算值
        算出来写死——估算函数微调一次,写死的期望就全错,而错误信息会
        指向测试而不是被测代码。
        """
        settings.CONTEXT_TOKEN_BUDGET = 120
        retriever = FakeRetriever(
            chunks=[chunk(1, "甲" * 30), chunk(2, "乙" * 400), chunk(3, "丙" * 30)]
        )
        prepared = await make_service(settings, retriever, FakeLLM()).prepare(
            "问题", kb_ids=[1]
        )
        assert [s.chunk_id for s in prepared.sources] == [1, 3]
        assert prepared.bundle.tokens <= 120
        assert [d.chunk_id for d in prepared.bundle.dropped] == [2]


class TestQueryRewriteReachesRetrieval:
    """多轮追问:改写后的句子去检索,**原话**去回答。

    这一条在单元测试里很容易看着对、实际错位——因为错位的症状是「检索
    质量变差」,而假 Retriever 无论收到什么都会返回预设结果。所以这里
    不去看结果,直接看**送进检索的到底是哪句话**。
    """

    async def test_retrieval_receives_the_rewritten_question(self, settings) -> None:
        """追问「它的库存呢」必须带着「它」是谁一起去检索。

        原话在语料里几乎必然召偏:BM25 只匹配到「库存」,与「它」指显卡还是
        显示器无关。向量路好一些,但代词本身没有语义,靠它区分不了两个商品。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 库存 12")])
        llm = FakeLLM(rewritten="RX9070 显卡的库存是多少")

        prepared = await make_service(settings, retriever, llm).prepare(
            "它的库存呢", kb_ids=[1], history=_history()
        )

        assert retriever.calls[0]["query"] == "RX9070 显卡的库存是多少"
        assert prepared.search_query == "RX9070 显卡的库存是多少"

    async def test_the_user_original_wording_survives_for_the_answer(self, settings) -> None:
        """**改写结果不能拿去回答。**

        拿去回答的话,模型看到的「问题」是它自己(或另一个提示词)重写过的
        句子,措辞跟着跑偏,而且用户看不出为什么答非所问。检索与回答是
        两个环节,分别要「完整」和「原话」。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 库存 12")])
        llm = FakeLLM(rewritten="RX9070 显卡的库存是多少")

        prepared = await make_service(settings, retriever, llm).prepare(
            "它的库存呢", kb_ids=[1], history=_history()
        )

        assert prepared.question == "它的库存呢"
        assert "它的库存呢" in prepared.messages[-1]["content"]
        assert "RX9070 显卡的库存是多少" not in prepared.messages[-1]["content"]

    async def test_no_history_means_no_rewrite_call_at_all(self, settings) -> None:
        """**没有历史就不该多花一次 LLM 调用。**

        第一问不可能有指代,改写是纯粹的浪费——而且它是一整次往返,
        在首字时延里占的比例不小。这里断言的不是「结果一样」,而是
        「根本没调」。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        llm = FakeLLM()

        prepared = await make_service(settings, retriever, llm).prepare(
            "RX9070 多少钱", kb_ids=[1]
        )

        assert llm.rewrite_calls == []
        assert retriever.calls[0]["query"] == "RX9070 多少钱"
        assert prepared.search_query == "RX9070 多少钱"


# ---------------------------------------------------------------------------
# 流式阶段
# ---------------------------------------------------------------------------


class TestStreamOrdering:
    async def test_sources_event_precedes_every_token(self, settings) -> None:
        """**用户要先看到「依据是什么」,再看到答案。**

        反过来(先流式再补 sources)功能上也能跑,但用户在整个生成期间
        都不知道答案是从哪来的——而这正是这个产品相对通用大模型的价值。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        service = make_service(settings, retriever, FakeLLM(pieces=("甲", "乙")))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        names = [type(e).__name__ for e in events]
        assert names[0] == "SourcesEvent"
        assert "TokenEvent" not in names[: names.index("SourcesEvent")]

    async def test_event_sequence_is_sources_tokens_done(self, settings) -> None:
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        service = make_service(settings, retriever, FakeLLM(pieces=("甲", "乙", "丙")))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        assert [type(e).__name__ for e in events] == [
            "SourcesEvent",
            "TokenEvent",
            "TokenEvent",
            "TokenEvent",
            "DoneEvent",
        ]

    async def test_tokens_are_forwarded_in_order(self, settings) -> None:
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        service = make_service(settings, retriever, FakeLLM(pieces=("你", "好", "呀")))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))
        assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "你好呀"

    async def test_standalone_reasoning_chunks_are_never_forwarded(self, settings) -> None:
        """**思考模型的推理是以 `text=""` 的独立分片流出来的。**

        这才是真实的泄露路径:按「有内容就转发」来写,推理会被原样送进
        正文——用户看到模型自言自语,而且那段话无法被任何资料引用。
        只在「text 和 reasoning 同一个分片」时测试是测不到这条路径的。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        service = make_service(
            settings,
            retriever,
            FakeLLM(pieces=("答案",), reasoning_pieces=("让我", "想想…", "该这么答:")),
        )
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        tokens = [e.text for e in events if isinstance(e, TokenEvent)]
        done = next(e for e in events if isinstance(e, DoneEvent))
        assert tokens == ["答案"], f"推理被当成正文送出去了:{tokens}"
        assert done.answer == "答案"

    async def test_reasoning_attached_to_a_text_chunk_is_also_dropped(self, settings) -> None:
        """个别版本会把 reasoning_content 挂在正文分片上,一并挡住。"""
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        service = make_service(
            settings, retriever, FakeLLM(pieces=("答案",), inline_reasoning="让我想想…")
        )
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))
        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.answer == "答案"
        assert "让我想想" not in done.answer


class TestDoneEvent:
    async def test_carries_the_accumulated_answer(self, settings) -> None:
        """`done.answer` 与已发出的 token 是同一份内容。

        带上它不是冗余:SSE 没有投递保证,客户端无法区分「流正常结束」和
        「连接中途断了」。有了完整答案,断流时前端至少能把已收到的内容拼回
        一致状态,而不是显示一段被截断的回答。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        service = make_service(settings, retriever, FakeLLM(pieces=("依据是 ", "[1]。")))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        done = next(e for e in events if isinstance(e, DoneEvent))
        streamed = "".join(e.text for e in events if isinstance(e, TokenEvent))
        assert done.answer == "依据是 [1]。"
        assert done.answer == streamed

    async def test_reports_the_query_that_was_actually_searched(self, settings) -> None:
        """`done` 带上**真正送去检索**的那句话。

        这是多轮链路上唯一能看见「指代消解到底做了什么」的地方。没有它,
        排查时只能看到「追问的检索质量不如第一问」,分不清是改写带偏了
        还是检索本身不行——而这两者的修法完全相反。`smoke_e2e.py` 对着
        真模型验的也正是这一条:假模型会照着提示词回话,证明不了真模型
        也会把「它」补全。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 库存 12")])
        llm = FakeLLM(rewritten="RX9070 显卡的库存是多少")
        service = make_service(settings, retriever, llm)
        prepared = await service.prepare("它的库存呢", kb_ids=[1], history=_history())

        events = await collect(service, prepared)
        done = next(e for e in events if isinstance(e, DoneEvent))

        assert done.search_query == "RX9070 显卡的库存是多少"
        assert done.history_messages == 2, "一问一答两条"

    async def test_reports_which_sources_were_actually_cited(self, settings) -> None:
        retriever = FakeRetriever(chunks=[chunk(1, "甲"), chunk(2, "乙"), chunk(3, "丙")])
        service = make_service(settings, retriever, FakeLLM(pieces=("结论 [3][1]",)))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.cited == [1, 3]

    async def test_invented_citation_index_is_dropped(self, settings) -> None:
        """只有 2 条资料却写出 `[9]` —— 若照单全收,前端会去渲染一张不存在的卡片。"""
        retriever = FakeRetriever(chunks=[chunk(1, "甲"), chunk(2, "乙")])
        service = make_service(settings, retriever, FakeLLM(pieces=("见 [1] 与 [9]",)))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.cited == [1]

    async def test_citation_range_follows_the_source_count_not_the_hit_count(
        self, settings
    ) -> None:
        """**引用上限必须是「实际装进上下文的资料条数」。**

        检索回来 2 条、去重后只剩 1 条时,若上限取的是检索结果条数,
        模型写出的 `[2]` 就会被判成合法引用——而编号 2 根本没有对应的
        卡片,前端要么渲染一张空的,要么点击时崩掉。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "一模一样的正文"), chunk(2, "一模一样的正文")])
        service = make_service(settings, retriever, FakeLLM(pieces=("结论 [1][2]",)))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        sources_event = next(e for e in events if isinstance(e, SourcesEvent))
        done = next(e for e in events if isinstance(e, DoneEvent))
        assert [s.index for s in sources_event.sources] == [1]
        assert done.cited == [1]

    async def test_total_covers_the_whole_request_not_just_generation(self, settings) -> None:
        """**`total` 要从「用户按下回车」那一刻算起。**

        多轮追问里,改写是一次完整的 LLM 往返——对着真实平台实测 1.28 秒,
        比整个生成段还长。若 total 只量生成段,面板上就会出现
        `total 1291ms` 而紧邻的 `rewrite 1280ms`、`retrieve 736ms` 更大:
        一个自相矛盾的指标,比没有这项数据更误导,而且它会让人误判
        「这套系统只要 1.3 秒」,实际用户等了两秒半。

        首 token 时延同理,而且更严重——用户盯着转圈的那段时间是从提问
        开始的,不是从模型开始吐字开始的。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        llm = FakeLLM(rewritten="RX9070 显卡的库存是多少", rewrite_delay=0.15)
        service = make_service(settings, retriever, llm)
        prepared = await service.prepare("它的库存呢", kb_ids=[1], history=_history())
        events = await collect(service, prepared)

        t = next(e for e in events if isinstance(e, DoneEvent)).stage_timings

        # 前提检查:改写确实花了可观测的时间,否则下面两条断言恒真。
        assert t["rewrite"] >= 100, f"假模型的延迟没生效?rewrite={t['rewrite']}"
        assert t["total"] >= t["rewrite"], "total 把改写那一段排除在外了"
        assert t["first_token"] >= t["rewrite"], "首 token 时延要从提问算起"

    async def test_reports_stage_timings_including_first_token(self, settings) -> None:
        """**首 token 时延是 SSE 体验的关键指标**,必须单独可观测。

        只看总耗时会把「模型慢」和「流式没起作用」混为一谈——两者的
        优化方向完全不同。
        """
        retriever = FakeRetriever(
            chunks=[chunk(1, "甲")], stage_timings={"vector": 4.0, "total": 12.0}
        )
        service = make_service(settings, retriever, FakeLLM())
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        done = next(e for e in events if isinstance(e, DoneEvent))
        t = done.stage_timings
        assert "first_token" in t
        assert "generate" in t
        assert "assemble" in t
        # 检索阶段的明细要一起带上,否则排障时还得去翻日志
        assert t["vector"] == 4.0
        # **检索自己的 total 在合并时必须改名。** 两个都叫 total 的话,
        # 排障时会把它当成整条链路的总耗时,而它其实只覆盖了检索一段——
        # 一个看着有值、实际量错东西的指标比没有更误导。
        assert t["retrieve"] == 12.0
        # `total` 必须是**整次请求**的耗时,不能是检索那个 total 的别名,
        # 也不能只是生成段的耗时(见 `test_total_covers_the_whole_request_...`)。
        # 与 latency_ms 相等就是在说这件事。
        assert t["total"] == done.latency_ms
        # **`first_token` 与 `generate` 现在各有各的原点**,两者之间不再有
        # 固定的大小关系:前者从提问算起(含改写与检索),后者从模型开始
        # 吐字算起(只量模型自己的速度)——本用例里 generate 反而更小。
        # 仍然成立、也仍然有意义的约束是:首 token 一定落在整条流结束之前。
        assert t["first_token"] <= t["total"]

    async def test_reports_usage_when_the_provider_gives_it(self, settings) -> None:
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        usage = Usage(prompt_tokens=100, completion_tokens=20, total_tokens=120)
        service = make_service(settings, retriever, FakeLLM(usage=usage))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.usage is not None
        assert done.usage.total_tokens == 120

    async def test_usage_is_none_when_the_provider_omits_it(self, settings) -> None:
        """流式下用量是**可选**的——网关不回声就是没有。

        不能在这里编一个数:token 用量的唯一可信来源是服务端的 usage,
        自己估一个填进去,指标面板就会显示一个看起来精确的错数字。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        service = make_service(settings, retriever, FakeLLM(usage=None))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.usage is None

    async def test_retrieval_metadata_travels_on_the_sources_event(self, settings) -> None:
        retriever = FakeRetriever(
            chunks=[chunk(1, "甲")],
            route_sizes={"vector": 20, "keyword": 3},
            reranked=True,
            degraded=False,
        )
        service = make_service(settings, retriever, FakeLLM())
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        sources_event = next(e for e in events if isinstance(e, SourcesEvent))
        assert sources_event.route_sizes["keyword"] == 3
        assert sources_event.reranked is True
        assert sources_event.degraded is False


class TestNoContextShortCircuit:
    """**检索为空时不调用模型,直接给固定答复。**

    这不是省钱的优化,是正确性:提示词里的「不要编造」是**措辞**层面的
    约束,而模型完全可以在「没有任何资料」的情况下用自己的知识把答案
    补得像个样子。用代码短路是**结构**层面的约束,不存在被绕过的可能。
    代价是这类问题得不到任何润色——对一个知识库问答工具来说,这是想要的。
    """

    async def test_llm_is_never_called(self, settings) -> None:
        llm = FakeLLM()
        service = make_service(settings, FakeRetriever(chunks=[]), llm)
        await collect(service, await service.prepare("问题", kb_ids=[1]))
        assert llm.calls == []

    async def test_still_emits_the_full_event_sequence(self, settings) -> None:
        """事件形状必须一致——前端不该为「没检索到」写一条特殊分支。"""
        service = make_service(settings, FakeRetriever(chunks=[]), FakeLLM())
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))
        assert [type(e).__name__ for e in events] == [
            "SourcesEvent",
            "TokenEvent",
            "DoneEvent",
        ]

    async def test_answer_says_it_found_nothing(self, settings) -> None:
        service = make_service(settings, FakeRetriever(chunks=[]), FakeLLM())
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))
        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.answer == NO_CONTEXT_ANSWER
        assert "没有找到" in done.answer
        assert done.cited == []
        assert done.grounded is False


class TestFailureHandling:
    async def test_llm_failure_becomes_an_error_event(self, settings) -> None:
        """**流一旦开始,HTTP 状态码就已经发出去了。**

        SSE 响应头在第一个字节时就已经提交,之后再抛异常没法变成 500——
        客户端只会看到连接莫名其妙断掉。所以生成期的故障必须作为
        `error` 事件送达,这是唯一能告诉用户「出事了」的通道。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        service = make_service(settings, retriever, FakeLLM(pieces=("甲",), fail_at=0))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        errors = [e for e in events if isinstance(e, ErrorEvent)]
        assert len(errors) == 1
        assert "上游模型炸了" in errors[0].message

    async def test_tokens_emitted_before_the_failure_are_kept(self, settings) -> None:
        """已经流出去的字收不回来,也不该假装没发生——拼一半的答案
        加上一条错误提示,比一个空白正文有用得多。"""
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        service = make_service(
            settings, retriever, FakeLLM(pieces=("前半", "后半"), fail_at=1)
        )
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        assert "".join(e.text for e in events if isinstance(e, TokenEvent)) == "前半"
        assert any(isinstance(e, ErrorEvent) for e in events)

    async def test_error_is_terminal_no_done_follows(self, settings) -> None:
        """出错后再补一个 done,前端会以为这是完整答案并把它存进历史。"""
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        service = make_service(settings, retriever, FakeLLM(pieces=("甲",), fail_at=0))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        assert not any(isinstance(e, DoneEvent) for e in events)
        assert isinstance(events[-1], ErrorEvent)

    async def test_sources_still_arrive_before_the_error(self, settings) -> None:
        """检索是成功的,引用信息就该给到——用户至少知道系统找过哪些资料。"""
        retriever = FakeRetriever(chunks=[chunk(1, "甲资料")])
        service = make_service(settings, retriever, FakeLLM(pieces=("甲",), fail_at=0))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        sources_event = next(e for e in events if isinstance(e, SourcesEvent))
        assert len(sources_event.sources) == 1

    async def test_an_error_carries_how_long_the_user_already_waited(self, settings) -> None:
        """**失败的那一轮也要有耗时**,否则它在面板上显示成瞬间返回。

        拿不到这个数的话,指标表里唯一缺耗时的行恰好是最慢的那些(超时、
        上游挂了),而 p95 会把它们当成 0 算进去——**越失败,面板越好看**。

        计时原点必须是 `prepare()` 的第一行(见 `PreparedAnswer.started_at`),
        不是流开始的那一刻:改写和检索都在那之前,多轮追问里一次改写就是
        一次完整的 LLM 往返。
        """
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        service = make_service(settings, retriever, FakeLLM(pieces=("甲",), fail_at=0))
        prepared = await service.prepare("问题", kb_ids=[1])
        prepared = replace(prepared, started_at=time.perf_counter() - 1.5)

        events = await collect(service, prepared)

        error = next(e for e in events if isinstance(e, ErrorEvent))
        assert error.elapsed_ms >= 1500.0


# ---------------------------------------------------------------------------
# 答案缓存接进编排
#
# 上面 `test_cache.py` 验的是缓存自身的正确性。**它有没有被接上、接在哪一步**,
# 是另一回事,而且是最容易出错的地方:
#
#   - 接在 `prepare` 之后 → 检索白跑了,省下的只有 LLM,而且流已经开始、错了
#     就回不去
#   - 接在 `stream` 里但**只在最后写** → 第一次问永远命中不了,缓存形同虚设
#   - 状态码/事件顺序与正常路径不一致 → 前端按命中与否走两条不同的渲染逻辑
#
# 所以这一组断言的是**编排**,不是缓存:假检索器数调用次数、假 LLM 数调用
# 次数,再看事件流。
# ---------------------------------------------------------------------------


@pytest.fixture
async def answer_cache(tmp_path):
    """答案缓存 + 与它配套的配置。**独立临时库**,绝不碰开发库。"""
    from app.core.config import Settings
    from app.db.base import Database
    from app.services.cache import AnswerCache, CacheStore, CorpusVersion

    cache_settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path / "data",
        CHROMA_DIR=tmp_path / "data" / "chroma",
        UPLOAD_DIR=tmp_path / "data" / "uploads",
        DATABASE_URL=f"sqlite:///{(tmp_path / 'data' / 'app.db').as_posix()}",
        DASHSCOPE_API_KEY="sk-test",
        CONTEXT_TOKEN_BUDGET=3000,
    )
    cache_settings.ensure_dirs()
    database = Database(cache_settings)
    await database.create_all()
    try:
        yield (
            AnswerCache(
                store=CacheStore(database),
                version=CorpusVersion(database),
                settings=cache_settings,
            ),
            cache_settings,
        )
    finally:
        await database.dispose()


class TestAnswerCacheIsWiredIntoRAG:
    async def test_a_repeated_question_never_reaches_the_llm(
        self, answer_cache, settings
    ) -> None:
        """**这一层存在的全部意义。** 问第二遍不该再花一次模型调用。"""
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        llm = FakeLLM(pieces=("售价 ", "4299 元 ", "[1]。"))
        service = make_service(settings, retriever, llm, answers=cache)

        for _ in range(2):
            await collect(service, await service.prepare("RX9070 多少钱", kb_ids=[1]))

        assert len(llm.calls) == 1

    async def test_a_hit_skips_retrieval_entirely(self, answer_cache, settings) -> None:
        """命中时**检索也不该跑**。

        只把 LLM 省掉的话,用户仍然要等编码 + 两路召回 + 重排(实测几百毫秒),
        而这恰恰是「同一个热门商品被反复问」时最能省下的一段。做法是把缓存
        查询放在 `prepare()` 的最前面:命中就直接返回,后面的步骤一步都不进。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        service = make_service(settings, retriever, FakeLLM(), answers=cache)

        await collect(service, await service.prepare("RX9070 多少钱", kb_ids=[1]))
        await collect(service, await service.prepare("RX9070 多少钱", kb_ids=[1]))

        assert len(retriever.calls) == 1

    async def test_a_follow_up_with_history_never_hits_the_cache(
        self, answer_cache, settings
    ) -> None:
        """**「RX9070 多少钱」和带着历史的同一句话,不是同一个问题。**

        单轮问一次把答案缓存起来,然后带着上下文再问一次一字不差的问题
        ——第二次必须重新生成。缓存只看得到问题文本、范围和配置,看不到
        历史,所以它无法区分这两者;**唯一的正确做法是不让它参与**。

        这条也不是假想的:用户在同一会话里把上一句重发一遍是很常见的操作
        (改了个错别字、或者上一条回答没看清),而两次的上下文不同、答案
        本来就可能不同。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        llm = FakeLLM()
        service = make_service(settings, retriever, llm, answers=cache)

        await collect(service, await service.prepare("RX9070 多少钱", kb_ids=[1]))
        await collect(
            service,
            await service.prepare("RX9070 多少钱", kb_ids=[1], history=_history()),
        )

        assert len(llm.calls) == 2

    async def test_a_hit_still_emits_sources_before_the_tokens(
        self, answer_cache, settings
    ) -> None:
        """事件顺序与正常路径**完全一致**。

        命中时直接跳过 sources、只发一个 done,功能上「也能渲染」——前端
        会走另一条分支,而那条分支平时几乎不跑,是 bug 的温床。而且用户
        会先看到答案、再看到依据,正好把这个产品的价值顺序倒过来。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        service = make_service(settings, retriever, FakeLLM(), answers=cache)

        await collect(service, await service.prepare("RX9070 多少钱", kb_ids=[1]))
        events = await collect(
            service, await service.prepare("RX9070 多少钱", kb_ids=[1])
        )

        kinds = [type(e).__name__ for e in events]
        assert kinds[0] == "SourcesEvent"
        assert "TokenEvent" in kinds
        assert kinds[-1] == "DoneEvent"

    async def test_a_hit_restores_the_sources_that_were_cited(
        self, answer_cache, settings
    ) -> None:
        """来源卡片要能原样还原——它是这个产品相对通用大模型的价值所在。

        命中时若来源丢了或只剩编号,答案正文里的 `[1]` 就成了一个点不开的
        锚点,而正文看起来完全正常。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        service = make_service(settings, retriever, FakeLLM(), answers=cache)

        first = await collect(service, await service.prepare("RX9070 多少钱", kb_ids=[1]))
        second = await collect(service, await service.prepare("RX9070 多少钱", kb_ids=[1]))

        first_sources = next(e for e in first if isinstance(e, SourcesEvent)).sources
        second_sources = next(e for e in second if isinstance(e, SourcesEvent)).sources
        assert second_sources == first_sources
        assert second_sources[0].content == "RX9070 售价 4299 元"

    async def test_a_hit_reports_no_usage(self, answer_cache, settings) -> None:
        """命中时 `usage` 必须是 `None`。

        **这一轮真的花了 0 个 token。** 把首次的用量原样报出去,成本看板上
        的消耗会随着缓存命中率上升而虚高——一个越优化数字越难看的指标,
        而且没人会想到去怀疑它。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        usage = Usage(prompt_tokens=800, completion_tokens=40, total_tokens=840)
        service = make_service(settings, retriever, FakeLLM(usage=usage), answers=cache)

        await collect(service, await service.prepare("问题", kb_ids=[1]))
        events = await collect(service, await service.prepare("问题", kb_ids=[1]))

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.usage is None

    async def test_the_hit_is_observable_and_distinct_from_the_retrieval_cache(
        self, answer_cache, settings
    ) -> None:
        """「这一轮是不是缓存给的」必须能从事件里读出来,而且要能分清是哪一层。

        只报一个 `cache_hit` 的话,「检索命中」和「答案命中」看起来一模一样
        ——而它们的排查方向完全相反:前者要问「检索结果对不对」,后者要问
        「这个问题凭什么能复用」。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        service = make_service(settings, retriever, FakeLLM(), answers=cache)

        first = await collect(service, await service.prepare("问题", kb_ids=[1]))
        second = await collect(service, await service.prepare("问题", kb_ids=[1]))

        first_done = next(e for e in first if isinstance(e, DoneEvent))
        second_done = next(e for e in second if isinstance(e, DoneEvent))
        assert first_done.answer_cache_hit is False
        assert second_done.answer_cache_hit is True
        # 答案命中的时候,来源也是从缓存里拿的——两个标记不能互相矛盾。
        assert second_done.cache_hit is True

    async def test_history_trimmed_away_by_the_budget_still_blocks_the_cache(
        self, answer_cache, settings
    ) -> None:
        """历史被 prompt 预算裁光,**仍然不能进缓存**。

        两个「历史条数」差一个字,判错的后果却是脏缓存:

        - `history_messages` = 塞进 prompt 的历史,会被 `HISTORY_TOKEN_BUDGET`
          和 `HISTORY_MAX_TURNS` 裁掉;
        - `history_supplied` = 用户这次**带了**历史,改写那一环读的是它
          (`rewrite_question` 用的是原始历史,不受 prompt 预算影响)。

        按前者判:预算小到把历史裁空时,一条**接着上下文**生成的答案会以
        「无历史」的键写进去;之后任何一个全新会话问出同一句话,拿到的都是
        它。按后者判才不会。这里把两个值都断言出来——用例的前提(裁空了)
        和被断言的行为(还是不进缓存)都必须是显式的。
        """
        cache, _cs = answer_cache
        tight = settings.model_copy(update={"HISTORY_MAX_TURNS": 0})
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        llm = FakeLLM(rewritten="改写后的问题")
        service = make_service(tight, retriever, llm, answers=cache)

        await collect(service, await service.prepare("问题", kb_ids=[1]))
        prepared = await service.prepare("问题", kb_ids=[1], history=_history())
        await collect(service, prepared)

        assert prepared.history_messages == 0, "预算没把历史裁空,这条用例的前提不成立"
        assert prepared.history_supplied == 2
        assert len(llm.calls) == 2, "按裁剪后的条数判缓存,上下文相关的答案被当成了无历史的"

    async def test_an_ungrounded_answer_is_not_cached(self, answer_cache, settings) -> None:
        """**「资料里没有」不进缓存。**

        这条固定答复不花模型调用,缓存它省不下什么;但它是**可恢复的**:
        用户换个问法、管理员补一份文档,同一句话的答案就该变。存下来之后,
        半小时内所有问同一句话的人都拿到那句「没找到」,而系统看起来一切
        正常。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[])
        service = make_service(settings, retriever, FakeLLM(), answers=cache)

        await collect(service, await service.prepare("库外的问题", kb_ids=[1]))
        await collect(service, await service.prepare("库外的问题", kb_ids=[1]))

        assert len(retriever.calls) == 2

    async def test_a_failed_generation_is_not_cached(self, answer_cache, settings) -> None:
        """生成中途炸了,**半截答案不许进缓存**。

        存进去的话,接下来半小时内所有问这句话的人都拿到那段被截断的文字,
        而且它带着完整来源、看起来像一份正常回答——比当场报错糟糕得多,
        因为报错至少会让人重试。
        """
        cache, _cs = answer_cache
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        service = make_service(
            settings, retriever, FakeLLM(pieces=("前半", "后半"), fail_at=1), answers=cache
        )

        first = await collect(service, await service.prepare("问题", kb_ids=[1]))
        second = await collect(service, await service.prepare("问题", kb_ids=[1]))

        assert any(isinstance(e, ErrorEvent) for e in first)
        assert any(isinstance(e, ErrorEvent) for e in second), "失败的那一轮把半截答案缓存了"

    async def test_the_cache_is_not_consulted_when_switched_off(
        self, answer_cache, settings, tmp_path
    ) -> None:
        """关掉之后必须**每次都重新生成**,而不是「还在用之前存下的」。"""
        from app.services.cache import AnswerCache

        cache, cache_settings = answer_cache
        disabled = AnswerCache(
            store=cache._store, version=cache._version, settings=cache_settings, enabled=False
        )
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        llm = FakeLLM()
        service = make_service(settings, retriever, llm, answers=disabled)

        await collect(service, await service.prepare("问题", kb_ids=[1]))
        await collect(service, await service.prepare("问题", kb_ids=[1]))

        assert len(llm.calls) == 2


# ---------------------------------------------------------------------------
# 语义缓存接进编排
# ---------------------------------------------------------------------------


class _ScriptedEmbedder:
    """按文本回放预设向量——**由用例决定哪两句算「近」**。

    这里不能用 mock embedder(字符 n-gram 哈希):它算的是字符重叠,而
    本组用例要钉的是「编排有没有在正确的位置调用这一层」,不是分词质量。
    真让它去判相似度,用例的成败会取决于几个中文字的笔画重合度,失败
    信息也指向一个不存在的问题。
    """

    def __init__(self, vectors: dict[str, list[float]]) -> None:
        self._vectors = vectors

    model_id = "scripted"

    @property
    def dim(self) -> int:
        return len(next(iter(self._vectors.values())))

    async def embed_query(self, text: str) -> list[float]:
        return list(self._vectors[text])

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [list(self._vectors[t]) for t in texts]


def _axis(angle_cos: float) -> list[float]:
    """与参照轴 `[1,0]` 夹角余弦恰为 `angle_cos` 的单位向量。"""
    return [angle_cos, max(0.0, 1.0 - angle_cos**2) ** 0.5]


@pytest.fixture
async def semantic_cache(tmp_path):
    """语义缓存 + 配套配置 + 脚本化编码器。**独立临时库**,绝不碰开发库。"""
    from app.core.config import Settings
    from app.db.base import Database
    from app.services.cache import CorpusVersion, SemanticCache

    cache_settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path / "data",
        CHROMA_DIR=tmp_path / "data" / "chroma",
        UPLOAD_DIR=tmp_path / "data" / "uploads",
        DATABASE_URL=f"sqlite:///{(tmp_path / 'data' / 'app.db').as_posix()}",
        DASHSCOPE_API_KEY="sk-test",
        CONTEXT_TOKEN_BUDGET=3000,
        # 这一层默认关闭(见 `Settings.SEMANTIC_CACHE_ENABLED`),要验它就得
        # 显式打开——顺带证明那个开关真的在读。
        SEMANTIC_CACHE_ENABLED=True,
    )
    cache_settings.ensure_dirs()
    database = Database(cache_settings)
    await database.create_all()
    embedder = _ScriptedEmbedder(_SEMANTIC_VECTORS)
    try:
        yield (
            SemanticCache(
                db=database,
                version=CorpusVersion(database),
                settings=cache_settings,
                embedder=embedder,
            ),
            cache_settings,
            embedder,
        )
    finally:
        await database.dispose()


# 参照轴取 `[1,0]`,于是每条向量与它的余弦**就是**上面那个数,不必手算夹角。
#   质保多久 ↔ 保修几年 = 0.99(近义改写,必须命中)
#   质保多久 ↔ 显示器多少钱 = 0.30(换了问的东西,必须不命中)
_SEMANTIC_VECTORS: dict[str, list[float]] = {
    "RX9070 显卡质保多久": _axis(1.0),
    "RX9070 显卡保修几年": _axis(0.99),
    "显示器多少钱": _axis(0.30),
    "它多少钱": _axis(1.0),
}


class TestSemanticCacheIsWiredIntoRAG:
    async def test_a_near_duplicate_question_never_reaches_the_llm(
        self, semantic_cache, settings
    ) -> None:
        """**这一层存在的全部理由,在编排层再钉一遍。** 换个问法不该再花一次模型调用。

        只测 `SemanticCache` 本身是不够的:那个类可能是完好的,却根本
        没被接进 `prepare()`——于是「实现了但没生效」,而它在指标上看起来
        和「命中率天然就低」一模一样。
        """
        cache, _cs, _emb = semantic_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        llm = FakeLLM(pieces=("质保 ", "三年 ", "[1]。"))
        service = make_service(settings, retriever, llm, semantics=cache)

        await collect(service, await service.prepare("RX9070 显卡质保多久", kb_ids=[1]))
        await collect(service, await service.prepare("RX9070 显卡保修几年", kb_ids=[1]))

        assert len(llm.calls) == 1

    async def test_a_hit_skips_retrieval_entirely(self, semantic_cache, settings) -> None:
        """命中时检索也不该跑——省下的必须是整段等待,不只是模型调用。"""
        cache, _cs, _emb = semantic_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        service = make_service(settings, retriever, FakeLLM(), semantics=cache)

        await collect(service, await service.prepare("RX9070 显卡质保多久", kb_ids=[1]))
        before = len(retriever.calls)
        await collect(service, await service.prepare("RX9070 显卡保修几年", kb_ids=[1]))

        assert len(retriever.calls) == before

    async def test_a_dissimilar_question_still_reaches_the_llm(
        self, semantic_cache, settings
    ) -> None:
        """**这一层唯一真正的风险的编排版**:把另一个问题的答案发出去。

        「命中率」这个指标看不出假阳性——它只会显示缓存工作得很好。
        """
        cache, _cs, _emb = semantic_cache
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        llm = FakeLLM()
        service = make_service(settings, retriever, llm, semantics=cache)

        await collect(service, await service.prepare("RX9070 显卡质保多久", kb_ids=[1]))
        await collect(service, await service.prepare("显示器多少钱", kb_ids=[1]))

        assert len(llm.calls) == 2

    async def test_a_semantic_hit_is_reported_as_an_answer_cache_hit(
        self, semantic_cache, settings
    ) -> None:
        """**复用了一份别人生成的答案,指标上必须看得出来。**

        `answer_cache_hit` 为假的话,这个答案在埋点里与「模型刚刚生成」
        完全一样——于是「缓存给出错答案」这件事永远查不到源头。
        """
        cache, _cs, _emb = semantic_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        service = make_service(settings, retriever, FakeLLM(), semantics=cache)

        await collect(service, await service.prepare("RX9070 显卡质保多久", kb_ids=[1]))
        events = await collect(
            service, await service.prepare("RX9070 显卡保修几年", kb_ids=[1])
        )

        done = next(e for e in events if isinstance(e, DoneEvent))
        assert done.answer_cache_hit is True
        assert done.cache_hit is True

    async def test_a_question_with_history_is_neither_served_nor_cached(
        self, semantic_cache, settings
    ) -> None:
        """带历史的提问一律绕过这一层,读和写两侧都要挡。

        在答案缓存上这是个谨慎的选择;在这里它是**必须的**——「它多少钱?」
        在两个会话里的向量完全相同,比措辞不同还更容易跨会话命中,而
        `[1,0]` 那种完全相同的向量正是这一层最擅长匹配的东西。
        """
        cache, _cs, _emb = semantic_cache
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        llm = FakeLLM(rewritten="它(显卡)多少钱")
        service = make_service(settings, retriever, llm, semantics=cache)

        for _ in range(2):
            await collect(
                service,
                await service.prepare("它多少钱", kb_ids=[1], history=_history()),
            )

        assert len(llm.calls) == 2

    async def test_the_layer_is_off_when_the_switch_is_off(
        self, semantic_cache, settings
    ) -> None:
        """**默认关闭是产品决定,不是漏配**——关掉后每次都必须真的重新生成。

        断言 `answer_cache_hit is False` 是不够的:一个「每次都查、每次都
        不命中」的实现同样满足它,而那比没有缓存更慢。
        """
        from app.services.cache import SemanticCache

        cache, cache_settings, embedder = semantic_cache
        disabled = SemanticCache(
            db=cache._db,
            version=cache._version,
            settings=cache_settings,
            embedder=embedder,
            enabled=False,
        )
        retriever = FakeRetriever(chunks=[chunk(1, "甲")])
        llm = FakeLLM()
        service = make_service(settings, retriever, llm, semantics=disabled)

        await collect(service, await service.prepare("RX9070 显卡质保多久", kb_ids=[1]))
        await collect(service, await service.prepare("RX9070 显卡保修几年", kb_ids=[1]))

        assert len(llm.calls) == 2

    async def test_a_corpus_write_invalidates_it(self, semantic_cache, settings) -> None:
        """知识库一写入,已缓存的答案必须立刻读不到。

        这一层比另外两层更要紧:文档删掉之后,缓存里那条答案的来源编号
        指向一个不存在的片段,而它照常显示成一张可展开的来源卡片。
        """
        from app.services.cache import CorpusVersion

        cache, _cs, _emb = semantic_cache
        retriever = FakeRetriever(chunks=[chunk(1, "RX9070 售价 4299 元")])
        llm = FakeLLM(pieces=("质保 ", "三年 ", "[1]。"))
        service = make_service(settings, retriever, llm, semantics=cache)

        await collect(service, await service.prepare("RX9070 显卡质保多久", kb_ids=[1]))
        # 代次由**库里那份**推进:服务用的是缓存自己持有的连接,
        # 换个新实例去 bump 才能证明它读的是库里的值,而不是内存里的。
        await CorpusVersion(cache._db).bump()
        await collect(service, await service.prepare("RX9070 显卡保修几年", kb_ids=[1]))

        assert len(llm.calls) == 2
