"""问答接口(SSE 流式)。

**顺序是这个文件里最重要的设计:检索先做完,再构造 `EventSourceResponse`。**

`EventSourceResponse` 一旦开始产出,响应头就已经提交。那之后无论抛什么,
都变不成 500——客户端只会看到连接莫名断掉。所以:

    检索 + 组装 + 提示词   ← 出错 → 正常的 JSON 错误,带正确的状态码
    ──────────────────── 响应头在此处提交 ────────────────────
    流式生成               ← 出错 → error 事件(这是唯一可用的通道)

把检索挪进生成器里看起来更「流式」,代价是丢掉上面那半段:向量库一挂,
用户拿到的是一个 200 开头的空连接,而不是「服务不可用,请稍后重试」。

**事件协议**(先 sources,再 token,最后 done):

    event: sources  data: {sources, route_sizes, reranked, degraded}
    event: token    data: {text}
    event: done     data: {answer, cited, usage, latency_ms, stage_timings, grounded,
                           search_query, history_messages, cache_hit, answer_cache_hit,
                           session_id, message_id}
    event: error    data: {code, message, elapsed_ms}   ← 终态,之后不再有事件

**先发 sources 再发 token**,是为了让用户提问后立刻看到「依据是哪几条」,
而不是等答案写完才知道这段话是从哪来的——而那正是这个产品相对通用
大模型的价值所在。
"""

from __future__ import annotations

import contextlib
import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, replace
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sse_starlette.sse import EventSourceResponse

from app.core.config import Settings
from app.core.deps import (
    enforce_chat_rate_limit,
    get_current_user,
    get_database_dep,
    get_db,
    get_settings_dep,
)
from app.core.logging import get_trace_id, log_extra
from app.db.base import Database
from app.db.models import ChatSession, RequestMetric, User
from app.db.repositories.document import ChunkRepository
from app.indexing.vectorstore import build_vector_store
from app.providers.embedding import build_embedder
from app.providers.llm import build_llm
from app.providers.reranker import build_reranker
from app.schemas.chat import ChatFilters, ChatRequest
from app.services.metrics import MetricsService
from app.services.rag import (
    DoneEvent,
    ErrorEvent,
    RagEvent,
    RAGService,
    SourcesEvent,
    TokenEvent,
)
from app.services.retrieval import RetrievalFilters
from app.services.session import SessionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["问答"])


# ---------------------------------------------------------------------------
# 编码
# ---------------------------------------------------------------------------

# 事件名在这里集中登记,而不是散在 if/elif 里。
# 名字是**对前端的公开契约**:前端按 `event:` 分支,改名等于破坏接口,
# 而漏掉一个分支的后果是那个事件被静默丢弃(SSE 客户端通常只忽略未知事件)。
_EVENT_NAMES: dict[type, str] = {
    SourcesEvent: "sources",
    TokenEvent: "token",
    DoneEvent: "done",
    ErrorEvent: "error",
}


def encode_event(event: RagEvent) -> dict[str, str]:
    """领域事件 → sse-starlette 的一帧。"""
    name = _EVENT_NAMES.get(type(event))
    if name is None:  # pragma: no cover - 新增事件类型时立刻炸,而不是静默丢弃
        raise ValueError(f"未登记的事件类型:{type(event).__name__}")

    # `ensure_ascii=False`:中文正文在报文里保持可读,排查线上问题时
    # 可以直接用 curl 看,不必先过一遍转码。
    return {"event": name, "data": json.dumps(asdict(event), ensure_ascii=False)}


async def encode_stream(events: AsyncIterator[RagEvent]) -> AsyncIterator[dict[str, str]]:
    """把领域事件流逐条编码。

    **必须写成 async 生成器。** 对异步迭代器套同步的生成器表达式
    (`(f(e) for e in agen)`)不会报错,它会构造出一个「迭代一个 async
    generator」的同步生成器,直到 EventSourceResponse 真正开始迭代时才炸,
    而那时响应头已经发出去了——错误表现为「连接突然断掉」,排查成本极高。
    """
    async for event in events:
        yield encode_event(event)


# ---------------------------------------------------------------------------
# 依赖装配
# ---------------------------------------------------------------------------


def get_rag_service(
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
    db: Database = Depends(get_database_dep),
) -> RAGService:
    """构造本次请求用的 RAG 服务。

    **做成依赖而不是在函数体里直接 new**,是为了让测试能整体换掉它
    (见 `test_chat_api.py` 的故障路径用例)。这与 `get_settings_dep`
    存在的理由是同一个。

    **三层缓存各套在自己的那一层上**:

    - `CachedEmbedder` 包住编码器——它在这条链路里省下的是一次**网络往返**,
      比省下几次数据库查询值得多;
    - `CachedRetriever` 包住检索(编码 + 两路召回 + 重排);
    - `AnswerCache` 交给 `RAGService`,它命中时**连检索都不跑**。

    为什么答案缓存不做成「包住 `RAGService` 的装饰器」:那样只能拦截输出,
    拦截不了输入——生成已经发生了。要真的省下模型调用,判断必须在
    `prepare()` 的最前面,而那是 `RAGService` 内部的位置。

    为什么向量缓存**套在编码器上而不是套在检索里**:同一段正文的向量在
    问答和入库两条路径上都要用,而入库根本不经过检索。套在编码器上,
    两层收益就都拿到了(见 `ingest.build_ingest_service`)。

    三者都拿 `db` 而不是本次请求的 `session`:缓存的写入发生在流式响应
    期间,而那时请求事务的状态由路由决定,缓存不该被卷进它的提交或回滚。

    **这里用独立的会话是安全的,但安全来自顺序,不是来自缓存本身。**
    SQLite 同一时刻只允许一个写事务,所以「请求事务持着写锁 + 缓存另开会话
    要写」= 死锁(摄取那边就是这么栽的,见 `ingest.build_ingest_service`)。这条路径
    上不会发生,因为请求事务在 `prepare()` 之前就已经提交(问题的写入),
    之后到流结束都不再写——落库统一在 `_persist_after` 的 `DoneEvent` 分支,
    而答案缓存的写入还在它之前(见 `RAGService._remember`)。**动这里的顺序
    要连带想一下缓存**:把落库提前到检索之前,就会把这条路径变成和摄取一样。
    """
    from app.services.cache import (
        AnswerCache,
        CachedEmbedder,
        CachedRetriever,
        CacheStore,
        CorpusVersion,
        SemanticCache,
    )
    from app.services.retrieval import RetrievalService

    store, version = CacheStore(db), CorpusVersion(db)
    embedder = CachedEmbedder(inner=build_embedder(settings), store=store, settings=settings)

    retrieval = RetrievalService(
        settings=settings,
        embedder=embedder,
        vector_store=build_vector_store(settings),
        chunk_source=ChunkRepository(session),
        reranker=build_reranker(settings),
    )

    return RAGService(
        settings=settings,
        retrieval=CachedRetriever(
            inner=retrieval, store=store, version=version, settings=settings
        ),
        llm=build_llm(settings),
        answers=AnswerCache(store=store, version=version, settings=settings),
        # **同一个 `CachedEmbedder` 实例交给两层用**,不是图省事:语义缓存
        # 每次查询都要一次问题向量,而这次编码要由向量缓存吸收,它才不至于
        # 把最常见的路径拖慢(见 `SemanticCache` 的类文档)。换一个裸的
        # `build_embedder()` 在这里,功能一模一样、测试全绿,只是每次提问
        # 都多一次网络编码。
        #
        # 它自己带开关且**默认关闭**(见 `Settings.SEMANTIC_CACHE_ENABLED`):
        # 另外两层给错了只是快慢,这一层给错了是「另一个问题的答案」。
        semantics=SemanticCache(
            db=db, version=version, settings=settings, embedder=embedder
        ),
    )


def _to_filters(filters: ChatFilters | None) -> RetrievalFilters | None:
    if filters is None:
        return None
    return RetrievalFilters(
        category=filters.category,
        price_min=filters.price_min,
        price_max=filters.price_max,
    )


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------


def _build_metric(
    *,
    trace_id: str,
    user_id: int,
    session_id: int | None,
    history_messages: int,
    outcome: str,
    total_ms: float,
    event: DoneEvent | None = None,
) -> RequestMetric:
    """从终态事件组装一行指标。

    `event` 为 `None` 表示这一轮**根本没生成出回答**——检索期就失败了,
    或者生成中途报错。那种情况下所有描述「答案长什么样」的字段保持
    默认值,读的人一眼能看出这行不是一次正常的问答。

    失败轮一律 `grounded=False`。这不是「它不 grounded」,而是**它没有
    产出任何可溯源的东西**——一个没有答案的轮次不该把「有据可依的比例」
    往上抬。

    失败轮的 `first_token_ms` 也一律为空,这不是遗漏:`ErrorEvent` 不带
    首 token 时刻,而面板上那个分位数说的是「正常的一轮多久开始出字」,
    失败轮慢在哪由 `total_ms` 说明。
    """
    timings = dict(event.stage_timings) if event is not None else {}
    usage = event.usage if event is not None else None
    return RequestMetric(
        trace_id=trace_id,
        user_id=user_id,
        session_id=session_id,
        outcome=outcome,
        total_ms=total_ms,
        first_token_ms=timings.get("first_token"),
        stage_timings=timings,
        retrieval_cache_hit=event.cache_hit if event is not None else False,
        answer_cache_hit=event.answer_cache_hit if event is not None else False,
        grounded=event.grounded if event is not None else False,
        history_messages=history_messages,
        prompt_tokens=usage.prompt_tokens if usage is not None else 0,
        completion_tokens=usage.completion_tokens if usage is not None else 0,
        # **`usage is None` 时留 `None`,不补 0。** 没有 `usage` 意味着
        # **根本没发过 LLM 请求**(命中答案缓存、或检索为空走固定答复,
        # 见 `RAGService.stream` 的两个早退分支)——我们没问过平台,自然
        # 没有读数。补 0 会被读成「平台说这次一条都没命中」,把这一窗的
        # 命中率拉成 0%,看起来像缓存坏了,而真相是这一窗没有带 prompt
        # 的调用。`prompt_tokens` 上面补 0 则是对的:0 是**这一轮花了多少**
        # 的如实回答,而缓存读数是**平台说了什么**。
        cached_prompt_tokens=usage.cached_tokens if usage is not None else None,
    )


async def _record_metric(db: AsyncSession, metric: RequestMetric) -> None:
    """写一条指标并提交。**任何失败都只记日志,绝不外抛。**

    走 `MetricsService.on(db)` 借用本轮会话,而不是让它自己开一个:这段
    代码跑在响应头已经提交之后,会话里可能还压着刚写入的助手消息,另开
    连接去写拿不到 SQLite 的写锁(见 `cache` 模块头部那段)。提交时机
    因此归这里——借用来的会话从不自己提交。

    **调用点必须排在消息提交之后。** 反过来的话,指标一失败就得回滚,
    而那次回滚会把用户刚看到的回答从历史里抹掉,两件事的代价完全不对等。
    """
    try:
        await MetricsService.on(db).record(metric)
        await db.commit()
    except Exception:
        logger.exception("指标写入失败", extra=log_extra(trace_id=metric.trace_id))
        # 回滚只可能丢掉这条指标本身:调用点保证消息已经提交过了。
        with contextlib.suppress(Exception):
            await db.rollback()


async def _persist_after(
    events: AsyncIterator[RagEvent],
    *,
    service: SessionService,
    chat_session: ChatSession | None,
    db: AsyncSession,
    trace_id: str,
    user_id: int,
    history_messages: int,
) -> AsyncIterator[RagEvent]:
    """转发事件流,并在终态事件上落库 + 记指标。

    **为什么落库放在流的尾巴上而不是生成之前**:答案在流跑完之前根本不存在。
    但用户的问题在生成**之前**就已经写好了(见 `chat_stream`),所以这里
    只补助手那半条。

    **落库失败不能影响已经发出去的内容。** 响应头早就提交了,此刻再抛异常
    只会让客户端看到连接莫名断掉——而它其实收到了一个完整的回答。所以
    这里吞掉异常只记日志:用户拿到了答案,历史少一条,是可接受的降级;
    反过来让整个回答作废,不可接受。

    **无会话时也照样走这里。** 「随口一问」是另一条分支(不落消息、不进
    历史),把埋点挂在落库那一步上的话,这条路径会整条消失在面板之外
    ——而它恰恰是没有登录态压力、最容易被脚本刷的那条。所以
    `chat_session` 是可选的,落库跳过、指标照记。
    """
    pieces: list[str] = []
    # 来源快照从 `sources` 事件里捞。**不能从 `done` 事件里取**——它不带这个
    # 字段。来源在流的最前面就已经发出去过一次,这里只是把它留下来落库。
    sources: list[dict[str, Any]] = []
    finished = False
    error: str | None = None
    session_id = chat_session.id if chat_session is not None else None

    async for event in events:
        if isinstance(event, TokenEvent):
            pieces.append(event.text)
            yield event
        elif isinstance(event, SourcesEvent):
            sources = [asdict(s) for s in event.sources]
            yield event
        elif isinstance(event, DoneEvent):
            finished = True
            if chat_session is not None:
                try:
                    message = await service.append_answer(
                        chat_session,
                        content=event.answer,
                        sources=sources,
                        cited=event.cited,
                        prompt_tokens=event.usage.prompt_tokens if event.usage else 0,
                        completion_tokens=event.usage.completion_tokens if event.usage else 0,
                        latency_ms=event.latency_ms,
                    )
                    await db.commit()
                    event = replace(event, session_id=session_id, message_id=message.id)
                except Exception:
                    logger.exception(
                        "问答落库失败", extra=log_extra(session_id=session_id)
                    )
                    event = replace(event, session_id=session_id)
            await _record_metric(
                db,
                _build_metric(
                    trace_id=trace_id,
                    user_id=user_id,
                    session_id=session_id,
                    history_messages=history_messages,
                    outcome="done",
                    total_ms=event.latency_ms,
                    event=event,
                ),
            )
            yield event
        elif isinstance(event, ErrorEvent):
            error = event.message
            # **失败的那一轮同样占了一条连接、同样让用户等了这么久。**
            # 只在成功时记一笔的话,最慢的那部分(慢到出错的)会整个从
            # 分母里消失,于是 p95 越优化越好看,而用户感受不到。
            await _record_metric(
                db,
                _build_metric(
                    trace_id=trace_id,
                    user_id=user_id,
                    session_id=session_id,
                    history_messages=history_messages,
                    outcome="error",
                    total_ms=event.elapsed_ms,
                ),
            )
            yield event
        else:
            yield event

    if finished:
        return

    # 流正常耗尽但没有 done:生成期出过错(`ErrorEvent` 已发)或因异常中断。
    # **半截答案照样落库**——用户已经在屏幕上看到它了,库里却没有的话,
    # 下次翻开历史会觉得「我记错了」。
    partial = "".join(pieces)

    # 但**一个字都没流出来时不落库**。存一条空的助手消息看着像是「记录更
    # 完整」,实际有两处害处:历史列表里多一条点开什么都没有的空行;更糟的
    # 是它会经 `build_history` 进入后续几轮的 prompt——模型看到自己「说过」
    # 一句空话,那是纯噪音,还挤占本就不多的历史预算。用户看到的是「提问在、
    # 回答没有」,而这**就是**事实:这一轮确实什么都没答出来。
    #
    # 走到这里说明生成期出过错(`ErrorEvent` 已发),而它的指标在发出时
    # 就记过了。**客户端中途断连不会走到这里**——那时 `CancelledError` 从
    # 上面那个 `async for` 直接穿过去,断连的一轮因此不留在指标里:
    # 它没有终态,也就没有一个诚实的「耗时」可以填。
    if not partial or chat_session is None:
        return
    try:
        await service.append_answer(
            chat_session,
            content=partial,
            error=error or "生成被中断",
        )
        await db.commit()
    except Exception:
        logger.exception("失败轮次的落库也失败了", extra=log_extra(session_id=session_id))


@router.post(
    "/stream",
    summary="知识库问答(SSE 流式,先推引用再推答案)",
    response_class=EventSourceResponse,
    # 用户级限流**只挂在这一个端点上**。理由写在依赖自己的文档里 ——
    # 一句话:这里是唯一花钱的那条路,而全局挂会把知识库页的轮询打成 429。
    dependencies=[Depends(enforce_chat_rate_limit)],
)
async def chat_stream(
    payload: ChatRequest,
    service: RAGService = Depends(get_rag_service),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
    user: User = Depends(get_current_user),
) -> EventSourceResponse:
    """检索 → 组装 → 流式生成。**检索在返回响应之前完成。**

    `kb_ids` 为 `None` 表示不限制范围:有会话时用会话记住的范围,
    没有会话时是全部知识库。空列表表示一个都不许看,会直接短路成
    「没有找到相关资料」——**空列表是显式值,不会被会话范围覆盖过去**。

    **当前版本所有登录用户都能读全部知识库**——需求 6 只把「知识库管理」
    限定给管理员,并没有要求按库授权。将来要做按库 ACL,改动点在这里:
    把 `payload.kb_ids` 与当前用户的可见库取交集,空交集就退化成空列表。
    这件事必须在**服务端**做,前端传来的范围只能当作「上限」而不是「授权」。
    """
    sessions = SessionService(session)
    chat_session: ChatSession | None = None
    history = []

    if payload.session_id is not None:
        # **归属校验在这里。** 拿不到就 404,且发生在响应头提交之前,
        # 所以它能变成一个正常的 JSON 错误而不是半截 SSE 流。
        chat_session = await sessions.require_owned(payload.session_id, user_id=user.id)
        # 先把历史读出来,再写这一问——顺序反了的话,刚写进去的问题会
        # 作为「历史」出现在自己的 prompt 里,而模型会看到两个一样的问题。
        history = await sessions.history(
            chat_session, limit=settings.HISTORY_MAX_TURNS * 2 + 4
        )
        await sessions.append_question(chat_session, payload.question)
        await session.commit()

    # 会话记住的范围只在调用方**没有明确指定**时生效。`[]` 是明确指定,
    # 必须原样传下去——把它当成 falsy 忽略掉,一个没有权限的用户就会被
    # 悄悄升级成「搜索全部知识库」。
    kb_ids = payload.kb_ids
    if kb_ids is None and chat_session is not None:
        kb_ids = chat_session.kb_ids

    logger.info(
        "问答请求",
        extra=log_extra(
            user_id=user.id,
            question=payload.question[:80],
            session_id=payload.session_id,
            kb_ids=kb_ids,
            history=len(history),
            filtered=payload.filters is not None,
        ),
    )

    # **异常在这里抛出,此时响应头还没发出去**,所以它会走 `AppError`
    # 处理器变成一个带正确状态码的 JSON 响应。见模块文档。
    #
    # 计时原点取在 `prepare()` **之前**,而不是 `prepared.started_at`——
    # 失败的那些轮次恰恰没有 `prepared`,而那正是这一段存在的理由。
    # 两者的差只是 `prepare()` 里最开始那几行,可以忽略。
    asked_at = time.perf_counter()
    try:
        prepared = await service.prepare(
            payload.question,
            kb_ids=kb_ids,
            filters=_to_filters(payload.filters),
            history=history,
        )
    except Exception:
        # **这一段是最容易漏报的失败。** 它发生在流开始之前,连
        # `_persist_after` 都碰不到,而它又是最慢、最贵、最可能因为额度
        # 耗尽而出现的一类失败(向量库或模型调用挂了)。按「在流尾记一笔」
        # 来埋点的话,它一次都不会出现在面板上,而面板上「p95 很健康」与
        # 「有一半请求根本没走到这里」可以同时成立。
        await _record_metric(
            session,
            _build_metric(
                trace_id=get_trace_id(),
                user_id=user.id,
                session_id=chat_session.id if chat_session is not None else None,
                history_messages=len(history),
                outcome="error",
                total_ms=(time.perf_counter() - asked_at) * 1000.0,
            ),
        )
        raise

    events = service.stream(prepared)
    # **没有会话时也包一层**:埋点不能挂在落库那一步上,否则「随口一问」
    # 这条路径整条消失在面板之外(见 `_persist_after`)。
    events = _persist_after(
        events,
        service=sessions,
        chat_session=chat_session,
        db=session,
        trace_id=get_trace_id(),
        user_id=user.id,
        history_messages=len(history),
    )
    return EventSourceResponse(encode_stream(events))
