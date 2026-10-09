"""RAG 编排:检索 → 组装 → 提示词 → 流式生成 → 引用溯源。

    检索(向量+BM25→RRF→重排)
        ▼
    按 token 预算组装上下文        ← 编号在这里产生
        ▼
    构造提示词(资料进 user 消息)
        ▼
    ── sources 事件先于一切 token 发出 ──
        ▼
    流式生成 → token 事件 → done 事件

**为什么把编排单独一层,而不是塞进路由**:这里有三处静默失效点(编号与
来源错位、检索为空仍调用模型、生成期异常打断整条流),每一处都不会抛异常、
只会让结果悄悄变差。独立成层才能把每一处单独钉住。

**检索在 `prepare()` 里做完,`stream()` 不碰数据库。** 两个理由:
① 检索期的异常发生在响应头发出**之前**,还能变成一个带正确状态码的 JSON
   错误;一旦开始 SSE,就只能作为 `error` 事件送达(见 `stream` 的注释)。
② 请求级数据库会话的生命周期与流式响应的生命周期关系微妙,让生成阶段
   完全不依赖它,就不必去赌哪个版本的 FastAPI 在何时关会话。
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from app.core.config import Settings
from app.core.logging import log_extra
from app.db.models import Message
from app.providers.base import LLMProvider, Usage
from app.services.citations import extract_citations
from app.services.context import ContextBundle, RetrievedChunk, assemble_context
from app.services.history import build_history, rewrite_question
from app.services.prompts import build_messages
from app.services.retrieval import RetrievalFilters, RetrievalResult

if TYPE_CHECKING:
    # 只在类型标注里用到(`from __future__ import annotations` 让标注不求值),
    # 于是这一层在导入期完全不依赖缓存层——两个模块谁先被导入都不会出问题。
    from app.services.cache import AnswerCache, CachedAnswer, SemanticCache

logger = logging.getLogger(__name__)

# 检索为空时的固定答复。
#
# **这里刻意不走模型。** 提示词里的「不要编造」是措辞层面的约束,而模型
# 完全可以在手上一份资料都没有的情况下,用自己的知识把答案补得像个样子——
# 读起来和真的一样,用户没有任何办法分辨。用代码短路是结构层面的约束,
# 绕过它需要改代码,而不是靠一句更严厉的提示词。
NO_CONTEXT_ANSWER = (
    "资料中没有找到相关内容。可以换一种问法,或者确认所选知识库中是否包含这类资料。"
)


class Retriever(Protocol):
    """RAG 层需要检索层提供的能力。

    只声明这一个方法而不是直接依赖 `RetrievalService`:检索质量由
    `test_retrieval*.py` 负责,这里的测试要能把检索换成回放器,
    否则一个编排 bug 会被检索的噪音淹没。`RetrievalService` 结构上已满足。
    """

    async def retrieve(
        self,
        query: str,
        *,
        kb_ids: list[int] | None,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> RetrievalResult: ...


# ---------------------------------------------------------------------------
# 引用载荷
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourceRef:
    """一条被引用的资料。`index` 就是正文里 `[index]` 的那个数字(从 1 起)。

    `content` 是**完整正文**而不是摘要:前端要支持「点开来源卡片看原文」,
    而摘要会让用户看不到支撑结论的那半句。一个分块最多几百 token,
    五条加起来也就几 KB,换 SSE 连接上完全划算。

    打分明细(vector_rank / keyword_rank / rerank_score)一并带上,是因为
    「这条是向量路找到的还是关键词路找到的」在融合之后就再也算不出来了,
    而它正是调检索参数时唯一有用的信息。
    """

    index: int
    chunk_id: int
    doc_id: int
    doc_name: str
    kb_id: int
    kb_name: str
    chunk_index: int
    content: str
    category: str | None = None
    price: float | None = None
    rerank_score: float | None = None
    fusion_score: float | None = None
    vector_rank: int | None = None
    keyword_rank: int | None = None

    @classmethod
    def of(cls, index: int, piece: RetrievedChunk) -> SourceRef:
        return cls(
            index=index,
            chunk_id=piece.chunk_id,
            doc_id=piece.doc_id,
            doc_name=piece.doc_name,
            kb_id=piece.kb_id,
            kb_name=piece.kb_name,
            chunk_index=piece.chunk_index,
            content=piece.content,
            category=piece.category,
            price=piece.price,
            rerank_score=piece.rerank_score,
            fusion_score=piece.fusion_score,
            vector_rank=piece.vector_rank,
            keyword_rank=piece.keyword_rank,
        )


# ---------------------------------------------------------------------------
# 事件
#
# 服务产出的是**领域事件**,不是 SSE 的字节串。传输格式(怎么写成
# `event: x\ndata: {...}`)是路由层的事——分开之后,事件本身可以脱离
# HTTP 单测,换传输(WebSocket、gRPC 流)也不必动这一层。
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SourcesEvent:
    """引用面板的全部内容。**在第一个 token 之前发出。**"""

    sources: list[SourceRef]
    route_sizes: dict[str, int] = field(default_factory=dict)
    reranked: bool = False
    degraded: bool = False


@dataclass(frozen=True, slots=True)
class TokenEvent:
    text: str


@dataclass(frozen=True, slots=True)
class DoneEvent:
    """终态。**它的出现本身就是「这轮正常结束了」的信号。**

    带完整 `answer` 不是冗余:SSE 没有投递保证,客户端无法区分「流正常
    结束」与「连接中途断了」。有了完整答案,断流时前端至少能把状态拼回一致,
    而不是留下一段被截断的回答。
    """

    answer: str
    cited: list[int]
    usage: Usage | None
    latency_ms: float
    stage_timings: dict[str, float]
    grounded: bool
    # 真正送去检索的那句话;无历史时与用户原话相同。
    #
    # **多轮链路上唯一能看见「指代消解做了什么」的地方。** 没有它,排查
    # 「追问为什么召不准」时只有两种猜测——改写带偏了,或检索本身不行——
    # 而这两者的修法完全相反。前端的检索详情面板也靠它把这一步显示出来。
    search_query: str = ""
    # 本次实际带进 prompt 的历史**消息条数**(不是轮数:一问一答算 2 条)。
    history_messages: int = 0
    # 检索结果是不是缓存给的。
    #
    # **它必须由生产结果的那一层给,不能在别处靠耗时去猜。** 「这次很快」
    # 和「这次命中了缓存」是两件事,而排障时第一个要回答的问题恰恰是
    # 「这个答案是不是缓存给的」——答错了,后面所有推断都是错的。
    cache_hit: bool = False
    # 整份答案(含来源)是不是直接从答案缓存拿的。
    #
    # **必须与 `cache_hit` 分开报。** 只有 `cache_hit` 一个标记时,「检索命中」
    # 和「答案命中」在事件里长得一模一样,而两者的排查方向相反:前者要问
    # 「检索结果对不对」,后者要问「这个问题凭什么能复用」。不变量:
    # `answer_cache_hit` 为真时 `cache_hit` 必然也为真——来源确实是从缓存来的。
    answer_cache_hit: bool = False
    # 会话落库后才有值。**`RAGService` 自己不填这两个字段**——它不碰数据库
    # (见模块文档),而是由路由层的持久化包装器用 `dataclasses.replace`
    # 补上去。无会话的「随口一问」两个都是 None,这正是前端判断
    # 「这轮要不要显示在历史里」的依据。
    session_id: int | None = None
    message_id: int | None = None


@dataclass(frozen=True, slots=True)
class ErrorEvent:
    """终态(出错版本)。**之后不会再有任何事件。**

    出错后再补一个 `done`,前端会把它当成完整答案存进历史,而用户永远
    不知道那句回答是半截的。
    """

    code: str
    message: str
    # 从用户提问到出错的毫秒数。**失败请求的时延必须记,而且只能在这里记。**
    #
    # 失败的那一轮同样占用一条连接、同样让用户等了这么久,而它在 `done`
    # 事件里根本没有落脚点——`done` 永远不会发出。只统计成功请求的延迟
    # 分布,会把最慢的那部分(慢到出错的)整个从分母里删掉,于是 p95 越
    # 优化越好看,而用户实际感受不到。
    #
    # 默认 0.0 而不是 None:老客户端读到它至少能显示个数,而 None 会让
    # 「取出耗时」这件事在每个消费端各写一次空值判断。
    elapsed_ms: float = 0.0


RagEvent = SourcesEvent | TokenEvent | DoneEvent | ErrorEvent


# ---------------------------------------------------------------------------
# 编排
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PreparedAnswer:
    """一次问答在「开始生成之前」的全部产物。"""

    question: str
    messages: list[dict[str, str]]
    sources: list[SourceRef]
    bundle: ContextBundle
    retrieval: RetrievalResult
    # 实际拿去检索的问题。无历史时与 `question` 相同,有历史时是指代消解
    # 之后的结果。**单独留一个字段是为了排查**:检索没命中时,第一件要看的
    # 就是「检索的到底是哪句话」,而那时改写已经过去了。
    search_query: str = ""
    # 这次提问的检索范围与过滤条件。**流式生成结束后要靠它们回写答案缓存**,
    # 而生成阶段不碰请求上下文,所以必须随 `PreparedAnswer` 带过去。
    kb_ids: list[int] | None = None
    filters: RetrievalFilters | None = None
    # 命中答案缓存时,这里就是那条缓存答案;没有命中是 `None`。
    #
    # **用「缓存里有什么」而不是「要不要查缓存」来表达状态。** 后者是一个
    # 布尔量,而它表达不了「命中了、内容是什么」——`stream()` 需要的是内容。
    cached_answer: CachedAnswer | None = None
    # **是消息条数,不是轮数。** 叫 turns 会让人在指标面板上把一问一答读成
    # 「聊了 2 轮」,而真实轮数是 1——一个看着有值、实际量错对象的数据,
    # 比没有这项数据更误导。
    history_messages: int = 0
    # 本次携带的历史条数,**未经预算裁剪**。
    #
    # 与上面那个字段差一个字,含义却是两回事,而且**混淆它们的后果是脏缓存**:
    # `history_messages` 是「塞进 prompt 的历史」(会被 `HISTORY_TOKEN_BUDGET`
    # 和 `HISTORY_MAX_TURNS` 裁),`history_supplied` 是「用户这次带了历史」
    # (改写那一环读的就是它,不受 prompt 预算影响)。答案缓存要判的是后者
    # ——按前者判,预算小到把历史裁空时,一条上下文相关的答案会以「无历史」
    # 的键写进去。
    history_supplied: int = 0
    stage_timings: dict[str, float] = field(default_factory=dict)
    # **整次请求的计时原点,在 `prepare()` 的第一行取。**
    #
    # 不能拿 `stream()` 的开始当原点:改写与检索都在那之前完成,而多轮追问
    # 里一次改写就是完整的 LLM 往返(实测 1.28 秒),比整个生成段还长。
    # 用晚了会得到一个自相矛盾的指标——`total` 比它底下的阶段还小,
    # 从而让人以为「这套系统只要 1.3 秒」,而用户实际等了两秒半。
    #
    # 默认值取构造时刻而不是 0.0:手工构造 `PreparedAnswer` 的场合(测试)
    # 拿到的是「约等于生成段」的 total,而不是一个从开机算起的荒谬数字。
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def grounded(self) -> bool:
        """有没有可依据的资料。决定走模型还是走固定答复。"""
        return bool(self.sources)

    @property
    def answer_cache_hit(self) -> bool:
        return self.cached_answer is not None


class RAGService:
    def __init__(
        self,
        *,
        settings: Settings,
        retrieval: Retriever,
        llm: LLMProvider,
        answers: AnswerCache | None = None,
        semantics: SemanticCache | None = None,
    ) -> None:
        self._settings = settings
        self._retrieval = retrieval
        self._llm = llm
        # 答案缓存是**可选**的:不传就是纯直通,所有既有行为一个字都不变。
        # 这样「缓存关掉之后系统还对不对」是默认状态而不是一个要另外测的
        # 分支(见 `test_the_cache_is_not_consulted_when_switched_off`)。
        self._answers = answers
        # 语义缓存同样可选,而且**它自己还带一层开关**(默认关闭,见
        # `Settings.SEMANTIC_CACHE_ENABLED`)。这里传进来的实例可能整个是
        # 关着的——`prepare` 不需要知道,`get`/`put` 会自己一步不走。
        self._semantics = semantics

    # -- 生成前 --------------------------------------------------------------

    async def prepare(
        self,
        question: str,
        *,
        kb_ids: list[int] | None,
        filters: RetrievalFilters | None = None,
        history: Sequence[Message] | None = None,
    ) -> PreparedAnswer:
        """跑检索并组装好提示词。**不碰数据库**——这一步出错还能返回正常的 HTTP 错误。

        `history` 是此前几轮的消息(正序)。它影响两处:
        ① 检索前先把追问补成独立问题,**检索用改写后的问题**;
        ② 拼 prompt 时作为对话前缀(预算另计,见 `HISTORY_TOKEN_BUDGET`)。
        """
        # 计时原点。放在最前面,让 `done` 里的 total / first_token 覆盖
        # 「用户按下回车」之后的全部耗时,而不只是模型开始吐字之后的那段。
        started_at = time.perf_counter()
        past = list(history or [])

        # -- 答案缓存 --------------------------------------------------------
        # **放在最前面**:命中就一步都不往下走。放在检索之后的话,省下的只有
        # 模型调用,用户仍然要等编码 + 两路召回 + 重排(实测几百毫秒)——
        # 而「同一个热门商品被反复问」正是最能省下这一段的时候。
        #
        # 传 `len(past)` 而不是裁剪后的条数:只要**带了**历史,改写就可能把
        # 问题指代消解成另一句话,答案也就不是这句话的函数了。裁剪是预算
        # 问题,不该影响「能不能复用」这个判断。
        if self._answers is not None:
            cached = await self._answers.get(
                question,
                history_messages=len(past),
                kb_ids=kb_ids,
                filters=filters,
            )
            if cached is not None:
                logger.info(
                    "答案缓存命中",
                    extra=log_extra(question=question[:80], sources=len(cached.sources)),
                )
                return self._cached(
                    question, cached, kb_ids=kb_ids, filters=filters, started_at=started_at
                )

        # -- 语义缓存 --------------------------------------------------------
        # 「换一种问法」也能命中。**排在精确缓存之后**:那一层只要一次按
        # 哈希的读,比这里的一次向量比较还便宜,先试它没有代价。
        #
        # **排在改写之前是安全的,而且不是巧合。** 这一层只在没有历史时
        # 生效(见 `SemanticCache._eligible`),而没有历史时 `rewrite_question`
        # 走早退分支原样返回问题(见 `history.py`)。两个位置对它可能命中的
        # 每一次请求都是等价的,放在前面只是省掉一次函数进出。
        #
        # 这一次 `embed_query` 由**向量缓存**吸收:`CachedEmbedder` 就包在
        # 检索链上,同一个问题向量在这一次请求里本来也要算。这也正是这一层
        # 必须排在向量缓存之后才能启用的原因——没有向量缓存时,每次未命中
        # 都要白搭一次网络编码,把最常见的路径拖慢。
        if self._semantics is not None:
            cached = await self._semantics.get(
                question,
                history_messages=len(past),
                kb_ids=kb_ids,
                filters=filters,
            )
            if cached is not None:
                logger.info(
                    "语义缓存命中",
                    extra=log_extra(question=question[:80], sources=len(cached.sources)),
                )
                # 复用 `_cached`:命中语义缓存与命中最外层答案缓存,产出的
                # `PreparedAnswer` 是同一个形状——`answer_cache_hit=True`、
                # 检索标记为缓存给的、不发模型调用。**必须走这条路**,
                # 否则这个答案在事件里与「模型刚刚生成」长得一模一样,
                # 而「缓存给出错答案」恰恰是最难查的那类问题。
                return self._cached(
                    question, cached, kb_ids=kb_ids, filters=filters, started_at=started_at
                )

        # -- 指代消解 --------------------------------------------------------
        # 注意接下来传下去的是 `search_query`,而 `PreparedAnswer.question`
        # 始终保留用户**原话**。两者不能混:改写结果拿去给模型看会污染答案的
        # 措辞,而原话拿去检索又召不准——它们服务的是不同的环节。
        mark_rewrite = time.perf_counter()
        search_query = await rewrite_question(
            question, past, llm=self._llm, settings=self._settings
        )
        rewrite_ms = _ms_since(mark_rewrite, time.perf_counter())

        retrieval_result = await self._retrieval.retrieve(
            search_query, kb_ids=kb_ids, filters=filters
        )

        mark = time.perf_counter()
        bundle = assemble_context(
            retrieval_result.chunks, max_tokens=self._settings.CONTEXT_TOKEN_BUDGET
        )
        # **编号建立在 `bundle.used` 之上,不是 `retrieval_result.chunks`。**
        # 组装会丢掉重复正文和超预算的分块;照着原始结果编号的话,被丢掉
        # 的那条仍占一个号,后面每条 `[n]` 全部错位一格,而正文里的编号
        # 看上去完全正常。
        sources = [SourceRef.of(i, piece) for i, piece in enumerate(bundle.used, start=1)]
        assemble_ms = _ms_since(mark, time.perf_counter())

        if not sources:
            logger.info(
                "检索无结果,将走固定答复",
                extra=log_extra(
                    question=question[:80],
                    route_sizes=retrieval_result.route_sizes,
                    dropped=len(bundle.dropped),
                ),
            )

        history_messages = build_history(
            past,
            max_tokens=self._settings.HISTORY_TOKEN_BUDGET,
            max_turns=self._settings.HISTORY_MAX_TURNS,
        )

        return PreparedAnswer(
            question=question,
            search_query=search_query,
            # 无资料时也照常构造消息。`stream()` 不会用到它,但保留下来
            # 让这个对象是「这次问答准备发出去的完整 prompt」,排查
            # 「模型为什么会这么答」时可以直接看,不必重新推演一遍。
            messages=build_messages(question, bundle.text, history=history_messages),
            sources=sources,
            bundle=bundle,
            retrieval=retrieval_result,
            history_messages=len(history_messages),
            history_supplied=len(past),
            kb_ids=kb_ids,
            filters=filters,
            stage_timings={"rewrite": rewrite_ms, "assemble": assemble_ms},
            started_at=started_at,
        )

    def _cached(
        self,
        question: str,
        cached: CachedAnswer,
        *,
        kb_ids: list[int] | None,
        filters: RetrievalFilters | None,
        started_at: float,
    ) -> PreparedAnswer:
        """命中答案缓存时的 `PreparedAnswer`。

        **`messages` 是空的,这是有意的。** 提示词里最重的那部分(组装好的
        资料正文)没有随答案缓存一起存——存它等于把缓存条目撑大成原来的
        几十倍,而它对**复用**毫无用处,只是排查时想看一眼。要还原它得重新
        跑一遍检索,那正是这一层要避免的事。

        `retrieval.cache_hit=True`:**来源确实是从缓存里拿的**,尽管检索一步
        都没跑。这里若报 `False`,读事件的人会以为「检索跑了、没走缓存」,
        而事实是相反的那个极端。
        """
        return PreparedAnswer(
            question=question,
            messages=[],
            sources=list(cached.sources),
            bundle=ContextBundle(text="", used=[], dropped=[], tokens=0),
            retrieval=RetrievalResult(
                query=question,
                chunks=[],
                route_sizes={},
                stage_timings={},
                cache_hit=True,
            ),
            # 这一层只在**没有历史**时启用,而无历史时 `rewrite_question`
            # 原样返回问题(`history.py` 的早退分支),所以这里就是检索本该
            # 拿到的那句话——如实填,不是猜的。
            search_query=question,
            kb_ids=kb_ids,
            filters=filters,
            cached_answer=cached,
            started_at=started_at,
        )

    # -- 生成 ----------------------------------------------------------------

    async def stream(self, prepared: PreparedAnswer) -> AsyncIterator[RagEvent]:
        started = time.perf_counter()
        yield SourcesEvent(
            sources=prepared.sources,
            route_sizes=dict(prepared.retrieval.route_sizes),
            reranked=prepared.retrieval.reranked,
            degraded=prepared.retrieval.degraded,
        )

        if prepared.cached_answer is not None:
            # 命中。整段答案一次发出——前端本来就是逐片追加渲染的,一片
            # 发完和十片发完画出来的东西一模一样,而这里不存在「先看到
            # 前半句」的价值:**它早就写好了**。
            #
            # usage 是 `None`,见 `test_a_hit_reports_no_usage`:这一轮真的
            # 没花 token,把首次的用量报出去会让成本看板越优化越难看。
            answer = prepared.cached_answer.answer
            yield TokenEvent(text=answer)
            yield self._done(prepared, answer, None, started, None)
            return

        if not prepared.grounded:
            yield TokenEvent(text=NO_CONTEXT_ANSWER)
            yield self._done(prepared, NO_CONTEXT_ANSWER, None, started, None)
            return

        pieces: list[str] = []
        usage: Usage | None = None
        first_token_at: float | None = None

        try:
            async for piece in self._llm.astream(prepared.messages):
                # 用量通常只在最后一片上出现;没出现就保持 None,不估算。
                if piece.usage is not None:
                    usage = piece.usage
                if piece.reasoning:
                    # 思考模型的推理过程单独承接、绝不进正文:混进去用户会
                    # 看到模型自言自语,而且那段话无法被任何资料引用。
                    logger.debug(
                        "模型返回推理内容,已丢弃",
                        extra=log_extra(chars=len(piece.reasoning)),
                    )
                if not piece.text:
                    continue
                if first_token_at is None:
                    first_token_at = time.perf_counter()
                pieces.append(piece.text)
                yield TokenEvent(text=piece.text)
        except Exception as exc:
            # **`except Exception` 是刻意的,且必须保持这个宽度。**
            #
            # 响应头在第一个字节时就已经提交,此刻再抛异常没法变成 500——
            # 客户端只会看到连接莫名断掉,连「出错了」都不知道。所以生成期
            # 的任何故障都得作为 error 事件送达,这是唯一可用的通道。
            #
            # 注意 `asyncio.CancelledError` 继承自 BaseException,**不会**
            # 被这里捕获:客户端断连时它必须继续向上传播,好让
            # sse-starlette 干净地回收这个生成器。
            logger.exception(
                "生成阶段失败",
                extra=log_extra(question=prepared.question[:80], emitted=len(pieces)),
            )
            # 已经流出去的字收不回来,也不该假装没发生——半截答案配上一条
            # 错误提示,比一片空白有用。
            yield ErrorEvent(
                code=getattr(exc, "code", "generation_failed"),
                message=f"生成回答时出错:{exc}",
                elapsed_ms=_ms_since(prepared.started_at, time.perf_counter()),
            )
            return

        answer = "".join(pieces)
        await self._remember(prepared, answer)
        yield self._done(prepared, answer, usage, started, first_token_at)

    async def _remember(self, prepared: PreparedAnswer, answer: str) -> None:
        """把这一轮成功生成的答案写进缓存。

        **`ErrorEvent` 之后不会走到这里**(上面那条 `return`),所以半截答案
        永远进不了缓存——它带着完整来源、看起来像一份正常回答,存进去比
        当场报错糟糕得多:报错至少会让人重试。

        **`grounded` 为假时不写**:那条固定答复本来就不花模型调用,缓存它
        省不下什么,却会把「这次没找到」固化半小时,而用户换个问法、管理员
        补一份文档之后它本就该变。

        **写缓存失败绝不能让这一轮失败。** 用户已经完整看到答案了,此刻
        抛异常只会把一条正常的流变成「连接莫名断掉」。缓存是纯优化,它坏了
        的代价必须是「下次慢一点」。
        """
        if prepared.answer_cache_hit or not prepared.grounded:
            return
        if self._answers is None and self._semantics is None:
            return
        from app.services.cache import CachedAnswer

        # 两个缓存层存的就是**同一份** `CachedAnswer`(共用一套字节格式,
        # 见 `cache._encode_answer`),构造一次、各自写一次。
        entry = CachedAnswer(answer=answer, sources=list(prepared.sources))
        common = {
            # **必须是「带没带历史」,不是「历史塞进 prompt 了没有」。**
            # 预算够大时两者相等,一旦小到把历史全裁掉就分叉了:prompt
            # 里确实没有历史,可改写那一环读的是**原始**历史,照样可能
            # 把问题消解成另一句话。按裁剪后的条数判断,这里会以 0 写进
            # 一条上下文相关的答案,而它的键和一个全新的提问完全相同。
            "history_messages": prepared.history_supplied,
            "kb_ids": prepared.kb_ids,
            "filters": prepared.filters,
            "answer": entry,
        }
        # 分开 try:**一层的写入失败不该带走另一层**。两层是独立的收益,
        # 精确层炸了(比如键超过列宽)而语义层本来能写成功,合在一个
        # try 里就白白丢掉了一层。
        for layer, label in ((self._answers, "答案"), (self._semantics, "语义")):
            if layer is None:
                continue
            try:
                await layer.put(prepared.question, **common)
            except Exception:
                logger.exception(
                    "%s缓存写入失败", label, extra=log_extra(question=prepared.question[:80])
                )

    def _done(
        self,
        prepared: PreparedAnswer,
        answer: str,
        usage: Usage | None,
        started: float,
        first_token_at: float | None,
    ) -> DoneEvent:
        timings = {
            key: value
            for key, value in prepared.retrieval.stage_timings.items()
            if key != "total"
        }
        # 检索自己的 `total` **必须改名**。两个都叫 total 的话,排障时会把
        # 它当成整条链路的总耗时,而它只覆盖了检索一段——一个看着有值、
        # 实际量错了对象的面板数据,比没有这项数据更误导。
        timings["retrieve"] = prepared.retrieval.stage_timings.get("total", 0.0)
        timings.update(prepared.stage_timings)
        now = time.perf_counter()
        # **`generate` 从流开始算,`total` / `first_token` 从提问算起。**
        #
        # 这两个原点不能混:generate 量的是「模型自己有多快」,把检索和改写
        # 算进去就再也看不出模型变慢了还是检索变慢了;而用户感知的等待从
        # 按下回车那一刻开始,用流开始当原点会把最贵的那段(多轮改写)漏掉。
        timings["generate"] = _ms_since(started, now)
        if first_token_at is not None:
            timings["first_token"] = _ms_since(prepared.started_at, first_token_at)
        timings["total"] = _ms_since(prepared.started_at, now)

        return DoneEvent(
            answer=answer,
            # 编号从 1 起,上限是**实际装进上下文的资料条数**——就这样
            # 顺带挡掉了 `[2024]`(年份)、`[9]`(模型臆造的编号)这类误匹配。
            cited=extract_citations(answer, max_index=len(prepared.sources)),
            usage=usage,
            latency_ms=timings["total"],
            stage_timings=timings,
            grounded=prepared.grounded,
            # 即使检索为空也照带上:那正是最需要它的时刻——「检索为什么
            # 没命中」的第一手材料就是「拿哪句话去检索的」。
            search_query=prepared.search_query,
            history_messages=prepared.history_messages,
            cache_hit=prepared.retrieval.cache_hit,
            answer_cache_hit=prepared.answer_cache_hit,
        )


def _ms_since(started: float, now: float) -> float:
    return round((now - started) * 1000, 3)
