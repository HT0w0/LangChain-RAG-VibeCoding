"""请求指标:埋点的写入,以及面板要的聚合。

这一层不参与回答,所以它自己出错时的容忍度**必须比缓存还高**:缓存写失败
只是下次慢一点,指标写失败连用户都不该察觉。但它又必须**记得住失败**——
一个只记成功请求的指标表,会把超时和重试全部漏掉,而它们恰恰是最该被看见
的那部分。面板上「p95 很健康」和「有一半请求根本没进来」是可以同时成立的。
"""

from __future__ import annotations

from datetime import timedelta

import pytest
from app.db.models import RequestMetric, utcnow

#: 问答接口。**这里刻意不 import `test_chat_api` 的辅助函数**:那会把这个
#: 文件和另一个测试模块绑在一起,而它要的只是一次真实的问答请求。
_CHAT = "/api/chat/stream"

#: 一份最小的商品资料。要验「模型被调用过的一轮」就必须让检索有结果,
#: 而检索有结果就必须真的建库、真的摄取——没有捷径。
_CSV = "商品名称,类目,价格,库存\nRX9070 显卡,显卡,4299,12\n".encode()


@pytest.fixture
async def metrics(tmp_path):
    """指标服务 + 独立临时库。**绝不碰开发库。**"""
    from app.core.config import Settings
    from app.db.base import Database
    from app.db.models import ChatSession, User
    from app.services.metrics import MetricsService

    metric_settings = Settings(
        _env_file=None,
        DATA_DIR=tmp_path / "data",
        CHROMA_DIR=tmp_path / "data" / "chroma",
        UPLOAD_DIR=tmp_path / "data" / "uploads",
        DATABASE_URL=f"sqlite:///{(tmp_path / 'data' / 'app.db').as_posix()}",
        DASHSCOPE_API_KEY="sk-test",
    )
    metric_settings.ensure_dirs()
    database = Database(metric_settings)
    await database.create_all()

    # `_metric()` 里写死了 user_id=1 / session_id=1,而 `RequestMetric` 对两者
    # 都有外键(SQLite 的外键在 `db/base.py` 里被显式打开)。所以这里必须真的
    # 建出这两行——不然每条用例都会卡在 `FOREIGN KEY constraint failed`,
    # 而那个错**被 `record` 吞掉**(它的职责之一就是不打扰调用方),症状是
    # 「表里一行都没有」,看不出和外键有任何关系。
    async with database.session() as session:
        session.add(User(id=1, username="metric-owner", password_hash="x", role="user"))
        await session.flush()
        session.add(ChatSession(id=1, user_id=1, title="指标用例的会话"))
        await session.commit()

    try:
        yield MetricsService(database), database
    finally:
        await database.dispose()


def _metric(**overrides) -> RequestMetric:
    """一条指标,字段全都可覆盖。默认值取得「正常完成的一轮问答」。"""
    base = {
        "trace_id": "t" * 32,
        "user_id": 1,
        "session_id": 1,
        "outcome": "done",
        "total_ms": 100.0,
        "first_token_ms": 50.0,
        "stage_timings": {"rewrite": 10.0, "retrieve": 20.0},
        "retrieval_cache_hit": False,
        "answer_cache_hit": False,
        "grounded": True,
        "history_messages": 0,
        "prompt_tokens": 100,
        "completion_tokens": 20,
    }
    base.update(overrides)
    return RequestMetric(**base)


async def _count(database) -> int:
    from sqlalchemy import func, select

    async with database.session() as session:
        return int(await session.scalar(select(func.count()).select_from(RequestMetric)) or 0)


class TestRecording:
    async def test_a_recorded_request_lands_in_the_table(self, metrics) -> None:
        service, database = metrics

        await service.record(_metric(total_ms=123.0, trace_id="a" * 32))

        assert await _count(database) == 1

    async def test_every_field_survives_the_round_trip(self, metrics) -> None:
        """JSON 字段尤其要验:SQLite 存 JSON 是当文本存的,写进去的是 dict,
        读出来可能是 str——而那时面板拿到的是一个字符串做不了聚合,
        报错却发生在读取端,离写入端很远。"""
        from sqlalchemy import select

        service, database = metrics
        await service.record(
            _metric(
                stage_timings={"rewrite": 12.5, "rerank": 40.25},
                outcome="error",
                first_token_ms=None,
                grounded=False,
            )
        )

        async with database.session() as session:
            row = (await session.execute(select(RequestMetric))).scalars().one()

        assert row.stage_timings == {"rewrite": 12.5, "rerank": 40.25}
        assert row.outcome == "error"
        assert row.first_token_ms is None
        assert row.grounded is False

    async def test_a_write_failure_never_escapes(self, metrics) -> None:
        """**指标写失败绝不能让一轮问答失败。**

        用户已经完整看到答案了,此刻抛异常只会把一条正常的流变成「连接莫名
        断掉」。做法是把会话换成一个必然失败的替身——比去构造一个真实的
        写冲突可靠,也不会因为 SQLite 版本的差异而时灵时不灵。
        """
        service, _database = metrics

        class _BrokenSession:
            async def execute(self, *args, **kwargs):  # pragma: no cover - 不该被调用
                raise AssertionError("这里不该走 execute")

        broken = type(service)(_BrokenSession())

        # 不抛,就是这条用例的全部断言。
        await broken.record(_metric())

    def test_the_table_stores_no_question_text(self) -> None:
        """**指标表不存问题正文。**

        存了就等于造了一份**影子聊天记录**:它在会话的归属校验之外,任何
        能读指标的人都能读到所有人的提问。而它要回答的问题——长短问题的
        延迟分布——一个字符数就够了。

        这条断言只能在有人加回一个同名列时报警,挡不住换个名字存正文
        (`raw`、`body`)。写在这里的价值是**把这条约束写下来**:没有它,
        「加一个 question 字段方便排查」是个看起来很合理的改动。
        """
        columns = set(RequestMetric.__table__.columns.keys())

        assert not columns & {"question", "content", "answer", "text", "query"}


class TestSummary:
    async def test_an_empty_window_is_all_zeros_not_an_error(self, metrics) -> None:
        """没有任何请求时面板必须能开,而不是除零。

        一个空窗口是**最常见的状态**——刚部署、刚清库、或者只是这一小时
        没人提问。它在除零崩溃和显示 0 之间,只有后者是对的。
        """
        service, _database = metrics

        summary = await service.summary(window_minutes=60)

        assert summary.requests == 0
        assert summary.error_rate == 0.0
        assert summary.cache_hit_rate_retrieval == 0.0
        assert summary.cache_hit_rate_answer == 0.0
        assert summary.grounded_rate == 0.0
        assert summary.total_ms.p50 == 0.0
        assert summary.first_token_ms is None
        assert summary.stages == {}

    async def test_the_window_leaves_out_older_requests(self, metrics) -> None:
        """窗口之外的行不参与聚合。

        这条要测是因为**写错的代价很隐蔽**:算成全表的话,面板显示的是
        「开服以来的平均延迟」,而它在系统越来越好时几乎不动——看起来像
        「优化没生效」。
        """
        from sqlalchemy import update

        service, database = metrics
        await service.record(_metric(total_ms=1000.0))
        await service.record(_metric(total_ms=1000.0))
        await service.record(_metric(total_ms=10.0))

        # 把两条推到窗口外。用 UPDATE 而不是在构造时传 created_at:
        # 后者会绕过列默认值,顺带也就绕过了「默认值确实是 utcnow」这件事。
        async with database.session() as session:
            await session.execute(
                update(RequestMetric)
                .where(RequestMetric.total_ms == 1000.0)
                .values(created_at=utcnow() - timedelta(minutes=120))
            )
            await session.commit()

        summary = await service.summary(window_minutes=60)

        assert summary.requests == 1
        assert summary.total_ms.avg == 10.0

    async def test_outcomes_are_counted_separately(self, metrics) -> None:
        service, _database = metrics
        await service.record(_metric(outcome="done"))
        await service.record(_metric(outcome="done"))
        await service.record(_metric(outcome="error"))

        summary = await service.summary(window_minutes=60)

        assert summary.requests == 3
        assert summary.outcomes == {"done": 2, "error": 1}
        assert summary.error_rate == pytest.approx(1 / 3)

    async def test_percentiles_use_nearest_rank(self, metrics) -> None:
        """分位数的**定义**必须钉死,不然它是个会漂的数。

        这里用最近秩:排序后取第 `ceil(k/100 × n)` 个(从 1 数)。n=10、
        取值 1..10 时 p50 落在第 5 个(值 5),p95 落在第 10 个(值 10)。

        换成插值法(线性插值)这两个数会变成 5.5 和 9.55——都「对」,
        所以它不能只是实现细节,得有一条用例把它固定住。
        """
        service, _database = metrics
        for value in range(1, 11):
            await service.record(_metric(total_ms=float(value)))

        summary = await service.summary(window_minutes=60)

        assert summary.total_ms.p50 == 5.0
        assert summary.total_ms.p95 == 10.0
        assert summary.total_ms.max == 10.0
        assert summary.total_ms.avg == 5.5

    async def test_a_single_sample_reports_itself_for_every_percentile(self, metrics) -> None:
        """n=1 是最容易算错的一档:向上取整后 `ceil(0.5)-1` 恰好是 0,
        但写成 `int(k/100*n)` 就会得到索引 0 之外的负偏移,而它只在
        样本极少时出错——也就是刚部署、面板上只有一行的时候。"""
        service, _database = metrics
        await service.record(_metric(total_ms=42.0))

        summary = await service.summary(window_minutes=60)

        assert summary.total_ms.p50 == 42.0
        assert summary.total_ms.p95 == 42.0

    async def test_first_token_is_none_when_nothing_streamed(self, metrics) -> None:
        """没有任何一条记录有首 token 时,报 `None` 而不是 0。

        **0 会被读成「首 token 是瞬间出来的」**——一个漂亮得离谱的数字,
        而事实是这一轮一个字都没流出来。这两种情况在面板上必须能区分。
        """
        service, _database = metrics
        await service.record(_metric(first_token_ms=None, outcome="error"))

        assert (await service.summary(window_minutes=60)).first_token_ms is None

    async def test_first_token_ignores_requests_that_had_none(self, metrics) -> None:
        """一旦有记录报过首 token,那些没报的就该被排除在**分母**之外。

        把 `None` 当成 0 参与平均,会把首 token 时延算得比实际低——而这是个
        系统性偏乐观的偏差,不会随机抵消。
        """
        service, _database = metrics
        await service.record(_metric(first_token_ms=200.0))
        await service.record(_metric(first_token_ms=400.0))
        await service.record(_metric(first_token_ms=None, outcome="error"))

        summary = await service.summary(window_minutes=60)

        assert summary.first_token_ms is not None
        assert summary.first_token_ms.avg == 300.0

    async def test_cache_hit_rates_are_computed_over_the_window(self, metrics) -> None:
        service, _database = metrics
        await service.record(_metric(retrieval_cache_hit=True, answer_cache_hit=False))
        await service.record(_metric(retrieval_cache_hit=True, answer_cache_hit=True))
        await service.record(_metric(retrieval_cache_hit=False, answer_cache_hit=False))
        await service.record(_metric(retrieval_cache_hit=False, answer_cache_hit=False))

        summary = await service.summary(window_minutes=60)

        assert summary.cache_hit_rate_retrieval == 0.5
        assert summary.cache_hit_rate_answer == 0.25

    async def test_stage_averages_ignore_requests_that_lacked_the_stage(self, metrics) -> None:
        """一个阶段的分母是「**记过这个阶段**的请求数」,不是总请求数。

        用总数当分母的话,每加一个只部分出现的阶段(重排被关掉、改写被
        跳过),它的平均值都会被那些根本没跑过它的请求拉低——于是关掉重排
        看起来像是「重排变快了」。
        """
        service, _database = metrics
        await service.record(_metric(stage_timings={"rewrite": 10.0, "rerank": 100.0}))
        await service.record(_metric(stage_timings={"rewrite": 20.0}))
        await service.record(_metric(stage_timings={"rewrite": 30.0}))
        await service.record(_metric(stage_timings={}))

        summary = await service.summary(window_minutes=60)

        assert summary.stages["rewrite"] == 20.0
        # 只有一条记了 rerank,平均值就该是它自己,而不是 100/4。
        assert summary.stages["rerank"] == 100.0

    async def test_token_totals_are_summed(self, metrics) -> None:
        service, _database = metrics
        await service.record(_metric(prompt_tokens=100, completion_tokens=20))
        await service.record(_metric(prompt_tokens=300, completion_tokens=40))

        summary = await service.summary(window_minutes=60)

        assert summary.prompt_tokens == 400
        assert summary.completion_tokens == 60

    async def test_cache_read_rate_is_none_when_nothing_reported_it(self, metrics) -> None:
        """一条都没报 prompt 缓存读数时,命中率是 `None`,**不是 0.0**。

        两者的含义相反:`0.0` 是「平台说这次一条都没命中」,`None` 是
        「平台没说」。合并之后,面板上会是一条平稳的 0% 命中率曲线——
        看起来像「缓存一直不工作」,而事实是我们根本没在听。
        与 `first_token_ms` 用 `None` 表示「一个字都没流出来」是同一条理由。
        """
        service, _database = metrics
        await service.record(_metric(prompt_tokens=100))  # 没带读数

        summary = await service.summary(window_minutes=60)

        assert summary.cache_read_rate is None
        assert summary.cached_prompt_tokens == 0

    async def test_cache_read_rate_divides_by_the_requests_that_reported(self, metrics) -> None:
        """分母是**报过读数的那些请求的 prompt token**,不是全部请求的。

        把没报读数的请求也算进分母,命中率会被系统性拉低——而拉低的方向
        恰好是「缓存看起来更没用」,与真实情况相反。中间那条报了 0:
        **它算报过**,必须进分母(它确实消耗了 100 个 prompt token,
        一个都没从缓存来)。
        """
        service, _database = metrics
        await service.record(_metric(prompt_tokens=100, cached_prompt_tokens=50))
        await service.record(_metric(prompt_tokens=100, cached_prompt_tokens=0))
        await service.record(_metric(prompt_tokens=100))  # 没报

        summary = await service.summary(window_minutes=60)

        assert summary.cached_prompt_tokens == 50
        # 50 / 200,而不是 50 / 300。
        assert summary.cache_read_rate == 0.25

    async def test_a_reported_zero_is_rate_zero_not_no_reading(self, metrics) -> None:
        """只报了 0 的时候,命中率是 `0.0`,而且**不是 `None`**。

        这条是上一条的边界,单独写是因为它是那个最容易写错的形状:
        写成 `if cached:` 的实现会给出 `None`,而那意味着「一个会报读数的
        平台,只要一直不命中,就会被读成不会报读数」——两种完全不同的
        处置(去查缓存为什么不开 vs 去查平台为什么不说)被合并成了一种。
        """
        service, _database = metrics
        await service.record(_metric(prompt_tokens=100, cached_prompt_tokens=0))

        summary = await service.summary(window_minutes=60)

        assert summary.cache_read_rate == 0.0
        assert summary.cache_read_rate is not None


class TestRetention:
    """**这张表是唯一会无限增长的**:每次问答写一行,而它答不了任何业务
    问题,只服务于面板。所以删除判据值得单独钉一条用例——它的符号写反
    一次,删掉的就是**最新**的那些行,而那正是面板唯一要看的区间。"""

    async def _age(self, database, total_ms: float, *, days: int) -> None:
        from sqlalchemy import update

        async with database.session() as session:
            await session.execute(
                update(RequestMetric)
                .where(RequestMetric.total_ms == total_ms)
                .values(created_at=utcnow() - timedelta(days=days))
            )
            await session.commit()

    async def test_purge_removes_only_rows_past_the_retention_window(self, metrics) -> None:
        service, database = metrics
        await service.record(_metric(total_ms=1.0))  # 刚才
        await service.record(_metric(total_ms=2.0))
        await service.record(_metric(total_ms=3.0))
        await self._age(database, 2.0, days=40)
        await self._age(database, 3.0, days=90)

        removed = await service.purge_older_than(days=30)

        assert removed == 2
        remaining = await service.summary(window_minutes=60)
        # 留下来的必须是**刚才那条**,不是「反正是三条里的一条」。
        assert remaining.requests == 1
        assert remaining.total_ms.max == 1.0

    async def test_purge_keeps_the_boundary_row(self, metrics) -> None:
        """恰好落在边界上的那条**留着**。

        判据是「早于 `days` 天」而不是「不晚于」:这两者在真机上差一条
        记录,而写反之后没有任何症状——面板少一行,没人看得出来。边界
        留一条用例,是因为它把「闭区间还是开区间」这个必须二选一的问题
        变成了一次断言,而不是留给下一个人去读 SQL。
        """
        service, database = metrics
        await service.record(_metric(total_ms=7.0))
        await self._age(database, 7.0, days=29)

        assert await service.purge_older_than(days=30) == 0


# ---------------------------------------------------------------------------
# 面板接口
# ---------------------------------------------------------------------------


class TestMetricsEndpoint:
    def test_a_normal_user_cannot_read_the_panel(self, client) -> None:
        """**普通人拿不到全站延迟与成本数据。**

        指标里含所有人的 token 消耗与流量形态,是运维信息。前端不显示
        这个页面只是体验,挡不住 curl。
        """
        from tests.helpers import auth_header, register

        token = register(client, "metrics-probe-user")["access_token"]
        r = client.get("/api/admin/metrics", headers=auth_header(token))

        assert r.status_code == 403, r.text

    def test_an_anonymous_request_cannot_read_the_panel(self, client) -> None:
        assert client.get("/api/admin/metrics").status_code == 401

    def test_admin_gets_a_summary_over_the_requested_window(self, client) -> None:
        from tests.helpers import admin_token, auth_header

        r = client.get(
            "/api/admin/metrics?window_minutes=120", headers=auth_header(admin_token(client))
        )

        assert r.status_code == 200, r.text
        body = r.json()
        assert body["window_minutes"] == 120
        assert body["requests"] == 0
        assert body["total_ms"]["p95"] == 0.0

    def test_the_panel_sees_the_rows_the_instrumentation_wrote(self, client) -> None:
        """**埋点和面板必须落在同一张表、同一个窗口定义上。**

        分开看,两组用例都可以全绿而它们是断开的:服务层那组用独立临时库
        自己造数据,接口那组只看一个空窗口。「记下来的东西面板看得见吗」
        是第三个问题,而它恰恰是唯一有产品意义的那个——埋点写对了表、
        面板查错了表,两边各自的测试都不会红。

        所以这条用例自己走一遍真实请求,再从面板把那一行读回来。
        **不建知识库**:要验的是「写进去的行读得出来」,一个没有资料的
        提问同样会走完检索、组装、生成并留下完整的一行,而它省掉了
        「建库 → 传文件 → 等摄取」那一整套与本题无关的构造。
        """
        from tests.helpers import admin_token, auth_header

        r = client.post(
            _CHAT, json={"question": "库是空的,随便问问"}, headers=auth_header(admin_token(client))
        )
        assert r.status_code == 200, r.text

        r = client.get("/api/admin/metrics", headers=auth_header(admin_token(client)))
        assert r.status_code == 200, r.text
        body = r.json()

        assert body["requests"] == 1, "面板看不见刚写进去的那一轮问答"
        assert body["outcomes"] == {"done": 1}
        assert body["total_ms"]["p50"] > 0
        # 阶段耗时也得跟过来:它只存在 JSON 列里,是「面板真的读到了那份
        # 明细」而不是「读到了某张只有计数的表」的证据。
        assert body["stages"], "面板上的阶段耗时是空的——明细没被读出来"

    def test_a_round_that_never_asked_the_model_reports_no_reading(self, client) -> None:
        """**没调模型的一轮,缓存读数是 `None`,不是 0。**

        知识库里没有资料时走固定答复([`NO_CONTEXT_ANSWER`])、**根本不发
        LLM 请求**。没有请求就没有回包,没有回包就没有读数——报 0 会被读成
        「平台说这次一条都没命中」,而事实是我们**没问过平台**。

        两者在面板上的后果相反:`0` 会把这一窗的命中率算成 0%,看起来像
        「缓存配置有问题」,引导人去查缓存为什么不开;而真相是这一窗里
        压根没发生过带 prompt 的调用。
        """
        from tests.helpers import admin_token, auth_header, latest_metric

        headers = auth_header(admin_token(client))
        r = client.post(_CHAT, json={"question": "库是空的,随便问问"}, headers=headers)
        assert r.status_code == 200, r.text

        row = latest_metric(client)
        assert row is not None, "连行都没写进去"
        assert row["prompt_tokens"] == 0, "这一轮不该有 prompt token"
        assert row["cached_prompt_tokens"] is None

    def test_the_reading_the_provider_gave_reaches_the_table_and_the_panel(self, client) -> None:
        """接了模型的一轮,读数要**一路走到面板**。

        Provider → `Usage` → 埋点 → 表 → 聚合,五跳。断在哪一跳都不报错,
        症状一律是「面板上那一格是空的」——而空读数看起来就是「缓存没命中」,
        不会有人去查接线。

        mock Provider 报 `0`(它有读数,只是没有缓存),所以断言 `0.0`
        而**不是 `None`**:接线漏掉时聚合拿到的是 `None`,这条就会红。
        真平台上这个数会不会变成非零取决于前缀缓存命中,**不在本用例的
        射程内**——那是 `scripts/prompt_cache_probe.py` 的职责。
        """
        import io

        from tests.helpers import admin_token, auth_header, latest_metric

        headers = auth_header(admin_token(client))
        kb = client.post("/api/kb", json={"name": "读数库"}, headers=headers)
        assert kb.status_code == 201, kb.text
        uploaded = client.post(
            f"/api/kb/{kb.json()['id']}/documents",
            files={"files": ("报价.csv", io.BytesIO(_CSV), "text/csv")},
            headers=headers,
        )
        assert uploaded.status_code == 202, uploaded.text

        r = client.post(_CHAT, json={"question": "RX9070 多少钱"}, headers=headers)
        assert r.status_code == 200, r.text

        row = latest_metric(client)
        assert row is not None
        assert row["prompt_tokens"] > 0, "这一轮该真的调用模型"
        assert row["cached_prompt_tokens"] == 0, "读数在写表这一步丢了"

        body = client.get("/api/admin/metrics", headers=headers).json()
        assert body["cache_read_rate"] == 0.0, "读数在聚合这一步丢了"

    def test_the_window_is_bounded(self, client) -> None:
        """窗口必须有上下界:不封顶的话 `window_minutes=100000000` 就是一次
        全表扫描,而这在单机 SQLite 上是一次可以从公网触发的停顿。"""
        from tests.helpers import admin_token, auth_header

        headers = auth_header(admin_token(client))

        assert client.get("/api/admin/metrics?window_minutes=0", headers=headers).status_code == 422
        assert (
            client.get("/api/admin/metrics?window_minutes=99999999", headers=headers).status_code
            == 422
        )
