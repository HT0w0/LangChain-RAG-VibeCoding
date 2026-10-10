"""用户级限流:滑动窗口 + 问答端点上的 429。

**这个文件补的是一条「文档承诺了、代码没做」的欠账。**
这一轮之前,`README.md` 的架构图写着中间件做「用户级限流」,`RATE_LIMIT_QPM`
和 `RateLimitedError` 也都在,但**全仓没有一处调用它们** —— 也就是说
配置项和异常类都在,线没接。(那行文档已经改掉了,现在写的是实际的做法:
限流挂在**问答端点**的依赖上,不是中间件。)

(这类欠账的可怕之处在于**它看起来是做完的**:配置项在那儿、异常类在那儿、
文档也写着。只有搜一遍调用点,才知道中间那一段从来没有过。)

两层测试:
  * `TestSlidingWindowLimiter` —— 限流器本身(时钟注入,不靠真实等待);
  * `TestChatRateLimit` —— 接到接口上之后真的返回 429,而且**不误伤别的端点**。
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from app.core import deps
from app.main import app
from app.services.ratelimit import RateDecision, SlidingWindowLimiter
from fastapi.testclient import TestClient

from tests.helpers import auth_header, register

CHAT = "/api/chat/stream"


@pytest.fixture
def tiny_limit() -> Iterator[SlidingWindowLimiter]:
    """把限流器换成额度很小的一个 —— 几条请求就能撞线,不用把默认额度打满。

    **换的是依赖,不是配置**:这样测的仍然是真实的那条链路
    (依赖 → `SlidingWindowLimiter` → `RateLimitedError` → 429),
    只是额度小了。
    """
    limiter = SlidingWindowLimiter(limit=2, window_seconds=60.0)
    app.dependency_overrides[deps.get_rate_limiter] = lambda: limiter
    try:
        yield limiter
    finally:
        app.dependency_overrides.pop(deps.get_rate_limiter, None)


class _Clock:
    """可以手动拨的时钟。**不用 `time.sleep` 去等窗口滑过去** ——
    那会让测试既慢又靠机器快慢。"""

    def __init__(self, now: float = 1000.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _limiter(limit: int, clock: _Clock, *, window: float = 60.0) -> SlidingWindowLimiter:
    return SlidingWindowLimiter(limit=limit, window_seconds=window, clock=clock)


class TestSlidingWindowLimiter:
    def test_额度内放行_超出就拒(self) -> None:
        clock = _Clock()
        limiter = _limiter(3, clock)

        assert all(limiter.check("u1").allowed for _ in range(3))
        decision = limiter.check("u1")

        assert decision.allowed is False
        # 还要等多久:**第一条**记录滚出窗口的时候。
        assert 59.0 < decision.retry_after_seconds <= 60.0

    def test_窗口滑过去之后恢复(self) -> None:
        clock = _Clock()
        limiter = _limiter(2, clock)

        assert limiter.check("u1").allowed
        assert limiter.check("u1").allowed
        assert limiter.check("u1").allowed is False

        clock.advance(60.1)

        assert limiter.check("u1").allowed is True

    def test_窗口是滑动的_不是整分钟清零(self) -> None:
        """第 1 秒和第 59 秒各用掉一次,额度 2 —— 到第 62 秒时**只该解禁第一条**。

        整分钟清零的实现会在这里给出错误答案(它会把两次都算在新的一分钟里)。
        """
        clock = _Clock()
        limiter = _limiter(2, clock)

        assert limiter.check("u1").allowed  # t=0
        clock.advance(59)
        assert limiter.check("u1").allowed  # t=59,用满
        assert limiter.check("u1").allowed is False  # t=59,被拒

        clock.advance(2)  # t=61:第 1 秒那条已经滚出窗口

        assert limiter.check("u1").allowed is True
        # 但第二条(t=59)还在窗口里,所以再要一次就该被拒
        assert limiter.check("u1").allowed is False

    def test_不同用户各算各的(self) -> None:
        clock = _Clock()
        limiter = _limiter(1, clock)

        assert limiter.check("u1").allowed
        assert limiter.check("u1").allowed is False
        # 另一个人不该被 u1 的额度拖累
        assert limiter.check("u2").allowed is True

    def test_被拒的请求不占用额度(self) -> None:
        """**被拒的那次不能也记一笔。**

        记了的话:一个持续重试的客户端会把自己的窗口一直往后推,
        于是它**永远解禁不了** —— 越急越出不来。
        """
        clock = _Clock()
        limiter = _limiter(1, clock)

        assert limiter.check("u1").allowed
        for _ in range(10):
            assert limiter.check("u1").allowed is False

        clock.advance(60.1)
        assert limiter.check("u1").allowed is True

    def test_长时间不来的用户不会把内存撑爆(self) -> None:
        """扫一遍清掉过期条目 —— 否则每个来过的用户都留一个永久条目。

        清扫是**按检查次数**触发的,所以要打够次数才能观察到它。
        这里用 `SWEEP_EVERY` 而不是写死一个数:清扫频率是实现的细节,
        测试不该把它钉死。
        """
        clock = _Clock()
        limiter = _limiter(5, clock, window=60.0)

        for i in range(300):
            limiter.check(f"user-{i}")
        assert limiter.tracked_keys == 300

        clock.advance(61)
        for _ in range(SlidingWindowLimiter.SWEEP_EVERY):
            limiter.check("probe")

        # 那 300 个都已经滚出窗口,应当只剩 probe 这一个
        assert limiter.tracked_keys == 1

    def test_凭空构造非法额度要当场报错(self) -> None:
        """额度 0 会被读成「谁都别想用」—— 而配置项写成 0 时人想的往往是
        「不限流」。这种歧义必须挡在构造处,不能等线上才发现。"""
        with pytest.raises(ValueError):
            SlidingWindowLimiter(limit=0, window_seconds=60.0)

    def test_决策对象带着还要等多久(self) -> None:
        clock = _Clock()
        limiter = _limiter(1, clock)
        limiter.check("u1")

        decision = limiter.check("u1")
        assert isinstance(decision, RateDecision)
        assert decision.allowed is False
        assert decision.retry_after_seconds > 0

        # 放行时不需要等
        assert limiter.check("u2").retry_after_seconds == 0.0


class TestChatRateLimit:
    def test_额度用完之后的问答返回429(self, client, tiny_limit: SlidingWindowLimiter) -> None:
        """用「你好」这种纯问候语:它命中的是固定答复,不走检索也不调模型,
        所以这条测试只测限流,不会慢在别处。"""
        headers = auth_header(register(client, "ratelimit-user")["access_token"])

        for _ in range(2):
            resp = client.post(CHAT, json={"question": "你好"}, headers=headers)
            assert resp.status_code == 200

        resp = client.post(CHAT, json={"question": "你好"}, headers=headers)

        assert resp.status_code == 429
        body = resp.json()
        assert body["code"] == "rate_limited"
        # 前端要能告诉用户「等多久」,所以这个数必须在响应体里
        assert body["details"]["retry_after_seconds"] > 0
        # **而通用客户端读的是响应头,不是响应体。** 只把「等多久」放在我们自己
        # 的 JSON 里,等于假设调用方一定读过我们的文档；`Retry-After` 是这件事
        # 的标准说法,代理和 SDK 都认。
        assert resp.headers["Retry-After"] == "60"

    def test_额度是按用户各算各的(self, client, tiny_limit: SlidingWindowLimiter) -> None:
        """**别人把额度用光了,不该连累我。**"""
        first = auth_header(register(client, "ratelimit-a")["access_token"])
        second = auth_header(register(client, "ratelimit-b")["access_token"])

        for _ in range(2):
            client.post(CHAT, json={"question": "你好"}, headers=first)
        assert client.post(CHAT, json={"question": "你好"}, headers=first).status_code == 429

        assert client.post(CHAT, json={"question": "你好"}, headers=second).status_code == 200

    def test_别的端点不受这个额度影响(self, client, tiny_limit: SlidingWindowLimiter) -> None:
        """**这条在钉一个真实后果,不是洁癖。**

        知识库管理页每 1.5 秒轮询一次文档状态(`pages/KnowledgeBases.tsx`),
        那是 **40 次/分钟** —— 要是把限流挂成全局中间件,页面会**自己把自己
        打成 429**,而用户什么都没多做。所以限流只挂在问答那一个端点上。
        """
        headers = auth_header(register(client, "ratelimit-browser")["access_token"])

        for _ in range(10):
            assert client.get("/api/sessions", headers=headers).status_code == 200

    def test_剩余不足一秒时提示不说等零秒(self, client) -> None:
        """取整的边界。

        **「请等 0 秒再试」是一句自相矛盾的话** —— 让你等,但等 0 秒。
        用户看到的每一句提示都该是能照着做的,所以向上取整、并且至少 1 秒。
        """
        short = SlidingWindowLimiter(limit=1, window_seconds=0.3)
        app.dependency_overrides[deps.get_rate_limiter] = lambda: short
        try:
            headers = auth_header(register(client, "ratelimit-rounding")["access_token"])
            first = client.post(CHAT, json={"question": "你好"}, headers=headers)
            assert first.status_code == 200

            resp = client.post(CHAT, json={"question": "你好"}, headers=headers)
            assert resp.status_code == 429
            # 剩余 0.3 秒不到,但不能写成「等 0 秒」
            assert "等 1 秒" in resp.json()["message"], resp.json()["message"]
        finally:
            app.dependency_overrides.pop(deps.get_rate_limiter, None)

    def test_限流关着的时候一条都不拦(self, client) -> None:
        """`RATE_LIMIT_QPM=0` → lifespan 不建限流器 → 依赖直接放行。"""
        app.dependency_overrides[deps.get_rate_limiter] = lambda: None
        try:
            headers = auth_header(register(client, "ratelimit-off")["access_token"])
            for _ in range(6):
                resp = client.post(CHAT, json={"question": "你好"}, headers=headers)
                assert resp.status_code == 200
        finally:
            app.dependency_overrides.pop(deps.get_rate_limiter, None)

    def test_登录接口连着试密码会被拦住(self, client) -> None:
        """**登录是唯一一个不需要先登录的入口**,也是最该限流的一个。

        用错的密码打 —— 因为暴力破解正是这个样子,而这条测试要证明的
        就是「试密码这件事本身会被拦住」,不是「登录坏了」。
        """
        small = SlidingWindowLimiter(limit=2, window_seconds=60.0)
        app.dependency_overrides[deps.get_login_rate_limiter] = lambda: small
        try:
            wrong = {"username": "nobody", "password": "wrong-pw-123"}
            for _ in range(2):
                assert client.post("/api/auth/login", json=wrong).status_code == 401

            resp = client.post("/api/auth/login", json=wrong)

            assert resp.status_code == 429
            assert resp.json()["code"] == "rate_limited"
        finally:
            app.dependency_overrides.pop(deps.get_login_rate_limiter, None)

    def test_登录限流与问答限流互不影响(self, client) -> None:
        """**两个额度是分开的**,别共用一个计数器。

        共用的后果很具体:一个人问了几轮问题,就把自己**登录**的额度也花掉了
        —— 于是他会被自己刚才的提问锁在登录页外面,而且看不出是为什么。
        """
        # ⚠️ **必须先把实例建出来再交出去。** 写成
        # `lambda: SlidingWindowLimiter(...)` 的话,**每个请求都会拿到一个新建的
        # 限流器**,计数永远从零开始 —— 于是这个测试会以为自己验证了隔离,
        # 实际上什么都没限。第一版就是这么写的,当场报「第二次登录还是 401」。
        login_limiter = SlidingWindowLimiter(limit=1, window_seconds=60.0)
        app.dependency_overrides[deps.get_login_rate_limiter] = lambda: login_limiter
        try:
            headers = auth_header(register(client, "ratelimit-independent")["access_token"])
            # 问答额度由 `RATE_LIMIT_QPM` 决定,远远用不完
            assert client.post(CHAT, json={"question": "你好"}, headers=headers).status_code == 200

            # 但登录额度是 1,试一次错的就满了
            wrong = {"username": "nobody", "password": "wrong-pw-123"}
            assert client.post("/api/auth/login", json=wrong).status_code == 401
            assert client.post("/api/auth/login", json=wrong).status_code == 429

            # 而问答那边**没有**被登录的次数拖累
            assert client.post(CHAT, json={"question": "你好"}, headers=headers).status_code == 200
        finally:
            app.dependency_overrides.pop(deps.get_login_rate_limiter, None)

    def test_启动时按配置建好了限流器(self, client, settings) -> None:
        """钉住 lifespan 那一步真的做了 —— 否则「配置项在、线没接」会重演。"""
        assert isinstance(app.state.rate_limiter, SlidingWindowLimiter)
        assert app.state.rate_limiter.limit == settings.RATE_LIMIT_QPM

    def test_配置为0时启动不建限流器(self, client, settings, monkeypatch) -> None:
        """`RATE_LIMIT_QPM=0` 那一支:**重进一次 lifespan**,确认限流器是 `None`。

        **为什么非得重进 lifespan**:限流器是**启动时**建的,而 `client` 夹具
        已经用默认配置起过一次了 —— 在同一个 app 上事后改 `app.state` 是测不出
        「启动那一支走对了没有」的(那测的是我们自己的赋值)。

        `client` 夹具在这里的用处**不是它给的那个 client**,而是它做的隔离接线:
        把 `Database` 和 settings 的每一份引用都换成临时的那套。**没有它,
        重进 lifespan 会建表建到开发库上** —— 那正是 conftest 里那段长注释
        在防的事。所以这条测试挂在 `client` 上,哪怕它一个请求都不发。

        **两头都钉**:只测「0 → None」的话,一个「永远返回 None」的实现也能过。
        """
        monkeypatch.setattr(settings, "RATE_LIMIT_QPM", 0)
        with TestClient(app):
            assert app.state.rate_limiter is None

        monkeypatch.setattr(settings, "RATE_LIMIT_QPM", 7)
        with TestClient(app):
            assert isinstance(app.state.rate_limiter, SlidingWindowLimiter)
            assert app.state.rate_limiter.limit == 7
