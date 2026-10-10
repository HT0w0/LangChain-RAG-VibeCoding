"""密码哈希的并发行为:挪出事件循环 + 并发闸门。

`test_auth.py` 测的是接口「走不走得通」;这里测的是 `AuthService` 里那两处
**看不见的**改动 —— 它们坏了接口照样 200,只有在并发下才露馅:

* argon2 是**同步**的,直接在 async 函数里调会独占事件循环 —— 期间所有其他
  请求(含正在流式的 SSE)一起卡住;
* argon2 每次哈希按 `ARGON2_MEMORY_COST` 吃内存(默认 64MB),不限并发会把
  内存吃爆 —— 那只是把「事件循环阻塞」换成了更难查的「内存压力」。

两条都属于 §5.1 说的「写错了但不报错」:单机低并发下测不出来。所以各有一条
测试钉住 —— 判据用「事件循环有没有继续转」「同时跑几个哈希」,都不用墙钟耗时,
这样不会因为机器快慢而偶发变红。
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import AsyncIterator

import pytest
from app.core.config import Settings
from app.db.base import Database
from app.services.auth import AuthService


@pytest.fixture
async def auth_service(settings: Settings) -> AsyncIterator[AuthService]:
    """绑定到临时库的服务实例。

    这里其实用不到库 —— `_hash` / `_verify` 不碰 session。但给一个真会话比传
    `None` 诚实:哪天这两个方法开始用 session,测试不会突然报一句不相干的错。
    """
    settings.ensure_dirs()
    database = Database(settings)
    await database.create_all()
    try:
        async with database.session() as session:
            yield AuthService(session, settings)
    finally:
        await database.dispose()


class TestHashingOffloadsEventLoop:
    async def test_哈希期间事件循环仍在转(
        self, auth_service: AuthService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """哈希必须在线程池里做,**不能占着事件循环**。

        判据是「哈希进行期间事件循环有没有继续走动」,不是耗时:把
        `hash_password` 换成一个固定阻塞 0.3 秒的桩,若它跑在循环里,心跳一次
        都不会跳;跑在线程里,心跳照跳。这样与机器快慢无关。
        """
        import app.services.auth as auth_module

        def blocking_hash(password: str, settings: Settings) -> str:
            time.sleep(0.3)
            return "probe"

        monkeypatch.setattr(auth_module, "hash_password", blocking_hash)

        ticks = 0
        stop = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.002)

        hb = asyncio.create_task(heartbeat())
        await auth_service._hash("pw")
        stop.set()
        await hb

        assert ticks > 5, (
            f"哈希期间事件循环只转了 {ticks} 拍 —— 说明 hash 是在事件循环里同步跑的,"
            "会连带卡住所有其他请求。应当经 anyio.to_thread 挪进线程池。"
        )

    async def test_校验哈希期间事件循环仍在转(
        self, auth_service: AuthService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`_verify` 与 `_hash` 是同一条路径上的两个入口,分别钉住。

        只测 `_hash` 的话,哪天有人把 `_verify` 改回同步直调,登录接口照样能用,
        但**每次登录都会卡住事件循环** —— 一份没被覆盖的静默回退。
        """
        import app.services.auth as auth_module

        def blocking_verify(password: str, stored: str, settings: Settings) -> bool:
            time.sleep(0.3)
            return False

        monkeypatch.setattr(auth_module, "verify_password", blocking_verify)

        ticks = 0
        stop = asyncio.Event()

        async def heartbeat() -> None:
            nonlocal ticks
            while not stop.is_set():
                ticks += 1
                await asyncio.sleep(0.002)

        hb = asyncio.create_task(heartbeat())
        await auth_service._verify("pw", "stored-hash")
        stop.set()
        await hb

        assert ticks > 5, (
            f"校验期间事件循环只转了 {ticks} 拍 —— verify 在事件循环里同步跑了。"
        )


class TestHashingConcurrencyGate:
    async def test_并发哈希数受内存预算限制(
        self, auth_service: AuthService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """同时进行的哈希数由内存预算算出来,不能无限并发。

        参数调小才观察得到闸门:测试默认参数(8MB/次)折算出 64,比线程池上限还高,
        永远不触发。这里把每次哈希折算成 256MB,512MB 预算就只够 2 个并发。

        断言「峰值恰好等于 2」同时钉住两件事:闸门**确实在限**(否则 8 个一起上,
        峰值是 8),而且**确实允许到预算值**(否则退化成 1,并发等于没有)。
        """
        import app.services.auth as auth_module

        tight = auth_service._settings.model_copy(update={"ARGON2_MEMORY_COST": 262144})
        # 闸门是进程级、只按第一次见到的参数建一次。不重置的话这里拿到的是
        # 别处缓存的旧闸门,参数改了也不跟着变 —— 那样测试测的是上一次的设置。
        monkeypatch.setattr(auth_module, "_hash_limiter", None)
        monkeypatch.setattr(auth_service, "_settings", tight)

        lock = threading.Lock()
        running = 0
        peak = 0

        def blocking_hash(password: str, settings: Settings) -> str:
            nonlocal running, peak
            with lock:
                running += 1
                peak = max(peak, running)
            time.sleep(0.2)
            with lock:
                running -= 1
            return "probe"

        monkeypatch.setattr(auth_module, "hash_password", blocking_hash)

        await asyncio.gather(*(auth_service._hash(f"pw-{i}") for i in range(8)))

        assert peak == 2, (
            f"同时跑了 {peak} 个哈希 —— 预算 512MB / 每次 256MB 应只允许 2 个并发。"
        )
