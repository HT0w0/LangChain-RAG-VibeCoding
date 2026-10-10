"""连接池容量够不够装下目标并发数。

**这个文件存在的理由是一次实测。**

压测(`scripts/loadtest.py`,2026-10-10)在并发 20 附近把连接池打爆:

    sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10 reached,
    connection timed out, timeout 30.00

池容量是 SQLAlchemy 的默认值 15(5 + 10),第 16 个并发请求开始排队,
**30 秒后**才超时抛 500。而 `db/base.py` 里当时的注释写着:

    # SQLite 文件库不需要连接池扩容;单机场景下连接数不是瓶颈

**压测把这句话证伪了。** 所以这里钉住的是一条**行为**,不是某个配置数字:
目标并发数的人一起持连接,不该有人在排队。
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

import pytest
from app.core.config import Settings
from app.db.base import Database
from sqlalchemy import text

#: 目标并发数 —— 需求里的口径是「100 人在线」。
TARGET_CONCURRENCY = 100

#: 每个持有者占住连接的时长。取得比一次请求的检索阶段长一点,
#: 这样才有机会把池子占满。
HOLD_SECONDS = 1.0

#: 多久看一眼池子的签出数。取得比 `HOLD_SECONDS` 小两个量级,
#: 保证峰值一定能被采到。
SAMPLE_INTERVAL = 0.005


@pytest.fixture
async def pool_db(settings: Settings) -> AsyncIterator[Database]:
    # `ensure_dirs()` 不能省:临时库的目录还不存在时,SQLite 报的是
    # `unable to open database file` —— 一句和「目录没建」毫无关联的话。
    # 项目里每个直接建库的夹具都先调它,照抄。
    settings.ensure_dirs()
    database = Database(settings)
    await database.create_all()
    try:
        yield database
    finally:
        await database.dispose()


async def _hold(db: Database) -> None:
    """占住一条连接 `HOLD_SECONDS` 秒,模拟「检索阶段一直握着连接」。

    **必须真的执行一条语句。** 只 `async with db.session()` 而不碰数据库的话,
    SQLAlchemy 不会签出连接 —— 那样这个测试测的是空气。
    """
    async with db.session() as session:
        await session.execute(text("SELECT 1"))
        await asyncio.sleep(HOLD_SECONDS)


class TestPoolCapacity:
    async def test_目标并发数能同时签出连接(self, pool_db: Database) -> None:
        """100 个并发持有者要能**同时**签出连接 —— 一个都不该在排队。

        **判据是「池子同时签出的峰值」,不是耗时。**

        ⚠️ 第一版用的是墙钟耗时(「整批做完要 < 3 秒」)。那是**会偶发变红**的:
        质量评审跑 9 次,有 1 次因为机器被别的负载拖慢、跑到 10.1 秒而失败。
        **一条装在提交闸门里、靠机器心情变红的测试比没有测试更坏** ——
        它红了之后人会开始习惯性放行,那时候闸门就废了。

        换成峰值签出数之后,判据**与机器快慢无关**:
          * 容量够(≥ 100):100 个持有者能同时拿到连接,峰值到 **100**;
          * 容量是默认的 15:峰值封顶在 **15**,后面的都在排队。

        为什么池子小了会表现为「慢」而不是「错」,见文件头:请求不会立刻失败,
        它会一直排到 `pool_timeout` 才炸 —— 所以这个 bug 在超时前一直是隐形的。
        """
        stop = asyncio.Event()
        peak = 0

        async def sample() -> None:
            nonlocal peak
            while not stop.is_set():
                peak = max(peak, pool_db.engine.pool.checkedout())
                await asyncio.sleep(SAMPLE_INTERVAL)

        sampler = asyncio.create_task(sample())
        try:
            # 超时只是兜底:容量不够时这一批**会**做完,只是排着队做完(约 7–15 秒),
            # 所以不会卡死在这里 —— 判定交给下面那句断言,它的话说得清楚得多。
            await asyncio.wait_for(
                asyncio.gather(*(_hold(pool_db) for _ in range(TARGET_CONCURRENCY))),
                timeout=60.0,
            )
        finally:
            stop.set()
            await sampler

        assert peak >= TARGET_CONCURRENCY, (
            f"池子同时只签出了 {peak} 条连接,而并发持有者是 {TARGET_CONCURRENCY} 个 —— "
            f"说明容量不够,后面的请求在排队。\n"
            f"  默认容量是 15(5 + 10),`app/db/base.py` 里显式给了 pool_size / max_overflow。"
        )
