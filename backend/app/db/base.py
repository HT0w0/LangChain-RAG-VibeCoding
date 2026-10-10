"""数据库引擎与会话管理。

**用 async SQLAlchemy + aiosqlite**,而不是同步 + 线程池。理由:本项目的
问答链路本身就是 async 的(SSE 流式 + httpx 调模型),如果 DB 是同步的,
每个环节都要 `to_thread` 包一层,漏掉一处就阻塞整个事件循环——而这种疏漏
在单机低并发下**测不出来**,只在真实并发下暴露。统一 async 让「漏包」这类
错误根本不可能发生。

SQLite 的同步写会串行化,但本项目写入量极小(单条会话/消息),读靠 WAL 并发,
瓶颈不在这里。
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import Settings
from app.core.config import settings as global_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """所有 ORM 模型的基类。"""


def build_async_url(settings: Settings) -> URL | str:
    """构造异步驱动可用的 URL。

    用 `URL.create()` 而非字符串拼接:仓库所在路径**可能带空格**,直接拼进
    URL 字符串会被 SQLAlchemy 解析器在空格处截断。`URL.create` 会正确转义。
    (这里原先写的是当时那台机器的完整路径 —— 它**已经在这台机器上踩过**,
     所以这个坑不是假设。但路径本身不必留在仓库里:换个目录就过期了,
     而仓库要公开。)
    """
    raw = settings.DATABASE_URL
    if raw.startswith("sqlite:///"):
        # 相对路径按项目根解析,不受 uvicorn 启动目录影响
        return URL.create("sqlite+aiosqlite", database=str(settings.sqlite_path))
    if raw.startswith("sqlite+aiosqlite:///"):
        return raw
    return raw


def _register_sqlite_pragmas(engine: AsyncEngine) -> None:
    """注册 SQLite 连接级 pragma。

    三个都要开:

    - `journal_mode=WAL` —— 读并发 + 后台写。默认的 rollback journal 下,
      一个写事务会阻塞所有读,而入库是后台长任务,会把问答查询全卡住。
    - `foreign_keys=ON` —— **SQLite 默认关闭外键约束**。不开的话
      `ON DELETE CASCADE` 完全不生效,删用户留一堆孤儿会话,而且不报错。
    - `synchronous=NORMAL` —— WAL 下这个档位是安全的(断电最多丢最后一个
      已提交事务,不会损坏库),性能比 FULL 好很多。
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_conn, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_conn.cursor()
        try:
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA synchronous=NORMAL")
            cursor.execute("PRAGMA busy_timeout=5000")  # 写锁竞争时等 5s 而非立即报错
        finally:
            cursor.close()


class Database:
    """引擎 + 会话工厂的持有者。

    做成类而非模块级全局,是为了让测试能起独立实例(各自一个临时库文件),
    互不干扰。模块末尾提供一个默认实例供应用使用。
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self.engine: AsyncEngine = create_async_engine(
            build_async_url(settings),
            echo=False,
            pool_pre_ping=True,
            # ⚠️ **连接池容量必须显式给。**
            #
            # 这里原先的注释是「SQLite 文件库不需要连接池扩容;单机场景下连接数
            # 不是瓶颈」—— **2026-10-10 的并发压测把这句话证伪了**:
            #
            #     sqlalchemy.exc.TimeoutError: QueuePool limit of size 5 overflow 10
            #     reached, connection timed out, timeout 30.00
            #
            # 默认容量是 15(5 + 10),而一次问答在**检索阶段**要独占一条连接。
            # 并发一上 20,第 16 个请求就开始排队,排到 `pool_timeout` 才报错 ——
            # 而那个默认值是 **30 秒**,所以用户看到的是「转半分钟然后 500」。
            # 压测里它精确对上:服务端 572 次池超时 ↔ 客户端 572 次 HTTP 500。
            #
            # 容量按**目标并发数**给:需求的口径是「100 人在线」。20 条常驻
            # (SQLite 连接很轻,常驻不占什么),另外 100 条应付突发。
            #
            # 代价说清楚:这**不是在治本**。真正的毛病是「一次请求从头到尾
            # 握着连接」,扩池只是把天花板抬到够高。要治本得缩短持有时间
            # (检索完就还、流式期间不持有),那是另一个改动。
            pool_size=20,
            max_overflow=100,
            # 池子真被打满时,**快点报错**。默认 30 秒意味着用户盯着转圈
            # 半分钟才等到一个 500 —— 一个 30 秒的 500 比一个 5 秒的 500
            # 糟得多,而且它掩盖了「是池子满了」这个原因(被当成网络慢)。
            pool_timeout=10.0,
            connect_args={"check_same_thread": False},
        )
        if str(self.engine.url).startswith("sqlite"):
            _register_sqlite_pragmas(self.engine)

        self.session_factory = async_sessionmaker(
            self.engine,
            class_=AsyncSession,
            expire_on_commit=False,  # commit 后仍可读对象属性,省掉大量 refresh
            autoflush=False,
        )

    async def create_all(self) -> None:
        # 必须先 import models 让映射类注册到 Base.metadata 上
        from app.db import models  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.debug("数据表已就绪", extra={"extra_fields": {"tables": sorted(Base.metadata.tables)}})

    async def drop_all(self) -> None:
        from app.db import models  # noqa: F401

        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)

    @asynccontextmanager
    async def session(self) -> AsyncIterator[AsyncSession]:
        """事务边界显式化的会话。

        异常时回滚。注意 `async with session_factory()` 退出时只关闭连接,
        **不会自动回滚未提交的事务**——对 SQLite 来说未提交事务会一直挂着
        写锁,把后续写入全堵死。所以这里显式处理。
        """
        session = self.session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    async def dispose(self) -> None:
        await self.engine.dispose()


database = Database(global_settings)
