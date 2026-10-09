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

    用 `URL.create()` 而非字符串拼接:项目路径里**有空格**
    (`D:\\Claude Code\\LangChainRAG项目`),直接拼进 URL 字符串会被
    SQLAlchemy 解析器在空格处截断。`URL.create` 会正确转义。
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
            # SQLite 文件库不需要连接池扩容;单机场景下连接数不是瓶颈
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
