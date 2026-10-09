"""测试常量与辅助函数。

单独放一个模块而不是塞进 conftest:conftest 里应该只有**夹具**。从 conftest
直接 import 会形成隐式依赖,而且 pytest 对 conftest 的加载有特殊处理,容易出现
「本地能过、换个入口跑就 ImportError」。

依赖方向是单向的:`conftest` → `helpers`,反向不成立。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

# 测试用的管理员凭据。刻意与 .env 里的值不同——依赖 .env 的话,
# 别人改了 .env 测试就会莫名其妙挂掉,而且看不出原因。
TEST_ADMIN_USERNAME = "admin"
TEST_ADMIN_PASSWORD = "admin-test-pw"

__all__ = [
    "TEST_ADMIN_PASSWORD",
    "TEST_ADMIN_USERNAME",
    "admin_token",
    "auth_header",
    "embedding_entries",
    "in_app_loop",
    "latest_metric",
    "login",
    "metric_rows",
    "register",
    "semantic_entries",
]


def in_app_loop(client: TestClient, coro_fn):  # type: ignore[no-untyped-def]
    """在应用自己的事件循环里跑一段协程,并返回结果。

    向量库和数据库引擎都绑定在 TestClient 启动的那个循环上,在测试进程里
    另起 `asyncio.run` 会因为跨循环使用同一个 Chroma/SQLAlchemy 客户端而炸。
    `client.portal` 是 TestClient 暴露的、通往那个循环的入口。

    测试要**绕开接口直接看落库结果**时都得走这里,所以它放在共享模块里,
    而不是各测试文件各写一份。
    """
    return client.portal.call(coro_fn)  # type: ignore[attr-defined]


def embedding_entries(client: TestClient) -> int:
    """缓存表里 `namespace='embedding'` 的条数。

    **这是「向量缓存有没有接上」唯一能从外部观测到的证据。** 这个域只有
    那一层会写,所以「一条都没有」是二值且无歧义的;而「命中与否」观测不到
    ——命中时写进去的是同一个键、同一个值,行数一样。所以这个辅助函数
    只能证明写入侧,读取侧由 `test_cache.py::TestEmbeddingCache` 里那个
    计数替身负责。

    app 模块在函数体里 import:本模块的定位是「不依赖应用」的常量与辅助,
    模块导入期拉进 SQLAlchemy 会让 `import tests.helpers` 变重。
    """
    from app.db.base import database
    from app.db.models import CacheEntry
    from sqlalchemy import func, select

    async def _count() -> int:
        async with database.session() as session:
            return int(
                await session.scalar(
                    select(func.count())
                    .select_from(CacheEntry)
                    .where(CacheEntry.namespace == "embedding")
                )
                or 0
            )

    return in_app_loop(client, _count)


def semantic_entries(client: TestClient) -> int:
    """`cache_semantic` 表的条数——语义缓存**写入侧**接没接上的证据。

    与 `embedding_entries` 同一个道理:`cache_semantic` 只有这一层会写,
    所以「一条都没有」是二值且无歧义的。读取侧这个函数看不见(命中与否
    写进去的都是一行),它由 `test_rag.py::TestSemanticCacheIsWiredIntoRAG`
    和 `test_chat_api.py` 里那条「改掉向量再问」的用例负责。
    """
    from app.db.base import database
    from app.db.models import CacheSemanticEntry
    from sqlalchemy import func, select

    async def _count() -> int:
        async with database.session() as session:
            return int(
                await session.scalar(
                    select(func.count()).select_from(CacheSemanticEntry)
                )
                or 0
            )

    return in_app_loop(client, _count)


def metric_rows(client: TestClient) -> list[dict]:
    """全部请求指标,按写入顺序。**从字段定义生成**,不是手抄一份列名。

    手抄的话,加一个字段时这里不会红,而用例会继续断言一个旧形状——
    下一个改这个字段的人得先发现这份抄本。
    """
    from app.db.base import database
    from app.db.models import RequestMetric
    from sqlalchemy import select

    columns = [c.name for c in RequestMetric.__table__.columns]

    async def _rows() -> list[dict]:
        async with database.session() as session:
            result = await session.execute(select(RequestMetric).order_by(RequestMetric.id))
            return [
                {name: getattr(row, name) for name in columns} for row in result.scalars()
            ]

    return in_app_loop(client, _rows)


def latest_metric(client: TestClient) -> dict | None:
    rows = metric_rows(client)
    return rows[-1] if rows else None


def auth_header(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def register(client: TestClient, username: str, password: str = "user-pw-123") -> dict:
    """注册并返回响应体(含 access_token)。"""
    r = client.post("/api/auth/register", json={"username": username, "password": password})
    assert r.status_code == 201, r.text
    return r.json()


def login(client: TestClient, username: str, password: str) -> dict:
    r = client.post("/api/auth/login", json={"username": username, "password": password})
    assert r.status_code == 200, r.text
    return r.json()


def admin_token(client: TestClient) -> str:
    return login(client, TEST_ADMIN_USERNAME, TEST_ADMIN_PASSWORD)["access_token"]
