"""应用装配冒烟测试。

覆盖的是「**能起来吗**」——路由挂载、中间件、异常处理器、生命周期。
这类问题在单元测试里看不出来,却会让服务根本起不来。

注意这里用的是真实 `settings`(读 .env),不是 conftest 的隔离夹具:
本文件的目的就是验证**真实装配路径**,隔离掉配置就失去意义了。
"""

from __future__ import annotations

import pytest
from app.core.exceptions import ForbiddenError, NotFoundError
from app.core.logging import get_trace_id
from app.main import app
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client() -> TestClient:
    # with 块才会触发 lifespan(建目录、初始化日志),不加就是个空壳
    with TestClient(app) as c:
        yield c


def test_liveness_is_cheap(client: TestClient) -> None:
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_readiness_reports_checks(client: TestClient) -> None:
    r = client.get("/api/health/ready")
    assert r.status_code == 200
    body = r.json()
    assert "checks" in body
    assert body["checks"]["config"]["llm_model"]


def test_trace_id_header_present(client: TestClient) -> None:
    """每个响应都必须带 trace_id——前端报错时靠它定位,省掉「复现一下」。"""
    r = client.get("/api/health")
    assert r.headers.get("X-Trace-Id")


def test_incoming_trace_id_is_propagated(client: TestClient) -> None:
    """上游传了 trace_id 就用它,不要另起一个——否则跨服务追不了链路。"""
    r = client.get("/api/health", headers={"X-Trace-Id": "abc123def456"})
    assert r.headers["X-Trace-Id"] == "abc123def456"


def test_timing_header_present(client: TestClient) -> None:
    r = client.get("/api/health")
    assert float(r.headers["X-Process-Time-Ms"]) >= 0.0


def test_app_error_becomes_structured_json(client: TestClient) -> None:
    """异常响应体恒为 {code, message},前端按 code 分支而非匹配文案。"""

    @app.get("/api/_test/forbidden")
    async def _boom() -> None:
        raise ForbiddenError("没有权限")

    @app.get("/api/_test/missing")
    async def _missing() -> None:
        raise NotFoundError("找不到")

    try:
        r = client.get("/api/_test/forbidden")
        assert r.status_code == 403
        assert r.json() == {"code": "forbidden", "message": "没有权限"}

        r2 = client.get("/api/_test/missing")
        assert r2.status_code == 404
        assert r2.json()["code"] == "not_found"
    finally:
        # 测试路由不能留在 app 上,否则污染后续测试与 /docs
        app.router.routes = [rt for rt in app.router.routes if not getattr(rt, "path", "").startswith("/api/_test")]  # type: ignore[attr-defined]


def test_trace_id_is_set_during_request(client: TestClient) -> None:
    """请求处理期间 contextvar 必须已就绪,否则日志里的 trace_id 全是 '-'。"""

    @app.get("/api/_test/trace")
    async def _trace() -> dict[str, str]:
        return {"trace_id": get_trace_id()}

    try:
        r = client.get("/api/_test/trace")
        assert r.json()["trace_id"] not in ("", "-")
    finally:
        app.router.routes = [rt for rt in app.router.routes if not getattr(rt, "path", "").startswith("/api/_test")]  # type: ignore[attr-defined]
