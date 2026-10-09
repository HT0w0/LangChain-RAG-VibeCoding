"""认证流程的接口测试。"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers import (
    TEST_ADMIN_PASSWORD,
    TEST_ADMIN_USERNAME,
    auth_header,
    register,
)


class TestRegister:
    def test_creates_user_and_returns_token(self, client: TestClient) -> None:
        body = register(client, "alice")
        assert body["access_token"]
        assert body["token_type"] == "bearer"
        assert body["expires_in"] > 0
        assert body["user"]["username"] == "alice"
        # 注册接口**永远**只能建普通用户,否则 RBAC 就是摆设
        assert body["user"]["role"] == "user"

    def test_duplicate_username_conflicts(self, client: TestClient) -> None:
        register(client, "bob")
        r = client.post("/api/auth/register", json={"username": "bob", "password": "another-pw"})
        assert r.status_code == 409
        assert r.json()["code"] == "conflict"

    def test_username_uniqueness_is_case_insensitive(self, client: TestClient) -> None:
        register(client, "Carol")
        r = client.post("/api/auth/register", json={"username": "carol", "password": "some-pw"})
        assert r.status_code == 409

    def test_rejects_short_password(self, client: TestClient) -> None:
        r = client.post("/api/auth/register", json={"username": "shorty", "password": "12345"})
        assert r.status_code == 422

    def test_rejects_bad_username(self, client: TestClient) -> None:
        for bad in ("ab", "x" * 33, "has space", "semi;colon", ""):
            r = client.post("/api/auth/register", json={"username": bad, "password": "valid-pw-1"})
            assert r.status_code == 422, f"{bad!r} 应被拒绝"

    def test_accepts_chinese_username(self, client: TestClient) -> None:
        """中文电商场景,强制英文用户名会让运营同事没法用。"""
        body = register(client, "客服小张")
        assert body["user"]["username"] == "客服小张"

    def test_client_cannot_self_assign_admin(self, client: TestClient) -> None:
        """请求体里塞 role 不应有任何效果。

        Pydantic 默认忽略未声明的字段(不是报错),所以这条不测「被拒绝」,
        而是测「没生效」——这才是真正要防的东西。
        """
        r = client.post(
            "/api/auth/register",
            json={"username": "sneaky", "password": "valid-pw-1", "role": "admin", "is_active": True},
        )
        assert r.status_code == 201
        assert r.json()["user"]["role"] == "user"


class TestLogin:
    def test_success(self, client: TestClient) -> None:
        register(client, "dave", "dave-password")
        r = client.post("/api/auth/login", json={"username": "dave", "password": "dave-password"})
        assert r.status_code == 200
        assert r.json()["user"]["username"] == "dave"

    def test_admin_seed_account_can_login(self, client: TestClient) -> None:
        """需求指定 admin/123456 可用(测试里用等价的自定义凭据)。"""
        r = client.post(
            "/api/auth/login",
            json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
        )
        assert r.status_code == 200
        assert r.json()["user"]["role"] == "admin"

    def test_wrong_password_rejected(self, client: TestClient) -> None:
        register(client, "erin", "erin-password")
        r = client.post("/api/auth/login", json={"username": "erin", "password": "wrong"})
        assert r.status_code == 401

    def test_unknown_user_gives_identical_response(self, client: TestClient) -> None:
        """**用户枚举防护**。

        若「用户名不存在」和「密码错误」返回不同的文案或状态码,接口就变成了
        账号探测器:攻击者先确认哪些账号存在,再针对性撞库。所以两者必须
        逐字节一致。
        """
        register(client, "frank", "frank-password")
        wrong_pw = client.post("/api/auth/login", json={"username": "frank", "password": "nope"})
        no_such = client.post("/api/auth/login", json={"username": "ghost", "password": "nope"})
        assert wrong_pw.status_code == no_such.status_code == 401
        assert wrong_pw.json() == no_such.json()

    def test_last_login_updated(self, client: TestClient) -> None:
        register(client, "gina", "gina-password")
        r = client.post("/api/auth/login", json={"username": "gina", "password": "gina-password"})
        assert r.json()["user"]["last_login_at"] is not None


class TestMe:
    def test_requires_token(self, client: TestClient) -> None:
        r = client.get("/api/auth/me")
        assert r.status_code == 401
        assert r.json()["code"] == "missing_token"

    def test_returns_current_user(self, client: TestClient) -> None:
        token = register(client, "henry")["access_token"]
        r = client.get("/api/auth/me", headers=auth_header(token))
        assert r.status_code == 200
        assert r.json()["username"] == "henry"

    def test_garbage_token_rejected(self, client: TestClient) -> None:
        r = client.get("/api/auth/me", headers=auth_header("not.a.token"))
        assert r.status_code == 401

    def test_timestamps_are_utc_qualified(self, client: TestClient) -> None:
        """时间必须带时区。

        不带偏移量的话,前端 `new Date("2026-10-08T12:00:00")` 会按**本地时区**
        解析,东八区用户看到的时间整体偏 8 小时。
        """
        token = register(client, "ivan")["access_token"]
        created = client.get("/api/auth/me", headers=auth_header(token)).json()["created_at"]
        assert created.endswith("+00:00") or created.endswith("Z")


class TestChangePassword:
    def test_success_and_old_password_stops_working(self, client: TestClient) -> None:
        token = register(client, "jane", "original-pw")["access_token"]

        r = client.post(
            "/api/auth/change-password",
            json={"old_password": "original-pw", "new_password": "brand-new-pw"},
            headers=auth_header(token),
        )
        assert r.status_code == 204

        assert client.post(
            "/api/auth/login", json={"username": "jane", "password": "original-pw"}
        ).status_code == 401
        assert client.post(
            "/api/auth/login", json={"username": "jane", "password": "brand-new-pw"}
        ).status_code == 200

    def test_wrong_old_password_forbidden(self, client: TestClient) -> None:
        token = register(client, "kyle", "kyle-pw-1")["access_token"]
        r = client.post(
            "/api/auth/change-password",
            json={"old_password": "not-it", "new_password": "whatever-2"},
            headers=auth_header(token),
        )
        assert r.status_code == 403
        assert r.json()["code"] == "wrong_old_password"

    def test_rejects_same_password(self, client: TestClient) -> None:
        token = register(client, "laura", "laura-pw-1")["access_token"]
        r = client.post(
            "/api/auth/change-password",
            json={"old_password": "laura-pw-1", "new_password": "laura-pw-1"},
            headers=auth_header(token),
        )
        assert r.status_code == 422

    def test_rejects_weak_new_password(self, client: TestClient) -> None:
        token = register(client, "mike", "mike-pw-1")["access_token"]
        r = client.post(
            "/api/auth/change-password",
            json={"old_password": "mike-pw-1", "new_password": "123"},
            headers=auth_header(token),
        )
        assert r.status_code == 422

    def test_requires_authentication(self, client: TestClient) -> None:
        r = client.post(
            "/api/auth/change-password",
            json={"old_password": "a", "new_password": "b" * 8},
        )
        assert r.status_code == 401
