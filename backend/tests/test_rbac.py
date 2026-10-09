"""权限控制测试。

**这一组是全项目最重要的回归测试。** 需求里「只有管理员能管理知识库」是一条
业务约束,但它的技术本质是「普通用户直接 curl 后端也拿不到」。前端路由守卫
挡不住任何人——所以这里全部绕过前端,直接打接口。
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from tests.helpers import (
    TEST_ADMIN_PASSWORD,
    TEST_ADMIN_USERNAME,
    admin_token,
    auth_header,
    login,
    register,
)

ADMIN_USERS = "/api/admin/users"


class TestAdminOnlyAccess:
    def test_admin_allowed(self, client: TestClient) -> None:
        r = client.get(ADMIN_USERS, headers=auth_header(admin_token(client)))
        assert r.status_code == 200
        assert r.json()["total"] >= 1

    def test_regular_user_forbidden(self, client: TestClient) -> None:
        token = register(client, "plain-user")["access_token"]
        r = client.get(ADMIN_USERS, headers=auth_header(token))
        assert r.status_code == 403
        assert r.json()["code"] == "forbidden"

    def test_anonymous_unauthorized(self, client: TestClient) -> None:
        r = client.get(ADMIN_USERS)
        assert r.status_code == 401

    def test_forged_token_unauthorized(self, client: TestClient) -> None:
        r = client.get(ADMIN_USERS, headers=auth_header("aaa.bbb.ccc"))
        assert r.status_code == 401

    def test_401_and_403_are_distinguishable(self, client: TestClient) -> None:
        """前端要靠状态码区分「该重新登录」和「该找管理员开权限」。

        混用的话,权限不足的用户会被反复弹回登录页,而真正过期的人
        看到的是「无权限」——两边都懵。
        """
        anon = client.get(ADMIN_USERS)
        user = client.get(ADMIN_USERS, headers=auth_header(register(client, "lucy")["access_token"]))
        assert anon.status_code == 401
        assert user.status_code == 403
        assert anon.json()["code"] != user.json()["code"]


class TestAccountState:
    def test_disabled_user_cannot_login(self, client: TestClient) -> None:
        reg = register(client, "to-be-disabled")
        uid = reg["user"]["id"]

        r = client.patch(
            f"{ADMIN_USERS}/{uid}/active",
            json={"is_active": False},
            headers=auth_header(admin_token(client)),
        )
        assert r.status_code == 200

        login_resp = client.post(
            "/api/auth/login", json={"username": "to-be-disabled", "password": "user-pw-123"}
        )
        assert login_resp.status_code == 401
        assert login_resp.json()["code"] == "account_disabled"

    def test_disabling_kills_existing_token_immediately(self, client: TestClient) -> None:
        """**这是「role 不放进 JWT」这个决定的直接后果,必须有测试守住。**

        如果权限信息编码在 token 里,被禁用的用户在 token 过期前(最长 12 小时)
        仍然能正常使用系统。对一个 admin 独占知识库管理的系统,这不可接受。
        """
        token = register(client, "revoked")["access_token"]
        uid = client.get("/api/auth/me", headers=auth_header(token)).json()["id"]

        assert client.get("/api/auth/me", headers=auth_header(token)).status_code == 200

        client.patch(
            f"{ADMIN_USERS}/{uid}/active",
            json={"is_active": False},
            headers=auth_header(admin_token(client)),
        )

        r = client.get("/api/auth/me", headers=auth_header(token))
        assert r.status_code == 401
        assert r.json()["code"] == "account_disabled"

    def test_deleted_user_token_rejected(self, client: TestClient) -> None:
        """token 签名有效但账号已被删除——不能当成有效登录。

        这条测的是「每个请求都回库查一次用户」的**代价换来了什么**:
        如果只信 token 里的 sub,一个已删除账号在 token 过期前仍是有效身份。
        """
        import app.core.deps as deps_module  # noqa: PLC0415
        from app.db.repositories.user import UserRepository  # noqa: PLC0415

        token = register(client, "ghost-user")["access_token"]
        uid = client.get("/api/auth/me", headers=auth_header(token)).json()["id"]

        async def _delete_user() -> None:
            async with deps_module.database.session() as s:
                user = await UserRepository(s).get_by_id(uid)
                assert user is not None
                await s.delete(user)
                await s.commit()

        # pytest 是同步的,应用跑在 TestClient 自己的事件循环里。
        # portal 就是那个循环的入口——直接用 asyncio.run 会开一个新循环,
        # 而 SQLAlchemy 的异步引擎绑定在原来那个上,必然报 "attached to a
        # different loop"。
        client.portal.call(_delete_user)  # type: ignore[attr-defined]

        r = client.get("/api/auth/me", headers=auth_header(token))
        assert r.status_code == 401
        assert r.json()["code"] == "user_not_found"


class TestRoleChangesTakeEffectImmediately:
    def test_promoted_user_gains_admin_access(self, client: TestClient) -> None:
        reg = register(client, "future-admin")
        token = reg["access_token"]
        uid = reg["user"]["id"]

        assert client.get(ADMIN_USERS, headers=auth_header(token)).status_code == 403

        r = client.patch(
            f"{ADMIN_USERS}/{uid}/role",
            json={"role": "admin"},
            headers=auth_header(admin_token(client)),
        )
        assert r.status_code == 200
        assert r.json()["role"] == "admin"

        # 同一个 token,权限立刻生效——不需要重新登录
        assert client.get(ADMIN_USERS, headers=auth_header(token)).status_code == 200

    def test_demoted_admin_loses_access_immediately(self, client: TestClient) -> None:
        """降权即时生效。这是把 role 放进 JWT 会破坏的性质。"""
        reg = register(client, "temp-admin")
        uid = reg["user"]["id"]
        admin_h = auth_header(admin_token(client))

        client.patch(f"{ADMIN_USERS}/{uid}/role", json={"role": "admin"}, headers=admin_h)
        temp_token = login(client, "temp-admin", "user-pw-123")["access_token"]
        assert client.get(ADMIN_USERS, headers=auth_header(temp_token)).status_code == 200

        client.patch(f"{ADMIN_USERS}/{uid}/role", json={"role": "user"}, headers=admin_h)
        assert client.get(ADMIN_USERS, headers=auth_header(temp_token)).status_code == 403


class TestSelfLockoutGuards:
    """防止一次误操作把系统变成「没有任何人能管理」。"""

    def test_admin_cannot_disable_self(self, client: TestClient) -> None:
        admin_h = auth_header(admin_token(client))
        me = client.get("/api/auth/me", headers=admin_h).json()

        r = client.patch(
            f"{ADMIN_USERS}/{me['id']}/active", json={"is_active": False}, headers=admin_h
        )
        assert r.status_code == 422
        assert r.json()["code"] == "cannot_disable_self"

    def test_admin_cannot_demote_self(self, client: TestClient) -> None:
        admin_h = auth_header(admin_token(client))
        me = client.get("/api/auth/me", headers=admin_h).json()

        r = client.patch(
            f"{ADMIN_USERS}/{me['id']}/role", json={"role": "user"}, headers=admin_h
        )
        assert r.status_code == 422
        assert r.json()["code"] == "cannot_demote_self"

    def test_cannot_disable_last_admin(self, client: TestClient) -> None:
        """另一个管理员不能把唯一的管理员禁掉。

        自锁防护只挡「操作自己」,挡不住「A 禁用 B,而 B 是最后一个管理员」。
        """
        admin_h = auth_header(admin_token(client))
        admin_id = client.get("/api/auth/me", headers=admin_h).json()["id"]

        # 造第二个管理员,用它来操作原管理员
        other = register(client, "second-admin")
        other_h = auth_header(other["access_token"])
        client.patch(
            f"{ADMIN_USERS}/{other['user']['id']}/role", json={"role": "admin"}, headers=admin_h
        )

        # 两个管理员都在,禁用其中一个应当允许
        assert client.patch(
            f"{ADMIN_USERS}/{admin_id}/active", json={"is_active": False}, headers=other_h
        ).status_code == 200

        # 现在只剩一个管理员,再禁用就应被拦下
        r = client.patch(
            f"{ADMIN_USERS}/{other['user']['id']}/active",
            json={"is_active": False},
            headers=other_h,
        )
        assert r.status_code == 422
        assert r.json()["code"] in {"cannot_disable_self", "last_admin"}


class TestInputValidation:
    def test_invalid_role_rejected_at_schema_layer(self, client: TestClient) -> None:
        """非法角色必须在入参层就被挡掉,不能流到服务层写进库。"""
        reg = register(client, "role-target")
        r = client.patch(
            f"{ADMIN_USERS}/{reg['user']['id']}/role",
            json={"role": "superuser"},
            headers=auth_header(admin_token(client)),
        )
        assert r.status_code == 422

    def test_unknown_user_404(self, client: TestClient) -> None:
        r = client.patch(
            f"{ADMIN_USERS}/999999/active",
            json={"is_active": False},
            headers=auth_header(admin_token(client)),
        )
        assert r.status_code == 404

    def test_seeded_admin_credentials_match_config(self, client: TestClient) -> None:
        r = client.post(
            "/api/auth/login",
            json={"username": TEST_ADMIN_USERNAME, "password": TEST_ADMIN_PASSWORD},
        )
        assert r.status_code == 200
        assert r.json()["user"]["role"] == "admin"
