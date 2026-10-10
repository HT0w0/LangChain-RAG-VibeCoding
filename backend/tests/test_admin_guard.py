"""生产环境里管理员口令不能是项目默认值。

**这个文件补的是与 `JWT_SECRET` 同款的一个口子 —— 上一轮只堵了一半。**

`admin` / `123456` 是项目自带的演示凭据,而且**写在公开的 README 里**。
只要这个服务被部署到别人能访问的地方(绑定 `0.0.0.0` 或放到服务器上),
任何读过这个仓库的人都能直接登录管理员账号。

**为什么查的是库里的哈希,而不是 `settings.ADMIN_PASSWORD`:**

`ensure_admin` 只在账号**不存在**时创建,已存在则一概不动(那是幂等性
刻意的设计 —— 否则管理员自己改过的密码会被重启冲掉)。于是有这么一条
很常见的路径:

    先部署(口令还是默认的)→ 运维发现不对 → 把 .env 改成强口令 → 重启

这条路径下,**配置是强的、库里的口令还是默认的**。只查配置会放行,
而运维会以为自己已经修好了 —— 那正是这个项目最想防的「看起来修好了、
其实没有」。查库里的哈希才拦得住。
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from app.core.config import DEFAULT_ADMIN_PASSWORD, Settings
from app.core.security import hash_password
from app.main import app
from app.services.auth import assert_admin_password_is_not_default
from fastapi.testclient import TestClient


def _admin_with(password: str, settings: Settings) -> SimpleNamespace:
    """只需要 `.password_hash` —— 判据不看别的字段,给个最小的替身就够。"""
    return SimpleNamespace(password_hash=hash_password(password, settings))


class TestAdminPasswordGuard:
    def test_生产环境里口令还是默认值就拦住(self, settings: Settings) -> None:
        prod = settings.model_copy(update={"APP_ENV": "prod"})

        with pytest.raises(RuntimeError, match="管理员"):
            assert_admin_password_is_not_default(prod, _admin_with(DEFAULT_ADMIN_PASSWORD, settings))

    def test_生产环境里换过口令就放行(self, settings: Settings) -> None:
        """拦的是「默认口令」,不是「生产环境」本身。"""
        prod = settings.model_copy(update={"APP_ENV": "prod"})

        # 不抛异常就是过
        assert_admin_password_is_not_default(prod, _admin_with("一个不是默认值的口令", settings))

    def test_开发环境不管(self, settings: Settings) -> None:
        """本地开发用默认口令是**刻意的**(README 就拿它当演示登录),
        拦起来只会让 clone 下来的人一脸茫然。"""
        assert_admin_password_is_not_default(
            settings, _admin_with(DEFAULT_ADMIN_PASSWORD, settings)
        )

    def test_管理员自己改过密码之后也放行(self, settings: Settings) -> None:
        """改密接口把口令换掉之后,启动不该再拦 —— 这是最常见的正常路径。"""
        prod = settings.model_copy(update={"APP_ENV": "prod"})

        assert_admin_password_is_not_default(prod, _admin_with("用户自己设的口令", settings))


#: 一个「来自别的机器」的来源地址。用 RFC 5737 的文档保留段 ——
#: 它永远不可能是真实主机,所以不会因为在别的网络里跑就改变含义。
REMOTE = ("203.0.113.9", 51234)


class TestRemoteAccessWithDefaultPassword:
    """**判据是「请求从哪来」,不是「配置写没写 prod」。**

    这一层是为一个具体的缝补的:`APP_ENV` 默认是 `dev`,所以「生产环境拒绝
    默认口令」那道拦截**只在有人显式写了 `APP_ENV=prod` 时才生效**。
    而「绑 `0.0.0.0` 让别人访问、却没设 `APP_ENV`」的人,多半根本不知道
    有这个开关 —— 于是 `admin`/`123456` 照样能用。

    所以这里不看配置,看**事实**:只要管理员口令还是项目默认值,
    **来自其他机器的请求一律拒绝**;本机(含测试客户端)照常。
    """

    def test_默认口令时别的机器连不上(self, client, monkeypatch) -> None:
        monkeypatch.setattr(app.state, "admin_password_is_default", True)
        remote = TestClient(app, client=REMOTE)

        r = remote.get("/api/health")

        assert r.status_code == 503
        assert r.json()["code"] == "insecure_default_credentials"
        # 提示必须是**能照着做**的
        assert "ADMIN_PASSWORD" in r.json()["message"]

    def test_默认口令时本机照常能用(self, client, monkeypatch) -> None:
        """**不能误伤本机。** `client` 夹具的来源是 `testclient`,
        归为本地 —— 否则本地开发和全部既有测试都会被这道拦截打爆。"""
        monkeypatch.setattr(app.state, "admin_password_is_default", True)

        assert client.get("/api/health").status_code == 200

    def test_换掉口令之后别的机器就能用了(self, client, monkeypatch) -> None:
        """两头都钉:一直返回 503 的实现也能过上面那条,过不了这条。"""
        monkeypatch.setattr(app.state, "admin_password_is_default", False)
        remote = TestClient(app, client=REMOTE)

        assert remote.get("/api/health").status_code == 200
