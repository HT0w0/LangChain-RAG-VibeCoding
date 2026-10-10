"""认证服务。"""

from __future__ import annotations

import functools
import logging
import threading

import anyio
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import DEFAULT_ADMIN_PASSWORD, Settings
from app.core.exceptions import AuthError, ConflictError, ForbiddenError
from app.core.logging import log_extra
from app.core.security import (
    create_access_token,
    hash_password,
    needs_rehash,
    verify_password,
)
from app.db.models import User, UserRole
from app.db.repositories.user import UserRepository
from app.schemas.auth import TokenResponse, UserOut

logger = logging.getLogger(__name__)

# 统一的失败文案。
# 「用户名不存在」与「密码错误」必须给出**完全相同**的响应,否则接口就成了
# 用户枚举器——攻击者能先批量确认哪些账号存在,再针对性地撞库。
_INVALID_CREDENTIALS = "用户名或密码错误"

# ---------------------------------------------------------------------------
# 密码哈希:挪出事件循环 + 并发闸门
# ---------------------------------------------------------------------------
#
# **两件事必须一起做,只做第一件会把问题换个形式。**
#
# 一、挪出事件循环。argon2 是**同步**的、每次约 50–100ms。直接在 async 函数里
#     调它会独占事件循环 —— 期间**所有其他请求一起卡住**,包括正在流式输出的 SSE。
#
#     依据是 `docs/LOAD-TEST.md` 第四节那个「登录风暴」实验:挂着几条 SSE 流,
#     同时打 30 个并发登录,量**事件循环可用性**(别的请求还能不能跑),
#     并用「CPU 饱和」「并发廉价请求」两条对照臂把原因钉死。
#
#     **这里刻意不复述具体数字。** 那些数每一轮压测都会变(报告是
#     `loadtest.py --write` 生成的),写死在注释里必然过期 —— 我在这里
#     栽过两次:第一次引的数在仓库里核不到,改成报告里的数之后,
#     报告又重生成了一遍,数字又对不上了。**要数字就看报告第四节。**
#
#     判据那一行会告诉你结论:修后它报「不成立」,那正是这个改动的目的。
#     反过来说,如果哪天它又报「成立」,说明这个改动被回退了。
#
# 二、加并发闸门。argon2 是**内存硬化**算法:`ARGON2_MEMORY_COST=65536` 是
#     **每次哈希 64MB**。而 anyio 的默认线程池上限约 40 —— 40 个并发登录就能
#     同时占住约 2.5GB,而这台机器只有 15.4GB。**不设闸门的话,这个修法只是
#     把「事件循环阻塞」换成了「内存压力」**,而后者更难查。
#
# 预算 512MB:在 64MB/次的默认参数下允许 8 个并发哈希(8 × 64MB = 512MB),
# 与线程池其余工作(Chroma/BM25)也能共存。参数变了闸门自动跟着变。
_HASH_MEMORY_BUDGET_MB = 512

_hash_limiter: anyio.CapacityLimiter | None = None
_hash_limiter_lock = threading.Lock()


def _hash_limiter_for(settings: Settings) -> anyio.CapacityLimiter:
    """进程级、只按第一次见到的参数建一次。

    必须在**运行时**建:`anyio.CapacityLimiter` 要绑定到当时的事件循环,
    在模块导入时建会绑到一个还不存在的循环上。
    """
    global _hash_limiter
    if _hash_limiter is None:
        with _hash_limiter_lock:
            if _hash_limiter is None:
                per_hash_mb = max(1, settings.ARGON2_MEMORY_COST // 1024)
                _hash_limiter = anyio.CapacityLimiter(
                    max(2, _HASH_MEMORY_BUDGET_MB // per_hash_mb)
                )
    return _hash_limiter


def is_default_admin_password(settings: Settings, admin: User) -> bool:
    """库里的管理员口令是不是项目那个默认值。

    单独抽出来,是因为它有**两个用处**:生产环境启动时据此拒绝启动;
    非生产环境则据此决定「要不要拒绝来自其他机器的请求」(见 `main.py` 的
    `block_remote_when_default_admin`)。两处各算一遍就会漂开。
    """
    return verify_password(DEFAULT_ADMIN_PASSWORD, admin.password_hash, settings)


def assert_admin_password_is_not_default(settings: Settings, admin: User) -> None:
    """生产环境里管理员口令仍是项目默认值 → **拒绝启动**。

    和 `JWT_SECRET` 那道拦截是同一类:默认值写在公开仓库里,谁都能读到。
    在别人能访问到这个服务的地方,那等于把管理员账号的钥匙挂在门口。

    **为什么比的是库里的哈希,而不是 `settings.ADMIN_PASSWORD`:**

    `ensure_admin` 只在账号不存在时创建,已存在则一概不动(那是刻意的幂等 ——
    否则管理员自己改过的密码会被重启冲掉)。所以「先部署 → 发现不对 →
    改 .env → 重启」这条路上,**配置是强的、库里的口令还是默认的**。
    只比配置会放行,而运维会以为自己已经修好了。

    比库里的哈希同时覆盖了两种情形,而且顺带覆盖了「管理员从界面上
    改过密码」—— 那种情况下哈希早就不是默认的了,自然放行。

    只在 `APP_ENV=prod` 时检查:本地开发用默认口令是**刻意的**,
    README 就拿它当演示登录,拦起来只会让 clone 下来的人一脸茫然。
    """
    if settings.APP_ENV != "prod":
        return
    if not is_default_admin_password(settings, admin):
        return
    raise RuntimeError(
        "生产环境(APP_ENV=prod)的管理员口令仍是项目默认值 —— "
        "那个口令公开写在 README 里,任何人拿它就能登录管理员账号。"
        "处理办法:改 .env 的 ADMIN_PASSWORD **之后删掉 data/app.db 重建**"
        "(已存在的账号不会被 ensure_admin 覆盖,只改配置是没用的),"
        "或者登录之后用「修改密码」把它换掉。"
    )


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._s = session
        self._settings = settings
        self._users = UserRepository(session)
        self._dummy_hash: str | None = None

    # ---------- 内部 ----------

    async def _hash(self, password: str) -> str:
        """哈希一个密码,**在线程池里做**。见文件头那一节。"""
        return await anyio.to_thread.run_sync(
            functools.partial(hash_password, password, self._settings),
            limiter=_hash_limiter_for(self._settings),
        )

    async def _verify(self, password: str, stored: str) -> bool:
        """校验密码,**在线程池里做**。见文件头那一节。"""
        return await anyio.to_thread.run_sync(
            functools.partial(verify_password, password, stored, self._settings),
            limiter=_hash_limiter_for(self._settings),
        )

    async def _get_dummy_hash(self) -> str:
        """用于「用户不存在」分支的假哈希。

        没有它的话,账号不存在时接口立即返回,而账号存在时要等 argon2 算完
        (约 50ms)。这个**响应时间差**足以枚举用户,而且完全不需要读报错文案。
        拿真哈希去校验一个必然错误的密码,把两条路径的耗时拉平。
        """
        if self._dummy_hash is None:
            self._dummy_hash = await self._hash("dummy-password-for-timing-equalization")
        return self._dummy_hash

    def _issue(self, user: User) -> TokenResponse:
        token, payload = create_access_token(user.id, self._settings)
        ttl_seconds = int((payload.expires_at - payload.issued_at).total_seconds())
        return TokenResponse(
            access_token=token,
            expires_in=ttl_seconds,
            user=UserOut.model_validate(user),
        )

    # ---------- 对外 ----------

    async def register(self, username: str, password: str) -> TokenResponse:
        """注册普通用户。

        **注册接口永远只能创建 user 角色。** 不提供「注册成管理员」的入口——
        否则整个 RBAC 就是摆设。管理员只能由种子数据创建。
        """
        if await self._users.get_by_username(username) is not None:
            raise ConflictError("用户名已被占用", username=username)

        user = await self._users.create(
            username=username,
            password_hash=await self._hash(password),
            role=UserRole.USER,
        )
        await self._s.commit()
        logger.info("用户注册", extra=log_extra(user_id=user.id, username=user.username))
        return self._issue(user)

    async def authenticate(self, username: str, password: str) -> User:
        user = await self._users.get_by_username(username)

        if user is None:
            await self._verify(password, await self._get_dummy_hash())
            raise AuthError(_INVALID_CREDENTIALS)

        if not await self._verify(password, user.password_hash):
            logger.warning(
                "登录失败", extra=log_extra(username=username, reason="bad_password")
            )
            raise AuthError(_INVALID_CREDENTIALS)

        # 先验密码再判禁用:倒过来的话,「账号已禁用」这个响应本身就泄露了
        # 「该用户名存在」,而且不需要正确密码就能拿到。
        if not user.is_active:
            raise AuthError("账号已被禁用,请联系管理员", code="account_disabled")

        # 参数升级:登录成功时顺手把旧参数哈希换成当前参数,用户无感
        if needs_rehash(user.password_hash, self._settings):
            await self._users.update_password(user, await self._hash(password))

        await self._users.touch_last_login(user)
        await self._s.commit()
        return user

    async def login(self, username: str, password: str) -> TokenResponse:
        user = await self.authenticate(username, password)
        logger.info("登录成功", extra=log_extra(user_id=user.id, username=user.username))
        return self._issue(user)

    async def change_password(self, user: User, old_password: str, new_password: str) -> None:
        """改密。

        注意:已签发的 token **不会失效**(JWT 是无状态的,有效期最长 12h)。
        要真正做到「改密即踢下线」需要一个 token 版本号或吊销表,见 README
        的已知限制。这里先如实记录,不假装它不存在。
        """
        if not await self._verify(old_password, user.password_hash):
            logger.warning("改密失败", extra=log_extra(user_id=user.id, reason="bad_old_password"))
            raise ForbiddenError("当前密码不正确", code="wrong_old_password")

        await self._users.update_password(user, await self._hash(new_password))
        await self._s.commit()
        logger.info("用户改密", extra=log_extra(user_id=user.id))

    async def ensure_admin(self, username: str, password: str) -> User:
        """启动时确保管理员账号存在。

        只在**不存在**时创建,已存在则一概不动。理由是幂等:每次启动都重置
        密码的话,管理员自己改过的密码会被服务重启悄悄冲掉。
        """
        existing = await self._users.get_by_username(username)
        if existing is not None:
            return existing

        admin = await self._users.create(
            username=username,
            password_hash=await self._hash(password),
            role=UserRole.ADMIN,
        )
        await self._s.commit()
        logger.warning(
            "已创建管理员账号,请立即修改初始密码",
            extra=log_extra(username=username),
        )
        return admin
