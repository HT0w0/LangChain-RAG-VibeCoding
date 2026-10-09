"""认证服务。"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
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


class AuthService:
    def __init__(self, session: AsyncSession, settings: Settings) -> None:
        self._s = session
        self._settings = settings
        self._users = UserRepository(session)
        self._dummy_hash: str | None = None

    # ---------- 内部 ----------

    async def _get_dummy_hash(self) -> str:
        """用于「用户不存在」分支的假哈希。

        没有它的话,账号不存在时接口立即返回,而账号存在时要等 argon2 算完
        (约 100ms)。这个**响应时间差**足以枚举用户,而且完全不需要读报错文案。
        拿真哈希去校验一个必然错误的密码,把两条路径的耗时拉平。
        """
        if self._dummy_hash is None:
            self._dummy_hash = hash_password("dummy-password-for-timing-equalization", self._settings)
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
            password_hash=hash_password(password, self._settings),
            role=UserRole.USER,
        )
        await self._s.commit()
        logger.info("用户注册", extra=log_extra(user_id=user.id, username=user.username))
        return self._issue(user)

    async def authenticate(self, username: str, password: str) -> User:
        user = await self._users.get_by_username(username)

        if user is None:
            verify_password(password, await self._get_dummy_hash(), self._settings)
            raise AuthError(_INVALID_CREDENTIALS)

        if not verify_password(password, user.password_hash, self._settings):
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
            await self._users.update_password(user, hash_password(password, self._settings))

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
        if not verify_password(old_password, user.password_hash, self._settings):
            logger.warning("改密失败", extra=log_extra(user_id=user.id, reason="bad_old_password"))
            raise ForbiddenError("当前密码不正确", code="wrong_old_password")

        await self._users.update_password(user, hash_password(new_password, self._settings))
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
            password_hash=hash_password(password, self._settings),
            role=UserRole.ADMIN,
        )
        await self._s.commit()
        logger.warning(
            "已创建管理员账号,请立即修改初始密码",
            extra=log_extra(username=username),
        )
        return admin
