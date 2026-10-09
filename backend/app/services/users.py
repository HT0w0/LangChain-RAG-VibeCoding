"""用户管理服务(管理员用)。

这里的两条「自锁防护」是刻意的:管理员能禁用/降权任意账号,包括自己和
其他管理员。如果不拦,一次误操作就能把系统变成**没有任何人能管理**的状态,
而且没有任何界面能恢复它——只能手动改数据库。
"""

from __future__ import annotations

import logging

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import InvalidInputError, NotFoundError
from app.core.logging import log_extra
from app.db.models import User, UserRole
from app.db.repositories.user import UserRepository
from app.schemas.auth import UserOut

logger = logging.getLogger(__name__)


class UserAdminService:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session
        self._users = UserRepository(session)

    async def list_users(self, *, page: int, page_size: int) -> tuple[int, list[UserOut]]:
        total = await self._users.count()
        rows = await self._users.list_paged(limit=page_size, offset=(page - 1) * page_size)
        return total, [UserOut.model_validate(u) for u in rows]

    async def _get_or_404(self, user_id: int) -> User:
        user = await self._users.get_by_id(user_id)
        if user is None:
            raise NotFoundError("用户不存在")
        return user

    async def set_active(self, *, actor: User, user_id: int, is_active: bool) -> UserOut:
        target = await self._get_or_404(user_id)

        if not is_active:
            if target.id == actor.id:
                raise InvalidInputError("不能禁用当前登录的账号", code="cannot_disable_self")
            if target.is_admin and await self._users.count_admins() <= 1:
                raise InvalidInputError(
                    "系统必须保留至少一个启用状态的管理员", code="last_admin"
                )

        await self._users.set_active(target, is_active)
        await self._s.commit()
        logger.info(
            "管理员修改账号状态",
            extra=log_extra(actor_id=actor.id, target_id=target.id, is_active=is_active),
        )
        return UserOut.model_validate(target)

    async def set_role(self, *, actor: User, user_id: int, role: str) -> UserOut:
        target = await self._get_or_404(user_id)

        if role != UserRole.ADMIN and target.is_admin:
            if target.id == actor.id:
                raise InvalidInputError("不能撤销自己的管理员权限", code="cannot_demote_self")
            if await self._users.count_admins() <= 1:
                raise InvalidInputError(
                    "系统必须保留至少一个管理员", code="last_admin"
                )

        await self._users.set_role(target, role)
        await self._s.commit()
        logger.info(
            "管理员修改角色",
            extra=log_extra(actor_id=actor.id, target_id=target.id, role=role),
        )
        return UserOut.model_validate(target)
