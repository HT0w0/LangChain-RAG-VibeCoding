"""用户仓储。"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
from app.db.models import User, UserRole, utcnow


class UserRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get_by_id(self, user_id: int) -> User | None:
        return await self._s.get(User, user_id)

    async def get_by_username(self, username: str) -> User | None:
        stmt = select(User).where(func.lower(User.username) == username.lower())
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def create(
        self,
        *,
        username: str,
        password_hash: str,
        role: UserRole | str = UserRole.USER,
    ) -> User:
        user = User(username=username, password_hash=password_hash, role=str(role))
        self._s.add(user)
        try:
            await self._s.flush()
        except IntegrityError as exc:
            # 唯一约束兜底。服务层已经先查过重名,但两个并发注册请求
            # 可能在「都查完、都还没插」的窗口里同时通过检查——
            # 数据库约束是唯一可靠的防线,不能只靠应用层预检。
            await self._s.rollback()
            raise ConflictError("用户名已被占用", username=username) from exc
        await self._s.refresh(user)
        return user

    async def list_paged(self, *, limit: int, offset: int) -> list[User]:
        stmt = select(User).order_by(User.id).limit(limit).offset(offset)
        return list((await self._s.execute(stmt)).scalars())

    async def count(self) -> int:
        return int((await self._s.execute(select(func.count(User.id)))).scalar_one())

    async def count_admins(self) -> int:
        stmt = select(func.count(User.id)).where(
            User.role == UserRole.ADMIN, User.is_active.is_(True)
        )
        return int((await self._s.execute(stmt)).scalar_one())

    async def touch_last_login(self, user: User) -> None:
        user.last_login_at = utcnow()
        await self._s.flush()

    async def update_password(self, user: User, password_hash: str) -> None:
        user.password_hash = password_hash
        await self._s.flush()

    async def set_active(self, user: User, active: bool) -> None:
        user.is_active = active
        await self._s.flush()

    async def set_role(self, user: User, role: UserRole | str) -> None:
        user.role = str(role)
        await self._s.flush()
