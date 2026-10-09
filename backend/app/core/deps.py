"""FastAPI 依赖注入。

认证链路是 `Bearer token → get_current_user → require_admin`。**每一层都可能
是攻击面**,所以顺序和失败语义都要明确:

- token 无效/过期 → 401,前端据此清登录态跳登录页
- token 有效但账号被禁用 → 401(不是 403),因为「你是谁」这个问题已经答不上来了
- 身份有效但权限不足 → 403,前端应提示「无权限」而不是跳登录页

把 401 和 403 混用会让前端无法区分「该重新登录」和「该找管理员开权限」。
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi import Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthError, ForbiddenError
from app.core.security import decode_access_token
from app.db.base import Database, database
from app.db.models import User
from app.db.repositories.user import UserRepository

# auto_error=False:默认行为是直接抛 HTTPException,那会绕开我们的 AppError
# 处理器,导致 401 的响应体形状和其他错误不一致。自己抛才能统一。
_bearer_scheme = HTTPBearer(
    auto_error=False,
    scheme_name="JWT",
    description="登录接口返回的 access_token,填在此处(不需要加 Bearer 前缀)",
)


async def get_db() -> AsyncIterator[AsyncSession]:
    async with database.session() as session:
        yield session


def get_settings_dep() -> Settings:
    """把 settings 也做成依赖,方便测试用 dependency_overrides 替换。"""
    return get_settings()


def get_database_dep() -> Database:
    """当前进程的 `Database` 实例。

    注意它给的**不是** `AsyncSession`,而是「能自己开会话」的那个入口。
    缓存层必须这样拿数据库:它的写入发生在**请求事务之外**(流式响应期间,
    请求事务的状态由路由决定)。借请求的会话去写缓存,会同时踩到两个坑——
    提交它等于替路由做决定,回滚它则缓存永远只有 miss,而后者在指标上
    和「缓存压根没配」长得一模一样。

    返回的是模块级那个对象(而非在函数体里重新构造),因为 `Database`
    持有连接池,每个请求新建一个等于每个请求新建一个引擎。
    """
    return database


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer_scheme),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> User:
    if credentials is None or not credentials.credentials:
        raise AuthError("请先登录", code="missing_token")

    payload = decode_access_token(credentials.credentials, settings)

    # 每次请求都查库,而不是把 role 塞进 JWT。
    # 代价是一次主键查询;换来的是「降权/禁用立即生效」——
    # 否则被撤职的管理员还能拿着没到期的 token 改知识库。
    user = await UserRepository(session).get_by_id(payload.user_id)
    if user is None:
        # token 签名有效但用户已被删除。不能当成「没登录」之外的任何东西。
        raise AuthError("账号不存在", code="user_not_found")
    if not user.is_active:
        raise AuthError("账号已被禁用", code="account_disabled")

    return user


async def require_admin(user: User = Depends(get_current_user)) -> User:
    """管理员专用接口的守卫。

    这是**真正的防线**,前端路由守卫只是体验优化——任何人都能直接 curl 后端。
    """
    if not user.is_admin:
        raise ForbiddenError("该操作需要管理员权限")
    return user
