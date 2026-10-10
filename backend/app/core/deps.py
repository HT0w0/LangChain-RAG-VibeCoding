"""FastAPI 依赖注入。

认证链路是 `Bearer token → get_current_user → require_admin`。**每一层都可能
是攻击面**,所以顺序和失败语义都要明确:

- token 无效/过期 → 401,前端据此清登录态跳登录页
- token 有效但账号被禁用 → 401(不是 403),因为「你是谁」这个问题已经答不上来了
- 身份有效但权限不足 → 403,前端应提示「无权限」而不是跳登录页

把 401 和 403 混用会让前端无法区分「该重新登录」和「该找管理员开权限」。
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings, get_settings
from app.core.exceptions import AuthError, ForbiddenError, RateLimitedError
from app.core.security import decode_access_token
from app.db.base import Database, database
from app.db.models import User
from app.db.repositories.user import UserRepository
from app.services.ratelimit import SlidingWindowLimiter

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



# ---------------------------------------------------------------------------
# 限流
# ---------------------------------------------------------------------------


def _deny_if_over(
    limiter: SlidingWindowLimiter | None, key: str, *, what: str
) -> None:
    """两处限流依赖共用的判定 + 拒绝。

    **抽出来不只是为了少写几行。** 两处各写一遍的话,「等多久怎么算」
    「要不要回 `Retry-After`」「`details` 里放什么」这些决定会各自漂移 ——
    而它们本该完全一致:B 处漏了响应头,就是一个只有某些客户端才读得到的
    不一致,而且没人会为此开一个工单。

    `what` 只是开头那句人话(「问得太快了」/「登录尝试太频繁」),
    剩下的措辞与字段在这里统一生成。
    """
    if limiter is None:
        return
    decision = limiter.check(key)
    if decision.allowed:
        return

    # **向上取整,而且至少 1 秒。** 直接 `:.0f` 的话,剩余 0.4 秒会显示成
    # 「请等 0 秒再试」—— 一句自相矛盾的话:让你等,但等 0 秒。
    wait = max(1, math.ceil(decision.retry_after_seconds))
    raise RateLimitedError(
        f"{what},请等 {wait} 秒再试。(限制:{limiter.limit} 次/分钟)",
        # 响应体里那个是给**我们自己的前端**看的;响应头这个是给**所有**
        # 调用方(代理、SDK、curl)看的 —— `Retry-After` 是这件事的标准说法。
        headers={"Retry-After": str(wait)},
        retry_after_seconds=round(decision.retry_after_seconds, 1),
        limit_per_minute=limiter.limit,
    )


def get_login_rate_limiter(request: Request) -> SlidingWindowLimiter | None:
    """登录接口的限流器。与问答那个**分开**:额度不同、键也不同。"""
    return getattr(request.app.state, "login_rate_limiter", None)


async def enforce_login_rate_limit(
    request: Request,
    limiter: SlidingWindowLimiter | None = Depends(get_login_rate_limiter),
) -> None:
    """登录接口的限流。**按 IP 计,不按用户名计。**

    ⚠️ **按用户名计是错的,而且错得很隐蔽。** 那样的话,攻击者只要拿别人的
    用户名反复输错密码,就能把**受害者自己的账号锁在门外** —— 攻击者拿不到
    任何东西,却成功造成了拒绝服务。而按 IP 计,被限制的始终是发起方。

    ⚠️ **代价说清楚**:这个应用直接跑在本机、也没配反向代理,所以
    `request.client.host` 就是真实来源。**一旦挂到反向代理后面**,它拿到的
    会是代理的地址 —— 那时要么配 `ProxyHeadersMiddleware`,要么改用
    `X-Forwarded-For`(而后者可以被伪造,要配合可信代理白名单)。
    在配好之前,**所有人会共用一个额度**,这一点在部署时要知道。

    (登录之所以要单独按 IP 计,是因为它是**唯一一个可以反复尝试凭据**的入口 ——
     注册、健康检查虽然也不需要先登录,但它们没有「猜密码」这件事可猜。)
    """
    if limiter is None:
        return
    host = request.client.host if request.client else "unknown"
    _deny_if_over(limiter, f"ip:{host}", what="登录尝试太频繁")


def get_rate_limiter(request: Request) -> SlidingWindowLimiter | None:
    """当前进程的限流器。**在 lifespan 里建,挂在 `app.state` 上。**

    为什么挂 `app.state` 而不是模块级单例:`state` 的生命周期跟着 lifespan 走,
    而**测试里每个用例都会重新进一次 lifespan** —— 于是每个用例拿到一个干净
    的计数器。模块级单例会把上一条用例用掉的名额漏给下一条,症状是偶发 429,
    而那种失败最难查(单跑永远绿)。

    返回 `None` 表示**限流关着**(`RATE_LIMIT_QPM=0`),调用方直接放行。
    """
    return getattr(request.app.state, "rate_limiter", None)


async def enforce_chat_rate_limit(
    user: User = Depends(get_current_user),
    limiter: SlidingWindowLimiter | None = Depends(get_rate_limiter),
) -> None:
    """问答端点的用户级限流。**只挂在这一个端点上。**

    ⚠️ **不要把它挂成全局中间件。** 这不是偷懒,是有具体后果的:知识库管理页
    每 1.5 秒轮询一次文档状态(见 `pages/KnowledgeBases.tsx` 的轮询),
    那是 **40 次/分钟** —— 全局按 `RATE_LIMIT_QPM` 去限,会让**页面自己把自己
    打成 429**,而用户什么都没多做。

    真正需要限的是**贵的那一个**:问答要调 embedding、rerank 和 LLM,
    每次都花钱;而列会话、读历史这些是本地 SQLite 查询。

    键是**用户 id** 而不是 IP:同一个办公室的人共用一个出口 IP,
    按 IP 限会把同事互相拖累。
    """
    if limiter is None:
        return
    _deny_if_over(limiter, f"user:{user.id}", what="问得太快了")
