"""管理员路由。

整个 router 挂 `require_admin` 依赖——**在路由级而不是逐个端点上挂**。
逐个挂的话,以后新增一个端点忘了加依赖,就是一个静默的越权漏洞;
挂在 router 上,新增端点默认就是受保护的(fail-closed)。
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_db, require_admin
from app.db.models import User
from app.schemas.admin import SetActiveRequest, SetRoleRequest
from app.schemas.auth import UserListResponse, UserOut
from app.schemas.metrics import MetricsSummary
from app.services.metrics import MetricsService
from app.services.users import UserAdminService

router = APIRouter(
    prefix="/admin",
    tags=["管理"],
    dependencies=[Depends(require_admin)],  # 默认全部受保护,fail-closed
)

DbDep = Annotated[AsyncSession, Depends(get_db)]
AdminDep = Annotated[User, Depends(require_admin)]


@router.get("/users", response_model=UserListResponse, summary="用户列表")
async def list_users(
    session: DbDep,
    page: Annotated[int, Query(ge=1)] = 1,
    page_size: Annotated[int, Query(ge=1, le=100)] = 20,
) -> UserListResponse:
    total, items = await UserAdminService(session).list_users(page=page, page_size=page_size)
    return UserListResponse(total=total, items=items)


@router.patch("/users/{user_id}/active", response_model=UserOut, summary="启用/禁用账号")
async def set_active(
    user_id: int, payload: SetActiveRequest, session: DbDep, actor: AdminDep
) -> UserOut:
    return await UserAdminService(session).set_active(
        actor=actor, user_id=user_id, is_active=payload.is_active
    )


@router.patch("/users/{user_id}/role", response_model=UserOut, summary="调整角色")
async def set_role(
    user_id: int, payload: SetRoleRequest, session: DbDep, actor: AdminDep
) -> UserOut:
    return await UserAdminService(session).set_role(actor=actor, user_id=user_id, role=payload.role)


@router.get("/metrics", response_model=MetricsSummary, summary="请求指标")
async def get_metrics(
    session: DbDep,
    # **窗口必须有上界。** 不封顶的话 `window_minutes=100000000` 就是一次
    # 全表扫描,而这是单机 SQLite 上可以从公网触发的一次停顿——和
    # `admin/users` 的 `page_size` 是同一类事:凡是能变成「扫全表」的入参,
    # 都要有个数。
    #
    # 上界取 24h 而不是明细的保留期(30 天):面板要回答的是「刚才那阵子
    # 怎么了」,更长的区间该先落成快照再看,而不是每次都重算一遍原始明细。
    window_minutes: Annotated[int, Query(ge=1, le=1440)] = 60,
) -> MetricsSummary:
    return await MetricsService(session).summary(window_minutes=window_minutes)
