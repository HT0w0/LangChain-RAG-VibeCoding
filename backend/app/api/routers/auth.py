"""认证路由。"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.deps import (
    enforce_login_rate_limit,
    get_current_user,
    get_db,
    get_settings_dep,
)
from app.db.models import User
from app.schemas.auth import (
    ChangePasswordRequest,
    LoginRequest,
    RegisterRequest,
    TokenResponse,
    UserOut,
)
from app.services.auth import AuthService

router = APIRouter(prefix="/auth", tags=["认证"])

DbDep = Annotated[AsyncSession, Depends(get_db)]
SettingsDep = Annotated[Settings, Depends(get_settings_dep)]
CurrentUser = Annotated[User, Depends(get_current_user)]


@router.post(
    "/register",
    response_model=TokenResponse,
    status_code=status.HTTP_201_CREATED,
    summary="注册普通用户",
)
async def register(payload: RegisterRequest, session: DbDep, settings: SettingsDep) -> TokenResponse:
    """注册后直接签发 token,省掉一次登录往返。

    注意这里**只能建普通用户**;管理员由种子数据创建,没有开放入口。
    """
    return await AuthService(session, settings).register(payload.username, payload.password)


@router.post(
    "/login",
    response_model=TokenResponse,
    summary="登录",
    # **登录是唯一一个可以反复尝试凭据的入口**(注册、健康检查也不需要先登录,
    # 但它们没有「猜密码」这件事可猜),所以它是最该限流的一个。
    # 按 IP 计,不按用户名 —— 那会让攻击者用「故意输错」把别人的账号锁死。
    # 理由写在依赖自己的文档里。
    dependencies=[Depends(enforce_login_rate_limit)],
)
async def login(payload: LoginRequest, session: DbDep, settings: SettingsDep) -> TokenResponse:
    return await AuthService(session, settings).login(payload.username, payload.password)


@router.get("/me", response_model=UserOut, summary="当前登录用户")
async def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)


@router.post(
    "/change-password",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="修改密码",
)
async def change_password(
    payload: ChangePasswordRequest,
    user: CurrentUser,
    session: DbDep,
    settings: SettingsDep,
) -> None:
    await AuthService(session, settings).change_password(
        user, payload.old_password, payload.new_password
    )
