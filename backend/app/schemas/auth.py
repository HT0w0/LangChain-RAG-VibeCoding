"""认证相关的请求/响应模型。"""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.common import UTCDateTime, UTCModel

# 用户名允许中文:这是中文电商场景,强制英文用户名会让运营同事很难用。
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9_\-一-鿿]{3,32}$")

# 最短 6 位是为了迁就需求里 `admin/123456` 的约定。生产部署应在首次登录后改密,
# 这条会在 README 里写明。
MIN_PASSWORD_LENGTH = 6
MAX_PASSWORD_LENGTH = 128


def _validate_username(v: str) -> str:
    v = v.strip()
    if not USERNAME_PATTERN.match(v):
        raise ValueError("用户名需为 3-32 位,仅限中英文、数字、下划线和连字符")
    return v


def _validate_password(v: str) -> str:
    if len(v) < MIN_PASSWORD_LENGTH:
        raise ValueError(f"密码至少 {MIN_PASSWORD_LENGTH} 位")
    if len(v) > MAX_PASSWORD_LENGTH:
        # 上限不是为了安全,是为了防止超长输入把 argon2 的内存/时间开销放大成 DoS
        raise ValueError(f"密码最长 {MAX_PASSWORD_LENGTH} 位")
    return v


class RegisterRequest(BaseModel):
    username: str
    password: str

    _v_username = field_validator("username")(_validate_username)
    _v_password = field_validator("password")(_validate_password)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)


class ChangePasswordRequest(BaseModel):
    old_password: str = Field(min_length=1, max_length=MAX_PASSWORD_LENGTH)
    new_password: str

    _v_new = field_validator("new_password")(_validate_password)

    @field_validator("new_password")
    @classmethod
    def _not_same_as_old(cls, v: str, info) -> str:
        # 改密等于没改,直接拒绝而不是默默放过
        if info.data.get("old_password") == v:
            raise ValueError("新密码不能与当前密码相同")
        return v


class UserOut(UTCModel):
    # from_attributes:直接从 ORM 对象构造。时间字段的 UTC 归一化由 UTCModel
    # 统一处理,这里不必再挂 field_serializer(见 schemas/common.py)。
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    role: str
    is_active: bool
    created_at: UTCDateTime
    last_login_at: UTCDateTime | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int  # 秒。前端据此决定何时刷新,不必自己解析 JWT
    user: UserOut


class UserListResponse(BaseModel):
    total: int
    items: list[UserOut]
