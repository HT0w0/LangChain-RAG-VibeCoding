"""密码哈希与 JWT。

**密码用 argon2id**(而非 md5/sha1/bcrypt):内存硬化,抗 GPU/ASIC 暴力破解,
且是当前密码哈希竞赛的推荐算法。挖矿级别的并行算力在 argon2 的内存开销面前
得不到便宜。

**JWT 里不放 role。** 这是个刻意的取舍:

| | role 写进 token | 每次查库 |
|---|---|---|
| 每请求开销 | 0 次查询 | 1 次主键查询 |
| 降权/禁用生效 | 要等 token 过期(最长 12h) | **立即** |
| 后果 | 被撤职的管理员还能改知识库 | — |

知识库管理是 admin 独占的敏感操作,「权限已收回但还能用」不可接受。
主键查询走索引,代价可忽略。

`jti` 仍然写进 token,为将来的「登出即失效」准备(需要一个吊销表)。
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from functools import lru_cache

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.core.config import Settings
from app.core.exceptions import AuthError

TOKEN_TYPE = "access"


@lru_cache(maxsize=8)
def _hasher(time_cost: int, memory_cost: int, parallelism: int) -> PasswordHasher:
    """按参数缓存 hasher。

    argon2 的 `PasswordHasher` 构造本身很便宜,但 `check_needs_rehash` 之类
    的比较依赖实例参数,重复构造容易让「当前参数」和「验证时参数」不一致。
    统一从这里取,保证参数只有一个来源。
    """
    return PasswordHasher(
        time_cost=time_cost,
        memory_cost=memory_cost,
        parallelism=parallelism,
    )


def get_hasher(settings: Settings) -> PasswordHasher:
    return _hasher(settings.ARGON2_TIME_COST, settings.ARGON2_MEMORY_COST, settings.ARGON2_PARALLELISM)


def hash_password(plain: str, settings: Settings) -> str:
    return get_hasher(settings).hash(plain)


def verify_password(plain: str, hashed: str, settings: Settings) -> bool:
    """校验密码。

    所有失败模式都收敛成 `False`——`VerifyMismatchError`(密码错)、
    `InvalidHashError`(哈希串损坏)、`VerificationError`(其他)。把它们分开
    向上抛只会让调用方写出「密码错提示 A、哈希损坏提示 B」这种**泄露内部状态**
    的差异响应。
    """
    try:
        return get_hasher(settings).verify(hashed, plain)
    except (VerifyMismatchError, InvalidHashError, VerificationError):
        return False


def needs_rehash(hashed: str, settings: Settings) -> bool:
    """哈希是否用了过时的参数(如调高过 time_cost)。

    用于「登录成功时顺手升级哈希」——用户无感,参数随版本演进。
    """
    try:
        return get_hasher(settings).check_needs_rehash(hashed)
    except (InvalidHashError, VerificationError):
        return True


@dataclass(frozen=True, slots=True)
class TokenPayload:
    user_id: int
    jti: str
    issued_at: datetime
    expires_at: datetime


def create_access_token(
    user_id: int,
    settings: Settings,
    *,
    expires_minutes: int | None = None,
) -> tuple[str, TokenPayload]:
    """签发 access token,同时返回解析后的载荷(供调用方回给前端 `expires_in`)。"""
    now = datetime.now(UTC)
    ttl = timedelta(minutes=expires_minutes or settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    expires_at = now + ttl
    jti = uuid.uuid4().hex

    claims = {
        "sub": str(user_id),  # JWT 规范要求 sub 是字符串;PyJWT 会校验
        "typ": TOKEN_TYPE,
        "jti": jti,
        "iat": now,
        "exp": expires_at,
    }
    token = jwt.encode(claims, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    return token, TokenPayload(
        user_id=user_id, jti=jti, issued_at=now, expires_at=expires_at
    )


def decode_access_token(token: str, settings: Settings) -> TokenPayload:
    """解析并校验 token。任何问题都转成 `AuthError`(401)。

    注意对外错误信息**不区分**「过期」与「签名不对」以外的东西——过期的可以
    明确告诉前端(它能据此触发刷新),其他一律「无效凭证」,避免给攻击者反馈。
    """
    try:
        claims = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            # 显式要求 exp,防止有人拿没有过期时间的 token 来用
            options={"require": ["exp", "sub"]},
        )
    except jwt.ExpiredSignatureError as exc:
        raise AuthError("登录已过期,请重新登录", code="token_expired") from exc
    except jwt.InvalidTokenError as exc:
        raise AuthError("无效的登录凭证", code="invalid_token") from exc

    if claims.get("typ") != TOKEN_TYPE:
        # 防止将来引入 refresh token 后,拿 refresh 当 access 用
        raise AuthError("令牌类型不匹配", code="invalid_token")

    try:
        user_id = int(claims["sub"])
    except (TypeError, ValueError) as exc:
        raise AuthError("无效的登录凭证", code="invalid_token") from exc

    return TokenPayload(
        user_id=user_id,
        jti=str(claims.get("jti", "")),
        issued_at=datetime.fromtimestamp(claims["iat"], tz=UTC) if claims.get("iat") else datetime.now(UTC),
        expires_at=datetime.fromtimestamp(claims["exp"], tz=UTC),
    )
