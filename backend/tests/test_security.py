"""密码哈希与 JWT 的单元测试。

这些是安全边界,不测接口就测不到——接口测试只覆盖「路径」,覆盖不了
「同一个密码两次哈希结果不同」「篡改一个字节就失效」这类性质。
"""

from __future__ import annotations

from datetime import UTC, datetime

import jwt
import pytest
from app.core.config import Settings
from app.core.exceptions import AuthError
from app.core.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    needs_rehash,
    verify_password,
)

# ---------------------------------------------------------------------------
# 密码哈希
# ---------------------------------------------------------------------------


def test_hash_then_verify_roundtrip(settings: Settings) -> None:
    h = hash_password("correct horse battery staple", settings)
    assert verify_password("correct horse battery staple", h, settings) is True


def test_wrong_password_rejected(settings: Settings) -> None:
    h = hash_password("correct horse battery staple", settings)
    assert verify_password("correct horse battery stapl", h, settings) is False


def test_same_password_hashes_differently(settings: Settings) -> None:
    """加盐的**可观测证据**。

    两次哈希相同说明没加盐——那样一张彩虹表就能一次撞穿所有用户,
    而且能从库里直接看出「哪两个人用了同一个密码」。
    """
    a = hash_password("same-password", settings)
    b = hash_password("same-password", settings)
    assert a != b
    assert verify_password("same-password", a, settings)
    assert verify_password("same-password", b, settings)


def test_hash_is_argon2id(settings: Settings) -> None:
    """哈希串前缀要能认出算法,换算法时才有依据判断。"""
    assert hash_password("x" * 12, settings).startswith("$argon2id$")


def test_corrupted_hash_returns_false_not_raises(settings: Settings) -> None:
    """哈希串损坏时返回 False,不向上抛异常。

    抛出去的话,调用方很容易写成「密码错 → 提示 A,哈希损坏 → 提示 B」,
    这个差异就是一个用户枚举/状态泄露的信道。
    """
    assert verify_password("whatever", "not-a-valid-argon2-hash", settings) is False
    assert verify_password("whatever", "", settings) is False


def test_needs_rehash_detects_weaker_params(settings: Settings) -> None:
    """参数调高后,旧哈希应被判定为需要重算。

    这是「用户无感升级」的依据:登录成功时顺手换掉旧参数哈希。
    """
    weak = Settings(
        _env_file=None,  # type: ignore[call-arg]
        ARGON2_TIME_COST=1,
        ARGON2_MEMORY_COST=8192,
        ARGON2_PARALLELISM=1,
    )
    strong = Settings(
        _env_file=None,  # type: ignore[call-arg]
        ARGON2_TIME_COST=2,
        ARGON2_MEMORY_COST=8192,
        ARGON2_PARALLELISM=1,
    )
    old_hash = hash_password("pw-for-rehash", weak)
    assert needs_rehash(old_hash, strong) is True
    assert needs_rehash(old_hash, weak) is False


def test_needs_rehash_on_garbage_is_true(settings: Settings) -> None:
    assert needs_rehash("garbage", settings) is True


# ---------------------------------------------------------------------------
# JWT
# ---------------------------------------------------------------------------


def test_token_roundtrip(settings: Settings) -> None:
    token, payload = create_access_token(42, settings)
    decoded = decode_access_token(token, settings)
    assert decoded.user_id == 42
    assert decoded.jti == payload.jti
    assert decoded.expires_at > datetime.now(UTC)


def test_expired_token_has_distinct_code(settings: Settings) -> None:
    """过期要给出**可区分**的错误码。

    前端靠它决定「静默刷新」还是「跳登录页」。和其他无效凭证混成一种,
    前端就只能一律跳登录,用户体验和安全性都更差。
    """
    token, _ = create_access_token(1, settings, expires_minutes=-1)
    with pytest.raises(AuthError) as ei:
        decode_access_token(token, settings)
    assert ei.value.code == "token_expired"
    assert ei.value.http_status == 401


def test_tampered_signature_rejected(settings: Settings) -> None:
    token, _ = create_access_token(1, settings)
    head, body, sig = token.split(".")
    forged = f"{head}.{body}.{'A' * len(sig)}"
    with pytest.raises(AuthError) as ei:
        decode_access_token(forged, settings)
    assert ei.value.code == "invalid_token"


def test_token_signed_with_other_secret_rejected(settings: Settings) -> None:
    # 密钥长度 ≥32 字节:HS256 用这个长度才不触发 PyJWT 的 InsecureKeyLengthWarning。
    # 这里验的是「签名密钥不同就拒收」,与密钥长短无关,但一直挂着的警告
    # 会让人养成忽略警告的习惯。
    other = Settings(  # type: ignore[call-arg]
        _env_file=None, JWT_SECRET="a-completely-different-secret-that-is-long-enough"
    )
    token, _ = create_access_token(1, other)
    with pytest.raises(AuthError):
        decode_access_token(token, settings)


def test_payload_tampering_rejected(settings: Settings) -> None:
    """把 sub 改成别人,签名就对不上——这是 JWT 存在的全部意义。

    用「同一 secret 手工签一个 sub=999」来验,确保校验的不是别的什么东西。
    """
    token, _ = create_access_token(1, settings)
    claims = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
    claims["sub"] = "999"
    forged = jwt.encode(
        claims, "wrong-secret-but-long-enough-for-hs256", algorithm=settings.JWT_ALGORITHM
    )
    with pytest.raises(AuthError):
        decode_access_token(forged, settings)


def test_token_without_exp_rejected(settings: Settings) -> None:
    """无过期时间的 token 等同于永久凭证,必须拒绝。"""
    no_exp = jwt.encode({"sub": "1", "typ": "access"}, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)
    with pytest.raises(AuthError):
        decode_access_token(no_exp, settings)


def test_wrong_token_type_rejected(settings: Settings) -> None:
    """现在没有 refresh token,但先把校验写死——等引入时不会忘记加。

    没有这条,一旦将来签出 `typ=refresh` 的长效令牌,它可以直接当 access 用,
    等于把「短效 access + 长效 refresh」的设计整个绕过去。
    """
    wrong_typ = jwt.encode(
        {"sub": "1", "typ": "refresh", "exp": datetime.now(UTC).timestamp() + 3600},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(AuthError) as ei:
        decode_access_token(wrong_typ, settings)
    assert ei.value.code == "invalid_token"


def test_non_numeric_sub_rejected(settings: Settings) -> None:
    bad = jwt.encode(
        {"sub": "not-an-int", "typ": "access", "exp": datetime.now(UTC).timestamp() + 3600},
        settings.JWT_SECRET,
        algorithm=settings.JWT_ALGORITHM,
    )
    with pytest.raises(AuthError):
        decode_access_token(bad, settings)


def test_garbage_token_rejected(settings: Settings) -> None:
    for bad in ("", "abc", "a.b.c", "Bearer xyz"):
        with pytest.raises(AuthError):
            decode_access_token(bad, settings)
