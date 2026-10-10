"""统一异常体系。

每个异常都带 `code`（给前端做分支的稳定标识）和 `http_status`（给 FastAPI
转响应）。让前端能按 code 写分支，而不是匹配中文报错文本——文案随时会改。

`ProviderError` 单独拎出来，是因为外部 API 失败和业务逻辑失败要有不同的
处置：前者该重试/降级，后者该直接告诉用户。
"""

from __future__ import annotations


class AppError(Exception):
    """所有业务异常的基类。"""

    code: str = "internal_error"
    http_status: int = 500

    def __init__(
        self,
        message: str = "",
        *,
        code: str | None = None,
        headers: dict[str, str] | None = None,
        **details: object,
    ) -> None:
        self.message = message or self.__class__.__doc__ or self.code
        if code:
            self.code = code
        #: 额外的响应头。**给的是「标准说法」的机会** —— 比如限流要回
        #: `Retry-After`,那是这件事的通用表达,代理与 SDK 都认,而
        #: 我们自己的 `details.retry_after_seconds` 只有读过文档的人才知道。
        #: 关键字参数,免得和 `**details` 撞名。
        self.headers = headers or {}
        self.details = details
        super().__init__(self.message)

    def to_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {"code": self.code, "message": self.message}
        if self.details:
            payload["details"] = self.details
        return payload


# ---------- 认证 / 授权 ----------


class AuthError(AppError):
    """认证失败。"""

    code = "unauthorized"
    http_status = 401


class ForbiddenError(AppError):
    """无权访问该资源。"""

    code = "forbidden"
    http_status = 403


class NotFoundError(AppError):
    """资源不存在。"""

    code = "not_found"
    http_status = 404


class ConflictError(AppError):
    """资源冲突，如用户名已存在。"""

    code = "conflict"
    http_status = 409


class InvalidInputError(AppError):
    """入参不合法。"""

    code = "invalid_input"
    http_status = 422


class RateLimitedError(AppError):
    """触发限流。"""

    code = "rate_limited"
    http_status = 429


# ---------- 外部依赖 ----------


class ProviderError(AppError):
    """外部模型服务调用失败。

    `retryable` 用于区分「网络抖动/限流，重试有意义」与「参数错/欠费，
    重试只会更糟」。注意百炼的欠费是 **400 Arrearage** 而非 401，
    不能靠状态码区分认证失败与余额不足，要看响应体的 error.code。
    """

    code = "provider_error"
    http_status = 502

    def __init__(
        self,
        message: str = "",
        *,
        provider: str = "",
        retryable: bool = False,
        upstream_code: str = "",
        **details: object,
    ) -> None:
        super().__init__(message, provider=provider, upstream_code=upstream_code, **details)
        self.retryable = retryable
        self.upstream_code = upstream_code


class QuotaExhaustedError(ProviderError):
    """额度耗尽 / 欠费。不可重试，需要用户去控制台处理。"""

    code = "quota_exhausted"
    http_status = 402

    def __init__(self, message: str = "", **details: object) -> None:
        super().__init__(message or "账户额度不足或已欠费", retryable=False, **details)


# ---------- 会话 / 检索 ----------


class SessionError(AppError):
    """会话相关错误。"""

    code = "session_error"
    http_status = 400


class IngestError(AppError):
    """文档摄取失败。"""

    code = "ingest_error"
    http_status = 400


class FileTooLargeError(AppError):
    """上传文件超过大小上限。

    单独一个 413 而不是塞进 422:前端要据此给出**不同的**提示——
    「格式不对,换个文件」和「文件太大,请拆分后再传」是两种完全不同的
    用户动作,共用一个错误码就只能靠匹配文案来区分。
    """

    code = "file_too_large"
    http_status = 413
