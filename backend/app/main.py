"""FastAPI 应用装配。

注意文件顶部的环境变量设置:**必须在 import chromadb 之前执行**。
chromadb 的匿名遥测是在模块 import 时读取环境变量的,之后再设无效。
这也是为什么不把这些变量放进 .env 由 pydantic 读——那时已经太晚了。
"""

from __future__ import annotations

import os

# ---- 必须在任何 chromadb import 之前 ----
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")
# 避免 tokenizers 在部分 Windows 环境下的多进程告警
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import logging  # noqa: E402
import time  # noqa: E402
from collections.abc import AsyncIterator  # noqa: E402
from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from app.core.config import settings  # noqa: E402
from app.core.exceptions import AppError  # noqa: E402
from app.core.logging import (  # noqa: E402
    get_trace_id,
    log_extra,
    new_trace_id,
    set_trace_id,
    setup_logging,
)

logger = logging.getLogger("app")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:  # noqa: ARG001
    setup_logging(debug=settings.DEBUG, json_output=settings.APP_ENV == "prod")
    settings.ensure_dirs()
    logger.info(
        "应用启动",
        extra=log_extra(
            env=settings.APP_ENV,
            data_dir=str(settings.resolved_data_dir),
            llm=settings.LLM_MODEL,
            embed=settings.EMBED_MODEL,
            llm_backend=settings.LLM_BACKEND,
            has_api_key=settings.has_api_key,
        ),
    )
    if not settings.has_api_key:
        logger.warning("未配置 DASHSCOPE_API_KEY,将无法调用云端模型。可设 LLM_BACKEND=mock 离线开发。")

    # 建表 + 种子管理员。
    # 这里用 create_all 而不是 Alembic:单机演示项目引入迁移工具的成本
    # 大于收益。代价是**改列不会自动迁移**,只能删库重建——所以本项目
    # 的阶段推进方式是「只新增表,不改已有列」。要上生产必须换 Alembic。
    from app.db.base import database
    from app.services.auth import AuthService

    await database.create_all()
    async with database.session() as session:
        await AuthService(session, settings).ensure_admin(
            settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD
        )

    yield

    await database.dispose()
    logger.info("应用关闭")


app = FastAPI(
    title=settings.APP_NAME,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],  # Vite 默认端口
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Trace-Id", "X-Process-Time-Ms"],
)


@app.middleware("http")
async def trace_and_timing(request: Request, call_next):
    """给每个请求打 trace_id 并计时。

    trace_id 回写进响应头,前端报错时能直接把 id 给过来,
    省掉「复现一下」这一步。
    """
    incoming = request.headers.get("X-Trace-Id")
    if incoming:
        set_trace_id(incoming)
        tid = incoming
    else:
        tid = new_trace_id()

    t0 = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    response.headers["X-Trace-Id"] = tid
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.1f}"
    if elapsed_ms > 2000:
        logger.warning(
            "慢请求",
            extra=log_extra(
                method=request.method,
                path=request.url.path,
                ms=round(elapsed_ms, 1),
            ),
        )
    return response


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """业务异常统一转响应。

    响应体恒为 {code, message, details?},前端按 code 分支,
    不去匹配 message 文案——文案是随时会改的东西。
    """
    if exc.http_status >= 500:
        logger.error(
            "业务异常",
            extra=log_extra(code=exc.code, path=request.url.path, message=exc.message),
        )
    else:
        logger.info(
            "请求被拒绝",
            extra=log_extra(code=exc.code, path=request.url.path),
        )
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_dict(),
        headers={"X-Trace-Id": get_trace_id()},
    )


# ---------------------------------------------------------------------------
# 路由(后续阶段逐步挂载)
# ---------------------------------------------------------------------------

from app.api.routers import admin, auth, chat, health, kb, sessions  # noqa: E402

app.include_router(health.router, prefix=settings.API_PREFIX, tags=["system"])
app.include_router(auth.router, prefix=settings.API_PREFIX)
app.include_router(admin.router, prefix=settings.API_PREFIX)
app.include_router(kb.router, prefix=settings.API_PREFIX)
app.include_router(sessions.router, prefix=settings.API_PREFIX)
app.include_router(chat.router, prefix=settings.API_PREFIX)
