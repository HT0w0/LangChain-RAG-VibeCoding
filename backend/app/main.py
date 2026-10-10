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

    # 限流器。**挂在 app.state 上,不是模块级单例** —— 测试每个用例都会重进
    # 一次 lifespan,于是各自拿到一个干净的计数器;模块级单例会让名额跨用例
    # 泄漏,症状是偶发 429(见 `deps.get_rate_limiter`)。
    #
    # `RATE_LIMIT_QPM=0` 表示**关掉限流**:压测要量的是系统的原始容量,
    # 而限流是一层策略 —— 开着它测出来的只是「30 × 用户数」,那是个算得出来的
    # 数,不是测出来的。`scripts/loadtest.py` 起临时环境时会把它设成 0。
    from app.services.ratelimit import SlidingWindowLimiter

    def _build_limiter(qpm: int) -> SlidingWindowLimiter | None:
        return SlidingWindowLimiter(limit=qpm, window_seconds=60.0) if qpm > 0 else None

    # 两个限流器分开:**问答按用户**(`RATE_LIMIT_QPM`)、**登录按 IP**
    # (`LOGIN_RATE_LIMIT_QPM`)。
    # 登录那个额度小得多 —— 正常人一分钟不会登录十次,而暴力破解正是一次次试。
    # (这里**写环境变量名而不是数值**:数值改了注释就过期,而变量名不会。)
    app.state.rate_limiter = _build_limiter(settings.RATE_LIMIT_QPM)
    app.state.login_rate_limiter = _build_limiter(settings.LOGIN_RATE_LIMIT_QPM)
    for label, limiter, qpm, env_key in (
        ("问答(按用户)", app.state.rate_limiter, settings.RATE_LIMIT_QPM, "RATE_LIMIT_QPM"),
        (
            "登录(按 IP)",
            app.state.login_rate_limiter,
            settings.LOGIN_RATE_LIMIT_QPM,
            "LOGIN_RATE_LIMIT_QPM",
        ),
    ):
        if limiter is None:
            logger.warning(f"{label}限流已关闭({env_key}=0),这一路没有任何节流")
        else:
            logger.info(f"已启用{label}限流", extra=log_extra(per_minute=qpm))

    # 建表 + 种子管理员。
    # 这里用 create_all 而不是 Alembic:单机演示项目引入迁移工具的成本
    # 大于收益。代价是**改列不会自动迁移**,只能删库重建——所以本项目
    # 的阶段推进方式是「只新增表,不改已有列」。要上生产必须换 Alembic。
    from app.db.base import database
    from app.services.auth import (
        AuthService,
        assert_admin_password_is_not_default,
        is_default_admin_password,
    )

    await database.create_all()
    async with database.session() as session:
        admin = await AuthService(session, settings).ensure_admin(
            settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD
        )
    # **生产环境里管理员口令不能还是项目默认值。**
    # 放在 `ensure_admin` 之后,是有意的:要比的是**库里那个账号的哈希**,
    # 而不是配置里的字符串 —— 两者的区别见那个函数自己的文档。
    assert_admin_password_is_not_default(settings, admin)

    # 非生产环境不拦启动(本地用默认口令是刻意的),但**要把这个事实记在
    # `app.state` 上** —— 上面那道中间件靠它决定「要不要拒绝来自其他机器的请求」。
    # 判据放在启动时算一次:每次请求都验一遍 argon2 太贵,而这件事在进程的
    # 生命周期里不会变。
    app.state.admin_password_is_default = is_default_admin_password(settings, admin)
    if app.state.admin_password_is_default:
        logger.warning("管理员口令仍是演示默认值 —— 只允许本机访问,来自其他机器的请求会被拒绝")

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


#: 来源算「本机」的地址。`testclient` 是 Starlette 测试客户端的默认来源 ——
#: 它本来就跑在同一个进程里,归为本地是对的(否则全部既有测试都会被拦)。
_LOCAL_HOSTS = frozenset({"127.0.0.1", "::1", "localhost", "testclient"})


@app.middleware("http")
async def block_remote_when_default_admin(request: Request, call_next):
    """**管理员口令还是项目默认值时,拒绝一切来自其他机器的请求。**

    这道拦截是给一个具体的缝补的:`APP_ENV` 的默认值是 `dev`,所以
    「生产环境拒绝默认口令」那道拦截**只在有人显式写了 `APP_ENV=prod` 时才生效**。
    而「绑 `0.0.0.0` 让别人访问、又没设 `APP_ENV`」的人,多半根本不知道
    有这个开关 —— 于是 `admin` / `123456`(公开写在 README 里)照样能登录管理员账号。

    所以这里**不看配置,看事实**:只要库里的管理员口令还是默认值,
    来自其他机器的请求一律拒绝。判据与「服务是怎么启动的」无关。

    ⚠️ **这道拦截必须放在 `trace_and_timing` 之前定义。** Starlette 的
    `add_middleware` 是**前插**,后定义的在更外层;把它定义在前,它就位于
    内层,于是外层那层追踪中间件会正常给这个 503 也带上 `X-Trace-Id`。
    顺序反了不会报错,只是被拒的响应少一个追踪头 —— 又一个静默失效。

    ⚠️ **已知的边界**:如果服务藏在本机的反向代理后面(nginx → uvicorn 走回环),
    那么来源永远是 `127.0.0.1`,这道拦截不会生效。那种是**刻意的部署**,
    该由 `APP_ENV=prod` 那道去管(它拦的是启动)。两道各管一段。
    """
    if getattr(request.app.state, "admin_password_is_default", False):
        client = request.client
        if client is not None and client.host not in _LOCAL_HOSTS:
            logger.warning(
                "拒绝了来自其他机器的访问(管理员仍是默认口令)",
                extra=log_extra(client=client.host, path=request.url.path),
            )
            return JSONResponse(
                status_code=503,
                content={
                    "code": "insecure_default_credentials",
                    "message": (
                        "这个服务还带着演示用的默认管理员口令,因此只允许本机访问。"
                        "要对外开放,请先把 ADMIN_PASSWORD 换成别的 ——"
                        "注意已存在的账号不会被重启覆盖,得删掉 data/app.db 重建,"
                        "或者登录之后用「修改密码」。"
                    ),
                },
            )
    return await call_next(request)


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
        # `X-Trace-Id` 写在最后 —— 字典字面量里**后面的键赢**,这样异常即便带了
        # 同名的头也覆盖不了它。
        #
        # ⚠️ **但要说清楚:在当前链路上这一步其实观察不到效果。** 上面那个
        # `trace_and_timing` 中间件在 `call_next` 之后也会设一次 `X-Trace-Id`
        # (见文件上方),它才是最终胜出的那个。所以这里的顺序**是纵深防御,
        # 不是当前行为的依据** —— 也正因如此,**没有为它写测试**:
        # 响应级的断言两种写法都会绿,那是一条假测试。
        #
        # 真正修掉的是**注释与代码说话相反**这件事:第一版的注释写「异常自己带的头
        # 排在后面,免得它把 X-Trace-Id 覆盖掉」,而代码恰恰把 X-Trace-Id 放在了
        # 会被覆盖的位置。**一条把代码描述反了的注释**,比没有注释更误导人 ——
        # 下一个来改这段的人会照着它去推理。
        headers={**exc.headers, "X-Trace-Id": get_trace_id()},
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
