"""健康检查。

刻意分成两个:存活探针必须**永远轻量**,就绪探针才做真实检查。
把两者合成一个的常见后果是——外部模型服务抖动一下,编排系统就把
整个应用判定为死亡然后重启,反而放大故障。
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from app import __version__
from app.core.config import settings

router = APIRouter()


@router.get("/health", summary="存活探针")
async def health() -> dict[str, Any]:
    return {"status": "ok", "version": __version__}


@router.get("/health/ready", summary="就绪探针")
async def ready() -> dict[str, Any]:
    """真实检查依赖可用性。

    Phase 0 只检查配置与目录;数据库与向量库的连通性检查在
    对应模块落地后接上。
    """
    checks: dict[str, Any] = {}

    checks["config"] = {
        "ok": settings.has_api_key,
        "llm_backend": settings.LLM_BACKEND,
        "embed_backend": settings.EMBED_BACKEND,
        "rerank_backend": settings.RERANK_BACKEND,
        "llm_model": settings.LLM_MODEL,
        "embed_model": settings.EMBED_MODEL,
        "rerank_enabled": settings.RERANK_ENABLED,
    }

    dirs_ok = True
    for label, path in (
        ("data_dir", settings.resolved_data_dir),
        ("chroma_dir", settings.resolved_chroma_dir),
        ("upload_dir", settings.resolved_upload_dir),
    ):
        exists = path.exists()
        dirs_ok = dirs_ok and exists
        checks[label] = {"ok": exists, "path": str(path)}

    all_ok = checks["config"]["ok"] and dirs_ok
    return {"status": "ready" if all_ok else "degraded", "checks": checks}
