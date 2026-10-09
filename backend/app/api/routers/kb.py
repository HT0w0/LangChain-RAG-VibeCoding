"""知识库管理接口。

**整个 router 挂在 `require_admin` 下**(见文件末尾的构造),这是需求 6
的技术落点:「只有管理员能管理知识库」。前端路由守卫只是体验优化,
真正的防线在这一行——普通用户直接 curl 也拿不到。

唯一的例外是 `GET /kb/options`:普通用户问答时要选知识库范围,
所以它只要求登录,不要求管理员。它单独用一个不含 admin 依赖的 router。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, BackgroundTasks, Depends, File, Query, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.deps import get_current_user, get_db, get_settings_dep, require_admin
from app.core.exceptions import (
    ConflictError,
    FileTooLargeError,
    IngestError,
    NotFoundError,
)
from app.core.logging import log_extra
from app.db import base as db_base
from app.db.models import Document, DocumentStatus, KnowledgeBase, User
from app.db.repositories.document import ChunkRepository, DocumentRepository
from app.db.repositories.knowledge_base import KnowledgeBaseRepository
from app.indexing.vectorstore import build_vector_store
from app.providers.embedding import build_embedder
from app.schemas.kb import (
    ChunkPreview,
    ChunkPreviewPage,
    DocumentListResponse,
    DocumentOut,
    KBCreateRequest,
    KBListResponse,
    KBOut,
    KBUpdateRequest,
)
from app.services.ingest import IngestService, store_upload
from app.services.parsing import classify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 普通用户可见的部分
# ---------------------------------------------------------------------------

public_router = APIRouter(prefix="/kb", tags=["知识库"])


@public_router.get(
    "/options",
    summary="可选知识库列表(问答页的范围选择器用)",
)
async def list_kb_options(
    session: AsyncSession = Depends(get_db),
    _user: User = Depends(get_current_user),
) -> dict:
    """只返回 id 与名称。

    **必须注册在 `/kb/{kb_id}` 之前**(见 `admin_router` 的说明),否则
    FastAPI 会先匹配到 `/kb/{kb_id}` 并试图把字符串 "options" 解析成 int,
    直接返回 422。

    这里刻意不返回 doc_count 之类的管理信息:普通用户需要的是「能选哪几个库」,
    库里有几篇文档、谁建的,属于管理面数据。
    """
    kbs = await KnowledgeBaseRepository(session).list_all()
    return {"items": [{"id": kb.id, "name": kb.name} for kb in kbs]}


# ---------------------------------------------------------------------------
# 管理员
# ---------------------------------------------------------------------------

admin_router = APIRouter(
    prefix="/kb",
    tags=["知识库管理"],
    dependencies=[Depends(require_admin)],
)


async def _get_kb_or_404(session: AsyncSession, kb_id: int) -> KnowledgeBase:
    kb = await KnowledgeBaseRepository(session).get_by_id(kb_id)
    if kb is None:
        raise NotFoundError("知识库不存在", kb_id=kb_id)
    return kb


async def _get_doc_or_404(session: AsyncSession, doc_id: int) -> Document:
    doc = await DocumentRepository(session).get_by_id(doc_id)
    if doc is None:
        raise NotFoundError("文档不存在", doc_id=doc_id)
    return doc


def _build_ingest(session: AsyncSession, settings: Settings) -> IngestService:
    """构造本次摄取用的服务。

    **编码器在这里也套一层向量缓存**,和 `chat.get_rag_service` 那边同一个
    包装器。这一侧才是收益最大的地方:一次上传就是几百个分块,而商品文案
    里重复句式很多。两边只装一处也不会报错——问答侧装上、这里漏掉,表现
    只是「入库比预期慢」,而没有任何断言会红,所以 `test_kb` 里专门有一条
    按分块数核对缓存条目的用例。

    **缓存借用调用方这个事务(`CacheStore.on(session)`),而不是另开会话。**
    这不是省一次开会话:摄取一边往 `chunks` 里写、一边要编码,那个未提交的
    写事务持着 SQLite 的写锁,另开会话的缓存写入**拿不到锁**,而持锁的事务
    要等编码结束才提交——**死锁**,`busy_timeout` 只能把失败推迟 5 秒。
    实测症状是「上传的文档莫名变成 failed」,报错在日志里是一句与缓存看不出
    关系的 `database is locked`。

    代价是缓存的可见性跟着摄取事务走:摄取回滚,那批向量也不落库。这是可
    接受的——向量只是 (文本, 模型) 的函数,下次重算即可;而反过来的选择
    (另开会话)根本跑不通。
    """
    from app.services.cache import CachedEmbedder, CacheStore

    store = CacheStore.on(session)
    return IngestService(
        session,
        settings,
        embedder=CachedEmbedder(
            inner=build_embedder(settings), store=store, settings=settings
        ),
        vector_store=build_vector_store(settings),
    )


@admin_router.post(
    "",
    response_model=KBOut,
    status_code=status.HTTP_201_CREATED,
    summary="新建知识库",
)
async def create_kb(
    payload: KBCreateRequest,
    session: AsyncSession = Depends(get_db),
    admin: User = Depends(require_admin),
) -> KnowledgeBase:
    repo = KnowledgeBaseRepository(session)
    if await repo.get_by_name(payload.name) is not None:
        # 提前查一次是为了给出比数据库约束更友好的报错;
        # 唯一约束仍然是并发下的最终防线(见 KnowledgeBaseRepository.create)。
        raise ConflictError("知识库名称已存在", name=payload.name)

    kb = await repo.create(
        name=payload.name, description=payload.description, created_by=admin.id
    )
    await session.commit()
    await session.refresh(kb)
    logger.info("知识库已创建", extra=log_extra(kb_id=kb.id, name=kb.name, by=admin.id))
    return kb


@admin_router.get("", response_model=KBListResponse, summary="知识库列表")
async def list_kbs(
    session: AsyncSession = Depends(get_db),
) -> KBListResponse:
    repo = KnowledgeBaseRepository(session)
    return KBListResponse(total=await repo.count(), items=await repo.list_all())


@admin_router.get("/{kb_id}", response_model=KBOut, summary="知识库详情")
async def get_kb(
    kb_id: int, session: AsyncSession = Depends(get_db)
) -> KnowledgeBase:
    return await _get_kb_or_404(session, kb_id)


@admin_router.patch("/{kb_id}", response_model=KBOut, summary="修改知识库")
async def update_kb(
    kb_id: int,
    payload: KBUpdateRequest,
    session: AsyncSession = Depends(get_db),
) -> KnowledgeBase:
    kb = await _get_kb_or_404(session, kb_id)
    await KnowledgeBaseRepository(session).update(
        kb, name=payload.name, description=payload.description
    )
    await session.commit()
    await session.refresh(kb)
    return kb


@admin_router.delete(
    "/{kb_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除知识库"
)
async def delete_kb(
    kb_id: int,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> None:
    kb = await _get_kb_or_404(session, kb_id)
    await _build_ingest(session, settings).delete_knowledge_base(kb)


# ---------------------------------------------------------------------------
# 文档
# ---------------------------------------------------------------------------

docs_router = APIRouter(
    prefix="/kb/{kb_id}/documents",
    tags=["知识库管理"],
    dependencies=[Depends(require_admin)],
)


@docs_router.post(
    "",
    response_model=list[DocumentOut],
    status_code=status.HTTP_202_ACCEPTED,
    summary="上传文档(可多选)",
)
async def upload_documents(
    kb_id: int,
    background: BackgroundTasks,
    files: list[UploadFile] = File(..., description="支持 PDF/Word/Excel/CSV/MD/TXT/HTML"),
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
    admin: User = Depends(require_admin),
) -> list[Document]:
    """落盘 + 建记录,返回 202;解析与向量化在后台进行。

    **为什么返回 202 而不是等处理完再返回 200**:一篇 PDF 的解析 + 向量化
    要几秒到几十秒,期间连接一直挂着。前端改为拿到 202 后轮询文档列表看
    `status` 字段,体验和可靠性都更好——浏览器刷新也不会丢掉进度。

    **返回值是「本次真正新建的文档」,不是「本次提交的文件」。** 内容重复
    的文件会被静默跳过,所以 `len(返回) < len(提交)` 时前端应当提示
    「N 个文件因内容重复已跳过」。全部重复时返回空列表——这是正常结果,
    不是失败。

    跳过策略没有开关。曾经有过一个 `dedup=false` 参数,但它**不可能生效**:
    `documents` 表上有 `UNIQUE(kb_id, file_hash)`,关掉应用层检查只会让
    插入撞上数据库约束、返回 500。与其留一个骗人的参数,不如让约束来表达
    这条业务规则——同一知识库里没有必要存两份完全相同的资料。
    """
    kb = await _get_kb_or_404(session, kb_id)
    repo = DocumentRepository(session)
    max_bytes = settings.MAX_UPLOAD_MB * 1024 * 1024

    accepted: list[Document] = []
    for upload in files:
        filename = upload.filename or "未命名文件"

        # 先按扩展名挡一道。在读进内存之前就拒绝,避免为一个 .zip
        # 白白吃掉上传带宽和内存。
        classify(filename)

        data = await upload.read()
        if len(data) == 0:
            raise IngestError("文件内容为空", filename=filename)
        if len(data) > max_bytes:
            raise FileTooLargeError(
                f"文件超过 {settings.MAX_UPLOAD_MB}MB 上限",
                filename=filename,
                size=len(data),
                limit_mb=settings.MAX_UPLOAD_MB,
            )

        stored = store_upload(data, upload_dir=settings.resolved_upload_dir,
                              original_filename=filename)

        # 同库同内容:不重复入库。重复入库会让同一段资料在检索结果里出现两遍,
        # 把 top-k 名额挤掉,而且引用卡片会显示两个来源。
        #
        # 这次查询只是**为了给出一条像样的日志和更快地返回**;真正的保证
        # 来自 `UNIQUE(kb_id, file_hash)`——并发上传同一文件时,两个请求都会
        # 通过这里的检查,然后其中一个在插入时被数据库挡住。
        existing = await repo.get_by_hash(kb.id, stored.file_hash)
        if existing is not None:
            stored.path.unlink(missing_ok=True)
            logger.info(
                "跳过重复文档",
                extra=log_extra(kb_id=kb.id, filename=filename, existing_id=existing.id),
            )
            continue

        doc = await repo.create(
            kb_id=kb.id,
            filename=filename,
            stored_path=str(stored.path),
            file_hash=stored.file_hash,
            mime=upload.content_type,
            size=stored.size,
            uploaded_by=admin.id,
        )
        accepted.append(doc)

    if accepted:
        await KnowledgeBaseRepository(session).apply_delta(kb, docs=len(accepted))
        await session.commit()
        doc_ids = [d.id for d in accepted]
        # 后台任务用的是**请求之外**的新会话:请求级会话在响应返回时就关了,
        # 拿它去跑后台任务会得到一个已关闭的 session。
        #
        # 这里通过 `db_base.database` 取而不是 import 具体实例,是为了让测试
        # 里的 monkeypatch(替换 app.db.base.database)能生效——直接
        # `from app.db.base import database` 会在 import 时就绑死旧对象。
        background.add_task(_ingest_in_background, doc_ids, settings)
    else:
        await session.commit()

    return accepted


async def _ingest_in_background(doc_ids: list[int], settings: Settings) -> None:
    """后台逐篇摄取。一篇失败不影响其余的。"""
    async with db_base.database.session() as session:
        service = _build_ingest(session, settings)
        for doc_id in doc_ids:
            doc = await DocumentRepository(session).get_by_id(doc_id)
            if doc is None:  # 上传后、处理前被删掉了
                continue
            await service.ingest_document(doc)


@docs_router.get("", response_model=DocumentListResponse, summary="文档列表")
async def list_documents(
    kb_id: int,
    session: AsyncSession = Depends(get_db),
    status_filter: str | None = Query(None, alias="status"),
) -> DocumentListResponse:
    await _get_kb_or_404(session, kb_id)
    docs = await DocumentRepository(session).list_by_kb(kb_id, status=status_filter)
    return DocumentListResponse(total=len(docs), items=docs)


@docs_router.get("/{doc_id}", response_model=DocumentOut, summary="文档详情")
async def get_document(doc_id: int, session: AsyncSession = Depends(get_db)) -> Document:
    return await _get_doc_or_404(session, doc_id)


@docs_router.delete(
    "/{doc_id}", status_code=status.HTTP_204_NO_CONTENT, summary="删除文档"
)
async def delete_document(
    doc_id: int,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> None:
    doc = await _get_doc_or_404(session, doc_id)
    await _build_ingest(session, settings).delete_document(doc)


@docs_router.post(
    "/{doc_id}/reindex",
    response_model=DocumentOut,
    status_code=status.HTTP_202_ACCEPTED,
    summary="重新索引(重试失败的文档 / 应用新的分块参数)",
)
async def reindex_document(
    doc_id: int,
    background: BackgroundTasks,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_settings_dep),
) -> Document:
    """重新走一遍解析 → 分块 → 向量化。

    有两个真实用途,不是「以后可能有用」的接口:
    ① 文档失败后重试(改了源文件、或当时是 Embedding 服务临时抖动);
    ② 调过 `CHUNK_SIZE` / 换了 embedding 模型之后,让已有文档按新参数重建。
    """
    doc = await _get_doc_or_404(session, doc_id)
    doc.status = DocumentStatus.PENDING
    doc.error = None
    await session.commit()
    await session.refresh(doc)

    background.add_task(_ingest_in_background, [doc.id], settings)
    return doc


@docs_router.get(
    "/{doc_id}/chunks", response_model=ChunkPreviewPage, summary="分块预览"
)
async def preview_chunks(
    doc_id: int,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db),
) -> ChunkPreviewPage:
    doc = await _get_doc_or_404(session, doc_id)
    chunks = await ChunkRepository(session).list_by_doc(doc.id)
    page = chunks[offset : offset + limit]
    return ChunkPreviewPage(
        document_id=doc.id,
        filename=doc.filename,
        total=len(chunks),
        items=[
            ChunkPreview(
                chunk_index=c.chunk_index,
                content=c.content,
                token_count=c.token_count,
                meta=c.meta or {},
            )
            for c in page
        ],
    )


# ---------------------------------------------------------------------------
# 挂载
# ---------------------------------------------------------------------------

# 普通用户可见的 router 挂在前面。
#
# **顺序在这里是有意义的**:FastAPI 按注册顺序匹配路由。如果 admin_router
# 先注册,`/kb/options` 会先撞上 `/kb/{kb_id}`,FastAPI 试图把 "options"
# 解析成 int 并返回 422——而不是落到那个真正处理 options 的函数上。
# 这类 bug 在本地跑通 CRUD 时完全看不出来,只在打开问答页时才暴露。
router = APIRouter()
router.include_router(public_router)
router.include_router(admin_router)
router.include_router(docs_router)
