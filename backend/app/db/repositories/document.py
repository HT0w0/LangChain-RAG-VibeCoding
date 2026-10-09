"""文档与分块仓储。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentStatus, KnowledgeBase


@dataclass(frozen=True, slots=True)
class ChunkRecord:
    """检索期用的分块行,已经带上引用溯源需要的全部字段。

    **为什么单独一个类型,而不是直接返回 ORM 的 `Chunk`**:检索链路需要
    `doc_name` 与 `kb_name` 来渲染来源卡片,而它们分别在 `documents` 和
    `knowledge_bases` 两张表上。返回裸 `Chunk` 的话,调用方要么去触发懒加载
    (异步会话里会直接抛错),要么自己再查一轮——把 JOIN 收在仓储里,
    调用方拿到的就是可直接渲染的一行。

    同时也让检索层不必依赖 ORM:测试里塞个返回本类型的假对象就能跑。
    """

    chunk_id: int
    content: str
    kb_id: int
    doc_id: int
    doc_name: str
    kb_name: str
    chunk_index: int
    meta: dict[str, Any] = field(default_factory=dict)


class DocumentRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get_by_id(self, doc_id: int) -> Document | None:
        return await self._s.get(Document, doc_id)

    async def get_by_hash(self, kb_id: int, file_hash: str) -> Document | None:
        """同库内按内容哈希查重。

        知识库内 (kb_id, file_hash) 有唯一约束,这里查的是「是不是已经传过」,
        用于给出友好提示而不是等数据库抛约束错误。
        """
        stmt = select(Document).where(
            Document.kb_id == kb_id, Document.file_hash == file_hash
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def create(
        self,
        *,
        kb_id: int,
        filename: str,
        stored_path: str,
        file_hash: str,
        mime: str | None,
        size: int,
        uploaded_by: int | None,
    ) -> Document:
        doc = Document(
            kb_id=kb_id,
            filename=filename,
            stored_path=stored_path,
            file_hash=file_hash,
            mime=mime,
            size=size,
            uploaded_by=uploaded_by,
            status=DocumentStatus.PENDING,
        )
        self._s.add(doc)
        await self._s.flush()
        await self._s.refresh(doc)
        return doc

    async def list_by_kb(
        self, kb_id: int, *, status: str | None = None, limit: int | None = None
    ) -> list[Document]:
        stmt = select(Document).where(Document.kb_id == kb_id)
        if status is not None:
            stmt = stmt.where(Document.status == status)
        stmt = stmt.order_by(Document.id.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        return list((await self._s.execute(stmt)).scalars())

    async def list_ready(self, kb_ids: Sequence[int] | None = None) -> list[Document]:
        """所有可供检索的文档。BM25 建索引时要用。"""
        stmt = select(Document).where(Document.status == DocumentStatus.READY)
        if kb_ids:
            stmt = stmt.where(Document.kb_id.in_(list(kb_ids)))
        return list((await self._s.execute(stmt)).scalars())

    async def count_by_kb(self, kb_id: int) -> int:
        stmt = select(func.count(Document.id)).where(Document.kb_id == kb_id)
        return int((await self._s.execute(stmt)).scalar_one())

    async def set_status(
        self,
        doc: Document,
        status: DocumentStatus | str,
        *,
        error: str | None = None,
        chunk_count: int | None = None,
    ) -> None:
        doc.status = str(status)
        doc.error = error
        if chunk_count is not None:
            doc.chunk_count = chunk_count
        await self._s.flush()

    async def delete(self, doc: Document) -> None:
        # chunks 由外键 ON DELETE CASCADE + passive_deletes 交给 SQLite 处理。
        # 走 ORM 逐个删的话,一篇 200 块的文档要发 200 条 DELETE。
        await self._s.delete(doc)
        await self._s.flush()

    async def delete_many(self, docs: Sequence[Document]) -> None:
        for doc in docs:
            await self._s.delete(doc)
        await self._s.flush()


class ChunkRepository:
    """分块仓储。

    注意 `meta` 字段名与 SQLAlchemy 的 `Base.metadata` 无关——模型里刻意
    用了 `meta` 而不是 `metadata`,因为 `metadata` 是 DeclarativeBase 的保留属性。
    """

    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def bulk_create(
        self, rows: list[dict[str, Any]], *, chunk_size: int = 500
    ) -> list[Chunk]:
        """批量插入分块。

        分批 `add_all` 而不是逐条 `add`:SQLAlchemy 的 `flush` 会把整个
        批次编成一条多值 INSERT,几千个分块逐条发 SQL 会慢到无法接受。
        但也不能一次把一万个对象全塞进去——那样 ORM 的身份映射表会很大,
        而且要等全部插入完才有 id 可用,失败时全盘回滚。
        """
        created: list[Chunk] = []
        for start in range(0, len(rows), chunk_size):
            batch = [Chunk(**row) for row in rows[start : start + chunk_size]]
            self._s.add_all(batch)
            await self._s.flush()
            created.extend(batch)
        return created

    async def list_by_doc(self, doc_id: int) -> list[Chunk]:
        stmt = select(Chunk).where(Chunk.doc_id == doc_id).order_by(Chunk.chunk_index)
        return list((await self._s.execute(stmt)).scalars())

    async def list_by_kb(self, kb_ids: Sequence[int]) -> list[Chunk]:
        """按知识库取全量分块。BM25 重建索引与检索质量分析用。"""
        stmt = (
            select(Chunk)
            .where(Chunk.kb_id.in_(list(kb_ids)))
            .order_by(Chunk.kb_id, Chunk.doc_id, Chunk.chunk_index)
        )
        return list((await self._s.execute(stmt)).scalars())

    async def get_by_ids(self, chunk_ids: Sequence[int]) -> list[Chunk]:
        if not chunk_ids:
            return []
        stmt = select(Chunk).where(Chunk.id.in_(list(chunk_ids)))
        return list((await self._s.execute(stmt)).scalars())

    # -- 检索期:带引用信息的分块行 ------------------------------------------

    def _record_stmt(self):
        return (
            select(Chunk, Document.filename, KnowledgeBase.name)
            .join(Document, Chunk.doc_id == Document.id)
            .join(KnowledgeBase, Chunk.kb_id == KnowledgeBase.id)
        )

    @staticmethod
    def _to_record(row) -> ChunkRecord:
        chunk, filename, kb_name = row
        return ChunkRecord(
            chunk_id=chunk.id,
            content=chunk.content,
            kb_id=chunk.kb_id,
            doc_id=chunk.doc_id,
            doc_name=filename,
            kb_name=kb_name,
            chunk_index=chunk.chunk_index,
            meta=dict(chunk.meta or {}),
        )

    async def list_records_by_kb(self, kb_ids: Sequence[int] | None) -> list[ChunkRecord]:
        """按知识库取全量分块(带引用信息)。BM25 重建索引用。

        `kb_ids=None` 表示**不限制范围**(管理员预览),不是空集合——
        传 `[]` 会得到一个空列表,这与检索层的语义一致。
        """
        stmt = self._record_stmt().order_by(Chunk.kb_id, Chunk.doc_id, Chunk.chunk_index)
        if kb_ids is not None:
            ids = list(kb_ids)
            if not ids:
                return []
            stmt = stmt.where(Chunk.kb_id.in_(ids))
        return [self._to_record(row) for row in (await self._s.execute(stmt)).all()]

    async def get_records_by_ids(self, chunk_ids: Sequence[int]) -> dict[int, ChunkRecord]:
        """按 id 取分块,**返回字典而非列表**。调用方(检索层)手里是融合后的
        顺序,需要的正是「按 id 回查」;返回列表的话它还得自己建一遍索引。
        查不到的 id 不会出现在返回值里——那正是「向量库里还留着、库里已经删了」
        的过期条目,由调用方跳过。
        """
        ids = list(chunk_ids)
        if not ids:
            return {}
        rows = (await self._s.execute(self._record_stmt().where(Chunk.id.in_(ids)))).all()
        return {record.chunk_id: record for record in map(self._to_record, rows)}

    async def count_by_kb(self, kb_id: int) -> int:
        stmt = select(func.count(Chunk.id)).where(Chunk.kb_id == kb_id)
        return int((await self._s.execute(stmt)).scalar_one())

    async def delete_by_doc(self, doc_id: int) -> int:
        """删除某文档的全部分块,返回删除条数。

        这里用显式 DELETE 而不是 ORM 的 delete-orphan:重新索引时需要
        「先清空旧分块再写新的」,而 ORM 的级联要求先把对象加载进内存。
        """
        result = await self._s.execute(delete(Chunk).where(Chunk.doc_id == doc_id))
        await self._s.flush()
        return int(result.rowcount or 0)

    async def delete_by_kb(self, kb_id: int) -> int:
        result = await self._s.execute(delete(Chunk).where(Chunk.kb_id == kb_id))
        await self._s.flush()
        return int(result.rowcount or 0)
