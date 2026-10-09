"""知识库仓储。"""

from __future__ import annotations

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import ConflictError
from app.db.models import Chunk, Document, KnowledgeBase


class KnowledgeBaseRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    async def get_by_id(self, kb_id: int) -> KnowledgeBase | None:
        return await self._s.get(KnowledgeBase, kb_id)

    async def get_by_name(self, name: str) -> KnowledgeBase | None:
        stmt = select(KnowledgeBase).where(func.lower(KnowledgeBase.name) == name.lower())
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def create(
        self, *, name: str, description: str | None, created_by: int | None
    ) -> KnowledgeBase:
        kb = KnowledgeBase(name=name, description=description, created_by=created_by)
        self._s.add(kb)
        try:
            await self._s.flush()
        except IntegrityError as exc:
            # 同 UserRepository.create:唯一约束是并发下的唯一可靠防线,
            # 服务层的预检只是为了让 99% 的情况有更友好的报错。
            await self._s.rollback()
            raise ConflictError("知识库名称已存在", name=name) from exc
        await self._s.refresh(kb)
        return kb

    async def list_all(self) -> list[KnowledgeBase]:
        stmt = select(KnowledgeBase).order_by(KnowledgeBase.id)
        return list((await self._s.execute(stmt)).scalars())

    async def list_paged(self, *, limit: int, offset: int) -> list[KnowledgeBase]:
        stmt = select(KnowledgeBase).order_by(KnowledgeBase.id).limit(limit).offset(offset)
        return list((await self._s.execute(stmt)).scalars())

    async def count(self) -> int:
        return int((await self._s.execute(select(func.count(KnowledgeBase.id)))).scalar_one())

    async def update(
        self, kb: KnowledgeBase, *, name: str | None = None, description: str | None = None
    ) -> None:
        if name is not None:
            kb.name = name
        if description is not None:
            kb.description = description
        try:
            await self._s.flush()
        except IntegrityError as exc:
            await self._s.rollback()
            raise ConflictError("知识库名称已存在", name=name) from exc

    async def apply_delta(self, kb: KnowledgeBase, *, docs: int = 0, chunks: int = 0) -> None:
        """增量调整冗余计数。

        用 `max(0, ...)` 兜底:计数一旦被减到负数,说明代码里有地方漏了
        配对的 +1/-1。负数比「稍微偏大」更难排查,因为它会让前端显示
        「-1 篇文档」这种明显不可能的状态。真正的对账交给 `recount()`。
        """
        kb.doc_count = max(0, kb.doc_count + docs)
        kb.chunk_count = max(0, kb.chunk_count + chunks)
        await self._s.flush()

    async def recount(self, kb: KnowledgeBase) -> KnowledgeBase:
        """按实际行数重算冗余计数。对账入口。"""
        kb.doc_count = int(
            (
                await self._s.execute(
                    select(func.count(Document.id)).where(Document.kb_id == kb.id)
                )
            ).scalar_one()
        )
        kb.chunk_count = int(
            (
                await self._s.execute(
                    select(func.count(Chunk.id)).where(Chunk.kb_id == kb.id)
                )
            ).scalar_one()
        )
        await self._s.flush()
        return kb
