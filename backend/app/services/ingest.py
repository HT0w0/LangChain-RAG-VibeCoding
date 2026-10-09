"""文档摄取:解析 → 分块 → 向量化 → 落库。

**数据流与「谁是事实来源」**

    data/uploads/<hash前2位>/<uuid>.<ext>   ← 原始文件,永久保留
              ↓ parse
    ParsedDocument (text 或 rows)
              ↓ chunk
    chunk 列表(带元数据)
              ↓ embed + 写 SQLite(拿到 chunk.id)
    chunks 表  +  vector_id = "c{chunk.id}"
              ↓ upsert
    Chroma collection

**写入顺序是刻意的:先 SQLite,后 Chroma。** 理由有两条:

1. Chroma 的主键由 SQLite 的自增 id 推导(`c{id}`),反过来做拿不到 id。
2. 更重要的是**失败时的姿态**。两步之间必然有窗口,问题是窗口裂在哪边:
   - 先 Chroma 后 SQLite → 断了会留下**孤儿向量**,检索命中一个
     数据库里不存在的 chunk_id,引用卡片渲染不出来。
   - 先 SQLite 后 Chroma → 断了会留下**没有向量的分块**,BM25 照样能命中,
     只是这一篇的语义检索暂时缺失,重新索引即可修复。

   第二个失败模式是「降级」,第一个是「损坏」。取舍很明确。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.exceptions import IngestError
from app.core.logging import log_extra
from app.db.models import Chunk, Document, DocumentStatus, KnowledgeBase
from app.db.repositories.document import ChunkRepository, DocumentRepository
from app.db.repositories.knowledge_base import KnowledgeBaseRepository
from app.indexing.chunking import ChunkPiece, rows_to_sentences, split_markdown, split_text
from app.indexing.vectorstore import ChromaStore, chunk_vector_id
from app.providers.base import Embedder
from app.services.cache import CorpusVersion
from app.services.parsing import ParsedDocument, classify, parse_document

logger = logging.getLogger(__name__)

# Chroma 的元数据只接受 str / int / float / bool。
# None、list、dict 会直接抛异常,而且报错信息不会告诉你是**哪个**字段的问题。
_ALLOWED_META_TYPES = (str, int, float, bool)


@dataclass(frozen=True, slots=True)
class StoredFile:
    path: Path
    file_hash: str
    size: int


def store_upload(data: bytes, *, upload_dir: Path, original_filename: str) -> StoredFile:
    """把上传内容落盘,返回存储信息。

    **绝不用原始文件名拼路径。** `../../.env` 这样的文件名可以直接把内容
    写到项目外面去,而 `filename` 是用户完全可控的。这里用 uuid 生成文件名,
    原始名只存进数据库用于展示。

    按哈希前两位分子目录:几千个文件堆在一个目录里,NTFS 的目录遍历会明显变慢,
    这是运维层面很常见的一课,成本只是一层目录。
    """
    file_hash = hashlib.sha256(data).hexdigest()
    ext = Path(original_filename).suffix.lower()
    # 扩展名也要过滤:虽然前面 classify() 已经挡过一轮不支持的类型,
    # 但这里再限制一次字符集,防止 `foo.p<span>df` 这类构造。
    if not ext.isascii() or not ext.replace(".", "").isalnum():
        ext = ""

    sub = upload_dir / file_hash[:2]
    sub.mkdir(parents=True, exist_ok=True)
    path = sub / f"{uuid4().hex}{ext}"
    path.write_bytes(data)

    return StoredFile(path=path, file_hash=file_hash, size=len(data))


def sanitize_metadata(meta: dict[str, Any]) -> dict[str, Any]:
    """把分块元数据裁剪成 Chroma 能接受的形状。

    丢弃 None(Chroma 不接受)、丢弃 list/dict(嵌套结构不接受)。
    这是**静默失败的高发区**:上游往 meta 里塞一个 None 就会让整篇文档
    入库失败,而错误信息只告诉你「metadata value 类型不对」。
    """
    out: dict[str, Any] = {}
    for key, value in meta.items():
        if value is None:
            continue
        if isinstance(value, _ALLOWED_META_TYPES):
            out[key] = value
        elif isinstance(value, (int, float)):
            out[key] = value
        # 其余类型(list/dict/datetime…)一律丢弃,不报错——
        # 元数据是**附加信息**,不该因为它让整篇文档入库失败。
    return out


def build_ingest_service(session: AsyncSession, settings: Settings) -> IngestService:
    """构造一次摄取用的服务。

    **放在服务层而不是上传路由里**,因为调用方不止一个:上传接口、
    重建索引、删除、以及 `scripts/seed_data.py`。留在路由里的话,脚本
    要么去 import 一个私有函数,要么自己拼一遍——而自己拼的那一份
    迟早会和这一份漂开(漏掉缓存包装、漏掉向量库配置),症状还只是
    「慢一点」或者「写进了另一个 collection」。

    **编码器在这里套一层向量缓存**,和 `chat.get_rag_service` 那边同一个
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
    # 延迟 import:`cache` 与 `vectorstore` 都要拖进 chromadb,
    # 而模块级的循环 import 在这里是真实存在的风险。
    from app.indexing.vectorstore import build_vector_store
    from app.providers.embedding import build_embedder
    from app.services.cache import CachedEmbedder, CacheStore

    return IngestService(
        session,
        settings,
        embedder=CachedEmbedder(
            inner=build_embedder(settings), store=CacheStore.on(session), settings=settings
        ),
        vector_store=build_vector_store(settings),
    )


class IngestService:
    def __init__(
        self,
        session: AsyncSession,
        settings: Settings,
        *,
        embedder: Embedder,
        vector_store: ChromaStore,
    ) -> None:
        self._s = session
        self._settings = settings
        self._embedder = embedder
        self._vs = vector_store
        self._docs = DocumentRepository(session)
        self._chunks = ChunkRepository(session)
        self._kbs = KnowledgeBaseRepository(session)

    # -- 分块 ---------------------------------------------------------------

    def build_chunks(self, parsed: ParsedDocument, *, filename: str) -> list[ChunkPiece]:
        """按解析结果的形态选分块策略。"""
        size = self._settings.CHUNK_SIZE
        overlap = self._settings.CHUNK_OVERLAP

        if parsed.is_tabular:
            return rows_to_sentences(parsed.rows, source=filename)

        ext = Path(filename).suffix.lower()
        if ext in (".md", ".markdown"):
            return split_markdown(parsed.text, chunk_size=size, chunk_overlap=overlap)
        return split_text(parsed.text, chunk_size=size, chunk_overlap=overlap)

    # -- 摄取 ---------------------------------------------------------------

    async def ingest_document(self, doc: Document) -> int:
        """处理一篇已落盘的文档。返回分块数。

        会自己管理文档状态与提交;失败时把原因写进 `doc.error` 并把状态置为
        FAILED,**不向上抛异常**——一批上传里挂一篇不该拖垮其余的,
        调用方需要的是「逐篇的结果」而不是一个整体异常。
        """
        try:
            await self._docs.set_status(doc, DocumentStatus.PARSING)
            await self._s.commit()

            parsed = parse_document(Path(doc.stored_path), filename=doc.filename)
            pieces = self.build_chunks(parsed, filename=doc.filename)
            if not pieces:
                raise IngestError("文档里没有提取到可索引的内容", filename=doc.filename)

            # 重新索引时先清掉旧分块与旧向量,否则会留下孤儿向量
            await self._purge_index(doc)

            rows = [
                {
                    "doc_id": doc.id,
                    "kb_id": doc.kb_id,
                    "chunk_index": piece.index,
                    "content": piece.content,
                    "token_count": 0,
                    "meta": {**piece.meta, "filename": doc.filename},
                }
                for piece in pieces
            ]
            created = await self._chunks.bulk_create(rows)
            for chunk in created:
                chunk.vector_id = chunk_vector_id(chunk.id)
            await self._s.flush()

            await self._embed_and_store(doc, created)

            await self._docs.set_status(
                doc,
                DocumentStatus.READY,
                error=None,
                chunk_count=len(created),
            )
            doc.warnings = list(parsed.warnings)

            # 知识库上的冗余计数要在这里补上。
            #
            # 删除路径(`_purge_index` / `delete_document`)一直在做减法,而这里
            # 如果只更新文档自己、不告诉知识库,`chunk_count` 就会**只减不增**:
            # 列表页的数字越来越小,却没有任何地方会报错。`doc_count` 由路由层
            # 加,两者分散确实不理想,但它们是两个不同的生命周期(文档记录在
            # 上传时就存在,分块要等后台摄取完)。
            await self._kbs.apply_delta(await self._kb_of(doc), chunks=len(created))
            # **换代次,并且和上面这些写入在同一个事务里。**
            # 分块集合变了,任何算在该集合上的缓存结果都作废。放在这个
            # commit 之前而不是之后,是为了让「分块写进去了」和「缓存失效了」
            # 要么一起成立、要么一起不成立——拆成两个事务的话,中间那一下
            # 崩掉就是一次缓存不失效的写入,也就是一个给出旧答案的问答。
            await CorpusVersion.on(self._s).bump()
            await self._s.commit()

            logger.info(
                "文档入库完成",
                extra=log_extra(
                    doc_id=doc.id, kb_id=doc.kb_id, filename=doc.filename, chunks=len(created)
                ),
            )
            return len(created)

        except Exception as exc:  # noqa: BLE001 - 见 docstring:失败要落库,不能抛
            # **先把这两个值取出来,再回滚。**
            #
            # `rollback()` 会让会话里的对象**全部过期**(`expire_on_commit=False`
            # 只挡得住 commit,挡不住 rollback),之后任何属性访问都要重新去
            # 数据库读一次。而那是一次**同步** IO,在 async 上下文里会直接抛
            # `sqlalchemy.exc.MissingGreenlet`。
            #
            # 后果很难看:**错误处理自己把错误处理搞崩了**。摄取失败本该返回 0
            # 并把状态标成 FAILED,结果向上抛出一个与真实故障毫无关系的
            # greenlet 异常,调用方看到的东西和实际发生的事完全对不上——
            # 真正的失败原因(那次 `RuntimeError`)只留在日志的上一条里。
            #
            # 这不是理论问题:下面这条 `logger.warning` 曾经就是崩溃点,
            # 而它只在「摄取中途失败」时才会走到,平时测试全绿。
            doc_id, doc_filename = doc.id, doc.filename
            await self._s.rollback()
            message = exc.message if isinstance(exc, IngestError) else f"{type(exc).__name__}: {exc}"
            logger.warning(
                "文档入库失败",
                extra=log_extra(doc_id=doc_id, filename=doc_filename, error=message),
            )
            # 回滚后 `doc` 已过期,重新取一次再写状态(上面取的是普通值,
            # 不受过期影响)。
            fresh = await self._docs.get_by_id(doc_id)
            if fresh is not None:
                await self._docs.set_status(fresh, DocumentStatus.FAILED, error=message)
                await self._s.commit()
            return 0

    async def _embed_and_store(self, doc: Document, chunks: list[Chunk]) -> None:
        """向量化并写入 Chroma。

        `embed_documents` 内部已按 `EMBED_BATCH_SIZE` 分批(云端实测单批上限 20,
        配置取 16 留余量),这里不再自己分批。
        """
        if not chunks:
            return

        texts = [c.content for c in chunks]
        vectors = await self._embedder.embed_documents(texts)
        if len(vectors) != len(chunks):
            raise IngestError(
                f"向量数量与分块数量不一致({len(vectors)} ≠ {len(chunks)})",
                filename=doc.filename,
            )

        ids = [c.vector_id or chunk_vector_id(c.id) for c in chunks]
        metadatas = [
            sanitize_metadata(
                {
                    # 这三个是检索期的过滤/回查主键,必须写进去
                    "kb_id": doc.kb_id,
                    "doc_id": doc.id,
                    "chunk_index": c.chunk_index,
                    "filename": doc.filename,
                    **(c.meta or {}),
                }
            )
            for c in chunks
        ]

        await self._vs.upsert(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadatas,
        )

    async def _purge_index(self, doc: Document) -> None:
        """清掉一篇文档已有的分块与向量。重新索引前调用。

        即使 `delete_by_doc` 返回 0 也照样清一次向量:上一次摄取可能
        在「分块已写、向量写了但事务回滚」的半途失败过,留下的孤儿向量
        只有这里能顺手清掉。一次按 doc_id 的删除很便宜,不差这一下。
        """
        removed = await self._chunks.delete_by_doc(doc.id)
        if removed:
            await self._kbs.apply_delta(await self._kb_of(doc), chunks=-removed)
        await self._vs.delete_where({"doc_id": doc.id})

    async def _kb_of(self, doc: Document) -> KnowledgeBase:
        """取文档所属知识库。

        走主键查询而不是从 `identity_map` 里捞:后者依赖「调用方一定
        已经把对象加载进当前会话」这个隐式约定,一旦不成立就会静默地
        不更新计数(而是抛错,如果实现得不好)。一次主键查询的成本可以忽略,
        换来的是不依赖调用顺序。
        """
        kb = await self._kbs.get_by_id(doc.kb_id)
        if kb is None:  # pragma: no cover - 外键约束保证不会发生
            raise IngestError("文档所属的知识库不存在", doc_id=doc.id, kb_id=doc.kb_id)
        return kb

    # -- 删除 ---------------------------------------------------------------

    async def delete_document(self, doc: Document) -> None:
        """删除一篇文档:向量 → 分块 → 文档 → 文件。

        顺序同样是从「派生」到「原始」:先删派生物,后删事实来源。
        反过来的话,中途失败会留下**指向已删文件的数据库记录**,
        重试时解析必然失败。
        """
        kb = await self._kb_of(doc)
        chunk_count = await self._chunks.delete_by_doc(doc.id)

        await self._vs.delete_where({"doc_id": doc.id})
        await self._docs.delete(doc)
        await self._kbs.apply_delta(kb, docs=-1, chunks=-chunk_count)
        # 同一个理由见 `ingest_document`:分块被删了,缓存必须跟着作废。
        await CorpusVersion.on(self._s).bump()
        await self._s.commit()

        # 文件删除放在最后,且失败不影响业务结果——数据库里已经没有这条记录了,
        # 留下一个孤儿文件只是占几 MB 磁盘,而抛异常会让用户看到「删除失败」
        # 却查不到任何文档,那更让人困惑。
        path = Path(doc.stored_path)
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning(
                "原始文件删除失败(数据库记录已清理)",
                extra=log_extra(doc_id=doc.id, path=str(path), error=str(exc)),
            )

        logger.info("文档已删除", extra=log_extra(doc_id=doc.id, kb_id=doc.kb_id))

    async def delete_knowledge_base(self, kb: KnowledgeBase) -> None:
        """删除整个知识库。

        向量按 `kb_id` 一次性清掉,比逐篇文档删快得多;文件则要逐篇取路径
        才能删,所以先查出来。
        """
        docs = await self._docs.list_by_kb(kb.id)

        await self._vs.delete_where({"kb_id": kb.id})
        # 分块与文档交给外键 ON DELETE CASCADE 处理(passive_deletes),
        # 这里只需删知识库本身。
        await self._s.delete(kb)
        # **级联删除同样要换代次。** 这条最容易被漏掉,因为代码里根本没出现
        # 「删分块」这个动作——分块是被外键顺带带走的。但结果一样:缓存里
        # 那些 chunk 现在指向不存在的行,而答案会照常引用它们的编号。
        await CorpusVersion.on(self._s).bump()
        await self._s.commit()

        for doc in docs:
            try:
                Path(doc.stored_path).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning(
                    "原始文件删除失败(知识库记录已清理)",
                    extra=log_extra(doc_id=doc.id, error=str(exc)),
                )

        logger.info("知识库已删除", extra=log_extra(kb_id=kb.id, docs=len(docs)))

    # -- 校验 ---------------------------------------------------------------

    @staticmethod
    def classify_upload(filename: str) -> str:
        """上传前置校验:尽早拒绝不支持的格式,不浪费一次落盘。"""
        return classify(filename)
