"""向量库封装(Chroma PersistentClient)。

**为什么直接用 `chromadb`,而不是 `langchain_chroma.Chroma`**

项目名义上「基于 LangChain」。但这一层刻意不走
`langchain_chroma.Chroma`,pyproject 里也因此**没有** langchain-chroma
(装了却不用,只会让那份「刻意不引入」的清单失去可信度)。
理由是三条具体的:

1. **相似度要算对。** `langchain_chroma` 的 `similarity_search_with_score()`
   返回的是 **distance 不是 similarity**——名字里带 similarity,返回的却是
   距离,是社区里反复踩的坑。本项目的语义缓存要靠「余弦 > 0.95」判定,
   rerank 前的粗筛阈值也要用相似度。要拿到真正的相似度,横竖得自己
   从 distance 换算,那不如一开始就自己管。

2. **元数据过滤要显式。** 预过滤条件(类目、价格区间、kb_id)是电商场景的
   核心能力。Chroma 原生的 `where={"$and": [{"price": {"$gte": 1000}}]}`
   写法直接可见、可直接测;套一层 LangChain 的 filter 翻译反而多一层
   不确定性。

3. **少一层就少一处版本耦合。** langchain-chroma 的接口随 LangChain 1.x
   还在演进,而 chromadb 的 Python API 已经稳定。

LangChain 依然是本项目的编排骨架(LCEL Runnable、文本分块器、ChatOpenAI),
只是**存储这一层选择贴着原生 API**。这是取舍,不是漏用。
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass, field
from typing import Any

import anyio
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.core.config import Settings
from app.core.logging import log_extra

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class VectorHit:
    """一次向量检索的命中。

    `similarity` 是**余弦相似度**(越大越像),不是距离。
    `raw_score` 保留 Chroma 原样返回的 distance,便于排查
    「为什么这条没被召回」这类问题。
    """

    vector_id: str
    chunk_id: int | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    similarity: float = 0.0
    raw_score: float = 0.0


def chunk_vector_id(chunk_id: int) -> str:
    """SQLite 的 chunk.id → Chroma 的主键。

    加 `c` 前缀是为了让两类 id 在日志里一眼可分——排查「取回来的是
    数据库 id 还是向量 id」时,少一次翻代码。
    """
    return f"c{chunk_id}"


def parse_chunk_id(vector_id: str) -> int | None:
    """`chunk_vector_id` 的逆运算。解析不了就返回 None,不抛异常。"""
    if vector_id.startswith("c") and vector_id[1:].isdigit():
        return int(vector_id[1:])
    return None


def build_where(
    kb_ids: list[int] | None = None,
    *,
    category: str | None = None,
    price_min: float | None = None,
    price_max: float | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    """拼 Chroma 的 where 子句。条件全空时返回 None(Chroma 不接受空 dict)。

    **这是服务端预过滤**,不是「先取回再筛」。区别很大:后过滤时如果
    前 20 条恰好全是别的类目,筛完可能一条不剩;预过滤是先缩小子集再取
    top-k,结果数量有保证。
    """
    conds: list[dict[str, Any]] = []

    # **`None` 与 `[]` 含义相反:前者「不限制」,后者「一个都不许」。**
    # 写成 `if kb_ids:` 会把空集合当成「没有条件」,于是 where=None ——
    # 检索跨全部知识库。一个没有任何知识库权限的用户因此能读到别人的
    # 资料,而且不报错、不留异常日志,只是悄悄多返回了几条。
    #
    # 为什么不在这里返回一个「匹配不到任何东西」的条件:实测 Chroma 对
    # `{"kb_id": {"$in": []}}` 直接抛 `ValueError: Expected where operand
    # value to be a non-empty list`,根本走不到查询。既然底层表达不出
    # 「空」,就必须由调用方在**进查询之前**处理掉(检索服务遇到空范围
    # 直接返回无结果);这里把「忘了处理」变成一次响亮的失败。
    if kb_ids is not None:
        if not kb_ids:
            raise ValueError(
                "kb_ids 为空集合:这表示「没有任何可访问的知识库」,"
                "调用方应当在检索前直接返回空结果,而不是让过滤条件消失。"
                "若确实想不限制范围,请显式传 None。"
            )
        # $in 而不是逐库查 N 次:多知识库检索在电商里是常态
        # (比如同时查「显卡」和「主板」两个库)。
        conds.append({"kb_id": {"$in": list(kb_ids)}})
    if category:
        conds.append({"category": category})
    if price_min is not None:
        conds.append({"price": {"$gte": price_min}})
    if price_max is not None:
        # **不能写成 {"price": {"$gte": a, "$lte": b}}。**
        # Chroma 的校验要求每个字段的操作符表达式**恰好一个**操作符,
        # 两个会直接抛 `Expected operator expression to have exactly one
        # operator`。写法上必须拆成两个独立条件再用 $and 合并——
        # 这与 MongoDB 的直觉相反,是个很容易踩的坑。
        conds.append({"price": {"$lte": price_max}})
    if extra:
        conds.extend({k: v} for k, v in extra.items())

    if not conds:
        return None
    if len(conds) == 1:
        return conds[0]
    return {"$and": conds}


class ChromaStore:
    """持久化向量库。

    **所有方法都是 async**,内部把同步阻塞的 chromadb 调用丢进线程池。
    直接在 `async def` 里调 chromadb 会阻塞整个 FastAPI 事件循环——
    一次查询几十毫秒,并发请求会串成一条线,这正是「企业级」和
    「能跑通」的分界线。
    """

    def __init__(self, settings: Settings) -> None:
        self._path = settings.resolved_chroma_dir
        self._name = settings.CHROMA_COLLECTION
        self._client: chromadb.ClientAPI | None = None
        self._collection: Any = None

    # -- 同步内核(只在线程池里跑) ------------------------------------------

    def _ensure_collection(self) -> Any:
        if self._collection is not None:
            return self._collection

        self._path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self._path),
            settings=ChromaSettings(
                anonymized_telemetry=False,
                # 遥测开关在 main.py 里靠环境变量也关了一次。这里再关一道:
                # 环境变量只在 import 时刻生效,任何一处 import 顺序变化
                # 都会让它失效,而隐身遥测是公司内网部署的红线。
                allow_reset=False,
            ),
        )
        self._collection = self._client.get_or_create_collection(
            name=self._name,
            # **必须显式指定余弦距离。** Chroma 默认是 l2。两者的区别不只是
            # 数值不同:l2 距离没有上界,而本项目的语义缓存阈值(>0.95)、
            # 检索质量分析都建立在「余弦相似度落在 [-1,1]」这个前提上。
            # 不写这一行,阈值就变成了一个凭感觉拍出来的魔数。
            metadata={"hnsw:space": "cosine"},
        )
        return self._collection

    async def _call(self, fn: Any, /, *args: Any, **kwargs: Any) -> Any:
        """把同步调用挪到线程池。

        `functools.partial` 而非 lambda:`anyio.to_thread.run_sync` 只接受
        位置参数,关键字要靠 partial 预先绑定。
        """
        return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))

    # -- 写入 ----------------------------------------------------------------

    async def upsert(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        if not ids:
            return
        if not (len(ids) == len(embeddings) == len(documents) == len(metadatas)):
            # 长度不一致时 chromadb 会抛出一条很难读的内部错误。
            # 在这里挡住,错误信息才能指出到底差多少。
            raise ValueError(
                f"写入向量库的参数长度不一致: "
                f"ids={len(ids)} embeddings={len(embeddings)} "
                f"documents={len(documents)} metadatas={len(metadatas)}"
            )

        col = self._ensure_collection()
        # upsert 而非 add:重新索引同一篇文档时用同样的 vector_id 覆盖,
        # 不会留下孤儿向量。add 遇到重复主键会直接报错。
        await self._call(
            col.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        logger.debug("向量库 upsert 完成", extra=log_extra(count=len(ids)))

    async def delete(self, *, ids: list[str]) -> None:
        if not ids:
            return
        col = self._ensure_collection()
        await self._call(col.delete, ids=ids)

    async def delete_where(self, where: dict[str, Any]) -> None:
        """按元数据条件删除。删整篇文档/整个知识库时用。"""
        col = self._ensure_collection()
        await self._call(col.delete, where=where)

    # -- 查询 ----------------------------------------------------------------

    async def query(
        self,
        *,
        embedding: list[float],
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[VectorHit]:
        col = self._ensure_collection()
        n = col.count()
        if n == 0:
            # count()==0 时 Chroma 查询会抛异常而不是返回空列表。
            # 空知识库是**正常的初始状态**,不该让问答接口 500。
            return []

        res = await self._call(
            col.query,
            query_embeddings=[embedding],
            n_results=min(top_k, n),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[VectorHit] = []
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]

        for i, vid in enumerate(ids):
            distance = float(dists[i]) if i < len(dists) else 1.0
            hits.append(
                VectorHit(
                    vector_id=vid,
                    chunk_id=parse_chunk_id(vid),
                    content=docs[i] if i < len(docs) else "",
                    metadata=dict(metas[i]) if i < len(metas) and metas[i] else {},
                    # cosine 空间下 distance = 1 - cos_sim,且已裁剪到 [0,2]。
                    # 夹到 [-1,1] 是为了防浮点误差给出 1.0000000000000002。
                    similarity=max(-1.0, min(1.0, 1.0 - distance)),
                    raw_score=distance,
                )
            )
        return hits

    # -- 运维 ----------------------------------------------------------------

    async def count(self) -> int:
        col = self._ensure_collection()
        return int(await self._call(col.count))

    async def reset(self) -> None:
        """清空整个 collection。仅用于测试与「重建全部索引」。

        走 delete_collection + 重建,而不是逐条 delete:后者在几万条
        向量上要跑很久,而且会留下一堆待合并的 HNSW 墓碑。
        """
        self._collection = None
        if self._client is None:
            self._ensure_collection()
        assert self._client is not None

        def _drop_and_recreate() -> None:
            try:
                self._client.delete_collection(self._name)  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001 - collection 不存在时删除会抛错,属正常
                logger.debug("collection 不存在,跳过删除")
            self._collection = self._client.get_or_create_collection(  # type: ignore[union-attr]
                name=self._name, metadata={"hnsw:space": "cosine"}
            )

        await self._call(_drop_and_recreate)
        logger.info("向量库已重置")


def build_vector_store(settings: Settings) -> ChromaStore:
    return ChromaStore(settings)
