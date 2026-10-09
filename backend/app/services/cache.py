"""缓存层:向量缓存、检索缓存、答案缓存、语义缓存。

**缓存是这个系统里唯一会「给出错误答案」的优化。** 慢和贵都是可接受的降级,
错不是。所以这一层的设计围绕两件事:键要包含全部影响结果的因素,以及
知识库写入必须让内容相关的缓存整体失效——**后者没有商量余地**,文档已删除
而缓存还在拿旧 chunk 回答,答案里引用的来源编号会指向一个不存在的片段,
而且完全不报错。

**每一层缓存都必须能被单独关掉。** 排障时第一件要做的事就是确认「这个问题
是缓存带来的」,而如果关不掉,就只能去翻代码推断。

分层与各自的收益(实测数据见 `docs/BENCHMARK.md`):

    层         键                                            失效率    收益
    ------------------------------------------------------------------------
    向量       模型名 + 维度 + 角色 + 正文哈希                模型变更   省一次网络编码
    检索       归一化问题 + 范围 + 过滤 + 语料代次            语料写入   省两路召回
    答案       改写后的问题 + 范围 + 语料代次                  语料写入   省一次 LLM 调用
    语义       问题向量最近邻(scope 同上)                     语料写入   省一次 LLM 调用

**前三层的键是精确匹配,第四层不是**——于是它有一种别人没有的失败模式:
把另一个问题的答案发出去。所以它**默认关闭**,阈值是测出来的而不是拍的
(见 `SemanticCache` 的类文档与 `docs/SEMANTIC-CACHE.md`)。

**「语料代次」是失效机制的核心**:每次分块表发生写入,代次 +1;所有内容相关
的缓存的键里都带着当前代次,于是旧条目自然读不到(而不是需要遍历删除)。
代价是旧条目要等过期清理才释放空间,换来的是失效**不可能漏**——没有
「忘记调 invalidate」这种失败模式,因为失效不是一次动作,而是键的一部分。
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import struct
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol

from sqlalchemy import delete, select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.logging import log_extra
from app.db.base import Database
from app.db.models import (
    CacheEntry,
    CacheSemanticEntry,
    CorpusGeneration,
    naive_utc,
    utcnow,
)

if TYPE_CHECKING:
    # 只为类型标注而导入。`rag.py` 是编排层,缓存层不该在导入期依赖它
    # (方向反了),方法与 `CachedRetriever._decode` 的局部导入一致。
    from app.providers.base import Embedder
    from app.services.rag import SourceRef

logger = logging.getLogger(__name__)

# namespace 集中登记。写成常量而不是散在调用点的字符串字面量:拼错一个
# namespace 不会报错,只会得到一个永远不命中的缓存——一个「优化了但没生效」
# 的状态,而它在指标上看起来和「缓存确实没用」一模一样。
NS_EMBEDDING = "embedding"
NS_RETRIEVAL = "retrieval"
NS_ANSWER = "answer"

# 缓存条目的键长度上限,与 `CacheEntry.key` 的列宽一致。撞上它说明有人在
# 拿整段正文当键——那既没有必要(哈希就够了)也是一个信号:键里混进了
# 本该被哈希掉的东西。
MAX_KEY_LENGTH = 128

# 批量写入时一条 INSERT 里塞多少行。SQLite 默认的绑定变量上限是 999,
# 而每行占 4 个(域、键、值、过期时间),所以 200 行是安全值。
# 撞上限的表现是 `too many SQL variables`——而它只在入库大批量时出现,
# 小数据集的测试永远碰不到。
_MAX_BATCH_ROWS = 200


# ---------------------------------------------------------------------------
# 向量编解码
# ---------------------------------------------------------------------------


def pack_vector(vec: Sequence[float]) -> bytes:
    """向量 → 字节串。

    **用 `<f` 显式声明小端 float32**,而不是 `array("f")` 的原生序:后者
    在大小端机器上产出不同的字节,于是缓存文件换台机器读出来是乱码级别的
    错误——数值还在,只是完全不对,而它照样能算出一个余弦相似度。
    """
    if not vec:
        # 空向量缓存下来,余弦相似度会变成除以零。宁可在写入时就炸。
        raise ValueError("空向量不能被缓存")
    return struct.pack(f"<{len(vec)}f", *vec)


def unpack_vector(blob: bytes, *, expected_dim: int | None = None) -> list[float]:
    """字节串 → 向量。长度不对就抛,不静默返回残缺结果。

    截断的 blob 若被静默接受,会得到一个**短向量**:算余弦时取决于两条
    谁更长,结果是「有时崩、有时只是不命中」。所以长度必须校验。
    """
    if not blob or len(blob) % 4 != 0:
        raise ValueError(f"向量字节串长度非法:{len(blob)}")
    dim = len(blob) // 4
    if expected_dim is not None and dim != expected_dim:
        # 换 embedding 模型但没换代次时会出现这种情况。放任下去,检索会拿
        # 两种维度的向量做相似度比较。
        raise ValueError(f"向量维度不符:缓存里是 {dim},期望 {expected_dim}")
    return list(struct.unpack(f"<{dim}f", blob))


def text_hash(text: str) -> str:
    """正文 → 稳定短哈希。

    不用 `hash()`:它带进程级随机盐(PYTHONHASHSEED),换个进程同样的文本
    算出不同的值,缓存命中率会莫名其妙变成 0,而单次运行内看起来完全正常。
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _expired(expires_at: datetime | None) -> bool:
    """条目是否过期。

    **SQLite 的 DATETIME 存不了时区**:写进去的 aware datetime 读出来是
    naive 的(见 `models` 模块头部的约定)。直接拿它和 `utcnow()` 比较会抛
    `TypeError: can't compare offset-naive and offset-aware datetimes`,
    而这条路径只在「条目恰好过期的那个瞬间」被执行——平时全绿,线上偶发。
    所以这里统一按 UTC 解释读出来的值。
    """
    if expires_at is None:
        return False
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    return expires_at <= utcnow()


# ---------------------------------------------------------------------------
# 通用 KV 存储
# ---------------------------------------------------------------------------


class CacheStore:
    """带 TTL 的分区 KV 缓存,落在 SQLite 上。

    **默认每条操作自己开一个会话、自己提交**,不借用请求的事务。两个理由:
    问答接口里缓存写入发生在流式响应期间,那时请求事务的状态由路由决定,
    缓存不该被卷进去(提交了它会连带提交别的东西,回滚了它自己就白写);
    而它一旦被回滚,缓存就永远只有 miss,表现和「缓存没生效」一模一样。

    **但有一种情形必须反过来:`CacheStore.on(session)`。** 调用方此刻正持着
    一个未提交的写事务时,另开会话拿不到 SQLite 的写锁,而这不是「稍后再试」
    能解决的——持锁的事务要等这次编码结束才提交,而编码正在等这把锁,
    是个死锁。`busy_timeout` 只是把失败推迟了 5 秒(见 `tests/test_cache.py`
    的 `TestBorrowedStore`)。文档摄取正是这个形状:它一边往 `chunks` 里写,
    一边要编码。所以借用与自持两种模式都要有,这也正是 `CorpusVersion`
    已经采用的形状。
    """

    def __init__(self, source: Database | AsyncSession) -> None:
        if isinstance(source, AsyncSession):
            self._db: Database | None = None
            self._borrowed: AsyncSession | None = source
        else:
            self._db = source
            self._borrowed = None

    @classmethod
    def on(cls, session: AsyncSession) -> CacheStore:
        """绑定到**调用方已有的那个事务**。

        两个后果,缺一不可:

        - **写进调用方的事务**,所以调用方回滚,缓存也跟着回滚;
        - **不自己提交**,提交由调用方决定。

        拿它当「省一次开会话」的语法糖来用是不对的:它把缓存的可见性交给了
        调用方的提交时机,只在「本次写入必须和调用方的写入同生共死」时才用。
        """
        return cls(session)

    @contextlib.asynccontextmanager
    async def _scope(self) -> AsyncIterator[AsyncSession]:
        """借来的会话还给调用方,自己开的自己收。"""
        if self._borrowed is not None:
            yield self._borrowed
            return
        async with self._db.session() as session:
            yield session

    async def _commit(self, session: AsyncSession) -> None:
        """自己的会话才提交。**借来的提交会把调用方半个事务也提交掉。**"""
        if self._borrowed is None:
            await session.commit()

    async def get(self, namespace: str, key: str) -> bytes | None:
        async with self._scope() as session:
            row = await session.get(CacheEntry, (namespace, key))
            if row is None or _expired(row.expires_at):
                return None
            # `bytes()` 而不是直接返回:某些驱动返回 memoryview,
            # 它长得像字节串但不能被 `struct.unpack` 接受。
            return bytes(row.value)

    async def get_many(self, namespace: str, keys: Sequence[str]) -> dict[str, bytes]:
        """批量读。命中的键才有条目,未命中的不在返回值里。

        给 embedding 用:一次入库几百个分块,逐个开会话去查会把「批量」
        这件事的意义抵消掉。
        """
        if not keys:
            return {}
        async with self._scope() as session:
            rows = (
                await session.execute(
                    select(CacheEntry).where(
                        CacheEntry.namespace == namespace,
                        CacheEntry.key.in_(list(keys)),
                    )
                )
            ).scalars()
            return {
                row.key: bytes(row.value) for row in rows if not _expired(row.expires_at)
            }

    async def set_many(
        self,
        namespace: str,
        items: Mapping[str, bytes],
        *,
        ttl_seconds: int | None,
    ) -> None:
        """批量写入。给 embedding 用,和 `get_many` 配成一对。

        **一次调用只发一条语句**,而不是循环 `set`。逐个写的话,一次入库
        几百个分块会开几百个会话、几百次提交,把「批量编码」省下来的时间
        又还回去;更麻烦的是每条各是一个事务,中途失败会留下一个**写了一半
        的缓存**——它不报错,只是命中率莫名其妙地低。

        **校验在任何一行写入之前全部做完**。写在循环里的话,超长键会在跑到
        一半时才炸,而那时前面几百条已经落库了。

        超过 `_MAX_BATCH_ROWS` 时切块:SQLite 的绑定变量有上限,而撞上它的
        表现(`too many SQL variables`)只在真正大批量入库时出现,小数据集
        的测试永远碰不到。
        """
        if not items:
            # 空批次也要提前返回:否则会发出一条没有 VALUES 的 INSERT,
            # 那是一句语法错误的 SQL。
            return
        overlong = [key for key in items if len(key) > MAX_KEY_LENGTH]
        if overlong:
            raise ValueError(f"缓存键过长({len(overlong[0])}):{overlong[0][:64]}…")

        expires = None if ttl_seconds is None else utcnow() + timedelta(seconds=ttl_seconds)
        # `created_at` 显式给值。多行 `VALUES` 是**一条**语句,列默认值不会
        # 逐行触发,而这一列是 NOT NULL——省掉它就是一句 IntegrityError。
        created = utcnow()
        rows = [
            {
                "namespace": namespace,
                "key": key,
                "value": value,
                "expires_at": expires,
                "created_at": created,
            }
            for key, value in items.items()
        ]

        async with self._scope() as session:
            for start in range(0, len(rows), _MAX_BATCH_ROWS):
                chunk = rows[start : start + _MAX_BATCH_ROWS]
                stmt = sqlite_insert(CacheEntry).values(chunk)
                stmt = stmt.on_conflict_do_update(
                    # 冲突判据必须写成主键的两列。只写 key 会让两个域互相
                    # 覆盖——embedding 的批量写入把 retrieval 的同名键冲掉,
                    # 而且不报错。
                    index_elements=[CacheEntry.namespace, CacheEntry.key],
                    set_={
                        "value": stmt.excluded.value,
                        "expires_at": stmt.excluded.expires_at,
                        # 覆盖时保留原来的 created_at:它记的是这条缓存
                        # 「活了多久」,被一次刷新重置会让 TTL 相关的排查失真。
                    },
                )
                await session.execute(stmt)
            await self._commit(session)

    async def set(
        self,
        namespace: str,
        key: str,
        value: bytes,
        *,
        ttl_seconds: int | None,
    ) -> None:
        """写入。`ttl_seconds=None` 表示永不过期(向量缓存就是这种)。

        先删后插而不是 `merge`:merge 会先发一条 SELECT,而缓存写入的热路径
        上多一次往返没有意义。删除+插入在同一个事务里,对外是原子的。
        """
        if len(key) > MAX_KEY_LENGTH:
            raise ValueError(f"缓存键过长({len(key)}):{key[:64]}…")
        expires = None if ttl_seconds is None else utcnow() + timedelta(seconds=ttl_seconds)
        async with self._scope() as session:
            await session.execute(
                delete(CacheEntry).where(
                    CacheEntry.namespace == namespace, CacheEntry.key == key
                )
            )
            session.add(
                CacheEntry(namespace=namespace, key=key, value=value, expires_at=expires)
            )
            await self._commit(session)

    async def purge_expired(self) -> int:
        """删掉所有已过期的条目,返回删除条数。

        过期条目在读取时就已被当作 miss,这里只是回收空间。**必须显式调用**,
        不在读路径上顺手删:读路径上做写操作会让缓存命中时也去抢写锁,
        而缓存的全部意义是让热路径更快。
        """
        async with self._scope() as session:
            result = await session.execute(
                delete(CacheEntry).where(
                    CacheEntry.expires_at.is_not(None),
                    CacheEntry.expires_at <= naive_utc(utcnow()),
                )
            )
            await self._commit(session)
            return int(result.rowcount or 0)

    async def clear(self, namespace: str | None = None) -> int:
        """清空一个域(或全部)。返回删除条数。"""
        async with self._scope() as session:
            stmt = delete(CacheEntry)
            if namespace is not None:
                stmt = stmt.where(CacheEntry.namespace == namespace)
            result = await session.execute(stmt)
            await self._commit(session)
            return int(result.rowcount or 0)

    async def stats(self) -> dict[str, int]:
        """每个域的条目数。指标面板用。"""
        async with self._scope() as session:
            rows = await session.execute(
                select(CacheEntry.namespace, CacheEntry.key).where(
                    CacheEntry.expires_at.is_(None)
                    | (CacheEntry.expires_at > naive_utc(utcnow()))
                )
            )
            out: dict[str, int] = {}
            for namespace, _key in rows:
                out[namespace] = out.get(namespace, 0) + 1
            return out


# ---------------------------------------------------------------------------
# 语料代次
# ---------------------------------------------------------------------------


class CorpusVersion:
    """语料代次:任何改变分块集合的写入都必须 `bump()`。

    **它是失效机制的根**:检索缓存与答案缓存的键里都带着当前代次,写入后
    代次一变,旧条目自然读不到。这样做而不是「写入时遍历删除相关条目」,
    是因为**失效不可以漏**:漏删一次就是一次错误的答案,而「忘记在哪三个
    地方调用 invalidate」是一个必然会发生的错误。代价是旧条目要等 TTL 才
    被回收,拿空间换正确性——这笔交易在任何时候都划算。
    """

    #: 单行表里的那一行。写死而不是「取最新」,理由见 `CorpusGeneration`。
    ROW_ID = 1

    def __init__(self, source: Database | AsyncSession) -> None:
        if isinstance(source, AsyncSession):
            self._db: Database | None = None
            self._borrowed: AsyncSession | None = source
        else:
            self._db = source
            self._borrowed = None

    @classmethod
    def on(cls, session: AsyncSession) -> CorpusVersion:
        """绑定到**调用方已有的那个事务**。

        知识库写入走这个入口,而不是让 `CorpusVersion(db)` 各开各的会话。
        区别不是风格问题:

        - **两个事务** → 分块可能提交了而代次没提交(**缓存不失效**,给出旧
          答案),或者反过来(白清一次缓存)。前者是本模块最不能接受的故障。
        - **一个事务** → 两者同生共死,不存在中间状态。

        两种用法共用 `_scope()`,所以 `current()` / `bump()` 只有一条代码路径。
        """
        return cls(session)

    @contextlib.asynccontextmanager
    async def _scope(self) -> AsyncIterator[AsyncSession]:
        """借来的事务不提交(由调用方决定),自己开的事务自己提交。"""
        if self._borrowed is not None:
            yield self._borrowed
            return
        async with self._db.session() as session:
            yield session
            await session.commit()

    async def current(self) -> int:
        """当前代次。没有行时是 0——**不能返回 None**,它会被拼进缓存键。"""
        async with self._scope() as session:
            value = await session.scalar(
                select(CorpusGeneration.value).where(CorpusGeneration.id == self.ROW_ID)
            )
            return int(value) if value is not None else 0

    async def bump(self) -> int:
        """代次 +1,返回新值。

        **自增在 SQL 里完成,不是「读出来加一再写回」。** 后者在并发下会丢
        更新:五路并发各自读到同一个值、各自写回同一个值,实测只数出 2
        (见 `tests/test_cache.py::TestCorpusVersion`)。丢了代次就意味着
        两次知识库写入只推进了一格,中间那次缓存条目会继续命中——一个不报错、
        只给出旧答案的故障,而它只在并发下出现。
        `INSERT ... ON CONFLICT DO UPDATE` 是一条语句,数据库保证它原子。
        """
        async with self._scope() as session:
            stmt = sqlite_insert(CorpusGeneration).values(
                id=self.ROW_ID, value=1, updated_at=utcnow()
            )
            stmt = stmt.on_conflict_do_update(
                index_elements=[CorpusGeneration.id],
                set_={"value": CorpusGeneration.value + 1, "updated_at": utcnow()},
            )
            await session.execute(stmt)
            # 同一个事务里读回,拿到的一定是刚写进去的那个值。
            # 用列查询而不是 `session.get(CorpusGeneration, ...)`:后者会命中
            # 身份映射,可能返回这个会话里早先缓存的旧对象。
            value = await session.scalar(
                select(CorpusGeneration.value).where(CorpusGeneration.id == self.ROW_ID)
            )
            return int(value or 0)


# ---------------------------------------------------------------------------
# 向量缓存
# ---------------------------------------------------------------------------


class CachedEmbedder:
    """给 `Embedder` 套一层缓存。**结构上满足 `app.providers.base.Embedder` 协议**。

    **这一层的键里没有语料代次,这是它和另外两层最本质的区别。** 一段文本
    的向量只是 (文本, 模型) 的函数,与知识库里有没有这段文本无关;知识库
    写入不该让任何一条向量失效。带上代次等于每次入库都把最贵的那部分编码
    全部重付一遍——而那正是这一层想省掉的东西。

    `ttl_seconds=None`(永不过期)同理:向量不会变质,给它设 TTL 只会在
    某个毫无道理的时刻让编码重跑一次。

    **角色(查询 / 文档)进键。** bge-zh 一类的模型查询侧要加指令前缀,
    两侧向量本就不同;就算当前模型不需要,把两者当同一条也是把「现在成立」
    当成「永远成立」——而换模型时不会有任何信号提示这里要改。

    **维度也进键。** 维度通常由模型决定,看着是冗余的,但有模型把输出维度
    做成可调参数(text-embedding-3 系列的 `dimensions`),那时同一模型名
    下的两种维度是两条不能混用的向量。键里带上它,代价是几个字符。
    """

    def __init__(
        self,
        *,
        inner: Embedder,
        store: CacheStore,
        settings: Settings,
        enabled: bool | None = None,
    ) -> None:
        self._inner = inner
        self._store = store
        self._enabled = settings.CACHE_ENABLED if enabled is None else enabled

    @property
    def dim(self) -> int:
        return self._inner.dim

    @property
    def model_id(self) -> str:
        return self._inner.model_id

    # -- 对外入口 ------------------------------------------------------------

    async def embed_query(self, text: str) -> list[float]:
        if not self._enabled:
            return await self._inner.embed_query(text)

        key = self._key("query", text)
        cached = await self._load_many({text: key})
        if text in cached:
            return cached[text]

        vector = await self._inner.embed_query(text)
        await self._store.set(NS_EMBEDDING, key, pack_vector(vector), ttl_seconds=None)
        return vector

    async def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """批量编码,只把**缺失的那些**发出去。

        这是这一层收益最大的一处:入库时一次几百个分块,而商品文案里
        「XX 型号,支持……,质保……年」这类句式重复度很高。

        返回值必须与入参**逐位对应**。向量是按位置被安到分块上的,顺序错了
        不会报错,只会让整个知识库的向量和正文错配——一个只在检索结果上
        显形、且看起来像「检索质量差」的故障。所以下面组装时用的是
        `texts` 而不是 `unique` 的顺序。
        """
        if not texts:
            # 提前返回:否则会给编码器发一个空批次,而它可能把空列表
            # 当成一次真实调用计费。
            return []
        if not self._enabled:
            return await self._inner.embed_documents(texts)

        # 去重但保持首次出现的顺序,于是「同一批里重复的文本只编码一次」,
        # 而编码器看到的顺序仍然与入参一致。
        unique = list(dict.fromkeys(texts))
        keys = {text: self._key("document", text) for text in unique}

        found = await self._load_many(keys)

        missing = [text for text in unique if text not in found]
        if missing:
            vectors = await self._inner.embed_documents(missing)
            payload: dict[str, bytes] = {}
            for text, vector in zip(missing, vectors, strict=True):
                # 一边存缓存一边回填 `found`:下面组装返回值时只认 `found`,
                # 漏了这一步会把刚编码出来的那批当成没拿到。
                found[text] = vector
                payload[keys[text]] = pack_vector(vector)
            await self._store.set_many(NS_EMBEDDING, payload, ttl_seconds=None)

        return [found[text] for text in texts]

    # -- 键 ----------------------------------------------------------------

    def _key(self, role: str, text: str) -> str:
        """`(模型, 维度, 角色, 正文)` → 键。

        正文整段进哈希、不截断:截断会让两段前缀相同的文案共用一条缓存,
        而算出来的向量是**其中一段的**。用 `\\x00` 分隔而不是拼接,是为了
        让 `("a", "bc")` 和 `("ab", "c")` 落不到同一个键上——这种歧义在
        纯拼接下要等一条缓存给出错向量时才会被发现。
        """
        payload = f"{self._inner.model_id}\x00{self._inner.dim}\x00{role}\x00{text}"
        return text_hash(payload)

    # -- 读 ----------------------------------------------------------------

    async def _load_many(self, keys: Mapping[str, str]) -> dict[str, list[float]]:
        """批量读,返回 `{正文: 向量}`,未命中或读坏的都不在里面。

        **坏条目退化成 miss,不让提问失败**,但留一条 warning——否则
        「缓存一直在坏」永远没人发现,它在指标上和「命中率天然就低」
        长得一模一样。

        `expected_dim` 必须传:一个截断的 blob 若被静默接受,会解出一个
        **长度合法的短向量**,算余弦时取决于两条谁更长,于是「有时崩、
        有时只是不命中」。4 字节的 `X'DEADBEEF'` 正好解出一个浮点数。
        """
        blobs = await self._store.get_many(NS_EMBEDDING, list(keys.values()))
        out: dict[str, list[float]] = {}
        for text, key in keys.items():
            blob = blobs.get(key)
            if blob is None:
                continue
            try:
                out[text] = unpack_vector(blob, expected_dim=self._inner.dim)
            except ValueError as exc:
                logger.warning("向量缓存条目无法反序列化,按未命中处理:%s", exc)
        return out


# ---------------------------------------------------------------------------
# 检索缓存
# ---------------------------------------------------------------------------


class RetrieverLike(Protocol):
    async def retrieve(
        self,
        query: str,
        *,
        kb_ids: list[int] | None,
        filters: Any = None,
        top_k: int | None = None,
    ) -> Any: ...


def _normalize_query(query: str) -> str:
    """归一化提问文本。

    **只合并空白,不做大小写折叠。** 折叠大小写看起来无害,其实会改变结果:
    BM25 的分词是靠字符的,语料里若是 `RX9070`,用户输入 `rx9070` 时两路
    召回都可能落空。既然「改大小写会改变结果」,那把大小写折叠进缓存键就
    等于让两种不同的检索共用一条缓存——正是这一层最不能犯的错。

    空白合并则是真正无关的:`"a  b"` 与 `"a b"` 分词后完全一致。
    """
    return " ".join(query.split())


def _filters_key(filters: Any) -> Any:
    """把过滤条件变成可哈希、可比较的纯数据。

    逐字段列出而不是 `asdict(filters)`:新增字段时**必须在这里显式加一行**,
    于是「忘了把新过滤条件算进缓存键」这件事会在写代码时就被迫面对一次。
    用 `asdict` 的话自动全带上,看着更省事,但那也意味着一个 `dataclass`
    字段的顺序/类型变化会**静默**改变键——出问题是往另一个方向。
    """
    if filters is None:
        return None
    return {
        "category": filters.category,
        "price_min": filters.price_min,
        "price_max": filters.price_max,
        "extra": filters.extra,
    }


def _config_fingerprint(settings: Settings) -> dict[str, Any]:
    """哪些配置变了会让同样的提问算出不同结果。

    **这份清单就是这类配置的全部**,漏一项就是一类静默串味——换了 rerank
    模型却还在用旧模型的排序,而「检索质量看着没变」无从察觉,因为它确实
    没变:新模型根本没跑。
    """
    return {
        "embed_model": settings.EMBED_MODEL,
        "embed_dim": settings.EMBED_DIM,
        "rerank_model": settings.RERANK_MODEL,
        "rerank_enabled": settings.RERANK_ENABLED,
        "rerank_backend": settings.RERANK_BACKEND,
        "rerank_top_n": settings.RERANK_TOP_N,
        "rerank_threshold": settings.RERANK_SCORE_THRESHOLD,
        "rrf_k": settings.RRF_K,
        "retrieve_top_k": settings.RETRIEVE_TOP_K,
    }


#: 缓存值里存一份键结构版本。改了下面 `_encode` 的字段布局就必须 +1,
#: 否则新代码会读到旧格式的字节串——反序列化失败虽然会退化成 miss,
#: 但那意味着**一次全量缓存静默失效**,而这和「缓存没配好」在指标上
#: 长得一模一样。
_SCHEMA_VERSION = 1


class CachedRetriever:
    """给检索层套一层缓存。**结构上满足 `rag.Retriever` 协议**。

    用装饰器而不是把缓存写进 `RetrievalService`:检索本身有三条路、两次
    融合、一次重排,已经是最复杂的一段;把「什么时候读缓存、什么时候
    跳过」混进去,会让「检索为什么返回这个结果」多出一个与检索无关的
    分支。分开之后,缓存层的正确性可以完全独立于检索算法来验证。
    """

    def __init__(
        self,
        *,
        inner: RetrieverLike,
        store: CacheStore,
        version: CorpusVersion,
        settings: Settings,
        enabled: bool | None = None,
    ) -> None:
        self._inner = inner
        self._store = store
        self._version = version
        self._settings = settings
        self._enabled = settings.CACHE_ENABLED if enabled is None else enabled

    async def retrieve(
        self,
        query: str,
        *,
        kb_ids: list[int] | None,
        filters: Any = None,
        top_k: int | None = None,
    ) -> Any:
        if not self._enabled:
            return await self._inner.retrieve(
                query, kb_ids=kb_ids, filters=filters, top_k=top_k
            )

        key = await self._key(query, kb_ids=kb_ids, filters=filters, top_k=top_k)

        mark = time.perf_counter()
        blob = await self._store.get(NS_RETRIEVAL, key)
        lookup_ms = round((time.perf_counter() - mark) * 1000, 3)

        if blob is not None:
            restored = self._decode(blob, query=query)
            if restored is not None:
                # 命中时也照实报耗时(就是刚测的这次查询),不写 0:
                # 一个恒为 0 的阶段耗时会被读成「检索不要钱」,而它其实
                # 只是「这次没算」——面板上的假数据比没有数据更误导。
                return replace(restored, stage_timings={"total": lookup_ms})

        result = await self._inner.retrieve(query, kb_ids=kb_ids, filters=filters, top_k=top_k)

        # 降级结果不缓存,见 `test_a_degraded_result_is_never_cached`。
        if getattr(result, "degraded", False):
            return result

        await self._store.set(
            NS_RETRIEVAL,
            key,
            self._encode(result),
            ttl_seconds=self._settings.RETRIEVAL_CACHE_TTL_SECONDS,
        )
        return result

    # -- 键 ----------------------------------------------------------------

    async def _key(
        self,
        query: str,
        *,
        kb_ids: list[int] | None,
        filters: Any,
        top_k: int | None,
    ) -> str:
        payload = {
            "schema": _SCHEMA_VERSION,
            "query": _normalize_query(query),
            # `None`(不限)与 `[]`(什么都不看)必须保持可区分。两者归一成
            # 同一个键,等于让「一个库都没勾」的提问命中一次全库检索的
            # 结果——一次不报错的越权读。
            "kb_ids": None if kb_ids is None else sorted(set(kb_ids)),
            # `None` 与显式传默认值是同一次检索:都走 `RETRIEVE_TOP_K`。
            "top_k": self._settings.RETRIEVE_TOP_K if top_k is None else top_k,
            "filters": _filters_key(filters),
            "generation": await self._version.current(),
            "config": _config_fingerprint(self._settings),
        }
        return text_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    # -- 编解码 -------------------------------------------------------------

    @staticmethod
    def _encode(result: Any) -> bytes:
        payload = {
            "schema": _SCHEMA_VERSION,
            "chunks": [asdict(chunk) for chunk in result.chunks],
            "route_sizes": dict(result.route_sizes),
            "reranked": bool(result.reranked),
        }
        return json.dumps(payload, ensure_ascii=False).encode("utf-8")

    @staticmethod
    def _decode(blob: bytes, *, query: str) -> Any:
        """反序列化。**任何异常都退化成 miss,绝不让提问失败。**

        缓存是纯优化:它坏了,代价应当是「这次慢一点」,而不是「用户问不了
        问题」。但也不静默吞掉——留一条 warning,否则「缓存一直在坏」这件
        事永远没人会发现,而它看起来和「缓存命中率天然就低」一模一样。
        """
        from app.services.context import RetrievedChunk
        from app.services.retrieval import RetrievalResult

        try:
            data = json.loads(blob)
            if data.get("schema") != _SCHEMA_VERSION:
                raise ValueError(f"缓存结构版本不符:{data.get('schema')}")
            chunks = [RetrievedChunk(**raw) for raw in data["chunks"]]
        except Exception as exc:
            logger.warning("检索缓存条目无法反序列化,按未命中处理:%s", exc)
            return None

        return RetrievalResult(
            query=query,
            chunks=chunks,
            route_sizes=dict(data.get("route_sizes") or {}),
            # 耗时由调用方用「本次缓存查询」的真实耗时覆盖,见 `retrieve`。
            stage_timings={},
            reranked=bool(data.get("reranked")),
            degraded=False,
            cache_hit=True,
        )


# ---------------------------------------------------------------------------
# 答案缓存
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CachedAnswer:
    """一条可以整份复用的答案。

    **没有 `cited` 字段,是刻意的。** 引用编号是可以从 `answer` 正文里重新
    数出来的(`extract_citations(answer, max_index=len(sources))`),存两份
    就有两份不一致的可能,而且不一致的时候正文说了算——那存的那一份是
    纯噪音。

    `grounded` 同理不存:检索为空时那条固定答复根本不进缓存(见 `put` 的
    调用方),所以这里出现的答案必然有来源。
    """

    answer: str
    sources: list[SourceRef]


def _encode_answer(value: CachedAnswer) -> bytes:
    """一条答案 → 字节串。

    **提到模块层是因为两个缓存类都要用**:精确答案缓存按问题哈希存它,
    语义缓存按问题向量存它。存的是**同一套字节**很重要——两份格式迟早
    会分叉,而分叉的表现是有一边读不出来、静默退化成永不命中。
    """
    payload = {
        "schema": _SCHEMA_VERSION,
        "answer": value.answer,
        "sources": [asdict(source) for source in value.sources],
    }
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


def _decode_answer(blob: bytes) -> CachedAnswer | None:
    """字节串 → 一条答案。**任何异常都退化成 miss,绝不让提问失败。**

    缓存是纯优化:它坏了,代价应当是「这次慢一点」,而不是「用户问不了
    问题」。但也不静默吞掉——留一条 warning,否则「缓存一直在坏」这件事
    永远没人会发现,而它看起来和「命中率天然就低」一模一样。
    """
    from app.services.rag import SourceRef

    try:
        data = json.loads(blob)
        if data.get("schema") != _SCHEMA_VERSION:
            raise ValueError(f"缓存结构版本不符:{data.get('schema')}")
        answer = data["answer"]
        if not isinstance(answer, str) or not answer:
            # 空答案尤其要挡:它会被当成一条正常的 done 事件发出去,
            # 用户看到的是**一个空白回答**,而不是模型重新生成一次。
            raise ValueError("缓存里的答案不是非空字符串")
        sources = [SourceRef(**raw) for raw in data["sources"]]
    except Exception as exc:
        logger.warning("答案缓存条目无法反序列化,按未命中处理:%s", exc)
        return None

    return CachedAnswer(answer=answer, sources=sources)


def _answer_fingerprint(settings: Settings) -> dict[str, Any]:
    """哪些配置变了会让同一个问题算出不同答案。

    在检索那份指纹之外**额外加上模型与提示词**,因为答案还要经过一次
    LLM:换了模型、改了温度、改了系统提示,同样的资料也会写出不同的字
    ——而这三样里任何一样变了却不失效,表现都是「新配置部署上去了,线上
    一点变化都没有」,最容易被误读成「改动没效果」。

    **提示词在调用时才读**(局部导入 `prompts` 模块再取属性)。若写成
    `from app.services.prompts import SYSTEM_PROMPT`,这个值会在模块导入
    时被绑死,之后编辑提示词文件读到的永远是旧的那一份——一个只有升级
    部署后才暴露、且看起来毫无道理的「新提示词不生效」。
    """
    from app.services import prompts

    return {
        **_config_fingerprint(settings),
        "llm_model": settings.LLM_MODEL,
        "llm_temperature": settings.LLM_TEMPERATURE,
        "llm_max_tokens": settings.LLM_MAX_TOKENS,
        "llm_backend": settings.LLM_BACKEND,
        "prompt": text_hash(prompts.SYSTEM_PROMPT),
    }


class AnswerCache:
    """缓存**最终答案**。这一层是三层缓存里最贵、也最危险的一层。

    贵:命中一次省掉的是一次完整的 LLM 调用,而不只是若干毫秒的检索。

    危险:检索缓存给错了顶多让引用的片段偏了;答案缓存给错了,用户看到的
    是一段措辞、编号、来源卡片全都齐整的错误回答——**所有表面信号都正常**。

    **带历史的一律不进这一层**,读和写都挡(见 `_eligible`)。理由是键构造
    不出历史来:「它多少钱?」在显卡会话里和在键盘会话里键完全一样,缓存
    拿到的只有问题文本、范围、配置三样。挡掉它比在键里加标记更可靠——
    标记是「记得加」,结构是「加不进去」。
    """

    def __init__(
        self,
        *,
        store: CacheStore,
        version: CorpusVersion,
        settings: Settings,
        enabled: bool | None = None,
    ) -> None:
        self._store = store
        self._version = version
        self._settings = settings
        self._enabled = settings.CACHE_ENABLED if enabled is None else enabled

    # -- 对外入口 ------------------------------------------------------------

    async def get(
        self,
        question: str,
        *,
        history_messages: int,
        kb_ids: list[int] | None,
        filters: Any = None,
    ) -> CachedAnswer | None:
        if not self._eligible(history_messages):
            return None
        blob = await self._store.get(
            NS_ANSWER, await self._key(question, kb_ids=kb_ids, filters=filters)
        )
        return None if blob is None else self._decode(blob)

    async def put(
        self,
        question: str,
        *,
        history_messages: int,
        kb_ids: list[int] | None,
        filters: Any = None,
        answer: CachedAnswer,
    ) -> None:
        if not self._eligible(history_messages):
            return
        await self._store.set(
            NS_ANSWER,
            await self._key(question, kb_ids=kb_ids, filters=filters),
            self._encode(answer),
            ttl_seconds=self._settings.ANSWER_CACHE_TTL_SECONDS,
        )

    def _eligible(self, history_messages: int) -> bool:
        """这一层要不要参与。

        条件只有两条,但**两条都必须在读和写两侧同时生效**:只挡读会写进
        一个带上下文的答案(它的键和一个没有历史的问题完全相同);只挡写
        则还是会把上一个人带历史生成的答案发给下一个人。
        """
        return self._enabled and history_messages == 0

    # -- 键 ----------------------------------------------------------------

    async def _key(self, question: str, *, kb_ids: list[int] | None, filters: Any) -> str:
        payload = {
            "schema": _SCHEMA_VERSION,
            "question": _normalize_query(question),
            # `None`(不限)与 `[]`(什么都不看)含义相反,必须保持可区分
            # ——归一成同一个键就是一次不报错的越权读。
            "kb_ids": None if kb_ids is None else sorted(set(kb_ids)),
            "filters": _filters_key(filters),
            "generation": await self._version.current(),
            "config": _answer_fingerprint(self._settings),
        }
        return text_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    # -- 编解码 -------------------------------------------------------------
    #
    # 实现在模块层(`_encode_answer` / `_decode_answer`),因为语义缓存存的是
    # 同一套字节。留这两个名字是为了调用点读起来仍然贴着这一层。

    _encode = staticmethod(_encode_answer)
    _decode = staticmethod(_decode_answer)


# ---------------------------------------------------------------------------
# 语义缓存
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """余弦相似度。

    **自己算,不走向量库**:这里的候选集是缓存条目,不是语料,量级差两三个
    数量级;为它引入一次 Chroma 往返(还有把缓存条目塞进向量库同步的复杂度)
    不划算。

    **两个向量都要归一化**,不能假设上游已经归一过。DashScope 的 embedding
    是否归一化没有写进合约,而一旦它不归一,「点积」和「余弦」在长度差异大的
    两条文本上会给出不同的排序——那是个只在长问题对上短问题时才显形的错。
    """
    if len(a) != len(b):
        raise ValueError(f"向量维度不一致:{len(a)} vs {len(b)}")
    dot = na = nb = 0.0
    for x, y in zip(a, b, strict=True):
        dot += x * y
        na += x * x
        nb += y * y
    if na <= 0.0 or nb <= 0.0:
        # 零向量没有方向,余弦无定义。返回 0 而不是让它除出 inf/nan——
        # nan 参与比较永远是 False,于是一个坏条目会让**整个候选集**里
        # 最好的那个也被判成未命中。
        return 0.0
    return dot / math.sqrt(na * nb)


class SemanticCache:
    """按问题向量的**最近邻**取答案。三层缓存里唯一会造成假阳性的一层。

    另外两层的键是精确匹配:键不一样就一定不命中,错了也只是没命中。
    这一层是「差不多就算命中」,于是它有一个别人没有的失败模式——
    **把另一个问题的答案发出去**,而且那份答案是完整的:措辞通顺、来源
    编号对得上、`grounded` 为真。所有表面信号都正常。

    **所以它默认关闭**(见 `Settings.SEMANTIC_CACHE_ENABLED`),打开需要
    接受一个实测前提:真实 embedding 下同义改写与「同实体不同属性」之间
    存在一个可用的阈值区间。危险是单侧的——假阳性给错答案,假阴性只是
    一次未命中——所以阈值取在实测零错配区间的上方。

    **它必须排在向量缓存之后启用。** 查一次要一次问题向量;没有向量缓存
    时,每次未命中都白搭一次网络编码,把最常见的路径拖慢。
    """

    def __init__(
        self,
        *,
        db: Database,
        version: CorpusVersion,
        settings: Settings,
        embedder: Embedder,
        enabled: bool | None = None,
    ) -> None:
        self._db = db
        self._version = version
        self._settings = settings
        self._embedder = embedder
        # 总开关 `CACHE_ENABLED` 也管这一层:排障时要能一次关掉全部,
        # 而不是记住这里还有一个独立的开关。
        self._enabled = (
            settings.CACHE_ENABLED and settings.SEMANTIC_CACHE_ENABLED
            if enabled is None
            else enabled
        )

    # -- 对外入口 ------------------------------------------------------------

    async def get(
        self,
        question: str,
        *,
        history_messages: int,
        kb_ids: list[int] | None,
        filters: Any = None,
    ) -> CachedAnswer | None:
        if not self._eligible(history_messages):
            return None

        vector = await self._embedder.embed_query(question)
        candidates = await self._candidates(
            await self._scope(kb_ids=kb_ids, filters=filters), dim=len(vector)
        )

        best: bytes | None = None
        # 初值必须是 `-inf` 而不是 `-1.0`:余弦的下确界就是 -1,而用 `>`
        # 比较时 `-1.0 > -1.0` 为假——恰好完全相反的两条向量会一条都取不到,
        # 表现成「明明有候选却永远未命中」。
        best_score = float("-inf")
        for stored_vec, value in candidates:
            score = _cosine(vector, stored_vec)
            if score > best_score:
                best, best_score = value, score

        if best is None or best_score < self._settings.SEMANTIC_CACHE_THRESHOLD:
            return None

        answer = _decode_answer(best)
        if answer is not None:
            logger.info(
                "语义缓存命中",
                extra=log_extra(question=question[:80], similarity=round(best_score, 4)),
            )
        return answer

    async def put(
        self,
        question: str,
        *,
        history_messages: int,
        kb_ids: list[int] | None,
        filters: Any = None,
        answer: CachedAnswer,
    ) -> None:
        """写入。**这一步会再编码一次问题**,而那次编码由向量缓存吸收。"""
        if not self._eligible(history_messages):
            return

        vector = await self._embedder.embed_query(question)
        expires = utcnow() + timedelta(seconds=self._settings.SEMANTIC_CACHE_TTL_SECONDS)

        async with self._db.session() as session:
            session.add(
                CacheSemanticEntry(
                    scope=await self._scope(kb_ids=kb_ids, filters=filters),
                    question_vec=pack_vector(vector),
                    # 存**实际长度**而不是 `embedder.dim`:后者是一个声明,
                    # 而这里要保证的是「读出来的向量长度」和它一致。
                    dim=len(vector),
                    model=self._embedder.model_id,
                    value=_encode_answer(answer),
                    expires_at=expires,
                )
            )
            await session.commit()

    async def purge_expired(self) -> int:
        """删掉已过期的条目,返回删除条数。和 `CacheStore.purge_expired` 同理。

        **过期条目不会被自动回收,只能靠这个**:知识库一写入,旧 scope 的
        条目就再也读不到了(scope 里带语料代次),但它们仍然占着空间。
        """
        async with self._db.session() as session:
            result = await session.execute(
                delete(CacheSemanticEntry).where(
                    CacheSemanticEntry.expires_at.is_not(None),
                    CacheSemanticEntry.expires_at <= naive_utc(utcnow()),
                )
            )
            await session.commit()
            return int(result.rowcount or 0)

    # -- 内部 ----------------------------------------------------------------

    def _eligible(self, history_messages: int) -> bool:
        """和答案缓存同一条规矩,**但在这里更要紧**。

        「它多少钱?」在两个会话里是不同的问题,而它们的向量**完全相同**
        ——比措辞不同还更容易跨会话命中。读和写两侧都要挡。
        """
        return self._enabled and history_messages == 0

    async def _scope(self, *, kb_ids: list[int] | None, filters: Any) -> str:
        """范围指纹:`kb_ids` + 过滤条件 + 语料代次 + 配置。

        用答案那份指纹(而不是检索那份):这一层存的是**答案**,所以影响
        答案的每一项(提示词、模型、温度)变了都得失效。
        """
        payload = {
            "schema": _SCHEMA_VERSION,
            "kb_ids": None if kb_ids is None else sorted(set(kb_ids)),
            "filters": _filters_key(filters),
            "generation": await self._version.current(),
            "config": _answer_fingerprint(self._settings),
        }
        return text_hash(json.dumps(payload, sort_keys=True, ensure_ascii=False))

    async def _candidates(self, scope: str, *, dim: int) -> list[tuple[list[float], bytes]]:
        """某个范围下的全部候选 `(向量, 答案字节)`。

        **排除条件都放在 SQL 里**,不是取出来再筛:模型与维度不符的条目
        算出来的余弦是两个向量空间的夹角——数值合法、含义为零,取出来再
        判断等于把它们和真候选混在一起比,而它们的分数完全可能更高。

        单条坏向量只跳过它自己:一个条目坏了不该让整个候选集不可用。
        """
        async with self._db.session() as session:
            rows = await session.execute(
                select(CacheSemanticEntry.question_vec, CacheSemanticEntry.value).where(
                    CacheSemanticEntry.scope == scope,
                    CacheSemanticEntry.model == self._embedder.model_id,
                    CacheSemanticEntry.dim == dim,
                    CacheSemanticEntry.expires_at.is_(None)
                    | (CacheSemanticEntry.expires_at > naive_utc(utcnow())),
                )
            )
            out: list[tuple[list[float], bytes]] = []
            for blob, value in rows:
                try:
                    out.append((unpack_vector(blob, expected_dim=dim), bytes(value)))
                except ValueError as exc:
                    logger.warning("语义缓存条目里的向量无法解析,跳过:%s", exc)
            return out


__all__: list[str] = [
    "MAX_KEY_LENGTH",
    "NS_ANSWER",
    "NS_EMBEDDING",
    "NS_RETRIEVAL",
    "AnswerCache",
    "CacheStore",
    "CachedAnswer",
    "CachedEmbedder",
    "CachedRetriever",
    "CorpusVersion",
    "SemanticCache",
    "pack_vector",
    "text_hash",
    "unpack_vector",
]
