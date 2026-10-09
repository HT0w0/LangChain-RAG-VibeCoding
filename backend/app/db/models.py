"""ORM 模型。

**时间戳约定**:全库统一存 UTC。SQLite 的 DATETIME 类型在存储时会**静默丢弃
时区信息**,aware datetime 写进去、读出来是 naive 的。所以这里统一:
写入用 `utcnow()`,读出的一律**当作 UTC 处理**,由 Pydantic 层负责补上 `+00:00`
再返回给前端。不这么做的话,服务器换个时区跑,历史时间就会整体偏移。
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


def utcnow() -> datetime:
    """当前 UTC 时间。统一入口,避免各处 `datetime.now()` 混用本地时区。"""
    return datetime.now(UTC)


def naive_utc(value: datetime) -> datetime:
    """给**数据库端**比较用的时间值。

    存储层用的是 naive UTC 字符串,拿 aware 的值去比会得到一种介于
    「能跑」和「比错」之间的结果(字符串前缀相同,后缀 `+00:00` 参与比较),
    所以比较的右侧也必须先去掉时区。

    放在这里而不是各个调用模块里:它是**存储层的约定**(见模块文档),
    凡是拿 `utcnow()` 去和某一列比大小的地方都要用同一份实现。各写一份
    的话,漏掉一次就是某条查询的结果整体偏移——而它在 UTC 机器上看
    完全正常。
    """
    return value.astimezone(UTC).replace(tzinfo=None) if value.tzinfo else value


class UserRole(StrEnum):
    ADMIN = "admin"
    USER = "user"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(16), default=UserRole.USER, nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (Index("ix_users_role_active", "role", "is_active"),)

    @property
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN

    def __repr__(self) -> str:
        return f"<User id={self.id} username={self.username!r} role={self.role}>"


# ---------------------------------------------------------------------------
# 知识库
# ---------------------------------------------------------------------------


class DocumentStatus(StrEnum):
    """文档摄取状态机。

    `FAILED` 是**终态但可重试**:解析失败的文档不该让整批上传回滚,
    用户应该能看到「哪一篇挂了、为什么」,然后单独重试。
    """

    PENDING = "pending"        # 已落盘,排队等解析
    PARSING = "parsing"        # 正在解析/分块/向量化
    READY = "ready"            # 可被检索
    FAILED = "failed"          # 失败,error 字段有原因


class KnowledgeBase(Base):
    """知识库。检索时的最小隔离单位——每个 chunk 都带 kb_id。"""

    __tablename__ = "knowledge_bases"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    # doc_count / chunk_count 是**冗余计数**。
    #
    # 直接 COUNT(*) 也能算,但知识库列表页要同时列出 N 个库的文档数,那就是
    # N 次子查询;而这两个数字在列表页每次刷新都要显示。代价是它们可能与
    # 真实值漂移,所以:① 只在摄取/删除的同一个事务里改,② 提供 recount()
    # 作为对账入口。不做这个取舍的话,列表页会随文档数增长而变慢。
    doc_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    documents: Mapped[list[Document]] = relationship(
        back_populates="knowledge_base",
        cascade="all, delete-orphan",
        # passive_deletes:删知识库时让 SQLite 自己级联删文档与分块。
        # 不加的话 ORM 会把每一行都加载进内存再逐条 DELETE——一个几百篇文档、
        # 上万分块的库,删一次要发上万条 SQL。
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<KnowledgeBase id={self.id} name={self.name!r} docs={self.doc_count}>"


class Document(Base):
    """一篇上传的原始文档。"""

    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True)
    kb_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # filename 保留原始文件名**仅用于展示和引用溯源**。
    # 落盘路径一律用 uuid(stored_path),绝不用原始名拼路径——否则
    # "../../../etc/passwd" 这类文件名可以直接写到项目外。
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_path: Mapped[str] = mapped_column(String(512), nullable=False)
    # 内容 sha256。用于重复上传检测:同一份商品手册传两次不该产生两份分块,
    # 否则检索结果里同一条资料会出现两遍,把 top-k 名额挤掉。
    file_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    mime: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)
    status: Mapped[str] = mapped_column(
        String(16), default=DocumentStatus.PENDING, nullable=False
    )
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    # 解析过程中的**非致命**提示,例如「第 3 页是图片页,内容不会被检索到」。
    # 单独一列而不是塞进 error:error 意味着失败,而 warnings 的文档是好用的,
    # 只是不完整。混在一起会让前端无法区分「要重传」和「知道一下就行」。
    warnings: Mapped[list[str]] = mapped_column(JSON, default=list, nullable=False)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    uploaded_by: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    chunks: Mapped[list[Chunk]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (
        # 同一个库内同一份内容只允许一份。跨库允许重复——两个库各自引用
        # 同一份商品手册是合理的业务场景。
        UniqueConstraint("kb_id", "file_hash", name="uq_documents_kb_hash"),
        Index("ix_documents_kb_status", "kb_id", "status"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id} filename={self.filename!r} status={self.status}>"


class Chunk(Base):
    """检索的最小单元。

    **正文在这里冗余存一份**,而不是只存 vector_id 去 Chroma 反查。三个理由:
    ① BM25 需要对全量语料重建索引,逐条查向量库代价高;
    ② 引用溯源要按 chunk_id 直接取原文渲染来源卡片,走 SQLite 比走 Chroma 快;
    ③ Chroma 的 metadata 不适合放长文本。代价是同一份正文存两处,
    写入时必须保证 SQLite 与 Chroma 一致(见 IngestService 的写入顺序)。
    """

    __tablename__ = "chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    doc_id: Mapped[int] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # kb_id 冗余自 documents——检索时按 kb_id 过滤是最频繁的操作,
    # 有它就不必每次 JOIN documents。
    kb_id: Mapped[int] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), nullable=False
    )
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # 商品元数据:product_name / category / price / source_heading ...
    # 检索时用于 Chroma 服务端预过滤(类目、价格区间),也是引用卡片的展示字段。
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict, nullable=False)
    # Chroma 里的主键。用 "c{chunk_id}" 形式,保证两个存储间可双向定位。
    vector_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    document: Mapped[Document] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("doc_id", "chunk_index", name="uq_chunks_doc_index"),
        Index("ix_chunks_kb_doc", "kb_id", "doc_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id} doc={self.doc_id} idx={self.chunk_index}>"


# ---------------------------------------------------------------------------
# 会话与消息
# ---------------------------------------------------------------------------


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


class ChatSession(Base):
    """一次持续的多轮对话。

    **类名是 `ChatSession` 而不是 `Session`**,因为后者是 SQLAlchemy 的核心概念
    (`sessionmaker` / `AsyncSession`),同名会让「这个 session 指的是哪个」
    在阅读时反复卡壳。表名保持 `sessions`,因为那是业务语义。
    """

    __tablename__ = "sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    # 归属。**所有读写都必须带上 user_id**,单独按 id 查等于越权。
    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    # 标题由首个问题自动截取,用户可改。不做「用 LLM 总结标题」——
    # 为了列表页好看而给每次新建会话加一次模型调用,不划算。
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    # 本会话默认检索范围。存下来是为了「继续上次的对话」时不必让用户
    # 重新勾一遍知识库;每次提问仍可覆盖。空列表与 NULL 语义不同:
    # NULL = 不限(全部库),[] = 什么都不看。这个区分是越权读的防线
    # (见 `retrieval._kb_filter`),序列化时必须原样保留。
    kb_ids: Mapped[list[int] | None] = mapped_column(JSON, nullable=True)
    # 冗余计数,理由同 KnowledgeBase.doc_count:会话列表页要显示每个会话
    # 有几条消息,逐个 COUNT(*) 就是 N 次子查询。只在追加消息的同一个
    # 事务里自增,不提供对账入口——真实值就是 COUNT(*),漂移了直接重算。
    message_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )
    # 与 updated_at 分开:**列表按这个排序**,而重命名不该把会话顶到最前面。
    # 用 updated_at 排序的话,改个标题就会打乱「最近聊过」的顺序。
    last_message_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    messages: Mapped[list[Message]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="Message.id",
    )

    __table_args__ = (
        # 与列表查询的 ORDER BY 对齐(user_id 等值 + last_message_at 排序),
        # 让「我的会话」分页排序走索引而不必全表掃描 + 临时排序。
        Index("ix_sessions_user_last", "user_id", "last_message_at"),
    )

    def __repr__(self) -> str:
        return f"<ChatSession id={self.id} user={self.user_id} title={self.title!r}>"


class Message(Base):
    """一条消息。用户问题与助手回答共用一张表,用 `role` 区分。"""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("sessions.id", ondelete="CASCADE"), nullable=False, index=True
    )
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    # **存完整的来源快照,不只存编号。**
    #
    # 编号单独存是够渲染「[1]」的,但重看历史会话时来源卡片就没了——卡片要的是
    # 文件名、知识库名、正文原文。而按 chunk_id 现查也是错的:文档可能已被删除
    # 或重新索引,那时用户看到的是「当时那句话引用了一份现在已被替换的资料」,
    # 归因就悄悄变了。历史答案应当展示**当时的证据**。
    # 代价是同一段正文在 messages 里又存了一份,可接受(SQLite 单机场景)。
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list, nullable=False)
    # 正文里真正引用到的编号。与 sources 冗余,但它回答的是另一个问题
    # 「这条回答有没有落到具体资料上」,列表页和评测要用,不值得每次去
    # 扫 sources 里的 index。
    cited: Mapped[list[int]] = mapped_column(JSON, default=list, nullable=False)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    # 生成中途失败时留下半截回答 + 原因。**半截内容照样落库**:
    # 用户已经在屏幕上看到它了,库里却没有,下次翻开历史会觉得「我记错了」。
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    session: Mapped[ChatSession] = relationship(back_populates="messages")

    __table_args__ = (
        # 取历史消息恒为「某会话内按时间正序」,索引与之对齐。
        Index("ix_messages_session_created", "session_id", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} session={self.session_id} role={self.role}>"


class CacheEntry(Base):
    """通用 KV 缓存,按 `namespace` 分区。

    **一张表而不是三张**(plan 里曾写作 cache_embedding / cache_retrieval /
    cache_semantic)。前两者的结构完全相同——键 → 一段字节 + 一个过期时间——
    差别只在「存的是什么」,而那是 namespace 的语义。拆成三张表的代价是
    三套仓储代码、三套测试,换来的是「知道它在哪张表」这一点点方便。

    `cache_semantic` 仍然是独立的一张(见 `CacheSemanticEntry`),因为它的
    访问方式本质不同:**不是按键取,而是按向量最近邻取**——需要把候选向量
    都读出来算余弦,而这里根本没有「键」可用。

    **独立成表省掉的不是扫描,而是扫错东西。** 那张表照样要遍历它自己的全部
    候选(最近邻取本来就是这个代价);区别在于候选集只包含问题向量,而不是
    把检索结果、答案正文这些**没有向量语义的字节串**一起解出来试图算余弦。
    混在一起的表现会很隐蔽:算出来一堆垃圾相似度,偶尔超过阈值就命中。

    **主键是 (namespace, key) 而不是自增 id**:同一键重复写入必须是
    「替换」而不是「多一行」。用自增 id 的话,重复写入会攒下多行同键数据,
    读的时候取到哪一行取决于扫描顺序——一个「有时对有时错」的缓存。
    """

    __tablename__ = "cache_entries"

    namespace: Mapped[str] = mapped_column(String(32), primary_key=True)
    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    # `None` = 永不过期。向量缓存必须用 `None`:同一个文本同一个模型,
    # 向量永远一样,给它设 TTL 只会白白丢掉重算代价最高的一层。
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    __table_args__ = (
        # 清理过期条目时按这个字段扫。不建索引的话,每次清理都是全表扫描,
        # 而清理恰恰发生在表最大的时候。
        Index("ix_cache_entries_expires", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<CacheEntry {self.namespace}/{self.key}>"


class CacheSemanticEntry(Base):
    """语义缓存:按问题向量的最近邻取答案。

    **和 `CacheEntry` 分开的唯一原因是取法不同,不是在归一化。** 那张表是
    「键 → 字节」,这里的查询没有键可给——只能把某个范围内的候选全部取出、
    逐条算余弦、取最大。混进 `CacheEntry` 的话,每次查语义缓存都要把检索
    结果和答案正文也解成向量算一遍,而它们根本不是向量。

    **只存向量,不存问题正文。** 正文不参与匹配;存下来只会让后来的人以为
    它是键的一部分,进而「顺手」拿它做精确匹配——那样这一层就退化成了第二
    份答案缓存,而且是一份键更弱的。

    **`scope` 是唯一能进 SQL 的条件**,它把 `kb_ids`、过滤条件、语料代次、
    配置指纹压成一个哈希。这样知识库一写入,旧条目就因为 scope 不再匹配而
    当场失效(和另外两层同一个机制),而它们**连被算余弦的机会都没有**——
    过滤发生在 SQL 里,不是取出后再筛。代价是这些读不到的条目要等 TTL 才
    被回收,和另外两层一样。
    """

    __tablename__ = "cache_semantic"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: 见类文档。查询时**必须**带上它,否则会跨知识库命中。
    scope: Mapped[str] = mapped_column(String(128), nullable=False)

    #: 问题向量。维度与模型名随之存下来:换模型或换维度后,旧条目算出来的
    #: 余弦是两个不同向量空间的夹角——**数值合法、含义为零**。靠这两个字段
    #: 在 SQL 里排除,比算完再判断可靠。
    question_vec: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    model: Mapped[str] = mapped_column(String(128), nullable=False)

    #: `CachedAnswer` 的编码(答案正文 + 来源列表),与 `cache_entries` 里
    #: 答案层用的是同一套字节格式(见 `cache._encode_answer`)。
    value: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)

    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    __table_args__ = (
        # 查询恒为「某 scope 下、未过期的候选」。把 scope 放在前面,过期时间
        # 放在后面:scope 的选择性高得多,先按它把候选砍到一个很小的集合。
        Index("ix_cache_semantic_scope", "scope", "expires_at"),
    )

    def __repr__(self) -> str:
        return f"<CacheSemanticEntry id={self.id} scope={self.scope[:8]}…>"


class CorpusGeneration(Base):
    """语料代次:单行、单调递增。

    **为什么单开一张表,而不是复用 `cache_entries`。** 两者的更新方式不同:
    缓存是「按键覆盖」,代次是「自增」。自增**必须在 SQL 侧完成**
    (`SET value = value + 1`),否则并发写入会丢代次——而丢一次代次就意味着
    一次知识库写入没能让缓存失效,正是本模块最不能接受的那类故障(见
    `services/cache.py` 的 `CorpusVersion.bump`)。`cache_entries.value` 是
    BLOB,在它上面做整数自增要套两层 CAST,脆弱且读不懂。

    表里**恒只有一行**(`id = 1`)。用固定主键而不是「取最新一行」,是为了让
    「当前代次是多少」这个查询不需要 ORDER BY / LIMIT——它发生在每次检索
    的最热路径上。
    """

    __tablename__ = "corpus_generation"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    value: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=utcnow, onupdate=utcnow, nullable=False
    )

    def __repr__(self) -> str:
        return f"<CorpusGeneration value={self.value}>"


class RequestMetric(Base):
    """一轮问答的埋点。**一行一次请求,只追加、不更新。**

    **它不存问题正文。** 存了就等于造了一份影子聊天记录:它在会话的归属
    校验之外,任何能读指标的人都能读到所有人问过什么。而它要回答的问题
    ——长短问题的延迟分布——一个字符数就够(目前连字符数都没存,因为还没
    有用例需要它)。

    **失败必须记。** 一张只记成功请求的指标表比没有指标表更危险:超时、
    重试、上游报错全都不在里面,于是「p95 很健康」和「有一半请求根本没
    走到这里」可以同时成立,而看图的人只会看到前半句。所以 `outcome` 是
    一等字段,失败的那一轮同样带耗时。

    **`stage_timings` 是 JSON 而不是拆成列。** 阶段的集合会变(加了改写、
    关了重排、以后可能再加一步),拆列意味着每次改动都要迁移一次表——
    而本项目用的是 `create_all`,改列不会自动迁移(见 `main.lifespan`)。
    代价是这个字段没法在 SQL 里直接聚合,聚合放在 Python 侧做,而窗口内
    的行数很小。

    **和 `messages` 的分工**:那边是**用户要看的东西**(回答、来源、引用),
    这边是**运维要看的东西**(耗时、命中、失败)。同一次请求在两处各留一份
    记录是有意的——那边的行会随会话被删除而消失,而「这一小时 p95 涨了」
    不该因为有人删了自己的会话就变了。
    """

    __tablename__ = "request_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)

    #: 与日志里那个 trace_id 是同一个值。**这是排查时把「日志里的慢请求」
    #: 和「表里的行」对上的唯一手段**,所以它从请求头 `X-Trace-Id` 来。
    trace_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)

    #: `SET NULL` 而不是 `CASCADE`:删掉用户不该改写历史流量统计——那会让
    #: 过去某个时间点的 p95 在事后悄悄变化。
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True
    )
    session_id: Mapped[int | None] = mapped_column(
        ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )

    #: `done` | `error`。**只有两种,而且必须两种都记**(见类文档)。
    outcome: Mapped[str] = mapped_column(String(16), nullable=False)

    #: 「用户按下回车」到「最后一个字到达」的全程,含改写、检索、重排、
    #: 生成。失败的那一轮同样有值——拿不到它,失败在面板上就是瞬间返回。
    total_ms: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    #: 首 token 时延。**`None` 表示这一轮一个字都没流出来**,而不是 0——
    #: 把「没流出来」写成 0,面板上会显示成一个漂亮得离谱的数字。
    first_token_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    #: 各阶段耗时:`{rewrite, retrieve, rerank, assemble, first_token, generate}`。
    #: **不同请求的阶段集合可以不同**(改写被跳过、重排被关掉),所以聚合时
    #: 每个阶段的分母是「记过这个阶段的请求数」,不是总请求数。
    stage_timings: Mapped[dict[str, float]] = mapped_column(JSON, default=dict, nullable=False)

    #: 三层缓存各自的命中情况。分开存而不是一个 `cache_hit` 布尔值:
    #: 「检索命中了」和「整份答案命中了」的排查方向相反——前者要问「检索
    #: 结果对不对」,后者要问「这个问题凭什么能复用」。
    retrieval_cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    answer_cache_hit: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    #: 检索有没有拿到资料。它为假时走的是固定答复,**不花模型调用**——
    #: 不单独记的话,这类请求会把生成段的平均值拉低,看起来像「生成变快了」。
    grounded: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)

    #: 带进 prompt 的历史条数。用来判断「多轮是不是让延迟变长了」,而这
    #: 在总平均值里是被单轮请求稀释掉的。
    history_messages: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    #: 上面那批 prompt token 里,有多少是服务端 prompt 缓存供给的。
    #:
    #: **可空,`None` 表示平台没报这个读数**——和 `first_token_ms` 同一个理由:
    #: `0` 是「平台说这次没命中」,`None` 是「平台没说」。合并之后,
    #: 「缓存一直不工作」和「我们根本没在听」在面板上长得一模一样,
    #: 而这两件事的排查方向完全相反。聚合命中率时**分母也只算报过的那些**。
    cached_prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, nullable=False)

    __table_args__ = (
        # 面板的每一个查询都是「最近 N 分钟」,索引必须与之一致。没有它,
        # 每次打开面板都是一次全表扫描,而这表只增不减。
        Index("ix_request_metrics_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<RequestMetric id={self.id} outcome={self.outcome} total_ms={self.total_ms:.0f}>"
