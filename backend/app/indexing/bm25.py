"""中文 BM25 关键词检索路。

**为什么不直接用 LangChain 的 `BM25Retriever`**

官方实现的默认分词是 `text.split()`。中文没有空格,一句话会被切成
**一个** token,于是任何查询都匹配不上——检索恒返回空,而且**不报错**,
看起来就像「知识库里确实没有相关内容」。要让它工作必须传
`preprocess_func`,既然横竖要写分词,不如连索引一起自己管:

- 不到 150 行,可单独测试(见 `tests/test_bm25.py`,含一条与
  `text.split()` 错误行为的**对照测试**);
- 能保留每一路的 rank 明细,供 `fusion.py` 融合与「关键词路到底有没有
  贡献」的调参分析;
- 不必引入 `langchain-community`(已冻结,官方 issue #674)。

**为什么向量路之外还要留这一路**

两者失效的场景不重叠。向量路对**型号、货号、专有名词**天然不敏感——
「RX9070」和「RX 9070」在嵌入空间里几乎重合,但用户要的是精确匹配;
反过来,用户问「打游戏用哪个」时,关键词路什么都搜不到,向量路才知道
「打游戏」和「游戏性能」是一回事。融合两路正是为了互补。

**已知限制:分词粒度决定了关键词路「只能整词匹配」。** 实测 jieba 把
「接口定义」切成**一个**词,于是查询「接口」召回不了含「接口定义」的分块——
这在中文里很常见(前两个字恰好是后一个长词的前缀)。要做子串匹配得额外
索引字符二元组,是**召回率与索引体积**的又一层取舍,当前规模下不值得:
漏掉的这类查询向量路通常能补上,这正是混合检索的价值。
记录在此,是为了它哪天真的成为瓶颈时,不必重新发现一遍。
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import jieba
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

# 停用词只收**纯功能词**,判断标准是「去掉它不改变句子的检索意图」。
# 「的/是/了/在」符合;「支持」「三年」不符合——它们是商品资料里真实的
# 信息点,删掉会让「质保几年」这类查询失去落点。宁可少收,不可多收。
DEFAULT_STOPWORDS: frozenset[str] = frozenset(
    {
        "的", "了", "是", "在", "和", "与", "及", "或", "有", "把", "被", "让",
        "给", "为", "对", "从", "到", "于", "之", "其", "就", "都", "也", "还",
        "很", "会", "能", "要", "我", "你", "他", "她", "它", "这", "那",
        "我们", "你们", "他们", "这个", "那个", "一个", "什么", "怎么", "如何",
        "多少", "哪些", "哪个", "吗", "呢", "吧", "啊", "请", "帮", "想",
        "需要", "可以",
    }
)


@dataclass(frozen=True, slots=True)
class BM25Document:
    """进入关键词索引的一篇分块。

    `chunk_id` 是 SQLite 里 `chunks.id`,也就是全链路通用的那个稳定 id——
    向量路、引用溯源、前端来源卡片都用同一个,不需要任何映射表。
    """

    chunk_id: int
    kb_id: int
    text: str


@dataclass(frozen=True, slots=True)
class BM25Hit:
    chunk_id: int
    score: float
    rank: int
    """1 起始的名次,直接喂给 RRF。见 `fusion.py` 对 1 起始的解释。"""


# 进程级的分词器缓存:词表 → 已预热好的 jieba Tokenizer。
#
# 用 dict 而不是 `functools.lru_cache`:键是一个可变长度的词表,而
# `lru_cache` 只接受可哈希参数——把词表塞进去要自己转 tuple,那不如
# 直接写清楚。容量**不需要上限**:词表来自配置,种类是常数量级
# (一个部署就是一份),给 LRU 只会平添一处可能失效的路径。
_TOKENIZERS: dict[tuple[str, ...], jieba.Tokenizer] = {}
_TOKENIZERS_LOCK = threading.Lock()


def _shared_tokenizer(user_words: Iterable[str] | None) -> jieba.Tokenizer:
    """取(必要时新建并预热)一个分词器。

    **不是模块级的 `jieba.lcut`。** 那个的词典是进程内全局状态:一个知识库
    `add_word('星曜显卡')` 会泄漏到所有其他知识库,测试之间也会互相污染
    (先跑的用例决定后跑的结果)。

    **按词表分组**而不是做成单一单例:同样是为了不串味——两个知识库各有
    自定义词表时,后建的那个必须拿到自己那份。

    **锁是必需的,不是防御性编程。** 关键词路跑在线程池里
    (`anyio.to_thread.run_sync`),两个并发请求会真的同时进这里。
    """
    key = tuple(sorted({w.strip() for w in (user_words or ()) if w and w.strip()}))
    with _TOKENIZERS_LOCK:
        tokenizer = _TOKENIZERS.get(key)
        if tokenizer is None:
            tokenizer = jieba.Tokenizer()
            for word in key:
                tokenizer.add_word(word)
            _TOKENIZERS[key] = tokenizer
        return tokenizer


def clear_tokenizer_cache() -> None:
    """清空分词器缓存。**只给测试用**——生产里没有「换一份词表」的场景
    (词表来自配置,改了配置就重启进程),所以不提供失效接口,只提供清理。"""
    with _TOKENIZERS_LOCK:
        _TOKENIZERS.clear()


class JiebaBM25:
    """jieba 分词 + `rank_bm25.BM25Okapi` 的内存关键词索引。

    **索引整份重建,不做增量。** 增量维护 BM25 的 IDF 需要重算全局文档
    频次——复杂度换不来可感知的收益,还会引入「删了文档索引里还有」这类
    难查的漂移。检索服务每次重建时整体替换,见 `index()`。

    **但分词器本身是要复用的,而且要按词表复用。** 曾经的注释写着「重建
    一次是毫秒级」——那是个没测过的假设。实测(见 `docs/BENCHMARK.md`):
    新建实例后的第一次分词要 250ms 上下,同一实例的第二次只要 20µs 左右,
    也就是说那 250ms **全是 jieba 加载主词典**,与语料大小无关;而检索
    服务每次查询都新建一个实例,于是每个问题都为同一本从不变化的词典
    重付一次冷启动。

    **关键不在这一项有多贵,而在它是固定成本。** 实测语料从 200 涨到 3000
    分块,建索引那一趟涨了十余倍,而这一项纹丝不动。所以缓存的粒度取在
    分词器上:按索引缓存会在每次语料写入后失效、把固定成本重新付一遍;
    按分词器缓存则跨越语料变更持续有效,同时保住了「索引每次按最新语料
    重建」这个不会漂移的性质。
    """

    def __init__(
        self,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        stopwords: Iterable[str] | None = None,
        user_words: Iterable[str] | None = None,
    ) -> None:
        if k1 <= 0:
            raise ValueError(f"k1 必须为正数,收到 {k1}")
        if not 0.0 <= b <= 1.0:
            raise ValueError(f"b 必须落在 [0,1],收到 {b}")

        self._k1 = k1
        self._b = b
        # 自定义停用词是**追加**而非替换。替换的话,调用方为了加一个词
        # 会顺手把「的」「是」放回索引——一个不报错的检索质量下降。
        extra = {w.strip().lower() for w in (stopwords or ()) if w and w.strip()}
        self._stopwords = DEFAULT_STOPWORDS | extra

        self._tokenizer = _shared_tokenizer(user_words)

        self._documents: dict[int, BM25Document] = {}
        self._tokens: dict[int, list[str]] = {}
        self._bm25: BM25Okapi | None = None

    # -- 分词 ----------------------------------------------------------------

    def tokenize(self, text: str) -> list[str]:
        """切词 → 去停用词 → 去纯标点 → 统一小写。

        统一小写是**有意的**:查询侧和索引侧必须用同一套规则,否则
        「RX9070」和「rx9070」是两回事,而用户不会记得型号的大小写。
        """
        if not text or not text.strip():
            return []

        tokens: list[str] = []
        for raw in self._tokenizer.lcut(text):
            token = raw.strip().lower()
            if not token or token in self._stopwords:
                continue
            # 纯标点不进词表。'。' ',' 这类符号在 BM25 里没有任何区分度,
            # 却会各占一个 IDF 位,稀释真正有信息量的词。
            # `str.isalnum()` 对汉字返回 True,所以这一条同时放行中文。
            if not any(ch.isalnum() for ch in token):
                continue
            tokens.append(token)
        return tokens

    # -- 索引 ----------------------------------------------------------------

    def index(self, documents: Sequence[BM25Document]) -> None:
        """整体替换索引。传空序列即清空。"""
        # 同 chunk_id 者**后者覆盖前者**。上游重建索引时若不小心把同一
        # 分块喂了两遍,重复项会在候选列表里占两个位置,而 RRF 会把它
        # 当成「两路都召回了」,凭空抬高它的融合分数。
        unique: dict[int, BM25Document] = {}
        for doc in documents:
            unique[doc.chunk_id] = doc

        self._documents = unique
        self._tokens = {cid: self.tokenize(d.text) for cid, d in unique.items()}

        # 语料顺序取自 self._documents,与 search() 里回填 chunk_id 的
        # 遍历顺序**必须一致**——那里靠下标对齐,错位不会报错,只会让
        # 分数张冠李戴。空 token 的文档保留占位,它的位置不能少。
        corpus = [self._tokens[cid] for cid in unique]
        # 一篇都没切出词时不建索引:BM25Okapi 的 avgdl 会除零。
        self._bm25 = BM25Okapi(corpus, k1=self._k1, b=self._b) if any(corpus) else None

    # -- 检索 ----------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        kb_ids: Iterable[int] | None = None,
    ) -> list[BM25Hit]:
        """返回按 BM25 分数降序的命中,`rank` 从 1 开始。

        `kb_ids` 为 `None` 表示**不限制范围**(管理员预览检索效果时用),
        为空集合表示**没有任何范围**——两者含义相反,见下方注释。
        """
        if self._bm25 is None or not self._documents:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # **`None` 是「不限制」,`[]` 是「一个都没有」。**
        # 若把空集合按 falsy 跳过过滤,一个没有任何知识库权限的用户
        # 会检索到全部内容——一个不报错的越权读。所以这里判的是 `is None`。
        allowed = None if kb_ids is None else set(kb_ids)
        if allowed is not None and not allowed:
            return []

        wanted = set(query_tokens)
        scores = self._bm25.get_scores(query_tokens)

        candidates: list[tuple[float, int]] = []
        for position, chunk_id in enumerate(self._documents):
            doc = self._documents[chunk_id]
            if allowed is not None and doc.kb_id not in allowed:
                continue
            # **筛的是「是否真的共现了查询词」,不是 `score > 0`。**
            # rank_bm25 在查询词出现于**全部**文档时 idf 为负(N=1 时必然
            # 如此),而它对负 idf 的「修正」是 `epsilon * average_idf`——
            # 平均值同样是负的,修正完照旧是负数(实测 -0.2747)。
            # 按分数正负过滤的话,只喂了一篇文档的知识库会**任何关键词都
            # 搜不到**,而且不报错。共现判断没有这个问题:分数只决定
            # 次序,不决定是否入选。
            if not (wanted & set(self._tokens[chunk_id])):
                continue
            candidates.append((float(scores[position]), chunk_id))

        # 次键 chunk_id:同分时的顺序不能依赖字典插入顺序,否则同一个
        # query 两次调用可能给出不同的引用顺序,语义缓存会因此产出与
        # 首次不同的来源列表。
        candidates.sort(key=lambda pair: (-pair[0], pair[1]))
        return [
            BM25Hit(chunk_id=chunk_id, score=score, rank=rank)
            for rank, (score, chunk_id) in enumerate(candidates[:top_k], start=1)
        ]

    # -- 状态 ----------------------------------------------------------------

    @property
    def size(self) -> int:
        """已索引的分块数(去重后)。"""
        return len(self._documents)
