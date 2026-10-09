"""中文 BM25 测试。

**这个模块存在的唯一理由是「中文不能不分词」。** 官方 `BM25Retriever`
默认 `text.split()`,而中文没有空格——整句话会变成**一个** token,
于是任何查询都匹配不上,检索结果恒为空。

最坏的地方在于它**不报错**:BM25 照常建索引、照常打分、照常返回空列表,
看起来就像「知识库里没有相关内容」。所以这里的第一个测试是拿
`text.split()` 的错误行为做对照,而不是只测自己。
"""

from __future__ import annotations

from app.indexing.bm25 import BM25Document, JiebaBM25
from rank_bm25 import BM25Okapi

# 语料**必须是 5 篇,不能是 4 篇**。BM25 的 idf = log((N-n+0.5)/(n+0.5)),
# N=4 而某个词恰好出现在 2 篇里时,它等于 log(2.5/2.5) = **0** ——
# 所有文档同分。此时排序断言会靠 chunk_id 的兜底次序「碰巧」通过,
# 看上去是绿的,实际上什么也没验证(把 idf 算错成别的值也照样绿)。
# 5 篇时 n=2 → log(3.5/2.5) = 0.3365 > 0,分数才真正反映匹配程度。
CORPUS = [
    BM25Document(chunk_id=1, kb_id=100, text="RTX 显卡支持三年质保,人为损坏除外。"),
    BM25Document(chunk_id=2, kb_id=100, text="主板支持三年质保,以换代修。"),
    BM25Document(chunk_id=3, kb_id=100, text="显示器参数说明与接口定义。"),
    BM25Document(chunk_id=4, kb_id=200, text="另一知识库里的显卡报价单。"),
    BM25Document(chunk_id=5, kb_id=100, text="机械键盘轴体说明与灯效设置。"),
]


def build(docs=CORPUS, **kwargs) -> JiebaBM25:
    index = JiebaBM25(**kwargs)
    index.index(docs)
    return index


def ids(hits) -> list[int]:
    return [h.chunk_id for h in hits]


class TestWhyTokenizationMatters:
    def test_naive_split_fails_where_jieba_succeeds(self) -> None:
        """**对照测试:证明 `text.split()` 在这个语料上真的检索不到东西。**

        不写这一条的话,「我们做了中文分词」只是句口号——没人知道不做会怎样。
        """
        texts = [d.text for d in CORPUS]
        naive = BM25Okapi([t.split() for t in texts])
        naive_scores = naive.get_scores("质保".split())
        assert max(naive_scores) <= 1e-6, (
            "按空格切词时整句是一个 token,所以「质保」匹配不到任何文档——"
            "这正是必须分词的原因"
        )

        assert ids(build().search("质保")) != []

    def test_a_chinese_word_matches_documents_containing_it(self) -> None:
        found = ids(build().search("质保"))
        assert set(found) == {1, 2}
        assert 3 not in found, "不含「质保」的文档不该被召回"

    def test_tokens_are_words_not_whole_sentences(self) -> None:
        index = JiebaBM25()
        tokens = index.tokenize("显卡支持三年质保")
        assert tokens == ["显卡", "支持", "三年", "质保"]
        assert len(tokens) > 1


class TestQueryBehaviour:
    def test_more_matching_terms_ranks_higher(self) -> None:
        """同时命中「显卡」和「质保」的文档,应当排在只命中一个的前面。"""
        hits = build().search("显卡 质保")
        assert hits[0].chunk_id == 1
        assert hits[0].score > 0

    def test_unmatched_query_returns_nothing(self) -> None:
        """**不能「返回点什么」**:BM25 对完全不含查询词的文档给 0 分,
        这类候选必须被丢掉,否则关键词路会对任何 query 都吐出噪音,
        再经 RRF 融合后污染最终结果。
        """
        assert build().search("洗衣机 烘干") == []

    def test_empty_query_returns_nothing(self) -> None:
        assert build().search("") == []
        assert build().search("   ") == []

    def test_single_document_index_still_matches(self) -> None:
        """**只有一个分块的知识库也必须能命中关键词。**

        `rank_bm25` 在「查询词出现在**全部**文档里」时 IDF 为负(N=1 时必然
        如此),而它对负 IDF 的「修正」是 `eps * average_idf`——平均值同样是
        负的,修正完还是负数(实测 -0.2747)。若按 `score > 0` 过滤,
        一个只喂了一篇文档的知识库会**任何关键词都搜不到**,而且不报错。

        所以筛选依据是「是否真的共现了词」,而不是分数的正负。
        """
        index = JiebaBM25()
        index.index([BM25Document(chunk_id=1, kb_id=1, text="主板支持三年质保")])
        hits = index.search("质保")
        assert ids(hits) == [1]
        assert hits[0].rank == 1

    def test_matching_is_case_insensitive(self) -> None:
        """用户不会记得型号的大小写写对没有。"""
        index = JiebaBM25()
        index.index([BM25Document(chunk_id=1, kb_id=1, text="RX9070 显卡报价单")])
        assert ids(index.search("rx9070")) == [1]
        assert ids(index.search("RX9070")) == [1]
        assert ids(index.search("Rx9070")) == [1]

    def test_query_of_only_stopwords_returns_nothing(self) -> None:
        """全是停用词时切完为空,不能退化成「匹配所有文档」。"""
        assert build().search("的 是 了") == []

    def test_top_k_limits_results(self) -> None:
        index = build()
        assert len(index.search("质保 显卡 主板 显示器", top_k=2)) <= 2

    def test_ranks_are_one_based_and_sequential(self) -> None:
        hits = build().search("质保 显卡 主板 显示器", top_k=10)
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))

    def test_scores_are_descending(self) -> None:
        hits = build().search("质保 显卡 主板 显示器", top_k=10)
        scores = [h.score for h in hits]
        assert scores == sorted(scores, reverse=True)


class TestKnowledgeBaseScope:
    def test_kb_ids_restricts_results(self) -> None:
        """多知识库隔离:关键词路必须和向量路受同样的范围约束。

        向量路有 Chroma 的 `where` 预过滤,BM25 这一路如果没有对应的过滤,
        普通用户就能通过关键词召回读到没授权知识库的内容。
        """
        index = build()
        assert set(ids(index.search("显卡", kb_ids={100}))) <= {1, 3}
        assert ids(index.search("显卡", kb_ids={200})) == [4]

    def test_empty_kb_scope_returns_nothing_not_everything(self) -> None:
        """**空集合表示「没有任何范围」,不是「不限制」。**

        把空集当成 falsy 而跳过过滤,会让一个没有任何知识库权限的用户
        检索到全部内容——一个静默的越权读。所以这里必须返回空。
        """
        assert build().search("显卡", kb_ids=set()) == []
        assert build().search("质保", kb_ids=[]) == []

    def test_none_scope_means_unrestricted(self) -> None:
        """显式 `None` 才是「不限制」——管理员预览检索效果时用。"""
        assert set(ids(build().search("显卡", kb_ids=None))) == {1, 4}


class TestIndexLifecycle:
    def test_empty_index_returns_nothing(self) -> None:
        assert JiebaBM25().search("显卡") == []
        assert JiebaBM25().size == 0

    def test_index_replaces_previous_documents(self) -> None:
        """重建索引必须整体替换。

        累加的话,文档删除后它的分块仍留在索引里,检索会命中一个
        数据库里已经不存在的 chunk_id——引用卡片点开是空的。
        """
        index = build()
        assert index.size == len(CORPUS)

        index.index([CORPUS[2]])
        assert index.size == 1
        assert ids(index.search("质保")) == []
        assert ids(index.search("显示器")) == [3]

    def test_empty_reindex_clears_everything(self) -> None:
        index = build()
        index.index([])
        assert index.size == 0
        assert index.search("质保") == []


class TestCustomDictionary:
    def test_user_words_keep_product_names_whole(self) -> None:
        """商品领域词典把「星曜显卡」当成一个词,而不是拆成「星曜」+「显卡」。

        拆开后「星曜」这个品牌词会变得毫无分量——它只出现在少数文档里,
        IDF 很高,却和「星曜显卡」这个查询词对不上,品牌检索就废了。
        """
        without = JiebaBM25()
        assert "星曜显卡" not in without.tokenize("星曜显卡限时特惠")

        with_dict = JiebaBM25(user_words=["星曜显卡"])
        assert "星曜显卡" in with_dict.tokenize("星曜显卡限时特惠")

    def test_user_dictionary_is_per_instance(self) -> None:
        """jieba 的全局词典是模块级状态,必须用独立 Tokenizer 隔离。

        不隔离的话,一个知识库加的词会泄漏到另一个知识库,
        而且测试之间会互相影响(先跑的测试决定后跑的结果)。
        """
        import jieba

        JiebaBM25(user_words=["星曜显卡"])
        assert "星曜显卡" not in jieba.lcut("星曜显卡限时特惠")

    def test_the_dictionary_is_loaded_once_per_word_list(self, monkeypatch) -> None:
        """**同一份自定义词表只应该加载一次 jieba 词典。**

        这条用例来自一次实测(数据见 `docs/BENCHMARK.md`):新建 `JiebaBM25`
        之后的**第一次**分词要 250ms 上下,而同一实例的第二次只要 20µs 左右——
        那 250ms 全部是 jieba 把主词典读进内存,和语料多大毫无关系。
        而检索服务此前**每次查询都新建一个 `JiebaBM25`**,于是每个问题
        都在为同一本从不变化的词典重付一次冷启动。

        这里数的是 `jieba.Tokenizer` 的构造次数,而不是测「跑得快不快」:
        时间断言在 CI 上会因为机器负载飘,而「构造了几次」是确定性的,
        它才是这条优化的契约本身。用了真实的 `jieba.Tokenizer`,只是
        在它外面套了一层计数——不 mock 掉行为,否则测的就不是真的了。
        """
        import app.indexing.bm25 as bm25_module
        import jieba
        from app.indexing.bm25 import clear_tokenizer_cache

        clear_tokenizer_cache()
        created: list[str] = []
        real = jieba.Tokenizer

        def counting(*args, **kwargs):
            created.append("tokenizer")
            return real(*args, **kwargs)

        monkeypatch.setattr(bm25_module.jieba, "Tokenizer", counting)

        JiebaBM25(user_words=["星曜显卡"])
        JiebaBM25(user_words=["星曜显卡"])

        assert len(created) == 1, "同样的词表把 jieba 词典又加载了一遍"

    def test_different_word_lists_get_different_tokenizers(self) -> None:
        """缓存必须按词表分组。

        按「有没有 user_words」粗分是不够的:两个知识库各有一份词表时,
        后建的那个会拿到前一个的分词器,于是**行业词串味**——A 库的
        「星曜显卡」在 B 库也被当成一个词,而 B 库的语料里根本没这个词。
        """
        from app.indexing.bm25 import clear_tokenizer_cache

        clear_tokenizer_cache()
        a = JiebaBM25(user_words=["星曜显卡"])
        b = JiebaBM25(user_words=["玄武电源"])

        assert "星曜显卡" in a.tokenize("星曜显卡限时特惠")
        assert "星曜显卡" not in b.tokenize("星曜显卡限时特惠")

    def test_a_reused_tokenizer_still_carries_its_own_dictionary(self) -> None:
        """复用不能变成「拿到别人的词典」。

        这条挡的是一个很容易写出来的错:缓存键只取了词表的长度、或者干脆
        只取「有没有传 user_words」。那时 `JiebaBM25(["星曜显卡"])` 和
        `JiebaBM25([])` 会共用同一个分词器,而它们本该切出不同的词。
        """
        from app.indexing.bm25 import clear_tokenizer_cache

        clear_tokenizer_cache()
        JiebaBM25(user_words=["星曜显卡"])  # 先把缓存写热

        warmed = JiebaBM25(user_words=["星曜显卡"])
        assert "星曜显卡" in warmed.tokenize("星曜显卡限时特惠")

        plain = JiebaBM25()
        assert "星曜显卡" not in plain.tokenize("星曜显卡限时特惠")

    def test_english_model_codes_survive(self) -> None:
        """型号是纯字母数字混排,jieba 本来就切得对——防止将来「优化」坏掉。

        断言小写形式是**有意的**:tokenize 统一转小写,查询侧和索引侧
        才会一致(见 `test_matching_is_case_insensitive`)。型号一旦被
        切碎成 `rx` + `9070`,「RX9070」和「RX 9070」就变成了两回事。
        """
        assert "rx9070" in JiebaBM25().tokenize("RX9070显卡报价")


class TestStopwords:
    def test_stopwords_are_removed_from_tokens(self) -> None:
        index = JiebaBM25()
        tokens = index.tokenize("这个商品的价格是多少")
        assert "的" not in tokens
        assert "是" not in tokens
        assert "价格" in tokens

    def test_default_stopwords_include_common_particles(self) -> None:
        index = JiebaBM25()
        assert index.tokenize("了 的 是 在") == []

    def test_custom_stopwords_are_merged_with_defaults(self) -> None:
        """自定义停用词是**追加**,不是替换。

        替换的话,调用方为了加一个词会不小心把「的」「是」这些
        重新放回索引里——一个不报错的检索质量下降。
        """
        index = JiebaBM25(stopwords=["显卡"])
        assert "显卡" not in index.tokenize("显卡质保")
        assert "的" not in index.tokenize("这个的价格")


class TestDuplicateChunkIds:
    def test_duplicate_chunk_id_keeps_the_last_occurrence(self) -> None:
        """同 id 重复出现时只保留一条,且**不产生两条召回结果**。

        重建索引时上游若不小心把同一分块喂了两遍,重复项会在候选里占两个
        位置——RRF 那边会把它当成「两路都召回了」,凭空抬高它的排名。
        """
        index = JiebaBM25()
        index.index(
            [
                BM25Document(chunk_id=1, kb_id=100, text="旧的内容 显卡"),
                BM25Document(chunk_id=1, kb_id=100, text="新的内容 主板"),
            ]
        )
        assert index.size == 1
        assert len(index.search("主板")) == 1
        assert index.search("显卡") == []
