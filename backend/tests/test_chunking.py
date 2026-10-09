"""分块器测试。

分块是 RAG 里**唯一一个错了也不会报错**的环节:切坏了照样入库、照样能检索,
只是答案开始变得不准。所以这些断言针对的都是「静默劣化」的场景。
"""

from __future__ import annotations

from app.indexing.chunking import (
    CHINESE_SEPARATORS,
    coerce_number,
    estimate_tokens,
    normalize_columns,
    normalize_row,
    row_to_sentence,
    rows_to_sentences,
    split_markdown,
    split_text,
)


class TestSeparators:
    def test_english_period_is_not_a_separator(self) -> None:
        """**这条是整个分块模块最重要的约束。**

        英文句点一旦进了分隔符表,`0.95`、`v1.2.3`、`e.g.` 全都会被切开,
        而且切完看起来「很正常」——只有检索「相似度 0.95」死活搜不到时
        才会有人去查,那时已经过去很久了。
        """
        assert "." not in CHINESE_SEPARATORS

    def test_decimal_numbers_survive_splitting(self) -> None:
        text = "相似度阈值取 0.95,温度设为 0.7,学习率 0.001。" * 5
        pieces = split_text(text, chunk_size=30, chunk_overlap=5)
        joined = "".join(p.content for p in pieces)
        assert "0.95" in joined
        assert "0.001" in joined
        # 更严格:不能出现被腰斩的「0.9」+「5」
        assert not any(p.content.endswith("0.9") for p in pieces)

    def test_version_numbers_survive(self) -> None:
        text = "当前版本 v1.2.3,下一版 v2.0.0 计划支持。其余说明见附录内容若干。" * 5
        pieces = split_text(text, chunk_size=30, chunk_overlap=5)
        joined = "".join(p.content for p in pieces)
        assert "v1.2.3" in joined
        assert "v2.0.0" in joined


class TestSplitText:
    def test_respects_token_budget(self) -> None:
        text = "这是一句用于测试分块的中文句子,长度适中,适合用来构造语料。" * 40
        pieces = split_text(text, chunk_size=100, chunk_overlap=20)
        assert len(pieces) > 1, "长文本应该被切成多块"
        for piece in pieces:
            # 允许一定超出:切分器按「分隔符边界」切,不会为了凑整而
            # 把一句话劈开。超出比例是分块质量的可观测指标。
            assert estimate_tokens(piece.content) <= 160, piece.content[:40]

    def test_consecutive_chunks_overlap(self) -> None:
        """overlap 的作用是防止答案正好落在切口上。

        overlap=0 时,「7 天无理由退货」如果被从中间切开,两边的块都答不了
        这个问题。这里断言相邻块确实共享内容,而不是各切各的。
        """
        sentences = [f"第{i}条:关于退换货的具体说明文字,需要足够长以便分块。" for i in range(12)]
        pieces = split_text("".join(sentences), chunk_size=60, chunk_overlap=20)
        assert len(pieces) >= 3

        shared = False
        for a, b in zip(pieces, pieces[1:], strict=False):
            tail = a.content[-12:]
            if tail and tail in b.content:
                shared = True
                break
        assert shared, "相邻分块之间应该有重叠内容"

    def test_indexes_are_sequential_from_zero(self) -> None:
        pieces = split_text("甲。乙。丙。丁。戊。己。庚。辛。" * 10, chunk_size=30, chunk_overlap=5)
        assert [p.index for p in pieces] == list(range(len(pieces)))

    def test_whitespace_only_yields_nothing(self) -> None:
        assert split_text("   \n\n   \t  ", chunk_size=100, chunk_overlap=10) == []
        assert split_text("", chunk_size=100, chunk_overlap=10) == []

    def test_metadata_is_propagated_to_every_piece(self) -> None:
        pieces = split_text("内容。 " * 60, chunk_size=30, chunk_overlap=5,
                            meta={"kb_id": 7, "source": "x.txt"})
        assert pieces
        assert all(p.meta["kb_id"] == 7 for p in pieces)
        assert all(p.meta["source"] == "x.txt" for p in pieces)


class TestSplitMarkdown:
    MD = (
        "# 商品手册\n\n"
        "## 售后政策\n\n"
        "### 退换货\n\n"
        "自签收之日起 7 天内可无理由退货,商品需保持完好,包装配件齐全。\n\n"
        "### 发票\n\n"
        "支持开具电子发票,下单时勾选即可。\n\n"
        "## 保修条款\n\n"
        "显卡整机保修 3 年,人为损坏不在保修范围内。"
    )

    def test_headers_land_in_metadata(self) -> None:
        pieces = split_markdown(self.MD, chunk_size=200, chunk_overlap=20)
        refund = next(p for p in pieces if "无理由退货" in p.content)
        assert refund.meta["h1"] == "商品手册"
        assert refund.meta["h2"] == "售后政策"
        assert refund.meta["h3"] == "退换货"

    def test_breadcrumb_is_prepended_to_content(self) -> None:
        """面包屑必须进正文,不能只留在 metadata。

        只放 metadata 的话,BM25 和向量都看不到它——而「这个商品的退换货」
        里的商品名,正是把不同商品的同名章节区分开的关键词。
        """
        pieces = split_markdown(self.MD, chunk_size=200, chunk_overlap=20)
        refund = next(p for p in pieces if "无理由退货" in p.content)
        assert refund.content.startswith("商品手册 › 售后政策 › 退换货")
        assert refund.meta["breadcrumb"] == "商品手册 › 售后政策 › 退换货"

    def test_sibling_sections_get_their_own_breadcrumbs(self) -> None:
        pieces = split_markdown(self.MD, chunk_size=200, chunk_overlap=20)
        warranty = next(p for p in pieces if "保修 3 年" in p.content)
        # 保修条款在 h2 下直接是正文,没有 h3
        assert warranty.meta["breadcrumb"] == "商品手册 › 保修条款"

    def test_long_section_without_subheadings_is_still_split(self) -> None:
        """**回归测试:`MarkdownHeaderTextSplitter` 没有 chunk_size 上限。**

        它不继承 TextSplitter,超长章节会被原样吐成一整块。少了二层切分,
        一个万字章节会:① 撑爆上下文预算 ② 向量被稀释成一个平均值,
        检索什么都沾一点、什么都不准。
        """
        md = "# 手册\n\n## 参数\n\n" + "这是该商品的详细参数说明文字,包含大量细节。" * 60
        pieces = split_markdown(md, chunk_size=120, chunk_overlap=20)
        assert len(pieces) > 1, "超长章节必须被二次切分"
        for piece in pieces:
            assert estimate_tokens(piece.content) <= 200

    def test_markdown_markers_do_not_reach_content(self) -> None:
        pieces = split_markdown(self.MD, chunk_size=200, chunk_overlap=20)
        for piece in pieces:
            assert not piece.content.lstrip().startswith("#")

    def test_empty_text_yields_nothing(self) -> None:
        assert split_markdown("", chunk_size=100, chunk_overlap=10) == []
        assert split_markdown("# 只有标题", chunk_size=100, chunk_overlap=10) == []


class TestTableRows:
    def test_column_aliases_are_normalized(self) -> None:
        cols = normalize_columns(["商品名称", "类目", "价格", "库存", "自定义列"])
        assert cols["商品名称"] == "product_name"
        assert cols["类目"] == "category"
        assert cols["价格"] == "price"
        # 认不出来的列原样保留,不能丢——丢了就等于删了用户的数据
        assert cols["自定义列"] == "自定义列"

    def test_row_becomes_readable_chinese_sentence(self) -> None:
        """**不能用 `product_name RX9070` 这种英文键直出。**

        用户问的是「这个显卡多少钱」,`price` 这个词在中文问句里不出现,
        也没有任何字面重叠;渲染成「价格 4299元」之后,字面与语义才同时对上。
        """
        sentence = row_to_sentence(
            {"product_name": "RX9070", "category": "显卡", "price": 4299}
        )
        assert sentence == "商品 RX9070,类目 显卡,价格 4299元。"
        assert "product_name" not in sentence
        assert "元" in sentence, "数值字段要带单位,否则「4299」和「4299 元」不是同一个词"

    def test_raw_chinese_headers_render_like_normalized_ones(self) -> None:
        """**回归测试:归一化必须发生在真实入参的形状上。**

        解析器实际吐出来的是 `{"商品名称": "RX9070", "价格": "4299"}`——
        中文表头、字符串数值,外加一个 `_sheet`。而此前单元测试只拿
        手工归一化过的 dict 喂 `row_to_sentence`,所以它一直是绿的,
        线上却渲染成「商品名称 RX9070,类目 显卡,价格 4299」:
        没有中文标签、没有单位、元数据里也没有 category/price。
        """
        raw = {"商品名称": "RX9070", "类目": "显卡", "价格": "4299", "_sheet": "报价"}
        assert row_to_sentence(raw) == "商品 RX9070,类目 显卡,价格 4299元。"

    def test_internal_keys_do_not_leak_into_the_sentence(self) -> None:
        """`_sheet` 是文件级信息,不能逐行渲染进正文。"""
        sentence = row_to_sentence({"商品名称": "甲", "_sheet": "Sheet1"})
        assert sentence == "商品 甲。"
        assert "Sheet1" not in sentence

    def test_empty_values_are_dropped(self) -> None:
        sentence = row_to_sentence({"product_name": "甲", "color": None, "spec": ""})
        assert sentence == "商品 甲。"

    def test_rows_are_grouped(self) -> None:
        rows = [
            {"product_name": f"型号{i}", "category": "显卡", "price": 1000 + i}
            for i in range(7)
        ]
        pieces = rows_to_sentences(rows, rows_per_chunk=3)
        assert [len(p.content.splitlines()) for p in pieces] == [3, 3, 1]

    def test_default_is_one_row_per_chunk(self) -> None:
        """**默认一行一个分块,每个分块带自己那一行的元数据。**

        多行合块时,块的 category/price 只能取块内第一行的值,于是
        「只看显示器」这类过滤会**漏掉**同块里别的类目的行——不报错、
        只是少返回几条。而类目/价格预过滤是本项目的卖点之一,精确性优先。

        代价是分块数变多、单行信息量低;补回来的是向量+BM25 混合检索
        (关键词路能抓住型号这类精确词)与「表格行转自然语言句式」
        (让单行不再是噪音)。这个取舍在 `test_retrieval_integration.py`
        里有端到端的回归。
        """
        rows = [
            {"product_name": f"型号{i}", "category": "显卡", "price": 1000 + i}
            for i in range(3)
        ]
        pieces = rows_to_sentences(rows)
        assert len(pieces) == 3
        # 关键断言:每块的元数据是**自己那一行**的,而不是第一行的副本。
        # 只断言数量的话,「三块都带着第一行的 price」照样能通过。
        assert [p.meta["price"] for p in pieces] == [1000, 1001, 1002]
        assert [p.meta["row_start"] for p in pieces] == [0, 1, 2]

    def test_metadata_carries_filterable_fields(self) -> None:
        """元数据要带上可过滤字段——类目/价格预过滤靠它。"""
        pieces = rows_to_sentences(
            [{"product_name": "RX9070", "category": "显卡", "price": 4299}],
            source="报价.xlsx",
        )
        assert pieces[0].meta["category"] == "显卡"
        assert pieces[0].meta["price"] == 4299
        assert pieces[0].meta["source"] == "报价.xlsx"

    def test_metadata_is_built_from_raw_parser_output(self) -> None:
        """**回归测试:`build_where` 支持按类目/价格过滤,但必须真的有值可过滤。**

        曾经整条链路上没有任何地方调用归一化,所以元数据里只有
        `source` / `row_start`——`ChromaStore.build_where(category=..., price_min=...)`
        是完整实现好的,却永远筛不出东西。
        """
        pieces = rows_to_sentences(
            [{"商品名称": "RX9070", "类目": "显卡", "价格": "4299", "_sheet": "报价"}],
            source="报价.xlsx",
        )
        meta = pieces[0].meta
        assert meta["product_name"] == "RX9070"
        assert meta["category"] == "显卡"
        assert meta["price"] == 4299, "价格必须是数字,否则 Chroma 的 $gte/$lte 筛不出东西"
        assert meta["sheet"] == "报价"

    def test_all_empty_rows_yield_nothing(self) -> None:
        assert rows_to_sentences([{}, {"a": None}]) == []


class TestNormalizeRow:
    def test_aliases_are_applied(self) -> None:
        assert normalize_row({"商品名称": "甲", "售价": "9"}) == {
            "product_name": "甲",
            "price": 9,
        }

    def test_unknown_headers_survive(self) -> None:
        """认不出的列原样保留——丢掉等于删了用户的数据。"""
        assert normalize_row({"自定义列": "x"}) == {"自定义列": "x"}

    def test_internal_keys_are_dropped(self) -> None:
        assert normalize_row({"甲": "1", "_sheet": "Sheet1"}) == {"甲": "1"}

    def test_is_idempotent(self) -> None:
        """归一化过的行再喂一遍不能变样——否则重索引会产出不同的分块。"""
        once = normalize_row({"商品名称": "甲", "价格": "4299"})
        assert normalize_row(once) == once

    def test_header_whitespace_is_trimmed(self) -> None:
        assert normalize_row({" 商品名称 ": "甲"}) == {"product_name": "甲"}


class TestCoerceNumber:
    def test_plain_integer(self) -> None:
        assert coerce_number("4299") == 4299
        assert isinstance(coerce_number("4299"), int)

    def test_decimal_is_kept_as_float(self) -> None:
        assert coerce_number("4299.50") == 4299.5

    def test_thousands_separator(self) -> None:
        assert coerce_number("4,299") == 4299

    def test_currency_prefix(self) -> None:
        assert coerce_number("¥4299") == 4299
        assert coerce_number("￥ 4,299") == 4299

    def test_negative(self) -> None:
        assert coerce_number("-50") == -50

    def test_non_numeric_is_returned_untouched(self) -> None:
        """「面议」必须原样留下,不能变成 None 被丢掉。

        丢掉等于把那行商品从知识库里删了,而用户只会觉得「问某某型号
        怎么没有」——一个不会报错的数据丢失。
        """
        assert coerce_number("面议") == "面议"
        assert coerce_number("暂无报价") == "暂无报价"

    def test_non_string_values_pass_through(self) -> None:
        assert coerce_number(4299) == 4299
        assert coerce_number(None) is None
        # bool 是 int 的子类,不能被当成数字处理
        assert coerce_number(True) is True

    def test_model_codes_are_not_touched(self) -> None:
        """只在 price/stock 上做数值化,所以「型号 12900K」这类不会被误伤。"""
        assert normalize_row({"型号": "12900K"}) == {"model": "12900K"}


class TestTokenEstimate:
    def test_cjk_counts_roughly_one_per_char(self) -> None:
        assert estimate_tokens("这是一段中文") == 6

    def test_latin_is_about_four_chars_per_token(self) -> None:
        assert estimate_tokens("abcdefgh") == 2

    def test_mixed_text(self) -> None:
        # 4 个汉字 + 8 个拉丁字符 → 4 + 2
        assert estimate_tokens("中文四个abcdefgh") == 6

    def test_empty_is_zero(self) -> None:
        assert estimate_tokens("") == 0
