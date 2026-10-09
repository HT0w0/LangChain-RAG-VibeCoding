"""上下文组装测试。

组装这一步决定了模型**看到什么**,以及用户看到的 `[1][2]` 到底指向哪条
资料——两者必须严格一致。编号错位是 RAG 里最隐蔽的一类 bug:
答案读起来完全正常,只是引用的来源是错的,而且**不会报错**。

预算截断也是同一类:截错了不会抛异常,只会让模型在缺资料的情况下
照常作答。所以这里的断言看的是「哪些分块被选中」,而不是只断言不崩。
"""

from __future__ import annotations

from app.services.context import RetrievedChunk, assemble_context, estimate_tokens


def chunk(cid: int, content: str, *, doc_name: str = "商品.xlsx", **kwargs) -> RetrievedChunk:
    kwargs.setdefault("kb_id", 1)
    kwargs.setdefault("doc_id", 1)
    kwargs.setdefault("kb_name", "商品库")
    kwargs.setdefault("chunk_index", cid)
    kwargs.setdefault("meta", {})
    return RetrievedChunk(chunk_id=cid, content=content, doc_name=doc_name, **kwargs)


def used_ids(bundle) -> list[int]:
    return [c.chunk_id for c in bundle.used]


def paragraph(index: int, size: int = 20) -> str:
    """生成**互不相同**的正文,长度固定为 size 个汉字。

    预算用例里必须让每篇内容不一样。内容相同的话,去重逻辑会先一步把它们
    丢掉,「预算不够」的用例就悄悄变成了「正文重复」的用例——断言照样是
    绿的,但验证的根本不是预算。这个坑我第一次就踩了。
    """
    return chr(0x4E00 + index) * size


class TestEstimateTokens:
    """**估算而非精确计数**,因为 `ChatOpenAI.get_num_tokens_from_messages()`
    对 qwen 未实现(抛 NotImplementedError),而 `trim_messages(token_counter=...)`
    正需要它。宁可自己估一个偏保守的数,也不要让上下文预算失控。"""

    def test_empty_text_is_zero(self) -> None:
        assert estimate_tokens("") == 0

    def test_ascii_is_about_four_chars_per_token(self) -> None:
        assert estimate_tokens("a" * 40) == 10

    def test_cjk_is_about_one_token_per_char(self) -> None:
        """中文一个字大致一个 token——按英文的 4:1 去估会严重低估,
        导致实际 prompt 远超预算,请求被服务端截断而内容悄悄丢失。"""
        assert estimate_tokens("显卡参数说明") == 6

    def test_mixed_text_sums_both_parts(self) -> None:
        # 4 个汉字 + "RX9070"(6 个 ASCII) → 4 + (6+3)//4 = 4 + 2
        assert estimate_tokens("显卡型号RX9070") == 6

    def test_is_monotonic(self) -> None:
        short = "显卡"
        assert estimate_tokens(short * 2) >= estimate_tokens(short)
        assert estimate_tokens(short * 10) > estimate_tokens(short)


class TestNumbering:
    """编号是引用溯源的锚点:正文里的 `[2]` 必须指向 `used[1]`。"""

    def test_chunks_are_numbered_from_one_in_input_order(self) -> None:
        bundle = assemble_context([chunk(7, "AAA"), chunk(9, "BBB")], max_tokens=1000)
        assert used_ids(bundle) == [7, 9]
        assert "[1]" in bundle.text
        assert "[2]" in bundle.text
        # 顺序:7 的正文必须出现在 9 的正文之前,否则 `[1]` 指到了第二条
        assert bundle.text.index("AAA") < bundle.text.index("BBB")

    def test_source_name_is_rendered_so_the_model_can_attribute(self) -> None:
        bundle = assemble_context([chunk(1, "支持三年质保", doc_name="售后政策.pdf")], max_tokens=1000)
        assert "售后政策.pdf" in bundle.text
        assert "支持三年质保" in bundle.text

    def test_meta_heading_is_included_when_present(self) -> None:
        """Markdown 的标题路径能把「这段在文档哪一节」告诉模型,提高归因准确率。"""
        bundle = assemble_context(
            [chunk(1, "正文", meta={"source_heading": "售后 > 质保"})], max_tokens=1000
        )
        assert "售后 > 质保" in bundle.text


class TestBudget:
    def test_result_never_exceeds_the_budget(self) -> None:
        # 每条约 18 个汉字(≈18 token)+ 表头 ≈25,预算 120 装得下 4 条。
        # 特意让「装得下」成立,否则全部被丢时这条断言会**平凡通过**,
        # 看起来是绿的却什么也没验证。
        chunks = [chunk(i, paragraph(i, 18)) for i in range(1, 6)]
        bundle = assemble_context(chunks, max_tokens=120)
        assert bundle.tokens <= 120
        assert 1 <= len(bundle.used) < len(chunks)

    def test_keeps_the_highest_ranked_chunks(self) -> None:
        """预算不够时先丢**排在后面**的,不能丢前面的。

        检索结果按相关性降序,尾部本来就更可能是噪音;反过来丢头部的
        话,最相关的那条资料反而进不了 prompt。
        """
        chunks = [chunk(i, paragraph(i)) for i in range(1, 6)]
        bundle = assemble_context(chunks, max_tokens=60)
        assert used_ids(bundle) == [1, 2]

    def test_oversized_chunk_is_skipped_but_later_ones_still_used(self) -> None:
        """**超大分块不能把后面的小分块一起挤掉。**

        若遇到放不下的就 `break`,一条 5000 token 的分块会把整个上下文
        清空——模型拿不到任何资料,却照样作答。所以是「跳过」而不是「中断」。
        """
        chunks = [
            chunk(1, "x" * 8000),  # 约 2000 token,远超预算
            chunk(2, "短期" * 5),
            chunk(3, "保修" * 5),
        ]
        bundle = assemble_context(chunks, max_tokens=100)
        assert used_ids(bundle) == [2, 3]
        assert [c.chunk_id for c in bundle.dropped] == [1]

    def test_nothing_fits_yields_an_empty_context(self) -> None:
        """全放不下时返回空串——由调用方去说「资料中没有找到相关内容」,
        而不是硬塞半条被截断的资料。半条资料比没有资料更危险:
        模型会基于残缺的句子编出一个看似有据的答案。
        """
        bundle = assemble_context([chunk(1, "x" * 8000)], max_tokens=50)
        assert bundle.used == []
        assert bundle.text == ""
        assert bundle.tokens == 0

    def test_dropped_chunks_are_reported(self) -> None:
        """被丢掉的分块要能查到,否则「为什么这条没进上下文」只能靠猜。"""
        chunks = [chunk(i, paragraph(i)) for i in range(1, 6)]
        bundle = assemble_context(chunks, max_tokens=60)
        assert [c.chunk_id for c in bundle.dropped] == [3, 4, 5]

    def test_used_and_dropped_partition_the_input(self) -> None:
        chunks = [chunk(i, paragraph(i)) for i in range(1, 6)]
        bundle = assemble_context(chunks, max_tokens=60)
        assert used_ids(bundle) + [c.chunk_id for c in bundle.dropped] == [1, 2, 3, 4, 5]


class TestDedup:
    def test_identical_content_is_included_once(self) -> None:
        """同一段正文被重复索引(比如同一份资料传了两次)时只保留一条。

        留两条会平白占掉一半预算,还会让模型以为「有两个来源都这么说」,
        从而给出比实际更有把握的语气。
        """
        bundle = assemble_context(
            [chunk(1, "显卡支持三年质保"), chunk(2, "显卡支持三年质保")], max_tokens=1000
        )
        assert used_ids(bundle) == [1]
        assert bundle.text.count("显卡支持三年质保") == 1

    def test_dedup_keeps_the_higher_ranked_one(self) -> None:
        bundle = assemble_context(
            [chunk(1, "同样的话", doc_name="甲.xlsx"), chunk(2, "同样的话", doc_name="乙.xlsx")],
            max_tokens=1000,
        )
        assert bundle.used[0].doc_name == "甲.xlsx"

    def test_different_content_is_not_deduped(self) -> None:
        bundle = assemble_context([chunk(1, "显卡质保三年"), chunk(2, "主板的质保是两年")], max_tokens=1000)
        assert used_ids(bundle) == [1, 2]


class TestEdgeCases:
    def test_empty_input(self) -> None:
        bundle = assemble_context([], max_tokens=1000)
        assert bundle.used == []
        assert bundle.dropped == []
        assert bundle.text == ""
        assert bundle.tokens == 0

    def test_zero_budget_yields_nothing(self) -> None:
        bundle = assemble_context([chunk(1, "内容")], max_tokens=0)
        assert bundle.used == []
