"""引用编号抽取。

答案正文里的 `[1]` 是**唯一**能把「模型说的这句话」和「哪条资料支撑它」
连起来的东西。前端靠它高亮来源卡片、后端靠它统计「这次问答真的有引用吗」。

它错起来也是静默的:抽多了一个,用户点开来源卡片看到的是无关资料;
抽漏了一个,一条真实的引用被显示成「无来源」。两种都不会报错。
"""

from __future__ import annotations

from app.services.citations import extract_citations


class TestBasicExtraction:
    def test_single_citation(self) -> None:
        assert extract_citations("价格是 4299 元 [1]。", max_index=3) == [1]

    def test_multiple_citations_are_sorted_and_deduped(self) -> None:
        """模型会把同一条引用写好几遍,也会倒着写。

        去重是必须的:前端按这个列表渲染「本条答案引用了 1、2 两条资料」,
        重复项会让计数变成 3。
        """
        assert extract_citations("[3] 参见 [1],另见 [3]。", max_index=3) == [1, 3]

    def test_adjacent_brackets(self) -> None:
        assert extract_citations("结论如下 [1][2][3]", max_index=3) == [1, 2, 3]

    def test_comma_separated_inside_one_bracket(self) -> None:
        """`[1,2]` 是很常见的写法,不能只认出其中的第一个。"""
        assert extract_citations("综合来看 [1,2] 都支持这个结论。", max_index=3) == [1, 2]

    def test_chinese_enumeration_comma_inside_bracket(self) -> None:
        assert extract_citations("[1、3]", max_index=3) == [1, 3]

    def test_citations_across_lines(self) -> None:
        text = "第一条依据 [1]。\n\n第二条依据 [2]。\n"
        assert extract_citations(text, max_index=3) == [1, 2]


class TestFullWidthBrackets:
    def test_fullwidth_brackets_are_recognized(self) -> None:
        """**中文模型经常吐全角方括号。**

        提示词里写的是 `[1]`,但模型把中括号写成 `【1】` 是常见现象。
        只认半角的话,这类答案会被判成「完全没有引用」——而答案本身
        看起来完全正常,只是来源卡片全是灰的。
        """
        assert extract_citations("价格是 4299 元【1】。", max_index=3) == [1]

    def test_mixed_width_brackets(self) -> None:
        assert extract_citations("[1] 与 【2】", max_index=3) == [1, 2]


class TestOutOfRangeIsRejected:
    def test_index_beyond_sources_is_dropped(self) -> None:
        """**模型会引用不存在的编号。**

        只给了 2 条资料却写出 `[9]`,指向一张不存在的卡片。放进列表里
        前端要么渲染出一个空卡片,要么点击时崩掉。宁可少一条,不能多一条。
        """
        assert extract_citations("[1] 和 [9]", max_index=2) == [1]

    def test_zero_is_dropped(self) -> None:
        """编号从 1 开始,`[0]` 永远是模型的错觉。"""
        assert extract_citations("[0] 不是合法引用", max_index=3) == []

    def test_no_sources_means_no_citations(self) -> None:
        assert extract_citations("依据 [1] 可知", max_index=0) == []

    def test_year_in_brackets_is_not_a_citation(self) -> None:
        """**`[2024]` 是年份,不是引用。**

        范围过滤顺带解决了这个问题——它比任何「像年份就排除」的启发式
        都可靠,因为它依赖的是一条硬约束:引用编号不可能超过资料条数。
        """
        assert extract_citations("该型号于 [2024] 年发布 [1]。", max_index=3) == [1]
        assert extract_citations("发布于 [2024] 年。", max_index=3) == []


class TestNonCitations:
    def test_plain_answer_has_none(self) -> None:
        assert extract_citations("资料中没有找到相关内容。", max_index=3) == []

    def test_empty_text(self) -> None:
        assert extract_citations("", max_index=3) == []

    def test_markdown_link_text_is_not_matched(self) -> None:
        """`[说明](http://x)` 里的方括号不含数字,不该被当成引用。"""
        assert extract_citations("详见 [文档](http://example.com)", max_index=3) == []

    def test_letters_inside_brackets_are_ignored(self) -> None:
        assert extract_citations("[a][b]", max_index=3) == []

    def test_bare_numbers_without_brackets_are_ignored(self) -> None:
        """正文里的数字到处都是(价格、库存、年份),没有方括号就不是引用。"""
        assert extract_citations("价格 4299 元,库存 12 件。", max_index=3) == []
