"""向量库过滤条件构造的测试。

只测 `build_where` 这个纯函数,不碰 Chroma——它是「权限范围」翻译成
「数据库过滤条件」的唯一入口,而**权限 bug 比检索质量 bug 严重得多**:
后者是答得不好,前者是把别人的数据读出去了,而且全程不报错。

`kb_ids` 的三种取值含义**互不相同**,这是本文件的核心:
`None` = 不限制(管理员预览)、非空列表 = 限定这些库、**空列表 = 一个都不许**。
把空列表错当成 falsy 而跳过过滤,就等于「没有任何权限」变成了「全部权限」。
"""

from __future__ import annotations

import pytest
from app.indexing.vectorstore import build_where


class TestKnowledgeBaseScope:
    def test_no_arguments_means_no_filter(self) -> None:
        """什么条件都没有时返回 None——Chroma 不接受空 dict 作为 where。"""
        assert build_where() is None

    def test_none_means_unrestricted(self) -> None:
        """显式 `None` = 不限制。管理员预览检索效果走这条。"""
        assert build_where(None) is None

    def test_scope_becomes_an_in_clause(self) -> None:
        """多知识库用 `$in` 一次查完,而不是逐库查 N 次再合并。"""
        assert build_where([1, 2]) == {"kb_id": {"$in": [1, 2]}}

    def test_empty_scope_is_rejected_not_silently_unfiltered(self) -> None:
        """**空集合必须抛错,绝不能退化成「不加过滤」。**

        这是本文件最要紧的一条。`if kb_ids:` 这种写法会把 `[]` 当成
        「没有条件」,于是 `where=None` —— 检索**跨全部知识库**。
        一个没有任何知识库权限的用户因此能读到所有人的资料,
        而且不会报错、不会有异常日志,只是悄悄多返回了几条。

        为什么不返回一个「匹配不到任何东西」的条件:实测 Chroma 对
        `{"kb_id": {"$in": []}}` 直接抛 `ValueError: Expected where operand
        value to be a non-empty list`,根本走不到查询。既然底层无法表达
        「空」,就必须让调用方在**进查询之前**处理掉(检索服务遇到空范围
        直接返回无结果),这里则把「忘了处理」变成一次响亮的失败。
        """
        with pytest.raises(ValueError, match="kb_ids"):
            build_where([])

    def test_empty_scope_rejected_even_with_other_filters(self) -> None:
        """带上别的条件也照样拒绝——别指望「类目过滤能兜住」。

        兜不住:空范围 + `category="显卡"` 仍然是「全部知识库里的显卡」。
        """
        with pytest.raises(ValueError, match="kb_ids"):
            build_where([], category="显卡")


class TestFieldConditions:
    def test_single_condition_is_not_wrapped_in_and(self) -> None:
        """只有一个条件时不套 `$and`:多一层包装没有语义,只让日志变难读。"""
        assert build_where(category="显卡") == {"category": "显卡"}

    def test_price_range_becomes_two_conditions(self) -> None:
        """**价格区间必须拆成两个独立条件。**

        Chroma 校验「每个字段的操作符表达式恰好一个操作符」,
        写成 `{"price": {"$gte": a, "$lte": b}}` 会直接抛
        `Expected operator expression to have exactly one operator`。
        这与 MongoDB 的直觉相反,是个很容易踩的坑,所以在这里钉死。
        """
        where = build_where(price_min=1000, price_max=5000)
        assert where == {"$and": [{"price": {"$gte": 1000}}, {"price": {"$lte": 5000}}]}

    def test_only_lower_bound(self) -> None:
        assert build_where(price_min=1000) == {"price": {"$gte": 1000}}

    def test_zero_is_a_real_bound_not_absent(self) -> None:
        """`price_min=0` 与「没传」是两回事。

        用 `if price_min:` 判断的话,0 会被当成没传,「0 元以上」
        这个条件静默消失——又是同一类 falsy 陷阱。
        """
        assert build_where(price_min=0) == {"price": {"$gte": 0}}


class TestCombination:
    def test_all_conditions_are_anded(self) -> None:
        where = build_where([1], category="显卡", price_min=1000, price_max=5000)
        assert where == {
            "$and": [
                {"kb_id": {"$in": [1]}},
                {"category": "显卡"},
                {"price": {"$gte": 1000}},
                {"price": {"$lte": 5000}},
            ]
        }

    def test_extra_conditions_are_merged(self) -> None:
        """`extra` 是给调用方留的逃生口(未来加品牌、上架状态等字段)。"""
        where = build_where([1], extra={"brand": "影驰"})
        assert where == {"$and": [{"kb_id": {"$in": [1]}}, {"brand": "影驰"}]}
