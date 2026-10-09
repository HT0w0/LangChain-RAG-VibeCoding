"""RRF 融合测试。

RRF 是**排序**算法,所以大部分 bug 不会让结果变空,只会让顺序悄悄变错——
而顺序错了,答案的引用来源就错了。因此这里的断言既看顺序,也看**分数本身**:
只断言顺序的话,k 偏移、权重失效、rank 从 0 开始这类错误全都测不出来。
"""

from __future__ import annotations

from app.indexing.fusion import reciprocal_rank_fusion


class TestCoreRanking:
    def test_document_in_both_lists_outranks_document_in_one(self) -> None:
        """RRF 的核心性质:两路都认可的文档,排在只有一路认可的前面。

        手算(k=60):
          10 → 1/61 + 1/62 = 0.032522
          30 → 1/63 + 1/61 = 0.032266
          20 → 1/62         = 0.016129
          40 → 1/63         = 0.015873
        """
        fused = reciprocal_rank_fusion(
            {"vector": [10, 20, 30], "keyword": [30, 10, 40]}
        )
        assert [h.chunk_id for h in fused] == [10, 30, 20, 40]

    def test_score_matches_hand_computed_value(self) -> None:
        """只断言顺序的话,`k` 用错值(比如 0 或 100)是发现不了的。"""
        fused = reciprocal_rank_fusion({"vector": [7]}, k=60)
        assert fused[0].score == 1 / 61

    def test_single_route_preserves_its_order(self) -> None:
        fused = reciprocal_rank_fusion({"vector": [3, 1, 2]})
        assert [h.chunk_id for h in fused] == [3, 1, 2]

    def test_rank_is_one_based(self) -> None:
        """rank 从 0 开始的话分数会整体错位,而排序却完全一样。

        所以必须断言数值:`1/(60+0)` 与 `1/(60+1)` 只差 1.6%,
        顺序上看不出来,但它是错的。
        """
        fused = reciprocal_rank_fusion({"vector": [7, 8]}, k=60)
        by_id = {h.chunk_id: h for h in fused}
        assert by_id[7].ranks == {"vector": 1}
        assert by_id[8].ranks == {"vector": 2}


class TestRankDetail:
    def test_each_route_rank_is_recorded(self) -> None:
        """保留每一路的 rank 明细是自研 RRF 的主要理由。

        官方 `EnsembleRetriever` 只给融合后的分数,看不到「这条是向量召回的
        还是关键词召回的」。没有这份明细,就没法回答「关键词路到底有没有
        贡献」——而那是调参时唯一有用的信息。
        """
        fused = reciprocal_rank_fusion(
            {"vector": [10, 20], "keyword": [20, 10]}
        )
        by_id = {h.chunk_id: h for h in fused}
        assert by_id[10].ranks == {"vector": 1, "keyword": 2}
        assert by_id[20].ranks == {"vector": 2, "keyword": 1}

    def test_contribution_per_route_is_recorded(self) -> None:
        """每路的分数贡献要能拆开,否则加权调参只能靠猜。"""
        fused = reciprocal_rank_fusion({"vector": [10], "keyword": [10]}, k=60)
        contrib = fused[0].contributions
        assert contrib["vector"] == 1 / 61
        assert contrib["keyword"] == 1 / 61

    def test_document_in_one_route_only_lists_that_route(self) -> None:
        fused = reciprocal_rank_fusion({"vector": [10], "keyword": [20]})
        by_id = {h.chunk_id: h for h in fused}
        assert by_id[10].ranks == {"vector": 1}
        assert by_id[20].ranks == {"keyword": 1}


class TestWeights:
    def test_weight_scales_contribution(self) -> None:
        fused = reciprocal_rank_fusion(
            {"vector": [10], "keyword": [20]}, weights={"vector": 2.0}
        )
        by_id = {h.chunk_id: h for h in fused}
        assert by_id[10].score == 2 / 61
        assert by_id[20].score == 1 / 61

    def test_unlisted_route_defaults_to_weight_one(self) -> None:
        """只给一路设权重时,其余各路必须保持原样而不是变成 0。

        变成 0 的话,那条召回路径就静默失效了——结果变少,
        但既不报错也不容易察觉。
        """
        fused = reciprocal_rank_fusion(
            {"vector": [10], "keyword": [20]}, weights={"vector": 3.0}
        )
        by_id = {h.chunk_id: h for h in fused}
        assert by_id[20].score == 1 / 61

    def test_weighted_route_can_overtake(self) -> None:
        """加权要能真的改变排序,否则它只是个装饰性的参数。"""
        rankings = {"vector": [10, 20], "keyword": [20, 10]}
        # 等权时 10 与 20 同分,靠 chunk_id 定序
        assert [h.chunk_id for h in reciprocal_rank_fusion(rankings)] == [10, 20]
        # 给 keyword 加权重后,keyword 排第一的 20 应当反超
        weighted = reciprocal_rank_fusion(rankings, weights={"keyword": 2.0})
        assert [h.chunk_id for h in weighted] == [20, 10]

    def test_weight_also_scales_recorded_contribution(self) -> None:
        """明细里记的必须是**加权后**的贡献,否则调参时看到的数是假的。"""
        fused = reciprocal_rank_fusion({"vector": [10]}, weights={"vector": 2.5})
        assert fused[0].contributions["vector"] == 2.5 / 61


class TestEdgeCases:
    def test_empty_input_returns_empty(self) -> None:
        assert reciprocal_rank_fusion({}) == []
        assert reciprocal_rank_fusion({"vector": [], "keyword": []}) == []

    def test_top_k_truncates_after_sorting(self) -> None:
        fused = reciprocal_rank_fusion({"vector": [10, 20, 30]}, top_k=2)
        assert [h.chunk_id for h in fused] == [10, 20]

    def test_top_k_larger_than_results_is_harmless(self) -> None:
        fused = reciprocal_rank_fusion({"vector": [10]}, top_k=99)
        assert len(fused) == 1

    def test_duplicate_within_one_route_uses_best_rank(self) -> None:
        """同一条被同一路召回两次时,只能计一次分,且取最好名次。

        计两次会让这一路对它的权重凭空翻倍——而调用方以为自己在做等权融合。
        """
        fused = reciprocal_rank_fusion({"vector": [10, 20, 10]})
        hit = next(h for h in fused if h.chunk_id == 10)
        assert hit.ranks == {"vector": 1}
        assert hit.score == 1 / 61
        assert len(fused) == 2, "重复项不该产生两条结果"

    def test_equal_scores_break_ties_by_chunk_id(self) -> None:
        """分数相同时的顺序必须**稳定**,不能依赖 dict 的插入顺序。

        不稳定的话,同一个问题两次提问可能给出不同的引用顺序——
        语义缓存会因此产出与首次不同的引用列表,测试也会随机挂。
        """
        first = reciprocal_rank_fusion({"vector": [5], "keyword": [3]})
        second = reciprocal_rank_fusion({"keyword": [3], "vector": [5]})
        assert [h.chunk_id for h in first] == [3, 5]
        assert [h.chunk_id for h in first] == [h.chunk_id for h in second]

    def test_k_must_be_positive(self) -> None:
        """k 为 0 或负数会让分数退化甚至除零,必须早失败。"""
        import pytest

        with pytest.raises(ValueError, match="k"):
            reciprocal_rank_fusion({"vector": [10]}, k=0)
        with pytest.raises(ValueError, match="k"):
            reciprocal_rank_fusion({"vector": [10]}, k=-61)

    def test_larger_k_compresses_score_gaps(self) -> None:
        """k 的语义:越大,头部名次之间的分差越小。

        这是 k 唯一的作用,所以直接断言分差而不是断言具体数值。
        """
        small = reciprocal_rank_fusion({"vector": [1, 2]}, k=10)
        large = reciprocal_rank_fusion({"vector": [1, 2]}, k=1000)
        small_gap = small[0].score - small[1].score
        large_gap = large[0].score - large[1].score
        assert large_gap < small_gap
