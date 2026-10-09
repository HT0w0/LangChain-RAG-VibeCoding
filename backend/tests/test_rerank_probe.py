"""重排阈值扫描的**判据**,不是它的数据。

`scripts/rerank_probe.py` 要联网、花钱、且结论随模型版本漂移,所以它自己
进不了单元测试。但它内部有一段纯函数——「给一批已标注的 (查询, 段落) 打分
和一组候选阈值,数出每个阈值下会发生什么」——那段是可以、也必须离线钉住的。

**为什么值得单独测:** 阈值这个数唯一的用处就是决定「哪些段落进上下文」,
而扫描表是选它的唯一依据。表算错一次,选出来的阈值就是个幌子——它看起来
有实测支撑,实际没有。而它错的方式特别安静:分母取错、边界取开区间、
截断顺序颠倒,这三种都不会抛异常,只会让表上少一行或多一行。

**这一段逻辑没有生产代码调用它。** 它只服务这一次测量,所以放在脚本里而
不是 `app/` 下——`app/` 里放一份没人调用的代码,是给下一个人制造「这里有
个入口」的误会。
"""

from __future__ import annotations

from scripts.rerank_probe import Judged, scan


def _j(query: str, passage: str, *, relevant: bool, score: float) -> Judged:
    return Judged(query=query, passage=passage, relevant=relevant, score=score)


class TestScan:
    def test_a_zero_threshold_keeps_everything(self) -> None:
        """阈值 0.0(当前默认)必须等于「不过滤」——一个都不许少。

        这是整个表的地基:如果连 0.0 都在丢东西,那说明丢的不是阈值干的,
        而表会把那部分损失记在别的阈值头上。
        """
        judged = [
            _j("q1", "p1", relevant=True, score=0.9),
            _j("q1", "p2", relevant=False, score=0.1),
            _j("q2", "p3", relevant=False, score=0.2),  # 一个谁也答不了的查询
        ]

        rows = scan(judged, [0.0])

        assert len(rows) == 1
        row = rows[0]
        assert row.relevant_kept == 1
        assert row.irrelevant_kept == 2
        assert row.answerable_kept == 1
        assert row.answerable_total == 1
        assert row.refused == 0
        assert row.offtopic_total == 1

    def test_a_query_is_answered_only_if_its_own_passage_survives(self) -> None:
        """**「答得出来」的判据是它自己那条标注段落活下来了,不是「还有东西活着」。**

        这条是整张表里最容易写错的一格。写成「留下来的非空就算答得出」的话,
        滤掉全部相关资料、只留下几条不相关的,照样会被记成「答得出」——而那
        恰恰是最坏的结局:模型拿着不相干的资料,被提示词要求只依据资料作答,
        于是编出一个像模像样的答案。扫描表会一路显示「可答率 100%」。
        """
        judged = [
            _j("q1", "p1", relevant=True, score=0.40),  # 相关的那条分更低
            _j("q1", "p2", relevant=False, score=0.80),
        ]

        row = scan(judged, [0.5])[0]

        assert row.answerable_kept == 0, "只剩不相关的段落,不该算答得出"
        assert row.irrelevant_kept == 1, "它确实留下了污染"
        assert row.offtopic_total == 0, "这个查询有标注段落,不是跑题问题"

    def test_an_off_topic_query_is_refused_only_when_nothing_survives(self) -> None:
        """跑题问题的判据是「一条都没留下」。

        只要留了一条,模型就有材料可编——所以这是全或无,没有中间态。
        """
        judged = [_j("q1", "p1", relevant=False, score=0.30)]

        assert scan(judged, [0.2])[0].refused == 0
        assert scan(judged, [0.5])[0].refused == 1

    def test_top_n_caps_what_the_threshold_can_ever_see(self) -> None:
        """排在 `top_n` 之外的段落,阈值再低也进不了上下文。

        **这一条不能写成「截断必须发生在过滤之前」。** 两个操作都对 score
        单调——截断取分数最高的 N 条,过滤卡 `score >= t`——所以它们**可交换**,
        先做哪个结果都一样。写成顺序断言的话,这条用例会永远绿,无论实现
        怎么写,它证明不了任何事。

        真正会变的是**候选集本身**:`top_n` 决定阈值能看见谁。生产是
        `rerank(top_n=5)` 拿到 5 条候选,阈值只在这 5 条里做减法——一条排在
        第 6 的相关资料,靠调低阈值救不回来。下面两条断言是一对,只写出
        前一条的话,一个「`top_n` 参数根本没被用上」的实现照样能通过。
        """
        judged = [
            _j("q1", "p1", relevant=False, score=0.9),
            _j("q1", "p2", relevant=False, score=0.8),
            _j("q1", "p3", relevant=True, score=0.7),  # 排第 3,落在 top_n 之外
        ]

        row = scan(judged, [0.0], top_n=2)[0]

        assert row.answerable_kept == 0, "阈值已是最低,仍救不回榜外的相关资料"
        assert scan(judged, [0.0])[0].answerable_kept == 1, "不截断就该留得住"
        assert row.irrelevant_kept == 2
        # 分母不受截断影响:标注就是标注,不会因为没进 top_n 就消失。
        # 拿截断后的数量当分母的话,「丢了多少」这件事会被自己抹掉。
        assert row.relevant_total == 1

    def test_the_boundary_is_inclusive(self) -> None:
        """分数**等于**阈值留着。生产写的是 `>= threshold`。

        差这一格的影响是系统性的:整张表的每一行都会平移一个样本的宽度,
        而它在任何一次肉眼检查里都看不出来。
        """
        judged = [_j("q1", "p1", relevant=True, score=0.5)]

        assert scan(judged, [0.5])[0].answerable_kept == 1
        assert scan(judged, [0.51])[0].answerable_kept == 0

    def test_chunk_level_denominators_count_every_labeled_pair(self) -> None:
        """段落级的两个分母是**标注总数**,不是幸存者数。

        用幸存者当分母的话,漏检会被算成「没发生过」——阈值越高,
        「留住的相关段落占比」越好看,直到分母变成零。那是一条自欺的曲线。
        """
        judged = [
            _j("q1", "p1", relevant=True, score=0.9),
            _j("q1", "p2", relevant=False, score=0.1),
            _j("q1", "p3", relevant=False, score=0.05),
        ]

        row = scan(judged, [0.5])[0]

        assert row.relevant_total == 1
        assert row.irrelevant_total == 2
        assert row.relevant_kept == 1
        assert row.irrelevant_kept == 0
