"""RRF(Reciprocal Rank Fusion)融合。

**为什么不直接用 LangChain 的 `EnsembleRetriever`**

它确实实现了 RRF,但只交给你一个融合后的排序。本项目的作品集要回答
「关键词这一路到底有没有贡献」「某个问题是谁召回的」——那需要**每一路的
rank 明细**。自己写不到 50 行,换来的是一份可直接展示、可单独测试、
可埋点的中间结果。

**公式**([Cormack et al., SIGIR 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)):

    score(d) = Σ_r  w_r / (k + rank_r(d))

`k=60` 取自原论文。它的作用是**压平头部**:k 越大,rank 1 与 rank 2 的
分数差越小,融合结果越不依赖任何单一路的头部判断。60 是论文在 TREC
数据上试出来的经验值,不是必须遵守的常数,所以这里做成参数。

**为什么用「排名」而不是「分数」融合**:向量检索给的是余弦相似度
(0~1),BM25 给的是无上界的正数,两者量纲不可比。做归一化再相加需要
知道各自的分布,而分布随 query 变化;只取排名则天然免除了这个问题——
这正是 RRF 在工业界流行的原因。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

# 原论文的经验值。见模块文档。
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusedHit:
    """融合后的一条候选。

    `ranks` 与 `contributions` 按路名(如 "vector" / "keyword")索引,
    只包含**真正召回了它**的那些路。
    """

    chunk_id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)
    contributions: dict[str, float] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[int]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    top_k: int | None = None,
) -> list[FusedHit]:
    """把多路排名融合成一路。

    参数 `rankings` 是 `{路名: [chunk_id, ...]}`,列表本身按相关性降序。
    `weights` 给每一路加权,未列出的路默认 1.0。

    返回按融合分数降序排列的 `FusedHit`,分数相同者按 chunk_id 升序
    (**排序必须是全序**,否则同一个 query 两次调用可能给出不同的引用顺序)。
    传 `top_k` 时只返回前 k 条,截断发生在排序之后。
    """
    if k <= 0:
        raise ValueError(f"k 必须是正整数,收到 {k}")

    weights = weights or {}
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    contributions: dict[int, dict[str, float]] = {}

    for route, chunk_ids in rankings.items():
        weight = weights.get(route, 1.0)
        # 同一路内重复召回只计一次,且取最好名次。计两次会让这一路的
        # 实际权重凭空翻倍,而调用方以为自己设的是 1.0。
        seen: set[int] = set()
        for position, chunk_id in enumerate(chunk_ids, start=1):
            if chunk_id in seen:
                continue
            seen.add(chunk_id)
            contribution = weight / (k + position)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            ranks.setdefault(chunk_id, {})[route] = position
            contributions.setdefault(chunk_id, {})[route] = contribution

    hits = [
        FusedHit(
            chunk_id=chunk_id,
            score=score,
            ranks=ranks[chunk_id],
            contributions=contributions[chunk_id],
        )
        for chunk_id, score in scores.items()
    ]
    # 次键 chunk_id 不是可有可无的:等分时的顺序若取决于 dict 插入顺序,
    # 结果就会随「哪一路先被遍历」而变。加它,排序才是全序。
    hits.sort(key=lambda h: (-h.score, h.chunk_id))
    return hits[:top_k] if top_k is not None else hits
