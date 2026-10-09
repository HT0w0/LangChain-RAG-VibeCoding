"""语义缓存阈值探针 —— `docs/SEMANTIC-CACHE.md` 的数据来源。

**这一层是三层缓存里唯一会给出错误答案的一层。** 另外两层键不同就一定
不命中,错了只是没命中;这一层是「差不多就算命中」,于是一个没测过的
阈值会把另一个问题的答案发出去——措辞、来源编号、`grounded` 全都齐整,
所有表面信号都正常。

所以那个阈值**必须测出来**,不能拍一个。这个脚本回答的问题很具体:

    在真实的 embedding 上,「同义改写」与「同实体不同属性」这两类问题对
    的余弦分布,有没有一个区间能把它们分开?

分不开的话,这一层就不该打开——而不是「先把阈值调低一点试试」。

**判据是扫描,不是两个极值相减。** 早期版本拿「应命中的最低分」减
「不应命中的最高分」,只要有一个离群点就得出「不存在安全阈值」的结论,
而那个结论对调参毫无用处。正确问法是:对每个候选阈值,数出多少漏检
(该命中的没命中)、多少错配(不该命中的命中了)。

**危险是单侧的**:错配给出错答案,漏检只是一次未命中。所以判据是
「零错配下取命中最多」,宁可往上取。

输入用的是一份**自己写的问题对**。它能证明「区间存在」,证明不了
「真实用户的问法都在区间里」——这正是这一层默认关闭的原因,也是为什么
`Settings.SEMANTIC_CACHE_THRESHOLD` 取在零错配区间**上方**。

用法:
    python backend/scripts/semantic_probe.py
    python backend/scripts/semantic_probe.py --thresholds 0.80 0.85 0.90

**联网、花钱、依赖模型版本。** 换 embedding 模型之后必须重跑:余弦的
绝对尺度随模型而变,旧阈值不作数。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

# backend/ 加入 sys.path,使 `import app.*` 生效(脚本直接跑时没有包上下文)
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.providers.embedding import build_embedder, l2_normalize  # noqa: E402

# Windows 默认控制台是 GBK(cp936),直接 print 中文会抛 UnicodeEncodeError,
# 而且是在跑到一半时抛,前面的结果全丢。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# 问题对
#
# 两类都刻意贴着电商客服的真实问法写。**关键在于构造方式**:每一对都
# 只在「问的是哪个属性」上不同,或者只在「换了哪种说法」上不同,而不是
# 随便挑两句长短不同的话——后者算出来的分数说明不了任何事。
# ---------------------------------------------------------------------------

# 应命中:同一个意图的不同问法。答案相同,复用是安全的。
HIT: list[tuple[str, str]] = [
    ("RX9070 显卡质保多久", "RX9070 的质保期是多长"),
    ("RX9070 显卡质保多久", "RX9070 显卡保修几年"),
    ("K870 机械键盘保修多久", "K870 键盘的质保期多长"),
    ("K870 机械键盘保修多久", "K870 这个键盘保修几年"),
    ("你们一般几天发货", "下单后多久能发出"),
    ("你们一般几天发货", "发货一般要几天"),
    ("支持七天无理由退货吗", "可以七天无理由退货吗"),
    ("支持七天无理由退货吗", "七天无理由退货能不能办"),
]

# 绝不能命中:同一个商品,问的是**另一个属性**,答案完全不同。
# 这一类是这个脚本存在的主要理由——它们的余弦有多高,决定了阈值能开多低。
MUST_NOT: list[tuple[str, str]] = [
    ("RX9070 显卡质保多久", "RX9070 显卡多少钱"),
    ("RX9070 显卡质保多久", "RX9070 显卡是什么接口"),
    ("RX9070 显卡质保多久", "RX9070 显卡功耗多少"),
    ("K870 机械键盘保修多久", "K870 机械键盘多少钱"),
    ("K870 机械键盘保修多久", "K870 键盘是什么轴"),
    ("K870 机械键盘保修多久", "K870 键盘支持蓝牙吗"),
    ("你们一般几天发货", "你们支持哪些快递"),
    ("你们一般几天发货", "发到新疆要几天"),
    ("支持七天无理由退货吗", "退货要自己出运费吗"),
    ("支持七天无理由退货吗", "退款几天到账"),
    # 不同商品、同一属性:句式几乎一样,只有商品名不同。
    # **这一组是最难的一类**——问法完全相同,只有实体换了。
    ("RX9070 显卡质保多久", "K870 机械键盘质保多久"),
    ("K870 机械键盘保修多久", "RX9070 显卡保修多久"),
]

# 扫描的候选阈值。密一点铺在低分区:那一段才有错配,而决策恰恰取决于
# 「最后一个还有错配的阈值」在哪里。
THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.78, 0.80, 0.82, 0.85, 0.88, 0.90, 0.95]


def cos(a: list[float], b: list[float]) -> float:
    """余弦相似度,**两侧都先归一化**。

    不假设上游已经归一过:DashScope 的 embedding 是否归一化没有写进合约,
    而一旦它不归一,点积与余弦在长度差异大的两条文本上会给出不同排序
    ——那是个只在长问题对上短问题时才显形的错。
    """
    a, b = l2_normalize(a), l2_normalize(b)
    return sum(x * y for x, y in zip(a, b, strict=True))


def _describe(name: str, scores: list[float]) -> None:
    ordered = sorted(scores)
    print(
        f"{name} n={len(scores)}  min={min(scores):.4f}  "
        f"median={ordered[len(ordered) // 2]:.4f}  max={max(scores):.4f}"
    )


async def main(*, thresholds: list[float]) -> int:
    print()
    print("=" * 72)
    print("  语义缓存阈值探针")
    print("=" * 72)
    print(f"embedding 模型:{settings.EMBED_MODEL}  backend={settings.EMBED_BACKEND}")
    if settings.EMBED_BACKEND != "dashscope":
        # mock 编码器算的是字符 n-gram 重叠,**不是语义**。用它扫出来的
        # 阈值与真实模型无关,照着它调参等于随机数。
        print("  ⚠ 当前不是 dashscope 后端 —— 这个结果不能用来定阈值")
    print()

    embedder = build_embedder(settings)
    texts = sorted({t for pair in HIT + MUST_NOT for t in pair})
    print(f"编码 {len(texts)} 条不同问题……")
    vectors = {text: await embedder.embed_query(text) for text in texts}

    hit_scores = [cos(vectors[a], vectors[b]) for a, b in HIT]
    bad_scores = [cos(vectors[a], vectors[b]) for a, b in MUST_NOT]

    print()
    _describe("应命中  ", hit_scores)
    _describe("不应命中", bad_scores)

    print()
    print("### 不应命中里分数最高的 6 条(最危险的错配)")
    print("    错配的代价是用户拿到另一个问题的答案,而且看起来完全正常。")
    for (a, b), score in sorted(zip(MUST_NOT, bad_scores, strict=True), key=lambda x: -x[1])[:6]:
        print(f"  {score:.4f}  {a}  ‖  {b}")

    print()
    print("### 阈值扫描")
    print(f"  {'阈值':>6} {'命中':>9} {'错配':>9}")
    best: tuple[float, int] | None = None
    for threshold in thresholds:
        tp = sum(1 for s in hit_scores if s >= threshold)
        fp = sum(1 for s in bad_scores if s >= threshold)
        mark = "  ← 零错配" if fp == 0 else ""
        print(f"  {threshold:>6.2f} {tp:>5}/{len(hit_scores):<3} {fp:>5}/{len(bad_scores):<3}{mark}")
        if fp == 0 and (best is None or tp > best[1]):
            best = (threshold, tp)

    # **零错配区间的下界是 `max(bad)`,不是「两类分布完全分得开」。**
    #
    # 这两个条件差得很远,混淆它们的后果是给出**方向相反的建议**:
    # 「分布分得开」要求 `min(命中) > max(错配)`,而只要有一个应命中的
    # 问题分数偏低(比如一句很短的改写),它就永远不成立——于是报告会得出
    # 「不存在安全阈值,只能关掉这一层」,而**它正上方的扫描表里明明写着
    # 0.80 起错配为零、命中还有 7/8**。
    #
    # 真正要问的是:哪个阈值能保住尽可能多的命中,同时一个错配都不要。
    # 下界取 `max(bad)`(超过它的阈值必然零错配),上界取**高于下界的
    # 最低命中分**(再往上就开始白丢命中了)。这两点之间的区间里,
    # 零错配与命中率同时成立,取舍为零。
    floor = max(bad_scores)
    keepable = [s for s in hit_scores if s > floor]
    if keepable:
        ceiling = min(keepable)
        print()
        print(f"零错配区间:({floor:.4f}, {ceiling:.4f}]  宽度 {ceiling - floor:.4f}")
        print(
            f"  区间内任意阈值都是:错配 0/{len(bad_scores)},"
            f"命中 {len(keepable)}/{len(hit_scores)} —— 取舍为零。"
        )
        print(
            f"  下界由最危险的那条错配决定({floor:.4f});高出上界 "
            f"{ceiling:.4f} 就开始拿命中率换安全——而这是**唯一安全的方向**。"
        )
        print(
            f"  取舍不对称,所以推荐取在区间上方:宁可少命中,不可错配。"
            f"当前配置的 {settings.SEMANTIC_CACHE_THRESHOLD} 就落在上方。"
        )
    else:
        # 连一条命中都保不住:说明这批样本上零错配的代价是这一层完全失效。
        # **这时正确的做法是关掉,而不是把阈值降下来凑命中率**——降下去
        # 换来的命中全是错配。
        print()
        print(f"**零错配区间为空**:不应命中的最高分 {floor:.4f} 之上没有任何一条应命中。")
        print("  也就是说,要想一个错配都不出,就得放弃全部命中——这一层在这个")
        print("  语料上没有任何价值。此时应当**关掉它**,而不是降阈值去凑命中率:")
        print("  降下去换来的命中,全部都是错配。")

    if best is not None:
        threshold, tp = best
        print(
            f"零错配下命中最多:阈值 {threshold:.2f} → 命中 {tp}/{len(hit_scores)},"
            f"错配 0/{len(bad_scores)}"
        )
    print()
    print(f"当前配置的阈值:{settings.SEMANTIC_CACHE_THRESHOLD}")
    print(f"当前配置的开关:SEMANTIC_CACHE_ENABLED={settings.SEMANTIC_CACHE_ENABLED}")
    print()

    aclose = getattr(embedder, "aclose", None)
    if aclose is not None:
        await aclose()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="语义缓存阈值探针")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=THRESHOLDS,
        help="要扫描的阈值列表(默认:0.60 起,低分区更密)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(thresholds=args.thresholds)))
