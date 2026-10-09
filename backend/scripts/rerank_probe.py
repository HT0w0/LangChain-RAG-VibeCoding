"""重排阈值探针 —— `docs/RERANK-THRESHOLD.md` 的数据来源。

阈值截断的机制早就在 `retrieval._apply_rerank` 里了,缺的从来不是代码,
而是一个**有依据的数字**。`RERANK_SCORE_THRESHOLD` 默认 0.0 等于不过滤,
意思是「不相关的资料别进上下文」这件事当前完全靠提示词约束模型。

为什么不能拍一个:重排分数是**查询相关**的,同一个段落在不同问题下分数
天差地别。配置注释里记着一手观测——一个完全跑题的问题在**相关段落**上拿到
0.1527,比一个相关问题里的**无关段落**(0.0637)还高。这类交错的分布没法靠
直觉定位,只能扫。

这个脚本回答的问题很具体:

    在真实的重排分数上,有没有一个阈值能同时做到
    「该答的问题还答得出来」和「跑题的问题一条资料都不留」?

**判据是扫描,不是两个极值相减。** 拿「应保留的最低分」减「应丢弃的最高分」
只要有一个离群点就得出「不存在安全阈值」,而那个结论对调参毫无用处。正确
问法是对每个候选阈值,数出三类后果:相关段落被误丢、无关段落被留下、
跑题问题没能被清干净。

**扫描必须连 `top_n` 一起复刻。** 生产是 `rerank(top_n=5)` 拿到 5 条候选,
阈值只在这 5 条里做减法——`top_n` 决定阈值能看见谁,阈值只在候选内取舍。
只看全量的表会高估召回:它默认阈值能触及榜外的段落,而线上触及不到。

    这里**不是**「两个筛子的先后」问题。截断取分数最高的 N 条、过滤卡
    `score >= t`,两者都对分数单调,因此**可交换**,先做哪个结果相同。
    真正起作用的是候选集的大小。把它写成顺序问题会写出一条永远绿的
    断言(见 `tests/test_rerank_probe.py` 里那条用例的说明)。

脚本因此对「全量」和「截断到 `RERANK_TOP_N`」各出一张表,后者才是线上
真实发生的事。

输入是一份**手写的、严格标注的语料**:每个可答问题只对应唯一一条含答案的
段落。这让「相关/无关」是一个干净的二元判断,代价是这批数据比真实语料
整齐——所以扫出来的阈值应当读作**乐观的上界**,而不是一个可以直接照搬的
最优解。这一点在文档里必须写明。

用法:
    python backend/scripts/rerank_probe.py
    python backend/scripts/rerank_probe.py --thresholds 0.0 0.05 0.10 0.2

**联网、花钱、依赖模型版本。** 换重排模型之后必须重跑:分数的绝对尺度
随模型而变,旧阈值不作数。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# backend/ 加入 sys.path,使 `import app.*` 生效(脚本直接跑时没有包上下文)
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import settings  # noqa: E402
from app.providers.reranker import build_reranker  # noqa: E402

# Windows 默认控制台是 GBK(cp936),直接 print 中文会抛 UnicodeEncodeError,
# 而且是在跑到一半时抛,前面的结果全丢。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover
        pass


# ---------------------------------------------------------------------------
# 语料
#
# 刻意做成真实切片的样子:一段就是一次问答里会被塞进上下文的那种长度,
# 内容取自商品详情页的三种版块(价格库存 / 规格参数 / 保修售后)和四份
# 通用政策。**语料必须够杂**:全是同一个商品的话,重排只要认出商品名
# 就能拿高分,扫出来的阈值会虚高,而线上是六个商品混在一起的。
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Passage:
    id: str
    doc: str  # 供报告指认「这是哪份资料」,不参与打分
    text: str


CORPUS: list[Passage] = [
    # ---- RX9070 显卡 ----
    Passage(
        "rx9070.price",
        "RX9070 显卡详情页",
        "RX9070 显卡,类目:显卡。官方指导价 4299 元,当前促销价 3999 元,"
        "活动截止到本月底。库存 12 件,下单后当天出库。",
    ),
    Passage(
        "rx9070.spec",
        "RX9070 显卡规格表",
        "RX9070 显卡采用 16GB GDDR6 显存,显存位宽 256bit,视频输出接口为 "
        "3×DP 2.1 + 1×HDMI 2.1,整卡功耗 220W,建议电源 650W,卡长 320mm,"
        "占用两个槽位。",
    ),
    Passage(
        "rx9070.warranty",
        "RX9070 显卡售后服务卡",
        "RX9070 显卡提供三年质保,自签收之日起计算。质保期内非人为损坏免费"
        "维修;显存故障、供电模块故障均在质保范围内。人为损坏(进液、"
        "私自拆解、外力破损)不在质保范围。",
    ),
    # ---- K870 机械键盘 ----
    Passage(
        "k870.price",
        "K870 机械键盘详情页",
        "K870 机械键盘,类目:键盘。售价 599 元,现有满 500 减 50 活动,"
        "到手价 549 元。库存 48 件。",
    ),
    Passage(
        "k870.spec",
        "K870 机械键盘规格表",
        "K870 机械键盘为 87 键紧凑布局,轴体为凯华 Box 白轴(段落轴),"
        "支持热插拔,键帽为 PBT 二色成型,连接方式为有线 Type-C,"
        "支持全键无冲,背光为单色白色。",
    ),
    Passage(
        "k870.warranty",
        "K870 机械键盘售后服务卡",
        "K870 机械键盘提供两年质保。轴体失灵、按键连击属于质保范围;"
        "键帽磨损、进水、摔落导致的损坏不在质保范围。质保期内可申请"
        "免费换轴。",
    ),
    # ---- AeroBook 14 笔记本 ----
    Passage(
        "aerobook.price",
        "AeroBook 14 笔记本详情页",
        "AeroBook 14 笔记本,类目:笔记本。标配版售价 5299 元,"
        "高配版(32GB 内存)售价 6299 元。标配版库存 7 台,高配版缺货,"
        "预计下月中旬到货。",
    ),
    Passage(
        "aerobook.spec",
        "AeroBook 14 笔记本规格表",
        "AeroBook 14 采用 14 英寸 2.8K 120Hz 屏幕,处理器为 8 核 16 线程,"
        "标配 16GB LPDDR5 内存,512GB PCIe 4.0 固态硬盘,电池容量 70Wh,"
        "整机重量 1.35kg,接口为 2×USB-C(支持 PD 充电)+ 2×USB-A + HDMI。",
    ),
    Passage(
        "aerobook.warranty",
        "AeroBook 14 笔记本售后服务卡",
        "AeroBook 14 笔记本整机保修两年,主要部件(主板、屏幕、处理器)"
        "保修三年。电池属于易损耗部件,保修一年。意外跌落与进液不在"
        "保修范围,可购买增值服务。",
    ),
    # ---- SoundPeak Pro 耳机 ----
    Passage(
        "soundpeak.price",
        "SoundPeak Pro 耳机详情页",
        "SoundPeak Pro 主动降噪耳机,类目:耳机。售价 899 元,"
        "会员价 849 元。库存 65 副。",
    ),
    Passage(
        "soundpeak.spec",
        "SoundPeak Pro 耳机规格表",
        "SoundPeak Pro 为入耳式主动降噪耳机,支持蓝牙 5.3 与 LDAC 编码,"
        "单次续航 8 小时,配合充电盒共 32 小时,降噪深度标称 45dB,"
        "具备 IPX4 级防水,单耳重量 5.2g。",
    ),
    Passage(
        "soundpeak.warranty",
        "SoundPeak Pro 耳机售后服务卡",
        "SoundPeak Pro 耳机提供一年质保。充电盒故障、单侧无声属于质保"
        "范围。耳塞套属于消耗品不保修;因汗液腐蚀导致的损坏,"
        "需经检测确认是否属于质保范围。",
    ),
    # ---- ViewMax 27 显示器 ----
    Passage(
        "viewmax.price",
        "ViewMax 27 显示器详情页",
        "ViewMax 27 显示器,类目:显示器。售价 1299 元,搭配整机购买可减 "
        "200 元。库存 30 台。",
    ),
    Passage(
        "viewmax.spec",
        "ViewMax 27 显示器规格表",
        "ViewMax 27 为 27 英寸 2K 显示器,分辨率 2560×1440,刷新率 165Hz,"
        "面板为 IPS,色域覆盖 99% sRGB,支持 HDR400,支架支持升降与旋转,"
        "接口为 1×DP 1.4 + 2×HDMI 2.0,支持壁挂(100×100mm)。",
    ),
    Passage(
        "viewmax.warranty",
        "ViewMax 27 显示器售后服务卡",
        "ViewMax 27 显示器整机保修三年。屏幕亮点超过三个(含三个)"
        "可申请换新;背光不均、闪烁属于质保范围。屏幕碎裂不在保修范围。",
    ),
    # ---- PowerCore 20000 移动电源 ----
    Passage(
        "powercore.price",
        "PowerCore 20000 移动电源详情页",
        "PowerCore 20000 移动电源,类目:充电配件。售价 249 元,"
        "两件八折。库存 120 件。",
    ),
    Passage(
        "powercore.spec",
        "PowerCore 20000 移动电源规格表",
        "PowerCore 20000 容量为 20000mAh,支持 65W 双向快充,"
        "输出接口为 2×USB-C + 1×USB-A,可同时为三台设备充电,"
        "支持笔记本供电,自重 420g。",
    ),
    Passage(
        "powercore.warranty",
        "PowerCore 20000 移动电源售后服务卡",
        "PowerCore 20000 提供 18 个月质保。电芯鼓包、无法充电属于质保"
        "范围。外壳开裂需确认是否人为;因使用非原装充电器导致的损坏"
        "不在质保范围。",
    ),
    # ---- 通用政策 ----
    Passage(
        "policy.shipping",
        "配送与发货说明",
        "订单在付款后 24 小时内出库,偏远地区(新疆、西藏、内蒙古部分"
        "地区)需额外 3 到 5 天。默认发顺丰陆运,大件商品走专线物流。"
        "节假日期间发货时间顺延。",
    ),
    Passage(
        "policy.return",
        "七天无理由退货说明",
        "自签收之日起七天内,商品保持完好且配件齐全的,可申请无理由"
        "退货。已激活的数码产品、已拆封的耳机(卫生原因)、"
        "定制类商品不支持无理由退货。",
    ),
    Passage(
        "policy.freight",
        "退货运费说明",
        "无理由退货的退回运费由买家承担;因质量问题退货的,"
        "往返运费均由我们承担。退货运费险可在下单时一并购买,"
        "保费为订单金额的 1%。",
    ),
    Passage(
        "policy.invoice",
        "发票说明",
        "所有订单均可开具增值税普通发票,下单时在备注里填写"
        "开票信息即可,发票随货寄出。企业客户可申请增值税专用发票,"
        "需提供营业执照与开票资料,开票周期为三个工作日。",
    ),
]


# ---------------------------------------------------------------------------
# 问题集与标注
#
# 每个可答问题**只标注一条**含答案的段落。这不是偷懒,是为了让「相关」
# 成为一个没有争议的二元量——一条段落要是「有点相关」,那它该不该被阈值
# 留下就变成了判断,而判断没法写进断言。
#
# 代价必须说清楚:真实语料里同一个问题往往有多条段落各自给出一部分答案
# (比如保修问题既在商品售后卡里,也在通用政策里)。这批数据没有这种
# 重叠,所以它算出来的召回是**上界**。
#
# 跑题问题一条标注都不给——它们的「正确答案」是拒答。
# ---------------------------------------------------------------------------

QUERIES: list[tuple[str, frozenset[str]]] = [
    # ---- RX9070 显卡 ----
    ("RX9070 显卡多少钱", frozenset({"rx9070.price"})),
    ("RX9070 的显存是多大", frozenset({"rx9070.spec"})),
    ("RX9070 显卡质保几年", frozenset({"rx9070.warranty"})),
    # ---- K870 机械键盘 ----
    ("K870 键盘卖多少钱", frozenset({"k870.price"})),
    ("K870 用的是什么轴", frozenset({"k870.spec"})),
    ("K870 机械键盘保修多久", frozenset({"k870.warranty"})),
    # ---- AeroBook 14 笔记本 ----
    ("AeroBook 14 售价是多少", frozenset({"aerobook.price"})),
    ("AeroBook 14 有多重", frozenset({"aerobook.spec"})),
    ("笔记本的电池保修多久", frozenset({"aerobook.warranty"})),
    # ---- SoundPeak Pro 耳机 ----
    ("SoundPeak Pro 多少钱", frozenset({"soundpeak.price"})),
    ("这个耳机的降噪深度是多少", frozenset({"soundpeak.spec"})),
    ("SoundPeak Pro 保修几年", frozenset({"soundpeak.warranty"})),
    # ---- ViewMax 27 显示器 ----
    ("ViewMax 27 显示器价格", frozenset({"viewmax.price"})),
    ("显示器是多少分辨率的", frozenset({"viewmax.spec"})),
    ("显示器屏幕有亮点能换吗", frozenset({"viewmax.warranty"})),
    # ---- PowerCore 20000 移动电源 ----
    ("PowerCore 20000 卖多少钱", frozenset({"powercore.price"})),
    ("这个充电宝能充笔记本吗", frozenset({"powercore.spec"})),
    ("充电宝质保多久", frozenset({"powercore.warranty"})),
    # ---- 通用政策 ----
    ("下单后多久发货", frozenset({"policy.shipping"})),
    ("耳机拆封了还能退吗", frozenset({"policy.return"})),
    ("退货的运费谁出", frozenset({"policy.freight"})),
    ("能不能开专票", frozenset({"policy.invoice"})),
    # ---- 跑题:语料里一条都答不了 ----
    ("iPhone 16 Pro 多少钱", frozenset()),
    ("你们卖洗衣机吗", frozenset()),
    ("支持花呗分期吗", frozenset()),
    ("你们的公司地址在哪里", frozenset()),
    ("可以货到付款吗", frozenset()),
    ("有没有以旧换新服务", frozenset()),
]

#: 扫描的候选阈值。低分区铺密——错配与漏检的交错全发生在那里,而决策
#: 恰恰取决于「最后一个还能保住召回率的阈值」落在哪。
DEFAULT_THRESHOLDS = [
    0.0,
    0.02,
    0.05,
    0.08,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
]


# ---------------------------------------------------------------------------
# 扫描:纯函数,离线可测(见 tests/test_rerank_probe.py)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Judged:
    """一条 (查询, 段落) 的打分与标注 —— 扫描的原子单位。"""

    query: str
    passage: str
    relevant: bool
    score: float


@dataclass(frozen=True)
class Row:
    """一个候选阈值下的全部后果。"""

    threshold: float
    #: 段落级:真的含有答案的那批段落里,有多少活过了阈值
    relevant_kept: int
    relevant_total: int
    #: 段落级:不含有答案的那批里,有多少被留了下来
    irrelevant_kept: int
    irrelevant_total: int
    #: 查询级:有答案可查的问题里,有多少还能至少剩一条自己的段落
    answerable_kept: int
    answerable_total: int
    #: 查询级:语料里答不了的问题里,有多少被清干净了
    refused: int
    offtopic_total: int


def scan(
    judged: Sequence[Judged],
    thresholds: Sequence[float],
    *,
    top_n: int | None = None,
) -> list[Row]:
    """对每个候选阈值数一遍会发生什么。

    **`top_n` 在这里是必须的,不是可选的优化。** 生产是
    `rerank(top_n=5)` 先拿 5 条候选、再由阈值过滤(`retrieval._apply_rerank`),
    不传 `top_n` 的表描述的是一个不存在的系统:在那个系统里阈值还能看见
    排在榜外的段落,于是「调低阈值换召回」看起来总是有效。

    两个筛子本身可交换(见模块文档),所以这里传的是**候选集大小**,
    不是执行顺序——`scan(..., top_n=N)` 等价于「先取前 N 名,再过滤」,
    也等价于「先过滤,再取前 N 名」,两者结果相同。
    """
    by_query: dict[str, list[Judged]] = {}
    for item in judged:
        by_query.setdefault(item.query, []).append(item)

    rows: list[Row] = []
    for threshold in thresholds:
        relevant_kept = relevant_total = 0
        irrelevant_kept = irrelevant_total = 0
        answerable_kept = answerable_total = 0
        refused = offtopic_total = 0

        for items in by_query.values():
            for item in items:
                if item.relevant:
                    relevant_total += 1
                else:
                    irrelevant_total += 1

            # 稳定排序:同分时保持输入顺序,避免同一次测量跑出两张不同的表。
            ranked = sorted(items, key=lambda j: -j.score)
            if top_n is not None:
                ranked = ranked[:top_n]
            kept = [j for j in ranked if j.score >= threshold]

            for item in kept:
                if item.relevant:
                    relevant_kept += 1
                else:
                    irrelevant_kept += 1

            has_answer = any(item.relevant for item in items)
            if has_answer:
                answerable_total += 1
                if any(item.relevant for item in kept):
                    answerable_kept += 1
            else:
                offtopic_total += 1
                if not kept:
                    refused += 1

        rows.append(
            Row(
                threshold=threshold,
                relevant_kept=relevant_kept,
                relevant_total=relevant_total,
                irrelevant_kept=irrelevant_kept,
                irrelevant_total=irrelevant_total,
                answerable_kept=answerable_kept,
                answerable_total=answerable_total,
                refused=refused,
                offtopic_total=offtopic_total,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# 采集:联网
# ---------------------------------------------------------------------------


async def collect(reranker, corpus: Sequence[Passage], queries: Sequence[tuple]) -> list[Judged]:
    """每个问题发一次请求,一次带上全部段落。

    每条问题一次调用(不是每对一次):重排接口本来就是「一个查询 + 一串
    文档」的形状,27 个问题就是 27 次调用。
    """
    documents = [p.text for p in corpus]
    judged: list[Judged] = []

    for query, gold in queries:
        hits = await reranker.rerank(query, documents, top_n=len(documents))
        scores = {hit.index: hit.score for hit in hits}

        # **缺一条就炸,不填 0。** 少回来的那条会被当成「0 分,不相关」,
        # 而 0 分意味着它会被所有阈值滤掉——一个让阈值看起来更安全的方向。
        # 这种偏差不会让任何断言变红,只会让结论偏乐观。
        missing = [i for i in range(len(documents)) if i not in scores]
        if missing:
            raise RuntimeError(
                f"重排只返回了 {len(hits)}/{len(documents)} 条,缺下标 {missing[:5]}"
                "——不可用的响应不能当成「不相关」继续往下算"
            )

        for index, passage in enumerate(corpus):
            judged.append(
                Judged(
                    query=query,
                    passage=passage.id,
                    relevant=passage.id in gold,
                    score=scores[index],
                )
            )
    return judged


# ---------------------------------------------------------------------------
# 报告
# ---------------------------------------------------------------------------


def _describe(name: str, scores: list[float]) -> None:
    if not scores:
        print(f"{name} n=0")
        return
    ordered = sorted(scores)
    print(
        f"{name} n={len(scores):<4} min={min(scores):.4f}  "
        f"median={ordered[len(ordered) // 2]:.4f}  max={max(scores):.4f}"
    )


def _table(rows: Sequence[Row], *, title: str) -> None:
    print()
    print(f"### {title}")
    print(
        f"  {'阈值':>6} {'相关段落':>11} {'无关段落':>11} {'可答率':>11} {'跑题清干净':>12}"
    )
    for row in rows:
        answer_rate = row.answerable_kept / row.answerable_total if row.answerable_total else 0.0
        refuse_rate = row.refused / row.offtopic_total if row.offtopic_total else 0.0
        flag = ""
        if row.irrelevant_kept == 0:
            flag += "  零污染"
        if row.answerable_kept < row.answerable_total:
            flag += "  有漏答"
        print(
            f"  {row.threshold:>6.2f} "
            f"{row.relevant_kept:>4}/{row.relevant_total:<6} "
            f"{row.irrelevant_kept:>4}/{row.irrelevant_total:<6} "
            f"{answer_rate:>9.0%}  "
            f"{refuse_rate:>10.0%}{flag}"
        )


def _relevant_ranks(judged: Sequence[Judged]) -> tuple[int, int, int]:
    """含答案的段落排在第几?返回 (第 1 名, 第 2-5 名, 第 6 名及以后)。

    **这决定 `top_n` 会不会误伤,而 `top_n` 和阈值是两个独立的筛子。**
    相关资料要是稳定排第一,那 `top_n` 调到几都无所谓;要是经常掉到第 6,
    那它就是真正的约束——而且调阈值救不回来。
    """
    by_query: dict[str, list[Judged]] = {}
    for item in judged:
        by_query.setdefault(item.query, []).append(item)

    first = middle = late = 0
    for items in by_query.values():
        ranked = sorted(items, key=lambda j: -j.score)
        for position, item in enumerate(ranked, start=1):
            if not item.relevant:
                continue
            if position == 1:
                first += 1
            elif position <= 5:
                middle += 1
            else:
                late += 1
    return first, middle, late


def _verdict(rows: Sequence[Row]) -> None:
    """把表读成一句话:有没有一个阈值同时保住可答率和拒答率。"""
    ordered = sorted(rows, key=lambda r: r.threshold)
    both = [
        r
        for r in ordered
        if r.answerable_kept == r.answerable_total and r.refused == r.offtopic_total
    ]

    print()
    if not both:
        # 两个目标一次都没同时成立。要指出是哪一边先崩的,否则读者只知道
        # 「没有好阈值」,不知道下一步该动的是提示词、分词还是重排模型。
        can_answer = [r for r in ordered if r.answerable_kept == r.answerable_total]
        can_refuse = [r for r in ordered if r.refused == r.offtopic_total]
        if not can_answer:
            print("**可答率一次都没满过** —— 阈值还没开始砍跑题,就已经在砍正确答案了。")
            print("  这种情况下该修的是检索与标注本身,不是阈值。")
        elif not can_refuse:
            print("**跑题问题从来没能被清干净** —— 任何阈值下,总有一个跑题问题")
            print("  能捞到一条不相干的资料。阈值买不到「不知道就说不知道」。")
        else:
            low, high = min(r.threshold for r in can_answer), max(r.threshold for r in can_answer)
            r_low, r_high = min(r.threshold for r in can_refuse), max(r.threshold for r in can_refuse)
            print(f"两边的可行区间不相交:可答率满格在 [{low:.2f}, {high:.2f}],")
            print(f"  跑题清干净在 [{r_low:.2f}, {r_high:.2f}]。")
            print("  **取舍躲不掉**,下面是它有多陡。")
        return

    low, high = both[0], both[-1]
    print(
        f"两个目标同时成立:{low.threshold:.2f} ~ {high.threshold:.2f}"
        f"({len(both)} 个候选点)"
    )
    print(
        f"  可答率 100%({low.answerable_kept}/{low.answerable_total}),"
        f"跑题全部拒答({low.refused}/{low.offtopic_total})。"
    )
    print(
        f"  区间内只剩一个差别 —— 残留的无关段落:{low.irrelevant_kept} 条 → "
        f"{high.irrelevant_kept} 条。"
    )
    print("  **所以取上界**:两个目标都不变,而带进上下文的噪音更少。")

    below = [r for r in both if r.threshold < low.threshold]
    if below:
        print(
            f"  低于 {low.threshold:.2f} 时跑题还能被捞到;高于 {high.threshold:.2f} "
            f"则开始丢答案。"
        )


async def main(*, thresholds: list[float]) -> int:
    print()
    print("=" * 78)
    print("  重排阈值探针")
    print("=" * 78)
    print(f"rerank 模型:{settings.RERANK_MODEL}  backend={settings.RERANK_BACKEND}")
    print(f"生产 top_n :{settings.RERANK_TOP_N}")
    if settings.RERANK_BACKEND != "dashscope":
        print("  ⚠ 当前不是 dashscope 后端 —— 这个结果不能用来定阈值")
    print(f"语料 {len(CORPUS)} 段 / 问题 {len(QUERIES)} 条"
          f"(其中跑题 {sum(1 for _, g in QUERIES if not g)} 条)")

    reranker = build_reranker(settings)
    print(f"\n发起 {len(QUERIES)} 次重排调用……")
    judged = await collect(reranker, CORPUS, QUERIES)

    as_close = getattr(reranker, "aclose", None)
    if as_close is not None:
        await as_close()

    relevant_scores = [j.score for j in judged if j.relevant]
    irrelevant_scores = [j.score for j in judged if not j.relevant]

    print()
    _describe("含答案的段落", relevant_scores)
    _describe("其余段落  ", irrelevant_scores)

    print()
    print("### 分最高的 8 条无关段落(污染的实际来源)")
    print("    它们顶着阈值,决定这个机制能开多低。**看它们像不像真的无关**——")
    print("    标注里的一句「其实也沾点边」,就能把下界整个抬上去。")
    for item in sorted((j for j in judged if not j.relevant), key=lambda j: -j.score)[:8]:
        print(f"  {item.score:.4f}  {item.passage:<20} ← {item.query}")

    print()
    print("### 分最低的 5 条含答案段落(决定阈值能开多高)")
    print("    它们一旦被砍掉,对应的问题就从「答得出」变成「资料中没有」——")
    print("    而用户看不出这是配置造成的。")
    for item in sorted((j for j in judged if j.relevant), key=lambda j: j.score)[:5]:
        print(f"  {item.score:.4f}  {item.passage:<20} ← {item.query}")

    first, middle, late = _relevant_ranks(judged)
    print()
    print(
        f"含答案段落的排名:第 1 名 {first} 条 / 第 2-5 名 {middle} 条 / "
        f"第 6 名及以后 {late} 条"
    )
    if late:
        print(
            f"  **有 {late} 条落在 top_n={settings.RERANK_TOP_N} 之外**,"
            f"它们调阈值救不回来——那要动的是 top_n。"
        )
    else:
        print(f"  全都在 top_n={settings.RERANK_TOP_N} 之内,截断这一步不会误伤。")

    full = scan(judged, thresholds)
    _table(full, title="全量(阈值能看到所有段落 —— 这不是线上发生的事)")
    _verdict(full)

    truncated = scan(judged, thresholds, top_n=settings.RERANK_TOP_N)
    _table(
        truncated,
        title=f"截断到 top_n={settings.RERANK_TOP_N} 后(这才是线上发生的事)",
    )
    _verdict(truncated)

    print()
    print(f"当前配置的阈值:{settings.RERANK_SCORE_THRESHOLD}"
          f"(0.0 = 不过滤,截断机制一次也不生效)")
    print()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="重排阈值探针")
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_THRESHOLDS,
        help="要扫描的阈值列表(默认:0.0 起,低分区更密)",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main(thresholds=args.thresholds)))
