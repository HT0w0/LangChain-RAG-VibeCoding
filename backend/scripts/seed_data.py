"""种子数据:一套仿真商品资料,让刚克隆下来的人立刻有东西可问。

**没有它,这个项目打开就是一个空壳。** 知识库列表是空的、问答永远是
「资料中没有找到相关内容」,而那不是「系统坏了」的样子——是「系统还没
内容」的样子,两者在界面上长得一模一样。新用户的第一反应会是去翻文档,
而文档里写的又是「上传商品资料」,于是他要先自己编几份 CSV 才能看到
这个产品真实的模样。

**走的是和上传接口完全同一条路径**(`store_upload` → `DocumentRepository.create`
→ `IngestService.ingest_document`),不另写一份「直接插库」的捷径。
捷径能跑,但它绕开了分块、向量化、代次失效——于是「种子数据能用、
上传的数据不能用」这种偏差永远不会被发现。

**编码器用的是配置里的那一个**,不是替身。这一点不能省:向量必须和
问答时用来编码问题的模型是同一个,否则检索永远召不回任何东西,而
表现只是「答得不准」,看不出是模型对不上。

用法:
    python backend/scripts/seed_data.py              # 已存在就跳过
    python backend/scripts/seed_data.py --reset      # 先删掉同名知识库再建

**`--reset` 会删数据。** 它只删这一个知识库(按名字),不碰别人的会话、
账号或其他知识库,但那个库里的文档和索引是不可恢复的。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import Settings  # noqa: E402
from app.core.logging import setup_logging  # noqa: E402

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover
        pass

SEED_KB_NAME = "示例商品资料"
SEED_KB_DESCRIPTION = "种子数据:六件商品 + 售后/配送/FAQ,用来演示引用溯源"

# ---------------------------------------------------------------------------
# 语料
#
# 刻意混了四种格式,**因为它们走的是四条不同的解析与分块路径**:
#   csv  → rows_to_sentences(参数表转成可检索的句子)
#   md   → MarkdownHeaderTextSplitter(标题层级进元数据)
#   txt  → RecursiveCharacterTextSplitter
#   html → beautifulsoup 抽正文
# 全是同一种格式的话,另外三条路径在演示里一次都不会被走到。
#
# 内容也刻意留了「答不了」的余地:商品只有六件,而问题可以是无穷的。
# 「资料中没有找到相关内容」这句话只有靠语料里真的没有才能演示出来。
# ---------------------------------------------------------------------------

CSV = """商品名称,类目,价格,库存,质保期
RX9070 显卡,显卡,4299,12,三年
K870 机械键盘,键盘,599,48,两年
AeroBook 14 笔记本,笔记本,5299,7,整机两年
SoundPeak Pro 耳机,耳机,899,65,一年
ViewMax 27 显示器,显示器,1299,30,三年
PowerCore 20000 移动电源,充电配件,249,120,18 个月
"""

GPU_MD = """# RX9070 显卡

## 核心规格

- 显存:16GB GDDR6,位宽 256bit
- 视频输出:3 × DisplayPort 2.1 + 1 × HDMI 2.1
- 整卡功耗:220W,建议搭配 650W 及以上电源
- 尺寸:长 320mm,占用两个 PCIe 槽位

## 价格与库存

官方指导价 4299 元,当前促销价 3999 元,活动截止到本月底。
库存 12 件,付款后当天出库。

## 兼容性说明

支持 PCIe 4.0 x16 接口,向下兼容 PCIe 3.0。机箱长度需要预留 330mm
以上,否则装不进去。显卡出厂即附带双 8pin 转接线。
"""

KEYBOARD_MD = """# K870 机械键盘

## 核心规格

- 布局:87 键紧凑型(无小键盘区)
- 轴体:凯华 Box 白轴,段落轴,支持热插拔
- 键帽:PBT 二色成型,不易打油
- 连接:有线 Type-C,支持全键无冲
- 背光:单色白色

## 价格与库存

售价 599 元,现有满 500 减 50 活动,到手价 549 元。库存 48 把。

## 使用建议

热插拔设计意味着换轴不需要焊接,用附带的拔轴器直接拔插即可。
原厂轴体保修两年,自行更换的轴体不在保修范围内。
"""

LAPTOP_MD = """# AeroBook 14 笔记本

## 核心规格

- 屏幕:14 英寸 2.8K 120Hz
- 处理器:8 核 16 线程
- 内存:标配 16GB LPDDR5,高配版 32GB
- 存储:512GB PCIe 4.0 固态硬盘
- 电池:70Wh
- 重量:1.35kg
- 接口:2 × USB-C(支持 PD 充电)+ 2 × USB-A + HDMI

## 价格与库存

标配版售价 5299 元,高配版(32GB 内存)售价 6299 元。
标配版库存 7 台;高配版缺货,预计下月中旬到货。

## 充电说明

USB-C 口支持 PD 协议,可用 65W 及以上氮化镓充电器供电,
出差时不必带原装适配器。但满载运行时功率会超过 65W,
那时电池会缓慢掉电。
"""

WARRANTY_MD = """# 售后与保修政策

## 保修期限

不同类目的保修期不一样,下单前请留意:

- 显卡:三年质保,自签收之日起算
- 键盘:两年质保
- 笔记本:整机两年,主要部件(主板、屏幕、处理器)三年,电池一年
- 耳机:一年质保
- 显示器:三年整机保修
- 移动电源:18 个月质保

## 保修范围

质保期内非人为损坏免费维修。**以下情况不在保修范围内**:

1. 进液、私自拆解、外力破损
2. 因使用非原装充电器导致的损坏
3. 耳塞套等消耗品
4. 键帽磨损

## 换新政策

显示器屏幕亮点超过三个(含三个)可申请换新。其余商品的换新
需经检测确认存在质量问题。
"""

SHIPPING_TXT = """配送与退换货说明

一、发货时效

订单在付款后 24 小时内出库。偏远地区(新疆、西藏、内蒙古部分地区)
需额外 3 到 5 天。默认发顺丰陆运,大件商品走专线物流。节假日期间
发货时间顺延,具体以商品页公告为准。

二、七天无理由退货

自签收之日起七天内,商品保持完好且配件齐全的,可以申请无理由退货。
以下商品不支持无理由退货:

- 已激活的数码产品(笔记本激活后视为已使用)
- 已拆封的耳机(卫生原因)
- 定制类商品

三、退货运费

无理由退货的退回运费由买家承担;因质量问题退货的,往返运费均由
我们承担。退货运费险可在下单时一并购买,保费为订单金额的 1%。

四、发票

所有订单均可开具增值税普通发票,下单时在备注里填写开票信息即可,
发票随货寄出。企业客户可申请增值税专用发票,需提供营业执照与
开票资料,开票周期为三个工作日。
"""

FAQ_HTML = """<!doctype html>
<html lang="zh-CN">
<head><meta charset="utf-8"><title>常见问题</title></head>
<body>
  <h1>常见问题</h1>

  <h2>下单之后多久能收到?</h2>
  <p>付款后 24 小时内出库,顺丰陆运一般 1 到 3 天到达。</p>

  <h2>可以开发票吗?</h2>
  <p>可以。普票随货寄出,专票需要提供营业执照,三个工作日内开出。</p>

  <h2>支持花呗分期吗?</h2>
  <p>暂不支持。</p>

  <h2>商品有质量问题怎么办?</h2>
  <p>七天之内可以直接申请退货,往返运费由我们承担;超过七天
  走保修流程,联系客服寄回检测。</p>

  <h2>怎么联系客服?</h2>
  <p>工作日 9:00 到 18:00 在线,也可以在订单详情页留言。</p>
</body>
</html>
"""

DOCUMENTS: list[tuple[str, bytes]] = [
    ("商品参数对照表.csv", CSV.encode()),
    ("RX9070 显卡产品说明.md", GPU_MD.encode()),
    ("K870 机械键盘产品说明.md", KEYBOARD_MD.encode()),
    ("AeroBook 14 笔记本产品说明.md", LAPTOP_MD.encode()),
    ("售后与保修政策.md", WARRANTY_MD.encode()),
    ("配送与退换货说明.txt", SHIPPING_TXT.encode()),
    ("常见问题 FAQ.html", FAQ_HTML.encode()),
]


async def main(*, reset: bool) -> int:
    from app.core.config import get_settings
    from app.db.base import database
    from app.db.repositories.document import DocumentRepository
    from app.db.repositories.knowledge_base import KnowledgeBaseRepository
    from app.db.repositories.user import UserRepository
    from app.services.auth import AuthService
    from app.services.ingest import build_ingest_service, store_upload
    from app.services.parsing import classify

    # `get_settings()` 就是应用自己在用的那个单例(lru_cache),所以种子
    # 写进的库、用的编码器,和问答时读的那个**必然是同一套**。
    # 不要在这里 `Settings()` 新建一个——那样两边只要有一处配置不同
    # (比如 .env 改动没重启),种子向量就会和检索对不上。
    settings: Settings = get_settings()
    settings.ensure_dirs()
    # **脚本要能独立跑,所以自己建表。** 不建的话,在一个从没启动过
    # 后端的目录上跑它会得到 `no such table`——而那看起来像脚本写错了,
    # 不像「后端还没起过」。`create_all` 是幂等的,已经有表时什么都不做。
    await database.create_all()

    print()
    print("=" * 72)
    print("  种子数据")
    print("=" * 72)
    print(f"知识库 : {SEED_KB_NAME}")
    print(f"文档   : {len(DOCUMENTS)} 份")
    print(f"编码器 : {settings.EMBED_MODEL} (backend={settings.EMBED_BACKEND})")
    print(f"数据库 : {settings.sqlite_path}")
    if settings.EMBED_BACKEND != "dashscope":
        print("  ⚠ 当前不是 dashscope 后端 —— 种子向量与真实问答对不上,只能算演示骨架")
    print()

    async with database.session() as session:
        kbs = KnowledgeBaseRepository(session)
        existing = await kbs.get_by_name(SEED_KB_NAME)
        if existing is not None:
            if not reset:
                print(f"「{SEED_KB_NAME}」已存在(id={existing.id},{existing.doc_count} 篇文档)。")
                print("要重建请加 --reset(会先删掉这个库)。")
                return 0
            print(f"删除已存在的「{SEED_KB_NAME}」(id={existing.id})……")
            await build_ingest_service(session, settings).delete_knowledge_base(existing)
            await session.commit()

        admin = await UserRepository(session).get_by_username(settings.ADMIN_USERNAME)
        if admin is None:
            # **调的是启动时用的同一个函数**(`main.lifespan` 也走它),
            # 不是脚本自己插一个用户。自己插会绕过 argon2 的参数与角色规则,
            # 而且「脚本建的 admin」和「应用建的 admin」迟早会长得不一样。
            # 它本身幂等:已存在就原样返回,不会重置密码。
            print(f"管理员「{settings.ADMIN_USERNAME}」不存在,按启动流程创建一个。")
            admin = await AuthService(session, settings).ensure_admin(
                settings.ADMIN_USERNAME, settings.ADMIN_PASSWORD
            )

        kb = await kbs.create(
            name=SEED_KB_NAME, description=SEED_KB_DESCRIPTION, created_by=admin.id
        )
        await session.commit()
        print(f"已建知识库 id={kb.id}")

        repo = DocumentRepository(session)
        created = []
        for filename, data in DOCUMENTS:
            # 和上传接口一样先按扩展名挡一道。这里永远不会失败(文件名
            # 是写死的),但**读起来是同一套规则**——接口改了,这里会跟着红。
            classify(filename)
            stored = store_upload(
                data, upload_dir=settings.resolved_upload_dir, original_filename=filename
            )
            doc = await repo.create(
                kb_id=kb.id,
                filename=filename,
                stored_path=str(stored.path),
                file_hash=stored.file_hash,
                mime=None,
                size=stored.size,
                uploaded_by=admin.id,
            )
            created.append(doc)
            print(f"  + {filename:<28} {stored.size:>6} 字节")

        await kbs.apply_delta(kb, docs=len(created))
        await session.commit()

        print()
        print("解析 + 分块 + 向量化……")
        service = build_ingest_service(session, settings)
        total_chunks = 0
        failures = 0
        for doc in created:
            chunks = await service.ingest_document(doc)
            total_chunks += chunks
            status = "OK  " if doc.status == "ready" else "FAIL"
            detail = f"{chunks} 块" if chunks else (doc.error or "")
            print(f"  [{status}] {doc.filename:<28} {detail}")
            if doc.status != "ready":
                failures += 1

    print()
    if failures:
        print(f"**{failures} 份文档没能入库。** 上面那几行有原因。")
        return 1

    print(f"完成:{len(created)} 份文档 / {total_chunks} 个分块。")
    print()
    print("接下来:")
    print("  1. 启动后端  uvicorn app.main:app --app-dir backend")
    print("  2. 启动前端  cd frontend && npm run dev")
    print(f"  3. 用 {settings.ADMIN_USERNAME} 登录,在问答页选上「{SEED_KB_NAME}」")
    print("  4. 试试这几句:")
    print("       RX9070 显卡多少钱          → 应召回到参数表和产品说明两处")
    print("       K870 键盘保修多久          → 保修期散在两份资料里,看引用怎么合")
    print("       支持花呗分期吗             → 答案在 FAQ 里,走的是 HTML 解析那条路")
    print("       你们卖洗衣机吗             → 应回答「资料中没有找到相关内容」")
    print("     最后一句答「没有找到」是**正确**的,不是故障——语料里确实没有。")
    print()
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="写入示例商品资料")
    parser.add_argument(
        "--reset",
        action="store_true",
        help=f"先删除同名知识库(「{SEED_KB_NAME}」)再重建,该库中的数据不可恢复",
    )
    args = parser.parse_args()
    # 和应用启动同一条设置(见 `main.lifespan`)。脚本的输出里要能看到
    # 摄取过程的失败原因,所以不加 `json_output`——那是给日志采集器的。
    from app.core.config import get_settings

    _settings = get_settings()
    setup_logging(debug=_settings.DEBUG, json_output=False)
    raise SystemExit(asyncio.run(main(reset=args.reset)))
