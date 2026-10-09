"""分块。

分块是整个 RAG 链路里**最不该被随手对付**的一步:切得太碎,一句话被腰斩,
检索到的片段没法作答;切得太整,一个 chunk 里混着三四个商品,重排模型
分不清用户问的是哪个,引用也指不准具体位置。

策略是**结构优先**:能识别结构的格式(Markdown 标题、表格行)先按结构切,
结构粒度不够或识别不了再退回定长切分。
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

# 中文分隔符层级:段落 → 行 → 句 → 分句 → 短语 → 字。
#
# **绝对不要把英文句点 `.` 放进来。** 中文技术文档里 `0.95`、`v1.2.3`、
# `e.g.`、`RX 9070 GRE 2.5GHz` 到处都是,按 `.` 切会把数字和小数点切开,
# 检索「支持 0.95 相似度」时原文已经变成了 `0` 和 `95` 两个片段。
# 中文的句末标点是全角的「。」,它已经覆盖了断句需求。
CHINESE_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", "、", " ", ""]

# 中日常韩统一表意文字 + 假名 + 谚文。这些字符在 BPE 分词器下通常
# 一字符≈一 token;拉丁字母约 4 字符≈一 token。
# 用转义码点而不是字面字符——字面写法在编辑器/编码转换里很容易被悄悄改掉。
_CJK_RE = re.compile(
    "["
    "぀-ヿ"  # 平假名 + 片假名
    "㐀-䶿"  # CJK 扩展 A
    "一-鿿"  # CJK 基本区
    "豈-﫿"  # CJK 兼容表意文字
    "가-힯"  # 谚文音节
    "]"
)


def estimate_tokens(text: str) -> int:
    """粗估 token 数。

    **为什么要自己估**:`langchain-openai` 的 `get_num_tokens_from_messages()`
    对 qwen 系列未实现,会直接抛 `NotImplementedError`——因为它内部要拿
    tiktoken 的编码表,而那套表对中文的分法与 qwen 自己的分词器并不一致。
    用错的分词器算预算,比用粗略但诚实的估算更危险。

    这个估算只用于**上下文预算控制**,不用于计费。真金白银的 token 用量
    一律以 API 响应里的 `usage` 字段为准。
    """
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    # -(-a // b) 是向下取整的整数上取整,避免引入数学库
    return cjk + -(-other // 4)


@dataclass(frozen=True, slots=True)
class ChunkPiece:
    """一个待入库的分块。index 在所属文档内连续且从 0 开始。"""

    index: int
    content: str
    meta: dict[str, Any] = field(default_factory=dict)


def _make_splitter(chunk_size: int, chunk_overlap: int) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        separators=CHINESE_SEPARATORS,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # "end" 让分隔符跟在上一段末尾,而不是成为下一段开头的「。」。
        # 引用卡片会直接展示 chunk 原文,开头挂一个句号很难看,
        # 而且会让 BM25 把「。」当成一个 token 计进去。
        keep_separator="end",
        length_function=estimate_tokens,
    )


def split_text(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    meta: Mapping[str, Any] | None = None,
) -> list[ChunkPiece]:
    """按中文标点层级定长切分。"""
    base = dict(meta or {})
    pieces = _make_splitter(chunk_size, chunk_overlap).split_text(text)
    return [
        ChunkPiece(index=i, content=p.strip(), meta={**base, "piece_index": i})
        for i, p in enumerate(pieces)
        if p.strip()
    ]


def _breadcrumb(headers: Mapping[str, str], order: Sequence[str]) -> str:
    """把标题层级拼成「商品手册 › 售后政策 › 退换货」这样的路径。

    单看 `{'h3': '退换货'}` 这一层,检索和生成都不知道这是哪个商品的退换货。
    """
    parts = [headers[k] for k in order if headers.get(k)]
    return " › ".join(parts)


def split_markdown(
    text: str,
    *,
    chunk_size: int,
    chunk_overlap: int,
    meta: Mapping[str, Any] | None = None,
    header_levels: Sequence[tuple[str, str]] = (
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
        ("####", "h4"),
    ),
) -> list[ChunkPiece]:
    """按 Markdown 标题切分,再把超长章节二次切分。

    **为什么必须做二次切分**:`MarkdownHeaderTextSplitter` 不继承
    `TextSplitter`,**没有 chunk_size 上限**——一个没有子标题的万字章节
    会被原样吐成一整块。直接入库的话,这一块既装不进上下文预算,
    向量也会被稀释成一个「什么都沾一点」的平均值,检索质量断崖式下跌。
    """
    base = dict(meta or {})
    order = [name for _, name in header_levels]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=list(header_levels),
        # strip_headers=True:标题从正文里摘掉,只留在 metadata。
        #
        # 这里看起来与「标题文字本身也有检索价值」矛盾,其实不是——下面
        # 会把**标题路径拼成面包屑前置回正文**。区别在于:
        #   strip_headers=False  → 正文里是 "## 退换货\n自签收之日起…"
        #                           只有当前这一层,且带着 markdown 记号
        #   面包屑前置            → 正文里是 "商品手册 › 售后政策 › 退换货\n自签收之日起…"
        #                           有完整祖先链,没有记号
        # 后者信息更多、噪音更少。两个都开只会让标题文字在正文里出现两遍。
        strip_headers=True,
    )

    sections = md_splitter.split_text(text)
    inner = _make_splitter(chunk_size, chunk_overlap)

    pieces: list[ChunkPiece] = []
    for section in sections:
        headers = {k: v for k, v in section.metadata.items() if k in order}
        breadcrumb = _breadcrumb(headers, order)
        body = section.page_content.strip()
        if not body:
            continue

        # 面包屑前置到正文。这样即使这个 chunk 被切得只剩半句话,
        # 它自己也带着「属于哪个商品、哪个章节」——向量和 BM25 都用得上。
        prefix = f"{breadcrumb}\n" if breadcrumb else ""
        section_meta = {**base, **headers, "breadcrumb": breadcrumb}

        for piece in inner.split_text(body):
            piece = piece.strip()
            if not piece:
                continue
            pieces.append(
                ChunkPiece(
                    index=len(pieces),
                    content=f"{prefix}{piece}",
                    meta={**section_meta, "piece_index": len(pieces)},
                )
            )

    return pieces


# ---------------------------------------------------------------------------
# 表格类数据:Excel / CSV
# ---------------------------------------------------------------------------

# 常见表头的中文别名 → 归一化字段名。
# 之所以要做映射而不是直接用原表头:同一份商品表在 A 商家叫「商品名称」、
# 在 B 商家叫「品名」,归一化之后元数据过滤(按 category 筛)才写得出来。
COLUMN_ALIASES: dict[str, str] = {
    "商品名称": "product_name",
    "商品名": "product_name",
    "品名": "product_name",
    "产品名称": "product_name",
    "名称": "product_name",
    "型号": "model",
    "商品型号": "model",
    "产品型号": "model",
    "类目": "category",
    "分类": "category",
    "商品分类": "category",
    "品类": "category",
    "价格": "price",
    "售价": "price",
    "商品价格": "price",
    "单价": "price",
    "库存": "stock",
    "颜色": "color",
    "规格": "spec",
    "材质": "material",
    "产地": "origin",
    "保修": "warranty",
    "描述": "description",
    "商品描述": "description",
    "卖点": "highlights",
}


def normalize_columns(columns: Sequence[str]) -> dict[str, str]:
    """原表头 → 归一化字段名。未识别的表头原样保留(去掉首尾空白)。"""
    return {c: COLUMN_ALIASES.get(c.strip(), c.strip()) for c in columns}


# 归一化字段名 → 渲染进句子的中文标签。
#
# 必须给中文标签,不能直接用 `product_name RX9070` 这种英文键。原因不是好看:
# 用户问的是「这个显卡多少钱」,而 `price` 这个词在中文问句里根本不出现。
# 渲染成「价格 4299 元」之后,字面词和语义才同时对上。
FIELD_LABELS: dict[str, str] = {
    "product_name": "商品",
    "model": "型号",
    "category": "类目",
    "price": "价格",
    "stock": "库存",
    "color": "颜色",
    "spec": "规格",
    "material": "材质",
    "origin": "产地",
    "warranty": "保修",
    "description": "描述",
    "highlights": "卖点",
}

# 数值字段带上单位,否则「4299」和「4299 元」在 BM25 里不是同一个词。
FIELD_UNITS: dict[str, str] = {"price": "元", "stock": "件"}

# 需要数值化的字段。
#
# **表格解析出来的一切都是字符串**——CSV 本来就没有类型,Excel 也被
# `_rows_from_matrix` 统一成了 `str`。而这两列要参与 Chroma 的
# `$gte` / `$lte` 范围过滤:元数据里存字符串 "4299",价格区间筛选要么
# 类型报错、要么按字典序比较,两种都是**静默失效**——过滤条件看着生效了,
# 返回的却是错的集合。所以必须在这里转成数字。
_NUMERIC_FIELDS = frozenset({"price", "stock"})

_CURRENCY_CHARS = "¥￥$€£"
_NUMBER_RE = re.compile(r"^-?\d+(\.\d+)?$")


def coerce_number(value: Any) -> Any:
    """「看起来是数字」的字符串 → int/float;转不了就**原样返回**。

    原样返回这一条是必须的:价格列出现「面议」「暂无报价」是真实场景,
    把它们丢掉等于把那行商品从知识库里删掉,而用户只会觉得「问某个型号
    怎么没有」。
    """
    if not isinstance(value, str):
        return value
    cleaned = value.strip().lstrip(_CURRENCY_CHARS).replace(",", "").replace(" ", "")
    if not _NUMBER_RE.match(cleaned):
        return value
    number = float(cleaned)
    return int(number) if number.is_integer() else number


def normalize_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """原始表格行 → 「入库形状」。这一步是可重入的(幂等)。

    做三件事:

    1. **表头归一化**:`商品名称` → `product_name`。不做的话
       `FIELD_LABELS` / `FIELD_UNITS` 全部匹配不上,渲染出来是
       「商品名称 RX9070,价格 4299」——没有「商品」前缀、没有「元」,
       而且元数据里既没有 `category` 也没有 `price`,**类目/价格预过滤
       整个失效**(`build_where` 支持它,但永远没有数据喂给它)。
    2. **数值化**:见 `_NUMERIC_FIELDS`。
    3. **丢掉 `_` 开头的内部键**:`_sheet` 是**文件级**信息,逐行渲染会
       在正文里凭空多出「_sheet Sheet1」。它由调用方提升到 meta。
    """
    out: dict[str, Any] = {}
    for raw_key, value in row.items():
        key = raw_key.strip()
        if key.startswith("_"):
            continue
        key = COLUMN_ALIASES.get(key, key)
        out[key] = coerce_number(value) if key in _NUMERIC_FIELDS else value
    return out


def _render_row(row: Mapping[str, Any]) -> str:
    """渲染**已归一化**的行。外部不要直接调,用 `row_to_sentence`。"""
    parts = [
        f"{FIELD_LABELS.get(key, key)} {value}{FIELD_UNITS.get(key, '')}"
        for key, value in row.items()
        if value not in (None, "", "nan")
    ]
    return ",".join(parts) + "。" if parts else ""


def row_to_sentence(row: Mapping[str, Any]) -> str:
    """单行 → 一句可检索的中文。原始表头也能直接喂进来。

        {"商品名称": "RX9070", "类目": "显卡", "价格": "4299"}
          → 「商品 RX9070,类目 显卡,价格 4299元。」

    归一化在这里面做,而不是要求调用方先调 `normalize_columns`——
    那个约定**没人会记得**,而且忘了不会报错,只会让检索悄悄变差
    (这正是它曾经坏掉的方式)。
    """
    return _render_row(normalize_row(row))


def rows_to_sentences(
    rows: Sequence[Mapping[str, Any]],
    *,
    rows_per_chunk: int = 1,
    source: str = "",
) -> list[ChunkPiece]:
    """把表格行转成可检索的自然语言句子,默认**一行一个分块**。

    **为什么不能把表格原样喂进去**:一行 `RX9070 | 显卡 | 4299 | 有货`
    在向量空间里几乎是噪音——没有动词、没有连接词,和「这个显卡多少钱」
    的语义距离很远。改写成「商品 RX9070,类目 显卡,价格 4299 元。」
    之后,问句和资料才落在同一个语义邻域里。这是表格类知识库检索
    质量的**决定性一步**,不是美化。

    **默认一行一块,是因为元数据只能取「这一块」的值。** 一个分块只有一份
    `category` / `price`,多行合块时它们只能是块内第一行的——于是一份
    混着显卡和显示器的商品表,按「只看显示器」过滤会**漏掉**显示器那些行
    (它们被算成了「显卡」)。不报错、只是少返回几条,是最难发现的那类
    缺陷。类目/价格预过滤是本项目的卖点,精确性优先。

    代价是分块数变多、单行信息量低。补回来的是:①向量+BM25 混合检索——
    单行里的型号能被关键词路精确抓住,而语义泛化由向量路负责;
    ②上面的句式改写让单行不再是噪音。

    传 `rows_per_chunk>1` 可以换回「块内同质、上下文更丰富」的取向,
    但**那个块就只按第一行的元数据被过滤**——这个取舍由调用方承担。
    """
    pieces: list[ChunkPiece] = []
    for start in range(0, len(rows), rows_per_chunk):
        lines: list[str] = []
        first_row: dict[str, Any] = {}
        sheet = ""

        for raw_row in rows[start : start + rows_per_chunk]:
            row = normalize_row(raw_row)
            sentence = _render_row(row)
            if not sentence:
                continue
            lines.append(sentence)
            if not first_row:
                first_row = row
                sheet = str(raw_row.get("_sheet") or "")

        if not lines:
            continue

        # 元数据取本块**第一行**的值,用于 Chroma 服务端预过滤。
        # 一个分块只有一份元数据,所以**块内多行时按第一个过滤并不精确**——
        # 这正是默认 rows_per_chunk=1 的原因(见函数文档)。调用方显式传了
        # 更大的值时,是自己选择了「块内容更丰富」而放弃「过滤精确」。
        meta: dict[str, Any] = {"source": source, "row_start": start}
        if sheet:
            # 从行里剥出来的 `_sheet` 提升到这里。多工作表的知识库里,
            # 「这个参数在哪张表上」是有用的定位信息,不该直接扔掉。
            meta["sheet"] = sheet
        for key in ("product_name", "category", "model"):
            if first_row.get(key) not in (None, ""):
                meta[key] = first_row[key]
        # 数值化之后这里才可能成立。存成字符串的话,Chroma 的
        # `{"price": {"$gte": 1000}}` 筛不出任何东西。
        if isinstance(first_row.get("price"), (int, float)):
            meta["price"] = first_row["price"]

        pieces.append(
            ChunkPiece(
                index=len(pieces),
                content="\n".join(lines),
                meta={**meta, "piece_index": len(pieces)},
            )
        )

    return pieces
