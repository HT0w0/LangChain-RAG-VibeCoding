"""上下文组装与 token 预算。

检索的结果最终要变成 prompt 里的一小段文字。这一步有两个不能出错的地方,
而它们出错时**都不会报错**:

1. **编号与来源必须对齐。** 正文里的 `[1]` 指向 `used[0]`,前端据此渲染
   来源卡片。编号错位时答案读起来完全正常,只是引用指向了别的资料——
   用户没有任何办法发现。
2. **预算截断不能截掉最相关的。** 检索结果按相关性降序,尾部本来就更像
   噪音;反过来丢头部,最相关的资料反而进不了 prompt。

**为什么自己估 token 而不调模型计数**:`ChatOpenAI.get_num_tokens_from_messages()`
对 qwen 未实现(抛 `NotImplementedError`),而 `trim_messages(token_counter=...)`
正需要它。估算偏保守是可以接受的——浪费几十个 token,好过让实际 prompt
超出上限被服务端截断,而截断是静默的。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# CJK 统一表意文字 + 扩展 A + 兼容表意文字。够覆盖中文商品资料。
_CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]")

_SEPARATOR = "\n\n"


def estimate_tokens(text: str) -> int:
    """粗略估算 token 数。**刻意偏保守**(宁可高估)。

    中文一个字大致一个 token;ASCII 大致 4 个字符一个。按英文的 4:1 去估
    中文会严重低估——实际 prompt 远超预算,请求被服务端截断,而截断是
    静默的:模型只是「少看到几段资料」,然后照常作答。
    """
    if not text:
        return 0
    cjk = len(_CJK_RE.findall(text))
    other = len(text) - cjk
    return cjk + (other + 3) // 4


@dataclass(frozen=True, slots=True)
class RetrievedChunk:
    """检索链路里流通的候选,携带引用溯源所需的**全部**信息。

    打分明细(vector_rank / keyword_rank / rerank_score ...)是刻意保留的:
    作品集要回答「关键词这一路到底有没有贡献」「这条是谁召回的」,
    而这些字段在融合后就再也算不出来了。
    """

    chunk_id: int
    content: str
    kb_id: int
    doc_id: int
    doc_name: str
    kb_name: str = ""
    chunk_index: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    vector_rank: int | None = None
    keyword_rank: int | None = None
    vector_similarity: float | None = None
    bm25_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    @property
    def category(self) -> str | None:
        value = self.meta.get("category")
        return str(value) if value else None

    @property
    def price(self) -> float | None:
        value = self.meta.get("price")
        return float(value) if isinstance(value, (int, float)) else None


@dataclass(frozen=True, slots=True)
class ContextBundle:
    """组装结果。

    `used[i]` 对应正文里的 `[i+1]`。`dropped` 是**未能进入上下文的输入**,
    包含两种原因:超出预算,或正文与已选中的重复(同一份资料被索引了两次)。
    分开统计会让接口更啰嗦,而排查「这条为什么没进去」时两种原因都指向
    同一次组装,合在一起够用。
    """

    text: str
    used: list[RetrievedChunk]
    dropped: list[RetrievedChunk]
    tokens: int


def render_chunk(index: int, piece: RetrievedChunk) -> str:
    """把一个分块渲染成 prompt 里的一段资料。`index` 从 1 开始。"""
    label = f"来源:{piece.doc_name}"
    heading = piece.meta.get("source_heading")
    if heading:
        # 标题路径告诉模型「这段在文档的哪一节」,归因更准。
        label += f" · {heading}"
    return f"[{index}] {label}\n{piece.content}"


def assemble_context(
    chunks: list[RetrievedChunk] | tuple[RetrievedChunk, ...],
    *,
    max_tokens: int,
) -> ContextBundle:
    """按相关性降序装入预算,返回渲染好的资料块与引用映射。

    **放不下的分块是「跳过」而不是「中断」。** 若遇到装不下的就 break,
    一条 5000 token 的分块会把整个上下文清空——模型手里一条资料都没有,
    却仍然会照常作答。跳过它、继续尝试后面的小分块,才是想要的行为。
    """
    used: list[RetrievedChunk] = []
    dropped: list[RetrievedChunk] = []
    parts: list[str] = []
    total = 0
    seen: set[str] = set()

    for piece in chunks:
        key = piece.content.strip()
        # 空正文和重复正文都直接丢弃。重复留两条会平白占掉一半预算,
        # 还会让模型以为「有两个来源都这么说」,语气比实际更有把握。
        if max_tokens <= 0 or not key or key in seen:
            dropped.append(piece)
            continue
        seen.add(key)

        rendered = render_chunk(len(used) + 1, piece)
        cost = estimate_tokens(rendered)
        if parts:
            cost += estimate_tokens(_SEPARATOR)
        if total + cost > max_tokens:
            dropped.append(piece)
            continue

        parts.append(rendered)
        used.append(piece)
        total += cost

    text = _SEPARATOR.join(parts)
    # 用**实际渲染结果**再估一次,而不是直接返回累加的 total:
    # 估算函数的整除取整不是严格可加的,真正的 prompt 长度以后者为准。
    # 累加的 total 是保守上界,所以不会出现「实际超了但检查没拦住」。
    return ContextBundle(text=text, used=used, dropped=dropped, tokens=estimate_tokens(text))
