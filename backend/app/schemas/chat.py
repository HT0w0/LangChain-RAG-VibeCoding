"""问答接口的请求模型。

响应不走 Pydantic——它是 SSE 事件流,由 `api/routers/chat.py` 直接编码。
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

# 2000 字的上限不是随便定的:它要能装下一段粘贴进来的商品描述或一整段
# 用户抱怨,同时挡住「把整篇文档贴进来当问题」这种用法——那会让检索到的
# 上下文和问题本身一起撑爆预算,而模型面对的是一团无法检索的噪音。
MAX_QUESTION_LENGTH = 2000


class ChatFilters(BaseModel):
    """检索期的结构化预过滤(电商场景的类目 / 价格筛选)。

    只在**分块元数据里真的有这些字段**时才有意义——纯文本和 PDF 大多
    提取不出类目和价格,此时过滤会把它们全部排除掉。这是刻意的:
    `matches_filters` 对缺失字段一律判为「不匹配」,否则过滤对大部分
    语料形同虚设。
    """

    category: str | None = None
    price_min: float | None = None
    price_max: float | None = None


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_LENGTH)

    # `None` 表示**不限制范围**,即「全部知识库」——这是问答页的默认选择。
    # 空列表则是另一个意思:**一个都不许看**,会直接短路成空结果。
    # 两者语义相反,所以默认值是 None 而不是 []。
    kb_ids: list[int] | None = None

    filters: ChatFilters | None = None

    # 带上它 = 这一问属于某个会话:会读取历史做指代消解,也会把问答落库。
    # 不带 = **随口问一句**,什么历史都不读、什么也不留下。
    #
    # 两种模式都要有。强制会话的话,「我就想快速查一下价格」也要先在库里
    # 建一条记录,而那条记录用户永远不会再打开——历史列表会被这种一次性
    # 会话塞满,真正要找回的对话反而被挤下去。
    session_id: int | None = None

    @field_validator("question")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        # `min_length=1` 挡不住 "   ":它长度够,但检索它只会得到一堆噪音,
        # 然后让模型对着一句空话作答。空白串必须在进链路之前就拒掉。
        stripped = value.strip()
        if not stripped:
            raise ValueError("问题不能为空")
        return stripped
