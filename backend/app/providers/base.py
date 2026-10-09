"""可插拔组件的接口定义。

每个能力（LLM / Embedding / Rerank）都抽象成 Protocol，理由有三：

1. **离线可开发**——MockLLM 让整条链路在 API 不可用、额度耗尽时照样跑测试。
2. **云端/本地可切换**——作品集要能讲「同一个接口两套实现」的取舍，不是空谈。
3. **换模型不动检索逻辑**——embedding 从云端换成 bge，检索层一行不用改。

Protocol 而非 ABC：结构化子类型，实现方不需要 import 本模块，测试里塞个
鸭子类型的假对象也能过。
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

# --------------------------------------------------------------------------
# Embedding
# --------------------------------------------------------------------------


@runtime_checkable
class Embedder(Protocol):
    """文本向量化。

    约定：`embed_documents` 与 `embed_query` 产出的向量必须落在同一空间且
    可直接做余弦相似度比较。若某模型对 query/passage 需要不同前缀（如
    bge-zh 系列），由实现内部处理，调用方无感。
    """

    @property
    def dim(self) -> int:
        """向量维度。用于建库校验与缓存键的一部分。"""
        ...

    @property
    def model_id(self) -> str:
        """模型标识，进入 embedding 缓存键。换模型必须让缓存自然失效。"""
        ...

    async def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """批量编码。实现内部负责按服务端上限分批。"""
        ...

    async def embed_query(self, text: str) -> list[float]:
        """单条查询编码。单独开一个方法是因为部分模型 query 侧要加指令前缀。"""
        ...


# --------------------------------------------------------------------------
# Rerank
# --------------------------------------------------------------------------


@dataclass(slots=True)
class RerankHit:
    """重排结果。

    `index` 指向传入 documents 列表的下标而非文档 id——重排服务只认位置，
    映射回 chunk 是调用方的事，这样 Provider 不必知道 chunk 的存在。
    """

    index: int
    score: float
    document: str | None = None


@runtime_checkable
class Reranker(Protocol):
    """候选重排。输入 query + N 条候选，输出按相关性降序的 M 条（M ≤ N）。"""

    @property
    def model_id(self) -> str: ...

    async def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int | None = None,
    ) -> list[RerankHit]: ...


# --------------------------------------------------------------------------
# LLM
# --------------------------------------------------------------------------


@dataclass(slots=True)
class ChatChunk:
    """流式增量。

    `reasoning` 与 `text` 分开：qwen3.7-plus 是思考模型，即使关掉思考，
    Provider 也应能承接 reasoning_content 而不污染正文。

    `usage` 是**可选的**：流式响应里 token 用量取决于网关有没有回声
    （OpenAI 协议的 `stream_options.include_usage`），百炼不保证给。
    拿不到时保持 `None` 而不是自己估一个——用量指标的唯一可信来源是
    服务端，填一个看着精确的估算值比留空更糟。
    """

    text: str = ""
    reasoning: str = ""
    usage: Usage | None = None


@dataclass(slots=True)
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    #: 其中有多少 prompt token 是**从服务端的 prompt 缓存里读出来的**,不是重算的。
    #:
    #: **`None` 表示平台没报这个读数,而不是「命中 0 条」。** 两者含义相反:
    #: `0` 是「平台说这次一条都没命中」,`None` 是「平台没说」。合并成 0 的话,
    #: 面板上会是一条平稳的 0% 命中率曲线——看起来像「缓存一直不工作」,而
    #: 事实是我们根本没在听。这与 `first_token_ms` 用 `None` 表示「一个字都
    #: 没流出来」是同一条理由。
    #:
    #: 它是 `prompt_tokens` 的**子集**,不是额外量:计费口径不变,变的只是
    #: 「这批 token 里有多少是缓存供给的」。
    cached_tokens: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ChatResult:
    """非流式调用的产物。流式调用由调用方自行累积。"""

    content: str
    usage: Usage = field(default_factory=Usage)


@runtime_checkable
class LLMProvider(Protocol):
    """对话模型。只暴露本项目真正需要的两个能力。"""

    @property
    def model_id(self) -> str: ...

    async def acomplete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        """一次性返回，用于查询改写等短任务。"""
        ...

    def astream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        """流式返回，用于答案生成。

        注意是普通 def 返回 AsyncIterator，不是 async def——调用方写
        `async for chunk in llm.astream(...)`，这样「构造」与「迭代」分离，
        异常能在迭代期而非 awaited 期抛出。
        """
        ...
