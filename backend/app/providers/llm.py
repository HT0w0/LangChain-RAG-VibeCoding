"""LLM Provider 实现。

主路径走 `langchain-openai.ChatOpenAI` 指向百炼的 OpenAI 兼容端点。

为什么不用现成的通义集成：
- `langchain-community.ChatTongyi` 随 community 冻结（官方 issue #674），不再维护
- `langchain-qwq.ChatQwen` 是第三方 beta 包，默认 base_url 指向**国际站**，
  且上游拒绝为其加 `init_chat_model` 支持

兼容端点本身就是 OpenAI 协议，用 ChatOpenAI 是支持最好、坑最少的一条路。

已知坑：`ChatOpenAI.get_num_tokens_from_messages()` 对 qwen 未实现，会抛
NotImplementedError，因此全项目**不使用** `trim_messages(token_counter=model)`，
上下文预算自行估算（见 `services/context_budget.py`）。
"""

from __future__ import annotations

import asyncio
import re
from collections.abc import AsyncIterator, Sequence
from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.core.config import Settings
from app.providers.base import ChatChunk, ChatResult, Usage

_ROLE_MAP = {"system": SystemMessage, "user": HumanMessage, "assistant": AIMessage}

# MockLLM 用的标记。与 `services/prompts.py` 保持一致。
_CONTEXT_HEADER = "【资料】"
_QUESTION_HEADER = "【问题】"
_MARKER_RE = re.compile(r"\[(\d+)\]")


def _cached_tokens(meta: dict[str, Any], token_usage: dict[str, Any]) -> int | None:
    """取 prompt 缓存的读数,两条路径都认,**取不到就是 `None`**。

    两条路径的形状不一样:LangChain 归一化过的是
    `usage_metadata.input_token_details.cache_read`,`response_metadata` 里
    网关原始回包是 `token_usage.prompt_tokens_details.cached_tokens`(嵌两层)。
    实测两条会同时出现,但归一化是 LangChain 自己的实现细节,版本一变就可能
    不再填——那时只有原始回包还留着这个数。

    **判断必须用 `is None`,不能写成 `a or b`。** 平台明确回 `cached_tokens: 0`
    是完全合法的(冷启动就是),而 `0 or b` 会把这个 0 当成缺失、转而去读
    第二条路径。于是「这次没命中」被读成「下一条路径的值」,而冷启动恰恰是
    最常见的一轮。
    """
    details = meta.get("input_token_details") or {}
    if details.get("cache_read") is not None:
        return int(details["cache_read"])

    raw = (token_usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    return int(raw) if raw is not None else None


def to_lc_messages(messages: Sequence[dict[str, str]]) -> list[BaseMessage]:
    """把朴素 dict 转成 LangChain 消息对象。

    对外只暴露 dict，是为了让 Provider 接口不泄漏 LangChain 类型——
    MockLLM 和未来的其他实现都不必依赖 langchain-core。
    """
    out: list[BaseMessage] = []
    for m in messages:
        cls = _ROLE_MAP.get(m.get("role", "user"))
        if cls is None:
            raise ValueError(f"未知 role: {m.get('role')!r}")
        out.append(cls(content=m.get("content", "")))
    return out


class DashScopeLLM:
    """百炼兼容端点上的对话模型。"""

    def __init__(self, settings: Settings) -> None:
        self._s = settings
        self._client = ChatOpenAI(
            model=settings.LLM_MODEL,
            api_key=settings.DASHSCOPE_API_KEY,  # type: ignore[arg-type]
            base_url=settings.DASHSCOPE_BASE_URL,
            temperature=settings.LLM_TEMPERATURE,
            max_tokens=settings.LLM_MAX_TOKENS,
            # 关闭思考：qwen3.7-plus 默认产出 reasoning_content，RAG 场景
            # 不需要长链推理，关掉显著降低首 token 时延。实测该字段生效。
            extra_body={"enable_thinking": settings.LLM_ENABLE_THINKING},
            timeout=120.0,
            max_retries=2,
        )

    @property
    def model_id(self) -> str:
        return self._s.LLM_MODEL

    def _bind(
        self, temperature: float | None, max_tokens: int | None, **kwargs: Any
    ) -> ChatOpenAI:
        overrides: dict[str, Any] = {}
        if temperature is not None:
            overrides["temperature"] = temperature
        if max_tokens is not None:
            overrides["max_tokens"] = max_tokens
        overrides.update(kwargs)
        return self._client.bind(**overrides) if overrides else self._client  # type: ignore[return-value]

    @staticmethod
    def _usage_from(msg: AIMessage) -> Usage:
        meta = getattr(msg, "usage_metadata", None) or {}
        response_meta = getattr(msg, "response_metadata", None) or {}
        token_usage = response_meta.get("token_usage") or {}
        return Usage(
            prompt_tokens=meta.get("input_tokens", token_usage.get("prompt_tokens", 0)) or 0,
            completion_tokens=meta.get("output_tokens", token_usage.get("completion_tokens", 0)) or 0,
            total_tokens=meta.get("total_tokens", token_usage.get("total_tokens", 0)) or 0,
            cached_tokens=_cached_tokens(meta, token_usage),
        )

    @classmethod
    def _maybe_usage(cls, chunk: AIMessage) -> Usage | None:
        """流式分片上的用量,**有就给,没有就 None**。

        用量取决于网关是否回声(OpenAI 协议的 `stream_options.include_usage`),
        百炼不保证给。拿不到时返回 `None` 而不是补一个 0:0 会被上层当成
        「这次用了 0 个 token」记进指标,而 None 会被当成「没上报」——
        两者在成本统计里的含义完全相反。
        """
        meta = getattr(chunk, "usage_metadata", None)
        response_meta = getattr(chunk, "response_metadata", None) or {}
        token_usage = response_meta.get("token_usage")
        if not meta and not token_usage:
            return None
        return cls._usage_from(chunk)

    async def acomplete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        llm = self._bind(temperature, max_tokens, **kwargs)
        msg: AIMessage = await llm.ainvoke(to_lc_messages(messages))
        content = msg.content if isinstance(msg.content, str) else str(msg.content)
        return ChatResult(content=content, usage=self._usage_from(msg))

    async def astream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        """流式生成。

        必须用 `astream()` 而不是 `stream()`——后者是同步迭代器，
        在 async 上下文里 `async for` 会直接 TypeError。
        """
        llm = self._bind(temperature, max_tokens, **kwargs)
        async for chunk in llm.astream(to_lc_messages(messages)):
            text = chunk.content if isinstance(chunk.content, str) else ""
            # 思考模型即使关掉思考，个别版本仍可能回吐 reasoning_content。
            # 单独承接，绝不混进正文。
            reasoning = ""
            if chunk.additional_kwargs:
                raw = chunk.additional_kwargs.get("reasoning_content")
                if isinstance(raw, str):
                    reasoning = raw
            usage = self._maybe_usage(chunk)
            # 用量往往落在**最后一片**上，而最后一片的正文可能是空的。
            # 这里不能用「有没有正文」来决定发不发，否则那次用量会被丢掉，
            # done 事件里就永远是 None。
            if text or reasoning or usage is not None:
                yield ChatChunk(text=text, reasoning=reasoning, usage=usage)


class MockLLM:
    """离线/无额度时的替身。

    存在的意义不是「占位」，而是让检索链路能被独立测试：RAG 的质量问题
    十有八九出在检索而非生成，MockLLM 把生成这一步固定住，检索的锅就不会
    甩到模型头上。

    **它会引用提示词里真实存在的资料编号**（而不是无脑写 `[1]`），这一点
    是刻意的：引用溯源要能被端到端验证，就得有一个会「按编号说话」的模型，
    否则 `cited` 恒等于 `[1]`，抽取逻辑写错了也看不出来。它引用几条取决于
    提示词里真的有几条资料。
    """

    def __init__(self, settings: Settings, *, delay: float = 0.01) -> None:
        self._s = settings
        self._delay = delay

    @property
    def model_id(self) -> str:
        return "mock-llm"

    @staticmethod
    def _compose(messages: Sequence[dict[str, str]]) -> str:
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"), ""
        )
        system = next((m["content"] for m in messages if m.get("role") == "system"), "")

        # 标记必须与 `services/prompts.py` 里的一致。写死而不 import，是
        # 为了让这个替身不依赖业务层——但它意味着改提示词格式时要同步这里，
        # 而 `test_prompts.py` 会先因为断言失效而提醒。
        has_context = _CONTEXT_HEADER in user or "[资料" in system or "参考资料" in system
        question = user.split(_QUESTION_HEADER, 1)[-1].strip() if _QUESTION_HEADER in user else user

        if not has_context:
            return f"[MOCK] 收到问题：{question}"

        # 按提示词里真实出现过的编号来引用，最多两条。
        indices: list[int] = []
        for raw in _MARKER_RE.findall(user):
            if int(raw) not in indices:
                indices.append(int(raw))
        refs = "".join(f"[{i}]" for i in indices[:2]) or "[1]"

        return (
            f"[MOCK] 依据检索到的资料，关于「{question}」的回答如下。"
            f"该结论来自资料片段 {refs}。"
            "（此文本由 MockLLM 生成，用于在无 API 额度时验证链路。）"
        )

    @staticmethod
    def _usage_for(text: str, messages: Sequence[dict[str, str]]) -> Usage:
        """确定性假用量。

        **只在 mock 里这么做。** 真实 Provider 拿不到用量时必须留 `None`
        （见 `ChatChunk`），因为 token 用量的唯一可信来源是服务端；而 mock
        的一切本来就是合成的，给出一个能算出来的数，是为了让「用量确实
        从 Provider 流到了 done 事件」这条路在测试里被真的走一遍。
        """
        prompt = sum(len(m.get("content", "")) for m in messages)
        return Usage(
            prompt_tokens=prompt,
            completion_tokens=len(text),
            total_tokens=prompt + len(text),
            # 替身没有缓存可言,但它**是一个会报读数的 Provider**,所以报 `0`
            # 而不是留 `None`。留空的话,「用量从 Provider 一路流到 done 事件
            # 再到指标表」这条路在离线测试里就少一段没人走过——而 `None` 和 `0`
            # 在那一层的处置完全不同(见 `Usage.cached_tokens`)。
            cached_tokens=0,
        )

    async def acomplete(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        text = self._compose(messages)
        return ChatResult(content=text, usage=self._usage_for(text, messages))

    async def astream(
        self,
        messages: Sequence[dict[str, str]],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatChunk]:
        text = self._compose(messages)
        usage = self._usage_for(text, messages)
        # 按 6 字切块模拟真实流式节奏，让前端能验证逐字渲染与取消逻辑
        for i in range(0, len(text), 6):
            if self._delay:
                await asyncio.sleep(self._delay)
            last = i + 6 >= len(text)
            yield ChatChunk(text=text[i : i + 6], usage=usage if last else None)


def build_llm(settings: Settings) -> DashScopeLLM | MockLLM:
    backend = settings.LLM_BACKEND
    if backend == "mock":
        return MockLLM(settings)
    if backend == "anthropic_compat":
        # 备选通路，留待需要演示「同逻辑换协议」时实现
        raise NotImplementedError(
            "AnthropicCompatLLM 尚未实现。工作空间 /apps/anthropic 网关实测认证通过，"
            "需要时可用 langchain-anthropic.ChatAnthropic + ANTHROPIC_COMPAT_BASE_URL 接入。"
        )
    return DashScopeLLM(settings)
