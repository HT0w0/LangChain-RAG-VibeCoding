"""Provider 层测试 —— 全部离线,不打真实 API。

这里测的是**解析与契约**,不是网络。历史教训:服务端不保证 `data` 数组
顺序与输入一致,而错位的向量不会报错,只会让检索质量悄悄变差,极难排查。
所以顺序还原必须有测试守着。
"""

from __future__ import annotations

import httpx
import pytest
from app.core.config import Settings
from app.core.exceptions import ProviderError, QuotaExhaustedError
from app.providers.embedding import (
    DashScopeEmbedder,
    _raise_for_upstream,  # noqa: PLC2701
    l2_normalize,
)
from app.providers.llm import DashScopeLLM, MockLLM
from app.providers.reranker import DashScopeReranker, NoopReranker

# ---------------------------------------------------------------------------
# 归一化
# ---------------------------------------------------------------------------


def test_l2_normalize_unit_length() -> None:
    v = l2_normalize([3.0, 4.0])
    assert v == pytest.approx([0.6, 0.8])
    assert sum(x * x for x in v) == pytest.approx(1.0)


def test_l2_normalize_handles_zero_vector() -> None:
    """全零向量不能除零崩掉——空文档或纯停用词会产出它。"""
    assert l2_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_l2_normalize_preserves_direction() -> None:
    a = l2_normalize([1.0, 2.0, 3.0])
    b = l2_normalize([10.0, 20.0, 30.0])
    assert a == pytest.approx(b)


# ---------------------------------------------------------------------------
# Embedding 响应解析
# ---------------------------------------------------------------------------


def _embed_response(pairs: list[tuple[int, list[float]]]) -> httpx.Response:
    return httpx.Response(
        200,
        json={"data": [{"index": i, "embedding": v} for i, v in pairs], "usage": {}},
    )


def test_parse_restores_order_when_server_shuffles(settings: Settings) -> None:
    """服务端乱序返回时,必须按 index 还原。

    这是本文件里最重要的一个测试:顺序错了不报错,只会让向量和文本错位。
    """
    emb = DashScopeEmbedder(settings)
    # 故意倒序返回
    resp = _embed_response([(2, [0.0, 0.0, 1.0]), (1, [0.0, 1.0, 0.0]), (0, [1.0, 0.0, 0.0])])
    out = emb._parse(resp, expected=3, is_query=False)  # noqa: SLF001
    assert out[0] == pytest.approx([1.0, 0.0, 0.0])
    assert out[1] == pytest.approx([0.0, 1.0, 0.0])
    assert out[2] == pytest.approx([0.0, 0.0, 1.0])


def test_parse_rejects_count_mismatch(settings: Settings) -> None:
    """条数不符必须立即失败。

    静默接受会让「第 3 条 chunk 拿到第 2 条的向量」这类错误流入库中。
    """
    emb = DashScopeEmbedder(settings)
    resp = _embed_response([(0, [1.0, 0.0])])
    with pytest.raises(ProviderError, match="条数不符"):
        emb._parse(resp, expected=3, is_query=False)  # noqa: SLF001


def test_parse_learns_actual_dim(settings: Settings) -> None:
    """实测维度与配置不符时以服务端为准,并更新自身状态。"""
    emb = DashScopeEmbedder(settings)
    assert emb.dim == 1024
    resp = _embed_response([(0, [0.1] * 768)])
    with pytest.warns(UserWarning, match="维度"):
        emb._parse(resp, expected=1, is_query=False)  # noqa: SLF001
    assert emb.dim == 768


def test_parse_normalizes_output(settings: Settings) -> None:
    """输出必须已归一化,否则余弦阈值(语义缓存 0.95)失去意义。"""
    emb = DashScopeEmbedder(settings)
    resp = _embed_response([(0, [3.0, 4.0] + [0.0] * 1022)])
    (vec,) = emb._parse(resp, expected=1, is_query=False)  # noqa: SLF001
    assert sum(x * x for x in vec) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 上游错误映射
# ---------------------------------------------------------------------------


def test_arrearage_maps_to_quota_error() -> None:
    """欠费是 400 而非 401。

    这条区分很重要:400 看起来像「参数写错了」,如果按状态码判断就会
    让人去翻请求体,而真正的问题是余额。
    """
    resp = httpx.Response(
        400, json={"error": {"code": "Arrearage", "message": "Access denied, please make sure..."}}
    )
    with pytest.raises(QuotaExhaustedError):
        _raise_for_upstream(resp, "test")


def test_invalid_key_is_not_retryable() -> None:
    resp = httpx.Response(
        401, json={"error": {"code": "InvalidApiKey", "message": "Invalid API-key provided."}}
    )
    with pytest.raises(ProviderError) as ei:
        _raise_for_upstream(resp, "test")
    assert ei.value.retryable is False


def test_429_is_retryable() -> None:
    resp = httpx.Response(429, json={"error": {"code": "Throttling", "message": "rate limited"}})
    with pytest.raises(ProviderError) as ei:
        _raise_for_upstream(resp, "test")
    assert ei.value.retryable is True


def test_html_error_page_does_not_crash() -> None:
    """网关故障时可能返回 HTML 而不是 JSON,解析不能炸。"""
    resp = httpx.Response(502, text="<html><body>Bad Gateway</body></html>")
    with pytest.raises(ProviderError, match="Bad Gateway"):
        _raise_for_upstream(resp, "test")


# ---------------------------------------------------------------------------
# Rerank 解析
# ---------------------------------------------------------------------------


def test_rerank_parse_sorts_by_score_desc() -> None:
    resp = httpx.Response(
        200,
        json={
            "output": {
                "results": [
                    {"index": 1, "relevance_score": 0.265},
                    {"index": 0, "relevance_score": 0.823},
                    {"index": 2, "relevance_score": 0.127},
                ]
            }
        },
    )
    hits = DashScopeReranker._parse(resp)  # noqa: SLF001
    assert [h.index for h in hits] == [0, 1, 2]
    assert hits[0].score == pytest.approx(0.823)


def test_rerank_parse_accepts_toplevel_results_fallback() -> None:
    """原生端点个别版本把 results 放在顶层,解析要兼容。"""
    resp = httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.5}]})
    hits = DashScopeReranker._parse(resp)  # noqa: SLF001
    assert len(hits) == 1


def test_rerank_parse_rejects_unknown_shape() -> None:
    resp = httpx.Response(200, json={"unexpected": True})
    with pytest.raises(ProviderError, match="响应结构"):
        DashScopeReranker._parse(resp)  # noqa: SLF001


@pytest.mark.asyncio
async def test_noop_reranker_preserves_order() -> None:
    r = NoopReranker()
    hits = await r.rerank("q", ["a", "b", "c", "d"], top_n=2)
    assert [h.index for h in hits] == [0, 1]


@pytest.mark.asyncio
async def test_noop_reranker_clamps_top_n() -> None:
    r = NoopReranker()
    hits = await r.rerank("q", ["a", "b"], top_n=99)
    assert len(hits) == 2


# ---------------------------------------------------------------------------
# MockLLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mock_llm_streams_in_chunks(settings: Settings) -> None:
    llm = MockLLM(settings, delay=0)
    chunks = [c.text async for c in llm.astream([{"role": "user", "content": "你好"}])]
    assert len(chunks) > 1, "应分多块产出,以便前端能验证逐字渲染"
    assert "你好" in "".join(chunks)


@pytest.mark.asyncio
async def test_mock_llm_cites_when_context_present(settings: Settings) -> None:
    """有资料时产出引用标记——让「引用溯源」这条链路在离线时也能端到端验证。"""
    llm = MockLLM(settings, delay=0)
    messages = [
        {"role": "system", "content": "参考资料:\n[资料1] 电池续航 8 小时。"},
        {"role": "user", "content": "续航多久?"},
    ]
    text = "".join([c.text async for c in llm.astream(messages)])
    assert "[1]" in text


@pytest.mark.asyncio
async def test_mock_llm_omits_citation_without_context(settings: Settings) -> None:
    llm = MockLLM(settings, delay=0)
    text = "".join([c.text async for c in llm.astream([{"role": "user", "content": "你好"}])])
    assert "[1]" not in text


@pytest.mark.asyncio
async def test_mock_llm_acomplete(settings: Settings) -> None:
    llm = MockLLM(settings, delay=0)
    result = await llm.acomplete([{"role": "user", "content": "测试"}])
    assert "测试" in result.content


# ---------------------------------------------------------------------------
# 用量里的 prompt 缓存读数
#
# 平台回一个 `prompt_tokens_details.cached_tokens`,它**此前被整个丢掉**——
# `Usage` 上根本没有承接它的字段,于是「这一轮的 prompt 有多少是缓存给的」
# 这个问题在日志、SSE、指标表里都无从回答。
#
# 下面两条走的是**两条不同的路径**,而它们的形状不一样:LangChain 归一化过的
# `usage_metadata.input_token_details.cache_read`(下划线风格),
# 与网关原始回包 `response_metadata.token_usage.prompt_tokens_details.cached_tokens`
# (驼峰风格,嵌两层)。只认其中一条的话,恰好另一条生效时读数是空的,
# 而空读数看起来就是「缓存没命中」——一个不会报错的错。
# ---------------------------------------------------------------------------


def _ai_message(*, usage_metadata=None, token_usage=None):
    from langchain_core.messages import AIMessage

    meta = {} if token_usage is None else {"token_usage": token_usage}
    return AIMessage(content="好", usage_metadata=usage_metadata, response_metadata=meta)


def test_cached_tokens_are_read_from_the_normalized_metadata() -> None:
    """LangChain 归一化路径:`input_token_details.cache_read`。"""
    msg = _ai_message(
        usage_metadata={
            "input_tokens": 1751,
            "output_tokens": 4,
            "total_tokens": 1755,
            "input_token_details": {"cache_read": 1664},
        }
    )

    usage = DashScopeLLM._usage_from(msg)

    assert usage.prompt_tokens == 1751
    assert usage.cached_tokens == 1664


def test_cached_tokens_fall_back_to_the_raw_gateway_payload() -> None:
    """网关原始路径:`prompt_tokens_details.cached_tokens`,嵌在两层里。

    这条**不是**上面那条的重复。真实回包里两条会同时出现,但 LangChain
    的归一化是它自己的实现细节,版本一变就可能不再填 `input_token_details`
    ——那时只有原始回包还带着这个数。两条都认,读数才不会随依赖升级消失。
    """
    msg = _ai_message(
        token_usage={
            "prompt_tokens": 1611,
            "completion_tokens": 4,
            "total_tokens": 1615,
            "prompt_tokens_details": {"cached_tokens": 1152},
        }
    )

    assert DashScopeLLM._usage_from(msg).cached_tokens == 1152


def test_a_silent_platform_reports_no_cache_reading_not_zero() -> None:
    """平台没报这个字段时是 `None`,**不是 0**。

    两者的含义相反:0 = 「平台说这次一条都没命中」,None = 「平台没说」。
    把没报当成 0,面板上会显示一条平稳的 0% 命中率曲线——看起来像
    「缓存一直不工作」,而事实是我们根本没在听。这与 `first_token_ms`
    用 `None` 表示「一个字都没流出来」是同一条理由。
    """
    msg = _ai_message(
        usage_metadata={"input_tokens": 100, "output_tokens": 5, "total_tokens": 105}
    )

    assert DashScopeLLM._usage_from(msg).cached_tokens is None


def test_a_reported_zero_is_kept_distinct_from_unreported() -> None:
    """平台明确回了 `cached_tokens: 0` 时,读数是 0,不是 None。

    这是上一条的镜像,单独写是因为**两个方向都容易写错**,而写错的那一侧
    总是把 `0` 和 `None` 合并掉。合并之后,「冷启动没命中」和「平台不报」
    在表里长得一模一样,而这两种情况的处置完全不同。
    """
    msg = _ai_message(
        usage_metadata={
            "input_tokens": 100,
            "output_tokens": 5,
            "total_tokens": 105,
            "input_token_details": {"cache_read": 0},
        }
    )

    assert DashScopeLLM._usage_from(msg).cached_tokens == 0
