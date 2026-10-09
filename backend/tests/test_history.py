"""多轮上下文:历史裁剪与指代消解。

`build_history` 是纯函数,可以直接喂字符串测;`rewrite_question` 要调模型,
用假 LLM。

这个文件里最要紧的两条是**边界**:
  * 窗口的左端不能停在助手的回答上(那段回答脱离问题就没意义了)
  * 改写失败必须退化成原问题,绝不能把整次提问带崩
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

from app.core.config import Settings
from app.providers.base import ChatChunk, ChatResult
from app.services.history import build_history, rewrite_question


def hist(*pairs: tuple[str, str]) -> list:
    """把 (问题, 回答) 序列铺成消息列表。"""
    from app.db.models import Message, MessageRole

    out = []
    for q, a in pairs:
        out.append(Message(session_id=1, role=str(MessageRole.USER), content=q))
        out.append(Message(session_id=1, role=str(MessageRole.ASSISTANT), content=a))
    return out


# ---------------------------------------------------------------------------
# build_history
# ---------------------------------------------------------------------------


class TestBuildHistory:
    def test_empty_history_yields_nothing(self) -> None:
        assert build_history([], max_tokens=1000, max_turns=5) == []

    def test_single_turn_keeps_both_sides_in_order(self) -> None:
        out = build_history(hist(("甲问题", "甲回答")), max_tokens=1000, max_turns=5)
        assert [m["role"] for m in out] == ["user", "assistant"]
        assert out[0]["content"] == "甲问题"
        assert out[1]["content"] == "甲回答"

    def test_chronological_order_is_preserved(self) -> None:
        """窗口是倒着取的(先拿最近的),但**交给模型时必须翻正**.

        顺序反了不会报错:模型会看到「先有回答、后有提问」,然后一本正经
        地接着演下去。这是最难从输出上看出根因的一类 bug。
        """
        out = build_history(
            hist(("第一问", "第一答"), ("第二问", "第二答")), max_tokens=1000, max_turns=5
        )
        assert [m["content"] for m in out] == ["第一问", "第一答", "第二问", "第二答"]

    def test_max_turns_keeps_the_most_recent(self) -> None:
        """**丢的是最早的,不是最新的。** 反过来丢,模型就会拿会话开头
        那几句陈年旧话去理解当前这一问。"""
        out = build_history(
            hist(("老问题", "老回答"), ("新问题", "新回答")), max_tokens=1000, max_turns=1
        )
        assert [m["content"] for m in out] == ["新问题", "新回答"]

    def test_token_budget_drops_the_oldest_first(self) -> None:
        long = "长" * 400
        out = build_history(
            hist((long, long), ("短问题", "短回答")), max_tokens=60, max_turns=10
        )
        contents = [m["content"] for m in out]
        assert "短问题" in contents, "最近的一轮被预算挤掉了"
        assert long not in contents, "超预算的老一轮没有被丢掉"

    def test_max_turns_window_starts_with_a_question(self) -> None:
        out = build_history(
            hist(("老问题", "老回答"), ("新问题", "新回答")), max_tokens=1000, max_turns=1
        )
        assert out[0]["role"] == "user"

    def test_orphan_leading_answer_is_dropped(self) -> None:
        """**窗口左端可能挂着一条没有问题的回答**,必须去掉。

        什么时候会出现:历史是按最近 N 条载入的,那条切片很可能正好从
        「上一次回答」的中间切开。没有问题的回答对模型是纯噪音,还会让它
        以为那是自己刚说过的话,于是顺着答下去。

        单独写一条用例,是因为上面那条(靠 max_turns 截断)走不到修剪
        分支——它的窗口本来就是从提问开始的。少这一条,修剪那段代码
        整个可以删掉而测试全绿(变异测试验出来的)。
        """
        from app.db.models import Message, MessageRole

        orphan = Message(session_id=1, role=str(MessageRole.ASSISTANT), content="孤零零的回答")
        out = build_history(
            [orphan, *hist(("新问题", "新回答"))], max_tokens=1000, max_turns=5
        )
        assert out[0]["role"] == "user"
        assert all(m["content"] != "孤零零的回答" for m in out), "孤立回答没有被丢掉"

    def test_zero_budget_yields_nothing(self) -> None:
        assert build_history(hist(("问", "答")), max_tokens=0, max_turns=5) == []

    def test_zero_turns_yields_nothing(self) -> None:
        assert build_history(hist(("问", "答")), max_tokens=1000, max_turns=0) == []


# ---------------------------------------------------------------------------
# rewrite_question
# ---------------------------------------------------------------------------


class FakeLLM:
    def __init__(self, reply: str = "改写后的问题", *, boom: bool = False) -> None:
        self.reply = reply
        self.boom = boom
        self.calls: list[Sequence[dict[str, str]]] = []

    @property
    def model_id(self) -> str:
        return "fake"

    async def acomplete(self, messages, **kwargs: Any) -> ChatResult:
        self.calls.append(messages)
        if self.boom:
            raise RuntimeError("模型挂了")
        return ChatResult(content=self.reply)

    async def astream(self, messages, **kwargs: Any) -> AsyncIterator[ChatChunk]:
        yield ChatChunk(text=self.reply)


def settings() -> Settings:
    return Settings(_env_file=None, JWT_SECRET="test")  # type: ignore[call-arg]


class TestRewriteQuestion:
    async def test_no_history_skips_the_model_entirely(self) -> None:
        """**第一问不调模型。** 没有历史就没有指代可消解,这一次调用纯属浪费
        ——而它是每轮都要付的成本,不是一次性的。"""
        llm = FakeLLM()
        out = await rewrite_question("RX9070 多少钱", [], llm=llm, settings=settings())
        assert out == "RX9070 多少钱"
        assert llm.calls == []

    async def test_history_triggers_a_rewrite(self) -> None:
        llm = FakeLLM(reply="RX9070 显卡的库存是多少")
        out = await rewrite_question(
            "它的库存呢", hist(("RX9070 多少钱", "4299 元")), llm=llm, settings=settings()
        )
        assert out == "RX9070 显卡的库存是多少"
        assert len(llm.calls) == 1

    async def test_rewrite_failure_falls_back_to_the_original(self) -> None:
        """**改写是优化,不是必经步骤。** 它挂了就当没做,继续用原问题检索。

        让改写失败升级成 500 是很糟的取舍:用户会为一个「本来只是想提升
        检索命中率」的辅助步骤丢掉整个回答。
        """
        llm = FakeLLM(boom=True)
        out = await rewrite_question(
            "它的库存呢", hist(("RX9070 多少钱", "4299 元")), llm=llm, settings=settings()
        )
        assert out == "它的库存呢"

    async def test_blank_rewrite_falls_back_to_the_original(self) -> None:
        """模型有时会返回空串或只有空白。拿它去检索会召回到随机的片段。"""
        llm = FakeLLM(reply="   ")
        out = await rewrite_question(
            "它的库存呢", hist(("RX9070 多少钱", "4299 元")), llm=llm, settings=settings()
        )
        assert out == "它的库存呢"

    async def test_rewrite_result_is_trimmed(self) -> None:
        llm = FakeLLM(reply="\n  RX9070 的库存 \n")
        out = await rewrite_question(
            "它的库存呢", hist(("RX9070 多少钱", "4299 元")), llm=llm, settings=settings()
        )
        assert out == "RX9070 的库存"

    async def test_rewrite_can_be_disabled_by_config(self) -> None:
        """关掉之后即使有历史也不该调模型——排查检索质量问题时需要一个
        「完全不改写」的基线。"""
        llm = FakeLLM()
        s = settings()
        s.REWRITE_ENABLED = False
        out = await rewrite_question("它的库存呢", hist(("问", "答")), llm=llm, settings=s)
        assert out == "它的库存呢"
        assert llm.calls == []
