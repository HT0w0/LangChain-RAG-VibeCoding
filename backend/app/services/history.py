"""多轮对话:历史窗口裁剪与指代消解。

这里处理的是「第二问之后」才出现的两个问题:

1. **历史占预算。** 一个聊了 30 轮的会话,全量回放会把检索资料挤出去
   ——而检索资料才是这个系统相对通用大模型的全部价值。资料优先,历史
   只在剩下的预算里装,且只装最近的。

2. **追问是残缺的。** 「它的库存呢」单独拿去检索,召回的是所有含「库存」
   的片段,与「它」指谁无关。多轮检索不做指代消解,第一问之后的检索
   质量会明显低于第一问——而用户会觉得「聊着聊着它变笨了」。

两者都不改模型的输出格式,所以出问题时不报错、只变差,只能靠测试盯住。
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from app.core.config import Settings
from app.core.logging import log_extra
from app.db.models import Message, MessageRole
from app.providers.base import LLMProvider
from app.services.context import estimate_tokens

logger = logging.getLogger(__name__)

REWRITE_SYSTEM_PROMPT = """你是一个查询改写器。用户正在与一个电商商品知识库问答系统多轮对话。

你的唯一任务:把用户的**最新一句话**改写成一个不依赖上下文、可以独立拿去检索的完整问题。

规则:
1. 把代词和省略补全。例:上文在聊「RX9070 显卡」,最新一句是「它的库存呢」,改写为「RX9070 显卡的库存是多少」。
2. **最新一句本身已经完整时,原样返回,不要画蛇添足。** 例:「27寸显示器多少钱」就是完整的。
3. 不要回答问题,不要解释,不要加引号或任何前缀。只输出改写后的那一句。
4. 如果最新一句与商品知识库无关(例如「谢谢」「你好」),原样返回。
"""


def build_history(
    messages: Sequence[Message],
    *,
    max_tokens: int,
    max_turns: int,
) -> list[dict[str, str]]:
    """把历史消息裁成一段能塞进 prompt 的对话前缀。

    **从最近往回取**,这样预算或轮数不够时丢掉的是会话开头那几句
    ——那几句与当前这一问的关系最远。

    返回顺序是**正序**(最早的在最前),与 `messages` 里聊天的自然顺序一致。
    倒序交给模型会让它看到「先有回答、后有提问」,然后顺着演下去。
    """
    if max_tokens <= 0 or max_turns <= 0 or not messages:
        return []

    turns: list[list[dict[str, str]]] = []
    current: list[dict[str, str]] = []
    for msg in messages:
        current.append({"role": msg.role, "content": msg.content})
        # 一条助手消息结束一轮。历史是「问-答-问-答」铺开的,所以这个切法
        # 能把它们配成对;若末尾挂着一个还没被回答的提问,它自成一组,
        # 在下面按「不满一轮」处理。
        if msg.role == str(MessageRole.ASSISTANT):
            turns.append(current)
            current = []
    if current:
        turns.append(current)

    kept: list[list[dict[str, str]]] = []
    used = 0
    for turn in reversed(turns):
        # 只有「问 + 答」都在的轮次才算一轮。末尾那个孤立的提问不占轮数名额,
        # 但它仍然要被带进 prompt 里——那是用户刚说的话。
        complete = len(turn) >= 2
        if complete and len(kept) >= max_turns:
            break
        cost = sum(estimate_tokens(m["content"]) for m in turn)
        if used + cost > max_tokens:
            break
        kept.append(turn)
        used += cost

    kept.reverse()
    flat = [m for turn in kept for m in turn]

    # 截断可能正好落在「提问与回答之间」,让窗口以一个孤零零的回答开头。
    # 没有问题的回答对模型是纯噪音,还会让它以为那是自己上一句说的话。
    while flat and flat[0]["role"] != str(MessageRole.USER):
        flat.pop(0)
    return flat


async def rewrite_question(
    question: str,
    history: Sequence[Message],
    *,
    llm: LLMProvider,
    settings: Settings,
) -> str:
    """有历史时把追问补成独立问题;**没有历史就原样返回,不调模型**。

    第一问没有指代可消解,调一次模型纯属浪费——而这是每轮都要付的成本。

    **任何失败都退化成原问题**,包括模型报错和模型返回空白。改写是提升
    检索命中率的辅助步骤,让它把整次问答带崩是很糟的取舍:用户会为一个
    自己根本不知道存在的步骤丢掉整个回答。
    """
    if not history or not settings.REWRITE_ENABLED:
        return question

    # 只把最近的几轮喂给改写器。改写要的是「最新一句在说什么」,
    # 给它整个会话既慢又容易让它改出一个跑偏的问题。
    recent = build_history(
        history, max_tokens=settings.REWRITE_TOKEN_BUDGET, max_turns=settings.REWRITE_MAX_TURNS
    )
    context_lines = []
    for msg in recent:
        speaker = "用户" if msg["role"] == str(MessageRole.USER) else "助手"
        context_lines.append(f"{speaker}:{msg['content']}")

    prompt = (
        "以下是对话历史:\n"
        + "\n".join(context_lines)
        + f"\n\n用户最新一句:{question}\n\n改写后的完整问题:"
    )

    try:
        result = await llm.acomplete(
            [
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=settings.REWRITE_MAX_TOKENS,
        )
    except Exception:
        logger.warning(
            "查询改写失败,退回原问题",
            extra=log_extra(question=question[:60]),
            exc_info=True,
        )
        return question

    rewritten = (result.content or "").strip()
    if not rewritten:
        logger.info("查询改写的返回为空,退回原问题", extra=log_extra(question=question[:60]))
        return question

    logger.debug("查询已改写", extra=log_extra(before=question[:40], after=rewritten[:40]))
    return rewritten


__all__ = ["REWRITE_SYSTEM_PROMPT", "build_history", "rewrite_question"]
