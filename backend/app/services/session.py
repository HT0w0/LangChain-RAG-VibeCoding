"""会话服务:归属校验、生命周期、消息落库。

**归属校验集中在这里的 `require_owned` 一个入口**,而不是散在路由里各写一遍
`if session.user_id != user.id: raise`。散写的问题不是啰嗦,是**漏**:
新增一个路由时忘记写那三行,代码照常跑、测试照常绿(如果恰好没为新路由
补越权用例的话),而漏洞是静默的。集中成一个必然被调用的方法,
新路由只要拿到会话就必须经过它。
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.exceptions import NotFoundError
from app.db.models import ChatSession, Message, MessageRole
from app.db.repositories.session import SessionRepository
from app.schemas.session import DEFAULT_SESSION_TITLE, MAX_TITLE_LENGTH

_WHITESPACE_RE = re.compile(r"\s+")


def derive_title(question: str) -> str:
    """从首个问题里截出会话标题。

    **把换行和连续空白压成单个空格。** 问题常常是多行的(用户从别处粘贴
    一段描述),原样截取会在列表页上渲染成一条断成几行、把行高撑开的
    标题,把整个列表挤变形。
    """
    collapsed = _WHITESPACE_RE.sub(" ", question).strip()
    if not collapsed:
        return DEFAULT_SESSION_TITLE
    if len(collapsed) <= MAX_TITLE_LENGTH:
        return collapsed
    # 截断加省略号,让人一眼看出这是被截过的,而不是就这么短。
    return collapsed[: MAX_TITLE_LENGTH - 1] + "…"


class SessionService:
    def __init__(self, session: AsyncSession) -> None:
        self._repo = SessionRepository(session)

    # -- 生命周期 ------------------------------------------------------------

    async def create(
        self,
        *,
        user_id: int,
        title: str | None = None,
        kb_ids: list[int] | None = None,
    ) -> ChatSession:
        return await self._repo.create(
            user_id=user_id,
            title=title or DEFAULT_SESSION_TITLE,
            kb_ids=kb_ids,
        )

    async def require_owned(self, session_id: int, *, user_id: int) -> ChatSession:
        """取会话,不属于当前用户就是**不存在**。

        抛 404 而非 403 是刻意的:403 等于告诉调用方「这个 id 有主,只是
        不给你看」,足以枚举出别人有多少会话。两种失败必须长得一样。
        """
        session = await self._repo.get_owned(session_id, user_id=user_id)
        if session is None:
            raise NotFoundError("会话不存在", session_id=session_id)
        return session

    async def list_for_user(
        self, user_id: int, *, limit: int, offset: int
    ) -> tuple[list[ChatSession], int]:
        items = await self._repo.list_for_user(user_id, limit=limit, offset=offset)
        return items, await self._repo.count_for_user(user_id)

    async def rename(self, session_id: int, *, user_id: int, title: str) -> ChatSession:
        session = await self.require_owned(session_id, user_id=user_id)
        await self._repo.rename(session, title)
        return session

    async def delete(self, session_id: int, *, user_id: int) -> None:
        session = await self.require_owned(session_id, user_id=user_id)
        await self._repo.delete(session)

    # -- 消息 ----------------------------------------------------------------

    async def messages(
        self, session_id: int, *, user_id: int, limit: int, offset: int
    ) -> tuple[list[Message], int]:
        await self.require_owned(session_id, user_id=user_id)
        items = await self._repo.list_messages(session_id, limit=limit, offset=offset)
        return items, await self._repo.count_messages(session_id)

    async def history(self, session: ChatSession, *, limit: int) -> list[Message]:
        """取最近 limit 条消息,供上下文窗口使用。

        **这里不重复做归属校验**,因为入参是已经过 `require_owned` 的
        `ChatSession` 对象而不是一个裸 id——类型本身就表明了它已经验过。
        """
        return await self._repo.recent_messages(session.id, limit=limit)

    async def append_question(self, session: ChatSession, question: str) -> Message:
        """记录用户提问。

        **先落库再检索**,理由是失败模式:如果检索或生成挂了,用户的问题
        仍然在历史里,下次打开能看到「我问过这句,当时没答上来」。
        反过来(答完再一起写)则一旦中途失败,这次提问就凭空消失了。
        """
        if session.title == DEFAULT_SESSION_TITLE:
            # 只在标题还是占位符时自动命名。见
            # `test_a_title_the_user_set_is_never_overwritten`。
            session.title = derive_title(question)
        return await self._repo.append_message(
            session, role=MessageRole.USER, content=question
        )

    async def append_answer(
        self,
        session: ChatSession,
        *,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        cited: list[int] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> Message:
        return await self._repo.append_message(
            session,
            role=MessageRole.ASSISTANT,
            content=content,
            sources=sources,
            cited=cited,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            error=error,
        )
