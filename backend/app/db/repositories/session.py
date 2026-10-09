"""会话与消息仓储。

**这个文件里每一条读方法都带 `user_id`。** 这不是重复劳动,而是把越权
防线放在唯一绕不过去的地方:只要调用方拿到的 `ChatSession` 是从这里来的,
它就已经验过归属了。若在服务层「先按 id 查、再比对 user_id」,那个比对
是可以被漏写、被短路、被一个提前 return 绕过的;写成查询条件则漏不掉。

代价是偶尔要多传一个参数。值得。
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import ChatSession, Message, MessageRole, utcnow


class SessionRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._s = session

    # -- 会话 ----------------------------------------------------------------

    async def create(
        self,
        *,
        user_id: int,
        title: str,
        kb_ids: list[int] | None = None,
    ) -> ChatSession:
        session = ChatSession(user_id=user_id, title=title, kb_ids=kb_ids)
        self._s.add(session)
        await self._s.flush()
        await self._s.refresh(session)
        return session

    async def get_owned(self, session_id: int, *, user_id: int) -> ChatSession | None:
        """按 id 取会话,**但只在它属于 user_id 时返回**。

        返回 None 表示「不存在**或**不属于你」——两种情况故意不可区分,
        上层一律转成 404。见 `test_sessions.py` 的越权用例。
        """
        stmt = select(ChatSession).where(
            ChatSession.id == session_id, ChatSession.user_id == user_id
        )
        return (await self._s.execute(stmt)).scalar_one_or_none()

    async def list_for_user(
        self, user_id: int, *, limit: int, offset: int
    ) -> list[ChatSession]:
        stmt = (
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            # 最近聊过的排最前。用 last_message_at 而不是 updated_at——
            # 后者会被「改标题」这种动作搅乱顺序(见 models.ChatSession)。
            # 次级排序键用 id 保证同一时刻的会话顺序稳定,否则分页会在
            # 两条 last_message_at 相同的记录之间来回跳。
            .order_by(ChatSession.last_message_at.desc(), ChatSession.id.desc())
            .limit(limit)
            .offset(offset)
        )
        return list((await self._s.execute(stmt)).scalars())

    async def count_for_user(self, user_id: int) -> int:
        stmt = select(func.count(ChatSession.id)).where(ChatSession.user_id == user_id)
        return int((await self._s.execute(stmt)).scalar_one())

    async def rename(self, session: ChatSession, title: str) -> None:
        session.title = title
        # **刻意不动 last_message_at。** 改标题不该把这个会话顶到列表最前面,
        # 那会让「最近聊过的」这个排序失去意义。
        await self._s.flush()

    async def delete(self, session: ChatSession) -> None:
        await self._s.delete(session)
        await self._s.flush()

    # -- 消息 ----------------------------------------------------------------

    async def append_message(
        self,
        session: ChatSession,
        *,
        role: MessageRole | str,
        content: str,
        sources: list[dict[str, Any]] | None = None,
        cited: list[int] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        error: str | None = None,
    ) -> Message:
        """追加一条消息,并在**同一个方法里**维护会话的冗余计数与时间戳。

        把计数放在这里而不是让调用方自己 `session.message_count += 1`,
        是因为那个自增一定会有人忘记,而忘了的后果是列表页显示「0 条消息」
        ——一个不会报错、只是慢慢变得不可信的数字。
        """
        message = Message(
            session_id=session.id,
            role=str(role),
            content=content,
            sources=sources or [],
            cited=cited or [],
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            error=error,
        )
        self._s.add(message)

        session.message_count += 1
        session.last_message_at = utcnow()
        await self._s.flush()
        await self._s.refresh(message)
        return message

    async def list_messages(
        self, session_id: int, *, limit: int, offset: int
    ) -> list[Message]:
        """按时间**正序**返回,供前端自上而下渲染对话。"""
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.id)
            .limit(limit)
            .offset(offset)
        )
        return list((await self._s.execute(stmt)).scalars())

    async def count_messages(self, session_id: int) -> int:
        stmt = select(func.count(Message.id)).where(Message.session_id == session_id)
        return int((await self._s.execute(stmt)).scalar_one())

    async def recent_messages(self, session_id: int, *, limit: int) -> list[Message]:
        """取**最近** limit 条,返回时已翻正为正序。

        这个方法是给上下文窗口用的,和 `list_messages` 取的是相反的一端:
        分页是从头往后翻,聊天历史要的是**离当前最近的几轮**。写反了不会
        报错,只会让模型拿到会话开头那几句陈年旧话,然后答非所问。
        """
        stmt = (
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.id.desc())
            .limit(limit)
        )
        rows = list((await self._s.execute(stmt)).scalars())
        rows.reverse()
        return rows
