"""会话管理接口(需求 ③④)。

**没有管理员例外。** 知识库有(admin 管全部),会话没有:需求 6 给管理员的
是知识库管理权,不是读别人对话的权力。所以这个 router 里每一个路由都
只依赖 `get_current_user`,归属校验一律走 `SessionService.require_owned`。

**每个路由都要传 `user.id`**,包括看起来「只需要 session_id」的那些。
这不是样板代码:少传一个,那条路由就能读到别人的会话,而它不会报错、
只会安静地返回不属于你的数据。
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, Query, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_current_user, get_db
from app.core.logging import log_extra
from app.db.models import User
from app.schemas.session import (
    MessageListResponse,
    MessageOut,
    SessionCreate,
    SessionListResponse,
    SessionOut,
    SessionRename,
)
from app.services.session import SessionService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions", tags=["会话"])


@router.post(
    "",
    response_model=SessionOut,
    status_code=status.HTTP_201_CREATED,
    summary="新建会话",
)
async def create_session(
    payload: SessionCreate,
    session: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionOut:
    """`kb_ids` 存成会话的默认检索范围,后续提问不必重复勾选。

    `None`(不限,即全部知识库)与 `[]`(什么都不看)是两个不同的意思,
    会原样存下来,不会把空列表「归一化」成 None。
    """
    created = await SessionService(session).create(
        user_id=user.id, title=payload.title, kb_ids=payload.kb_ids
    )
    # 服务层只 flush,提交由路由负责——与 kb 路由的既有约定一致
    # (`get_db` 的 session 上下文管理器不会自动提交,见 db/base.py)。
    await session.commit()
    logger.info("新建会话", extra=log_extra(user_id=user.id, session_id=created.id))
    return SessionOut.model_validate(created)


@router.get("", response_model=SessionListResponse, summary="我的会话列表")
async def list_sessions(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionListResponse:
    items, total = await SessionService(session).list_for_user(
        user.id, limit=limit, offset=offset
    )
    return SessionListResponse(
        total=total, items=[SessionOut.model_validate(s) for s in items]
    )


@router.get("/{session_id}", response_model=SessionOut, summary="会话详情")
async def get_session(
    session_id: int,
    session: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionOut:
    owned = await SessionService(session).require_owned(session_id, user_id=user.id)
    return SessionOut.model_validate(owned)


@router.patch("/{session_id}", response_model=SessionOut, summary="重命名会话")
async def rename_session(
    session_id: int,
    payload: SessionRename,
    session: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> SessionOut:
    renamed = await SessionService(session).rename(
        session_id, user_id=user.id, title=payload.title
    )
    await session.commit()
    return SessionOut.model_validate(renamed)


@router.delete(
    "/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除会话(连带消息)",
)
async def delete_session(
    session_id: int,
    session: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> Response:
    await SessionService(session).delete(session_id, user_id=user.id)
    await session.commit()
    logger.info("删除会话", extra=log_extra(user_id=user.id, session_id=session_id))
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{session_id}/messages",
    response_model=MessageListResponse,
    summary="会话消息(按时间正序)",
)
async def list_messages(
    session_id: int,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_db),
    user: User = Depends(get_current_user),
) -> MessageListResponse:
    """历史记录找回的入口(需求 4)。

    返回的 `sources` 是**当时那份来源快照**,不是按 chunk_id 现查的结果。
    文档可能已被删除或重新索引,现查会让历史答案指向一份当时并不存在的
    资料,归因就悄悄变了。见 `models.Message.sources`。
    """
    items, total = await SessionService(session).messages(
        session_id, user_id=user.id, limit=limit, offset=offset
    )
    return MessageListResponse(
        total=total, items=[MessageOut.model_validate(m) for m in items]
    )
