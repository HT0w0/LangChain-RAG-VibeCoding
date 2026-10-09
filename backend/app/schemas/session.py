"""会话与消息的请求/响应模型。

`user_id` **出现在出参里是刻意的**:前端拿到它就能确认「这条确实是我的」,
而不是盲信服务端已经过滤过。它本身不是秘密(客户端自己的 id),
泄露不了任何东西;真正要防的是 id 被人拿去当查询条件,而那个防线在仓储层。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from app.schemas.common import UTCModel

MAX_TITLE_LENGTH = 200
DEFAULT_SESSION_TITLE = "新会话"


class SessionCreate(BaseModel):
    title: str | None = Field(default=None, max_length=MAX_TITLE_LENGTH)
    # None = 不限(全部知识库);[] = 什么都不看。两者语义不同,不可互换。
    kb_ids: list[int] | None = None

    @field_validator("title")
    @classmethod
    def _not_blank(cls, value: str | None) -> str | None:
        if value is None:
            return None
        stripped = value.strip()
        if not stripped:
            # 与 `chat.ChatRequest.question` 同一个坑:`min_length=1` 放得过
            # `"   "`,而一个纯空白的标题在列表页上就是一条点不开的空行。
            raise ValueError("标题不能为空")
        return stripped


class SessionRename(BaseModel):
    title: str = Field(min_length=1, max_length=MAX_TITLE_LENGTH)

    @field_validator("title")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("标题不能为空")
        return stripped


class SessionOut(UTCModel):
    model_config = {"from_attributes": True}

    id: int
    user_id: int
    title: str
    kb_ids: list[int] | None = None
    message_count: int
    created_at: datetime
    updated_at: datetime
    last_message_at: datetime


class SessionListResponse(UTCModel):
    total: int
    items: list[SessionOut]


class MessageOut(UTCModel):
    model_config = {"from_attributes": True}

    id: int
    session_id: int
    role: str
    content: str
    # 来源快照。**类型是 `list[dict]` 而不是某个 SourceRef 模型**,因为它是
    # 一个**历史存档**:存进去时是那个版本的结构,读出来时不该被今天的
    # schema 变更影响。用强类型模型的话,哪天给 SourceRef 加一个必填字段,
    # 所有旧消息都会在反序列化时炸掉——而它们本来是好端端的。
    sources: list[dict[str, Any]] = Field(default_factory=list)
    cited: list[int] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None
    created_at: datetime


class MessageListResponse(UTCModel):
    total: int
    items: list[MessageOut]
