"""知识库与文档的请求/响应模型。"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.common import UTCDateTime, UTCModel

MAX_KB_NAME = 128
MAX_KB_DESCRIPTION = 512


def _clean_name(v: str) -> str:
    v = v.strip()
    if not v:
        # 全空格的名字在列表里看起来像一条空白记录,但唯一约束又会因为
        # 「多个空格串」而互相冲突。必须在入口挡掉。
        raise ValueError("知识库名称不能为空")
    if len(v) > MAX_KB_NAME:
        raise ValueError(f"知识库名称最长 {MAX_KB_NAME} 个字符")
    return v


class KBCreateRequest(BaseModel):
    name: str
    description: str | None = Field(default=None, max_length=MAX_KB_DESCRIPTION)

    _v_name = field_validator("name")(_clean_name)


class KBUpdateRequest(BaseModel):
    name: str | None = None
    description: str | None = Field(default=None, max_length=MAX_KB_DESCRIPTION)

    @field_validator("name")
    @classmethod
    def _v_name(cls, v: str | None) -> str | None:
        # None 表示「不改名」,与「改成空名」是两回事
        return _clean_name(v) if v is not None else None


class KBOut(UTCModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    description: str | None = None
    created_by: int | None = None
    doc_count: int
    chunk_count: int
    created_at: UTCDateTime
    updated_at: UTCDateTime


class KBListResponse(UTCModel):
    total: int
    items: list[KBOut]


class DocumentOut(UTCModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    kb_id: int
    filename: str
    mime: str | None = None
    size: int
    status: str
    error: str | None = None
    warnings: list[str] = Field(default_factory=list)
    chunk_count: int
    uploaded_by: int | None = None
    created_at: UTCDateTime
    updated_at: UTCDateTime


class DocumentListResponse(UTCModel):
    total: int
    items: list[DocumentOut]


class ChunkPreview(UTCModel):
    """分块预览。让管理员在入库前确认切分是否合理。

    这是「分块参数调优」的可用性支撑:chunk_size 是全局配置,管理员改不了,
    但至少要能**看见**默认值切出来长什么样,才知道该不该找开发调。
    """

    chunk_index: int
    content: str
    token_count: int
    meta: dict = Field(default_factory=dict)


class ChunkPreviewPage(UTCModel):
    document_id: int
    filename: str
    total: int
    items: list[ChunkPreview]
