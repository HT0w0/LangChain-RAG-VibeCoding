"""响应模型的公共基类。

**为什么时间要有统一出口**:库里存的是 naive 的 UTC(SQLite 的 DATETIME
类型会静默丢掉 tzinfo),直接序列化得到的是没有偏移量的字符串,前端
`new Date("2026-10-08T12:00:00")` 会按**本地时区**解析——东八区用户看到的
时间整体偏 8 小时。

## 做法与它交过的学费

第一版是挂一个通配的 `@field_serializer("*")`,好处是新增字段自动被覆盖
(见下面「为什么不用 model_serializer」)。运行时行为一直是对的,两条
`+00:00` 的断言也一直绿着。

**但它把生成的 JSON Schema 里所有字段的类型全抹平了。** pydantic 在生成
序列化模式的 schema 时,会拿序列化器的返回标注去描述**每一个**字段,而那是
`Any`——于是 OpenAPI 里 `"id": {"title": "Id"}`,没有 `type`;`/docs` 页面上
类型全空。**没有任何测试红过**,因为没人断言过 schema,而「坏了的东西长得和
好的东西一模一样,只是信息更少」。直到要照着 OpenAPI 写前端类型时才发现。

## 为什么不用 `model_serializer(mode="wrap")`

那条路实测走不通,而且原因不显然:FastAPI 走的是 `model_dump(mode="json")`,
pydantic 会**先把 datetime 转成不含时区的字符串**,再交给 wrap 包装器——
那时拿到的已经是 `str` 而不是 `datetime`,时区永远补不上。它的 schema 是好的,
值是错的,比原版更难发现。

## 现在用的是 `UTCDateTime`

```python
created_at: UTCDateTime          # 而不是 datetime
last_login_at: UTCDateTime | None = None
```

`Annotated` + `PlainSerializer`:既保住时区,又让 schema 里这个字段仍然是
`string`。两个目标同时成立,代价是**要逐字段挂**。

漏挂一个,那个字段就退回成裸 datetime、输出不带时区——而症状只在特定时刻
才被注意到。所以由 `tests/test_schemas.py::TestEveryDatetimeUsesTheUtcAlias`
兜底:递归遍历所有 `UTCModel` 子类,发现裸 datetime 就红。**把「记得挂」
交给测试,而不是交给记性。**
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import BaseModel, PlainSerializer


def to_utc_iso(v: datetime) -> str:
    """naive 时间按 UTC 解释,输出带偏移量的 ISO 8601。"""
    if v.tzinfo is None:
        v = v.replace(tzinfo=UTC)
    return v.astimezone(UTC).isoformat()


#: 所有 datetime **出参**都用它,不要用裸 `datetime`。
#:
#: `when_used="json"` 是关键:Python 模式下的 `model_dump()` 仍然拿到
#: `datetime` 对象(给内部代码用),只有 JSON 模式才转成字符串(给线和前端用)。
UTCDateTime = Annotated[
    datetime, PlainSerializer(to_utc_iso, return_type=str, when_used="json")
]

#: 所有 datetime **入参**用它。目前请求体里没有时间字段,留着是为了让
#: 「出参和入参是两个不同的类型」这件事有个明确的名字。
#:
#: 入参侧不做自动换算:客户端给什么时区就按什么时区解析,由业务代码决定
#: 怎么处理——把入参也悄悄改时区,会让「用户填的 2026-01-01 是哪一天」
#: 这类问题变得无法回答。
UTCDateTimeIn = datetime


class UTCModel(BaseModel):
    """所有**出参**模型的基类。

    它自己不再做任何序列化处理——时区由字段上挂的 `UTCDateTime` 负责。
    留这个基类是为了给「哪些模型是响应模型」一个可枚举的标记:
    `tests/test_schemas.py` 靠它递归遍历,才能保证新增的响应模型
    自动被那条「不许有裸 datetime」的用例覆盖。
    """
