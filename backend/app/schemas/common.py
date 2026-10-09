"""响应模型的公共基类。

**为什么时间要有统一出口**:库里存的是 naive 的 UTC(SQLite 的 DATETIME
类型会静默丢掉 tzinfo),直接序列化得到的是没有偏移量的字符串,前端
`new Date("2026-10-08T12:00:00")` 会按**本地时区**解析——东八区用户
看到的时间整体偏 8 小时。

散在每个模型里各写一个 `field_serializer` 的问题是:新增字段时很容易
忘记挂上去,而忘记的症状(时间偏 8 小时)只在北京时间下午才会被注意到。
基类用 `mode="wrap"` 遍历所有输出字段自动补时区,新字段默认就是对的。
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, field_serializer


def to_utc_iso(v: datetime) -> str:
    """naive 时间按 UTC 解释,输出带偏移量的 ISO 8601。"""
    if v.tzinfo is None:
        v = v.replace(tzinfo=UTC)
    return v.astimezone(UTC).isoformat()


class UTCModel(BaseModel):
    """所有出参模型的基类。自动把 datetime 字段规范成带时区的 ISO 字符串。"""

    # 通配符字段序列化器,子类新增的 datetime 字段自动被覆盖,不必逐个登记。
    #
    # 注意**不能用 `model_serializer(mode="wrap")` 做这件事**:
    # FastAPI 走的是 `model_dump(mode="json")`,而在 JSON 模式下 pydantic
    # 会先把 datetime 转成不含时区的字符串,再交给 wrap 包装器——
    # 那时拿到的已经是 str 而不是 datetime,时区永远补不上。
    @field_serializer("*")
    def _normalize_datetime(self, value: Any) -> Any:
        return to_utc_iso(value) if isinstance(value, datetime) else value
