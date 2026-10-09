"""管理员操作的请求/响应模型。"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class SetActiveRequest(BaseModel):
    is_active: bool


class SetRoleRequest(BaseModel):
    # 用 Literal 而不是 str:非法角色在**入参校验**阶段就被挡掉,
    # 不会流到服务层变成一次「写入了未知角色」的静默数据损坏。
    role: Literal["admin", "user"]
