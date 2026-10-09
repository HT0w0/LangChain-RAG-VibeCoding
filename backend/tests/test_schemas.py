"""出参模型的形状,以及**生成的 OpenAPI 到底有没有类型**。

这一组用例的存在理由是一次真实的教训:

`UTCModel` 为了让 datetime 带上时区,挂了一个通配的
`@field_serializer("*")`。运行时行为是对的——两条 `+00:00` 的断言
(`test_auth` / `test_kb`)一直绿着。但 pydantic 在生成**序列化模式**的
JSON Schema 时,会拿序列化器的返回标注去描述**每一个**字段,于是
`-> Any` 把整份响应模型的类型全抹平了:

    修复前:`"id": {"title": "Id"}`        ← 没有 type
    修复后:`"id": {"type": "integer"}`

**没有任何测试红过**,因为没人断言过 schema;`/docs` 页面上类型全空也
没人看。直到要照着 OpenAPI 写前端类型时才发现——**坏了的东西长得和
好的东西一模一样**,只是信息更少。

所以这里除了验行为,还守着两件事:响应模型的 schema 不许再被抹平,
以及新增 datetime 字段时必须挂上 UTC 序列化器(`UTCDateTime`)——
后者是那个「忘记挂上去」的老问题,现在由测试来提醒,而不是靠记性。
"""

from __future__ import annotations

import types
import typing
from datetime import UTC, datetime

from app.schemas.common import UTCDateTime, UTCModel
from pydantic import BaseModel, PlainSerializer


def _unwrap_optional(annotation: object) -> object:
    """剥掉 `X | None` 的外层,拿到 X。不为 Optional 时原样返回。"""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return args[0]
    return annotation


def _base_is_datetime(annotation: object) -> bool:
    """标注的底是不是 datetime。裸 `datetime` 和 `Annotated[datetime, ...]` 都算。"""
    if annotation is datetime:
        return True
    args = typing.get_args(annotation)
    return bool(getattr(annotation, "__metadata__", None)) and args[0] is datetime


def _is_utc_guarded(field: object) -> bool:
    """这个字段是不是「datetime 且挂了把时间转成带时区字符串的序列化器」。

    **`Annotated` 的元数据有两处可能,必须都看。** pydantic 建模时会把
    `Annotated[X, meta]` 拆成 `annotation=X` + `metadata=[meta]`,但**只在
    `Annotated` 位于顶层时**这么拆;写成 `Annotated[X, meta] | None` 的,
    整个 `Annotated` 留在 annotation 里。

    于是同一个别名、同一份代码,两种写法在 `model_fields` 上长得完全不一样:

        created_at: UTCDateTime          → annotation=datetime, metadata=[...]
        last_login_at: UTCDateTime | None → annotation=Annotated[datetime, ...]

    只判其中一处的检查会把另一处**整批误判**成「没挂」——而且误判的方向
    是随写法而变的,不是随机漏几个。
    """
    base = _unwrap_optional(field.annotation)
    if not _base_is_datetime(base):
        return False
    metadata = [*field.metadata, *getattr(base, "__metadata__", ())]
    return any(isinstance(m, PlainSerializer) for m in metadata)


def _all_subclasses(cls: type) -> list[type]:
    out: list[type] = []
    for sub in cls.__subclasses__():
        out.append(sub)
        out.extend(_all_subclasses(sub))
    return out


class TestUtcSerialization:
    def test_a_naive_datetime_comes_out_with_an_offset(self) -> None:
        """库里存的是 naive UTC(SQLite 的 DATETIME 会静默丢掉 tzinfo)。

        不带偏移量的话,前端 `new Date("2026-10-08T12:00:00")` 会按**本地
        时区**解析——东八区用户看到的时间整体偏 8 小时,而这个症状只在
        北京时间下午才会被注意到。
        """
        from app.schemas.auth import UserOut

        out = UserOut(id=1, username="u", role="user", is_active=True, created_at=datetime(2026, 1, 2, 3, 4, 5))

        assert out.model_dump(mode="json")["created_at"] == "2026-01-02T03:04:05+00:00"

    def test_an_aware_datetime_is_converted_to_utc(self) -> None:
        """已经是 aware 的要换算到 UTC,而不是原样输出当地时间。"""
        from app.schemas.auth import UserOut

        aware = datetime(2026, 1, 2, 11, 4, 5, tzinfo=UTC)
        out = UserOut(id=1, username="u", role="user", is_active=True, created_at=aware)

        assert out.model_dump(mode="json")["created_at"] == "2026-01-02T11:04:05+00:00"


class TestOpenApiKeepsTypes:
    """**生成的 schema 必须带类型。** 前端的类型定义照它写,`/docs` 也看它。"""

    def test_response_models_are_not_wiped_out(self) -> None:
        """这条是那次教训的回归测试。

        修复之前 `UserOut.id` 在 OpenAPI 里只有 `{"title": "Id"}`——
        FastAPI 的响应 schema 走**序列化模式**,而通配序列化器把每个字段的
        类型都换成了它自己的返回标注 `Any`。

        断言取的是最小的形状(`id` 是整数),因为这里要守的是
        「类型信息在不在」,不是「某个字段恰好是什么类型」——后者由每条
        接口自己的用例去管。
        """
        from app.main import app

        props = app.openapi()["components"]["schemas"]["UserOut"]["properties"]

        assert props["id"].get("type") == "integer", f"字段类型被抹平了:{props['id']}"
        assert props["username"].get("type") == "string"

    def test_datetime_fields_are_declared_as_strings(self) -> None:
        """datetime 出参在 JSON 里是字符串,而 schema 得说清楚这一点。

        它和上一条不是一回事:上一条守「类型没被抹平」,这条守
        「datetime 被序列化成了字符串,而不是被写成某个对象类型」——
        前端要按字符串解析它。
        """
        from app.main import app

        props = app.openapi()["components"]["schemas"]["UserOut"]["properties"]

        assert props["created_at"].get("type") == "string"


class TestTheDatetimeGuard:
    """先验守卫**本身**,再拿它去扫真实模型。

    顺序是有意的:一个只会说「没问题」的检查器,套在任何代码上都绿。
    下面两条用一个不进 `UTCModel` 注册表的临时模型(所以不会污染真正的
    那一遍扫描)把检查器的两个方向都钉住。
    """

    class _Sample(BaseModel):
        naked: datetime
        guarded: UTCDateTime
        guarded_optional: UTCDateTime | None = None

    def test_it_accepts_the_alias_in_both_of_its_shapes(self) -> None:
        """**同一个别名有两种可见形状,检查器必须两种都认。**

        这条是踩出来的:pydantic 会把 `Annotated[X, meta]` 拆成
        `annotation=X` + `metadata=[meta]`,但**只在 `Annotated` 位于顶层时**
        这么拆;写成 `X | None` 的,整个 `Annotated` 留在 annotation 里。

        只认一种形状的检查器会把另一种**整批**误判成「漏挂了」——而且
        误判方向随写法而定,不是随机漏几个。第一版就是这么错的。
        """
        fields = self._Sample.model_fields

        assert _is_utc_guarded(fields["guarded"])
        assert _is_utc_guarded(fields["guarded_optional"])

    def test_it_rejects_a_bare_datetime(self) -> None:
        """另一个方向:裸 datetime 必须被判为「没挂」。"""
        field = self._Sample.model_fields["naked"]

        assert _base_is_datetime(_unwrap_optional(field.annotation))
        assert not _is_utc_guarded(field)

    def test_no_schema_declares_a_bare_datetime(self) -> None:
        """**每一处 datetime 出参都必须用 `UTCDateTime`。**

        这是那个「忘记挂上去」问题的守门人。改用 `Annotated` 别名之后,
        行为不再由基类的通配序列化器兜底——漏挂一个字段,它就退回成
        没有时区的裸 datetime,而症状(时间偏 8 小时)只在特定时刻才被
        注意到。测试来提醒,比靠记性可靠。

        递归遍历所有 `UTCModel` 子类,所以新增一个响应模型时它自动被覆盖。
        """
        offenders: list[str] = []
        for cls in _all_subclasses(UTCModel):
            for name, field in cls.model_fields.items():
                base = _unwrap_optional(field.annotation)
                if _base_is_datetime(base) and not _is_utc_guarded(field):
                    offenders.append(f"{cls.__name__}.{name}")

        assert not offenders, (
            "这些 datetime 字段没用 UTCDateTime,会输出不带时区的时间:"
            + ", ".join(offenders)
        )

    def test_the_walk_actually_sees_the_models(self) -> None:
        """上一条靠遍历子类,所以**遍历本身**也要能失败。

        遍历写错(比如 `__subclasses__` 拿到的是空列表)的话,上一条会
        毫无怨言地绿——那是「一个从不检查任何东西的守卫」。这条用一个
        已知存在的模型钉住遍历是活的。
        """
        names = {c.__name__ for c in _all_subclasses(UTCModel)}

        assert "UserOut" in names, f"遍历没走到真实的模型上,只看到:{sorted(names)}"
