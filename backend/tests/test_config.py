"""配置层测试。

重点在**路径解析**——这是 Windows 上最容易出「本机跑得通、换台机器就找不到
文件」的地方,而它平时不报错,只在某个偏僻分支上炸。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from app.core.config import PROJECT_ROOT, Settings
from pydantic import ValidationError


def test_resolved_paths_are_absolute(settings: Settings) -> None:
    assert settings.resolved_data_dir.is_absolute()
    assert settings.resolved_chroma_dir.is_absolute()
    assert settings.resolved_upload_dir.is_absolute()


def test_relative_paths_anchor_at_project_root() -> None:
    """相对路径按项目根解析,不受 cwd 影响。

    uvicorn 从哪个目录启动是不确定的,如果按 cwd 解析,
    `data/` 会在不同目录下散落出好几份。
    """
    s = Settings(
        _env_file=None,  # type: ignore[call-arg]
        DATA_DIR=Path("data"),
        CHROMA_DIR=Path("data/chroma"),
        UPLOAD_DIR=Path("data/uploads"),
    )
    assert s.resolved_data_dir == PROJECT_ROOT / "data"


def test_absolute_paths_are_left_alone(tmp_path: Path, settings: Settings) -> None:
    assert settings.resolved_data_dir == tmp_path / "data"


def test_sqlite_path_extraction(settings: Settings) -> None:
    assert settings.sqlite_path.name == "app.db"
    assert settings.sqlite_path.parent.name == "data"


def test_native_rerank_url_shape(settings: Settings) -> None:
    """重排只有原生端点。兼容端点实测 404,拼错前缀会直接 404 而不是报错。"""
    url = settings.native_rerank_url
    assert url == (
        "https://dashscope.aliyuncs.com/api/v1"
        "/services/rerank/text-rerank/text-rerank"
    )


def test_the_rerank_threshold_is_actually_on() -> None:
    """**结构性的第二道防线必须是开着的。**

    `RERANK_SCORE_THRESHOLD=0.0` 时,阈值截断的机制照样在跑,只是**一次也不
    生效**——等价于「不相关的资料别进上下文」全靠提示词约束模型。这个状态下
    代码看起来完全正常,测试也全绿(验过滤的那条用例自己传 0.6),
    只有库里的回答开始答非所问时才看得出来。

    钉的是「这一刀要落下」这个决定,不是某个精确数字:0.4 还是 0.7 属于
    `docs/RERANK-THRESHOLD.md` 里那个取舍,改它的时候该一起改那份文档。
    这条用例红了,意思是有人把第二道防线关掉了——先问为什么。
    """
    assert Settings(_env_file=None).RERANK_SCORE_THRESHOLD > 0.0  # type: ignore[call-arg]


def test_has_api_key_rejects_placeholder() -> None:
    assert Settings(_env_file=None, DASHSCOPE_API_KEY="").has_api_key is False  # type: ignore[call-arg]
    assert (  # type: ignore[call-arg]
        Settings(_env_file=None, DASHSCOPE_API_KEY="sk-ws-xxxxxxxx").has_api_key is False
    )
    assert (  # type: ignore[call-arg]
        Settings(_env_file=None, DASHSCOPE_API_KEY="sk-ws-real").has_api_key is True
    )


def test_default_secret_warns() -> None:
    """默认 JWT 密钥必须告警,否则它会被一路带到生产。"""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        Settings(_env_file=None, JWT_SECRET="dev-only-insecure-secret-change-me")  # type: ignore[call-arg]
    assert any("JWT_SECRET" in str(w.message) for w in caught)


def test_默认密钥在生产环境必须拦住启动() -> None:
    """**告警不够,生产环境必须起不来。**

    密钥就是那个「只有服务器知道的印章」——它证明一张登录凭证确实是服务器
    签发的。而这个默认值是**写在公开仓库里的**,全世界都能看到。

    所以「生产环境 + 默认密钥」等于:任何读过这个仓库的人都能**自己签一张
    凭证**,声称自己是管理员,而服务器会盖章放行。

    为什么必须**拦住启动**而不是只告警:这条错误**没有任何别的症状** ——
    服务跑得好好的,直到有人拿管理员权限进来。而启动日志里的警告是最容易
    被忽略的东西,云平台上刷一下就过去了。**只有「起不来」会逼人当场处理。**
    """
    with pytest.raises(ValidationError, match="JWT_SECRET"):
        Settings(  # type: ignore[call-arg]
            _env_file=None,
            APP_ENV="prod",
            JWT_SECRET="dev-only-insecure-secret-change-me",
        )


def test_生产环境换成真密钥就正常启动() -> None:
    """拦的是「默认值」,不是「生产环境」本身 —— 别把正常部署也拦死。"""
    s = Settings(  # type: ignore[call-arg]
        _env_file=None,
        APP_ENV="prod",
        JWT_SECRET="a-real-secret-that-is-not-the-default-one",
    )
    assert s.APP_ENV == "prod"


def test_开发环境用默认密钥只告警不拦() -> None:
    """本地开发不受影响 —— 拦只在「明确声明这是生产」时才生效。"""
    import warnings

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        s = Settings(  # type: ignore[call-arg]
            _env_file=None, APP_ENV="dev", JWT_SECRET="dev-only-insecure-secret-change-me"
        )
    assert s.APP_ENV == "dev"


def test_ensure_dirs_creates_everything(settings: Settings) -> None:
    settings.ensure_dirs()
    assert settings.resolved_data_dir.is_dir()
    assert settings.resolved_chroma_dir.is_dir()
    assert settings.resolved_upload_dir.is_dir()
    assert settings.sqlite_path.parent.is_dir()
