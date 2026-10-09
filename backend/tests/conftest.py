"""pytest 公共夹具。

两条原则:

1. **单元测试绝不打真实 API。** 真实调用既慢、又花钱、又依赖网络,三者任一抖动
   都会让 CI 变成噪音源。云端 Provider 的验证交给 `scripts/check_api.py`。

2. **测试绝不碰开发库。** 每个测试用一个独立的临时 SQLite 文件。共用文件的话,
   `test_register` 第二次跑就会因「用户名已占用」失败——这种测试会训练人忽略
   失败信息,比没有测试更糟。
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import Settings  # noqa: E402

from tests.helpers import (  # noqa: E402
    TEST_ADMIN_PASSWORD,
    TEST_ADMIN_USERNAME,
    admin_token,
    auth_header,
    register,
)

__all__ = ["admin_token", "auth_header", "register"]

# 测试专用哨兵。写成一等常量,是因为下面有一条断言拿它做「测试没读 .env」的证据,
# 而断言里再抄一遍字面量的话,改夹具的人改了字面量就会顺手改断言,守卫当场失效。
TEST_API_KEY = "sk-test-not-a-real-key"


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """隔离的配置实例:临时目录 + 可预测的模型名 + 快速哈希参数。"""
    # as_posix() 而非 str()——Windows 的 `sqlite:///C:\\...` 不是合法 URL,
    # 反斜杠会让 SQLAlchemy 的 URL 解析出错。URL 里一律用正斜杠。
    db_url = f"sqlite:///{(tmp_path / 'data' / 'app.db').as_posix()}"
    return Settings(
        _env_file=None,  # type: ignore[call-arg]  # 绕开 .env,测试必须确定性
        DATA_DIR=tmp_path / "data",
        CHROMA_DIR=tmp_path / "data" / "chroma",
        UPLOAD_DIR=tmp_path / "data" / "uploads",
        DATABASE_URL=db_url,
        DASHSCOPE_API_KEY=TEST_API_KEY,
        # 本地确定性 embedding,不发网络请求。
        #
        # **不能用「随便返回个常量」的假实现**:向量检索、RRF、重排
        # 都要拿向量做区分,假向量不区分输入的话,这几步在测试里等于没跑
        # (任何输入都"通过")。MockEmbedder 用字符 n-gram 哈希,字面越像
        # 向量越近,检索类断言才有意义。
        EMBED_BACKEND="mock",
        # **LLM 也必须走 mock。** 不设的话默认是 "dashscope",问答接口的
        # 测试会拿一把假 key 去连真实的百炼端点——慢、看网络脸色、而且
        # 一旦有人把真 key 留在 .env 里,测试就开始花钱。这与 EMBED_BACKEND
        # 是同一个坑,单独标出来是因为问答接口是本项目唯一直接调 LLM 的地方。
        LLM_BACKEND="mock",
        # **第三个联网入口。** 前两个(embedding / llm)关掉之后,问答接口
        # 仍然会去连真实的 rerank 端点——`build_reranker` 按 RERANK_ENABLED
        # 判断,而它默认是 True。症状是测试里冒出一条
        # `dashscope-rerank 调用失败 [401] InvalidApiKey`,看起来像断言写错,
        # 实际是测试在发真实网络请求。
        #
        # 用 "none" 而不是换成假实现:NoopReranker **按原顺序返回**,
        # 于是重排这一环的代码路径(重排 → 阈值截断)照样被真实执行,
        # 只是打分的来源不是网络。
        RERANK_BACKEND="none",
        # 向量库用临时目录。Chroma 是持久化的,不隔离的话测试之间会互相
        # 看到对方写入的向量,出现「单独跑能过、一起跑就挂」。
        JWT_SECRET="test-secret-do-not-use-in-production",
        ADMIN_USERNAME=TEST_ADMIN_USERNAME,
        ADMIN_PASSWORD=TEST_ADMIN_PASSWORD,
        # 默认参数每次哈希约 100ms,几十个用例会拖慢整个套件。
        # 测试要验的是「哈希能验对」,不是「哈希有多难爆破」。
        ARGON2_TIME_COST=1,
        ARGON2_MEMORY_COST=8192,
        ARGON2_PARALLELISM=1,
    )


@pytest.fixture
def client(settings: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    """接入了临时数据库的完整应用。

    做法是把 `Database` 实例整体替换掉,而不是 override `get_db` 依赖——
    因为 lifespan(建表 + 种子管理员)发生在依赖注入之外,只 override 依赖的话
    建表会建到开发库上,测试依然不隔离。

    两个替换点都要打:`app.db.base.database` 供 lifespan 使用(它在函数体内
    import),`app.core.deps.database` 供请求期使用(模块级 import 已绑定名字)。
    """
    import app.core.config as config_module
    import app.core.deps as deps_module
    import app.db.base as db_base_module
    import app.main as main_module
    from app.api.routers import health as health_module
    from app.db.base import Database
    from app.main import app

    settings.ensure_dirs()
    test_db = Database(settings)

    monkeypatch.setattr(db_base_module, "database", test_db)
    monkeypatch.setattr(deps_module, "database", test_db)
    # lifespan 里的 ensure_admin / 路径都走这个 settings,不换的话种子管理员
    # 会用真实 .env 的密码,测试就与外部的 .env 隐式耦合了
    monkeypatch.setattr(main_module, "settings", settings)

    # ---- settings 的每一份引用都要换掉,不能只换 app.main 那份 ----
    #
    # 曾经这里只打了 `app.main.settings`,而**测试照样全绿**——问题是它们
    # 绿得很不干净:`app.core.deps` 在 import 时就把 `get_settings` 绑进了
    # 自己的命名空间,`get_settings_dep()` 直接返回 lru_cache 里那个由真实
    # `.env` 构造的单例。于是:
    #   * 文档摄取用真实 DASHSCOPE_API_KEY 发网络请求(慢、花钱、不确定)
    #   * 向量写进真实的 data/chroma(污染开发库)
    # 而断言本身看不出异常,因为向量确实写进去了、文档确实 ready 了。
    #
    # 只改 `app.main.settings` 是在改「一个命名空间里的名字」,而每个
    # `from app.core.config import settings` 的模块都持有自己那份引用。
    # 所以下面逐个点名,而不是写个循环去扫——扫描看起来聪明,但哪天多一个
    # 导入点就会静默漏掉,而这里的每一行都是「漏了就会偷偷用真实配置」。
    monkeypatch.setattr(config_module, "settings", settings)
    monkeypatch.setattr(config_module, "get_settings", lambda: settings)
    monkeypatch.setattr(deps_module, "get_settings", lambda: settings)
    monkeypatch.setattr(health_module, "settings", settings)

    # 走依赖注入的路(`Depends(get_settings_dep)`)最稳的关法是 override,
    # 这也正是 get_settings_dep 存在的理由(见其 docstring)。
    monkeypatch.setitem(app.dependency_overrides, deps_module.get_settings_dep, lambda: settings)

    # 兜底断言:上面任何一行失效,这里立刻炸,而不是等到某天发现测试
    # 账单上多出几百次真实 API 调用。测试的可信度依赖它。
    assert deps_module.get_settings_dep() is settings
    assert main_module.settings is settings

    # **所有能发网络请求的 Provider 后端都必须是本地的。**
    # 这三条是逐个踩出来的:embedding、llm、rerank 各自默认指向云端,
    # 每漏关一个,症状都是「某个测试莫名其妙失败」,而根因在配置里,
    # 不在断言里。列成清单比散在三处注释可靠——将来再加一个 Provider,
    # 这条断言会提醒把它也算进来。
    assert settings.EMBED_BACKEND == "mock", "测试不能调真实 embedding 接口"
    assert settings.LLM_BACKEND == "mock", "测试不能调真实 LLM 接口"
    assert not settings.RERANK_ENABLED or settings.RERANK_BACKEND != "dashscope", (
        "测试不能调真实 rerank 接口"
    )
    # **不要拿 `has_api_key` 来断言这件事。** 它对 key 的要求是「字段非空且不等于
    # `.env.example` 里的占位符」,而本夹具的哨兵 `sk-test-not-a-real-key` 满足这两条,
    # 于是它会返回 True——写 `assert not settings.has_api_key` 只会得到一条
    # 「配置里不该带真实密钥」的失败信息,指向一个根本没发生的错误。
    #
    # 真正要守住的是另一个问题:**测试有没有偷偷读走开发机的 `.env`**。
    # 一旦有人拿掉 `_env_file=None`,夹具就会从磁盘捡起一把真钥匙、以及
    # `EMBED_BACKEND=dashscope` 这类真实配置,测试开始联网而不自知。
    # 上面三条后端断言管住了后半句,这条钉住前半句:钥匙必须是哨兵本身。
    # 两条合起来,「读了 .env」这个故障模式无论从哪一头露头都会当场炸。
    assert settings.DASHSCOPE_API_KEY == TEST_API_KEY, (
        "测试配置读到了非哨兵的 API key,检查 `_env_file=None` 是否还在"
    )

    # with 块才会触发 lifespan(建表、种子管理员)。不加的话库是空的。
    with TestClient(app) as c:
        yield c


@pytest.fixture
def admin_headers(client: TestClient) -> dict[str, str]:
    """管理员请求头。异步测试里拿不到 sync fixture 时用 `admin_token(client)` 直接调。"""
    return auth_header(admin_token(client))
