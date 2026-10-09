"""全局配置。

所有模型名与端点一律走配置，代码里绝不硬编码——换模型、换网关不应改代码。

路径类字段统一在 Settings 里解析成绝对路径（`resolved_*`），因为工作目录
会随 uvicorn 启动位置变化，相对路径会导致「本地跑得好、换个目录就找不到库」。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import computed_field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/app/core/config.py -> backend/app/core -> backend/app -> backend -> 项目根
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ---------- 应用 ----------
    APP_NAME: str = "电商知识库 RAG 问答系统"
    APP_ENV: Literal["dev", "prod"] = "dev"
    # 默认 False:漏配 .env 时应该少说话,而不是把 DEBUG 日志和详细报错暴露出去。
    # 本地开发由 .env 里的 `DEBUG=true` 显式打开。
    DEBUG: bool = False
    API_PREFIX: str = "/api"

    # ---------- 安全 ----------
    JWT_SECRET: str = "dev-only-insecure-secret-change-me"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 720

    # argon2 参数。默认值按「单机、低并发」调过：64MB 内存 / 3 轮。
    # 测试里会调低——默认参数下每次哈希约 100ms，几十个测试用例会拖慢整个套件，
    # 而测试要验的是「哈希能验对」而不是「哈希有多难爆破」。
    ARGON2_TIME_COST: int = 3
    ARGON2_MEMORY_COST: int = 65536
    ARGON2_PARALLELISM: int = 4

    ADMIN_USERNAME: str = "admin"
    ADMIN_PASSWORD: str = "123456"

    # ---------- 阿里云百炼 ----------
    DASHSCOPE_API_KEY: str = ""
    DASHSCOPE_BASE_URL: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DASHSCOPE_NATIVE_URL: str = "https://dashscope.aliyuncs.com/api/v1"

    # ---------- 模型 ----------
    LLM_MODEL: str = "qwen3.7-plus"
    # qwen3.7-plus 是思考模型，默认产出 reasoning_content。实测关掉可显著降首 token 时延。
    LLM_ENABLE_THINKING: bool = False
    LLM_TEMPERATURE: float = 0.3
    LLM_MAX_TOKENS: int = 2048

    EMBED_MODEL: str = "qwen3.7-text-embedding"
    EMBED_DIM: int = 1024
    EMBED_BATCH_SIZE: int = 10  # check_api.py 探测到上限后可回填

    RERANK_MODEL: str = "qwen3.7-text-rerank"
    RERANK_ENABLED: bool = True
    RERANK_TOP_N: int = 5

    # ---------- Provider 后端 ----------
    # mock = 确定性本地向量化,不联网。测试与离线开发用;见 providers/embedding.py
    EMBED_BACKEND: Literal["dashscope", "local", "mock"] = "dashscope"
    RERANK_BACKEND: Literal["dashscope", "local", "none"] = "dashscope"
    LLM_BACKEND: Literal["dashscope", "anthropic_compat", "mock"] = "dashscope"

    # 工作空间的 Anthropic 兼容网关（实测认证通过），作为备选通路展示「同逻辑换协议」
    ANTHROPIC_COMPAT_BASE_URL: str = ""

    # ---------- 检索 ----------
    CHUNK_SIZE: int = 640
    CHUNK_OVERLAP: int = 96
    RETRIEVE_TOP_K: int = 20
    RRF_K: int = 60
    # **实测值,不是拍的。** 机制在 `retrieval._apply_rerank` 里,这里给的是
    # 它终于开始生效的那个数。完整的扫描表与分析见 `docs/RERANK-THRESHOLD.md`。
    #
    # 为什么是 0.5(探针 `scripts/rerank_probe.py`,22 段语料 / 28 个问题,
    # qwen3.7-text-rerank,实测 2026-10-09):
    #   * 0.50 是**第一个**让跑题问题全部拒答的阈值。再往下探没有意义——
    #     0.40 能保住的东西 0.50 一样保得住,却多放进 14 条无关段落;
    #   * 可答率一直到 0.70 都还是 22/22,所以 0.5 离观测到的最差相关段落
    #     (0.7837)留了 0.28 的余量。
    #
    # **余量是故意留的,因为两个方向的失败不对称:**
    #   * 偏低 → 多带一条无关段落,由提示词兜底(资料是数据,不是指令);
    #   * 偏高 → 丢掉本该答得出的资料,**没有任何兜底**。用户只会看到一句
    #     「资料中没有找到相关内容」,看不出这是配置造成的。
    # 真实问法必然比手写问题乱(错别字、一句问三件事、指代不明),
    # 上界只会下移,所以宁可把余量留在下面。
    #
    # 反直觉的一点(这正是不靠感觉给数的理由):跑题问题在**相关段落**上的
    # 分数(0.1527)可以高于正常问题里**无关段落**的分数(0.0637)。
    # 两个目标是交错的,不是单调的,只能扫出来。
    RERANK_SCORE_THRESHOLD: float = 0.5
    # 自行估算上下文体量，不用 trim_messages——ChatOpenAI 对 qwen 的
    # get_num_tokens_from_messages 未实现，会抛 NotImplementedError。
    CONTEXT_TOKEN_BUDGET: int = 3000

    # ---------- 多轮 ----------
    # 历史窗口的预算**与检索资料分开计**,不共用一个总预算。
    #
    # 合在一起的话,一个聊了几十轮的会话会把预算吃光,留给检索资料的
    # 只剩几百 token——而检索资料才是这个系统相对通用大模型的全部价值,
    # 历史只是让追问能被听懂。所以宁可让历史先被牺牲:它是锦上添花,
    # 资料是立身之本。
    HISTORY_TOKEN_BUDGET: int = 1500
    HISTORY_MAX_TURNS: int = 6
    # 改写是每轮多付的一次额外 LLM 调用。可以关掉——排查「检索为什么
    # 没命中」时,需要一个完全不改写的基线,否则分不清是改写带偏了
    # 还是检索本身不行。
    REWRITE_ENABLED: bool = True
    REWRITE_MAX_TURNS: int = 3
    # 喂给改写器的历史要比喂给回答的短得多:它只需要知道「最新一句在聊什么」。
    REWRITE_TOKEN_BUDGET: int = 600
    # 改写结果是一句话,给 128 够用。给多了模型有空间开始解释、加引号、
    # 输出「改写后的问题:...」这类前缀,而那些都会污染检索。
    REWRITE_MAX_TOKENS: int = 128

    # ---------- 存储 ----------
    DATA_DIR: Path = Path("data")
    DATABASE_URL: str = "sqlite:///data/app.db"
    CHROMA_DIR: Path = Path("data/chroma")
    UPLOAD_DIR: Path = Path("data/uploads")
    CHROMA_COLLECTION: str = "rag_chunks"
    MAX_UPLOAD_MB: int = 50

    # ---------- 缓存 ----------
    CACHE_ENABLED: bool = True
    RETRIEVAL_CACHE_TTL_SECONDS: int = 600
    # 答案缓存的 TTL 比检索缓存长:它省下的是一次完整的 LLM 调用(贵得多,
    # 而且用户要等),命中率的价值更高。**代价是错的答案也留得更久**,所以
    # 这一层靠的不是 TTL 而是代次——知识库一写入就立刻读不到了,TTL 只负责
    # 回收空间。
    ANSWER_CACHE_TTL_SECONDS: int = 1800
    # **语义缓存默认关闭,这是刻意的。** 另外两层给错了只是「慢一点还是
    # 快一点」的问题;这一层给错了,用户拿到的是**另一个问题的答案**——
    # 措辞、来源、编号全都齐整,因为它本来就是一份真答案。它只是回答错了。
    #
    # 打开它需要先接受一个测出来的前提:真实 embedding 下,「同义改写」与
    # 「同实体不同属性」的余弦能不能被一个阈值分开。实测(8 对应命中 +
    # 12 对应错配,阈值扫描)存在零错配区间 (0.7924, 0.8491],详见
    # `docs/SEMANTIC-CACHE.md`。阈值取 0.95 落在这个区间上方约 0.16,
    # 是偏保守的一侧——**代价是那 8 对里只命中 5 对**,别指望它很能省。
    #
    # 危险是**单侧**的:假阳性给出错答案,假阴性只是一次未命中。所以宁可
    # 把阈值往高里放——这也是它敢默认关闭的原因,关闭时唯一损失是命中率。
    SEMANTIC_CACHE_ENABLED: bool = False
    SEMANTIC_CACHE_TTL_SECONDS: int = 3600
    SEMANTIC_CACHE_THRESHOLD: float = 0.95

    # ---------- 限流 ----------
    RATE_LIMIT_QPM: int = 30

    @field_validator("JWT_SECRET")
    @classmethod
    def _warn_default_secret(cls, v: str) -> str:
        if v.startswith("dev-only-insecure"):
            import warnings

            warnings.warn(
                "JWT_SECRET 仍是默认值，生产环境必须替换。"
                '生成：python -c "import secrets; print(secrets.token_urlsafe(48))"',
                stacklevel=2,
            )
        return v

    # ---------- 派生路径 ----------
    # 相对路径统一按项目根解析，避免受 cwd 影响。

    def _resolve(self, p: Path) -> Path:
        return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_data_dir(self) -> Path:
        return self._resolve(self.DATA_DIR)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_chroma_dir(self) -> Path:
        return self._resolve(self.CHROMA_DIR)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def resolved_upload_dir(self) -> Path:
        return self._resolve(self.UPLOAD_DIR)

    @property
    def sqlite_path(self) -> Path:
        """从 DATABASE_URL 抽出物理路径，供建目录用。"""
        url = self.DATABASE_URL
        prefix = "sqlite:///"
        if url.startswith(prefix):
            return self._resolve(Path(url[len(prefix) :]))
        raise ValueError(f"暂不支持的 DATABASE_URL: {url}")

    def ensure_dirs(self) -> None:
        """启动时创建全部运行时目录。"""
        for d in (self.resolved_data_dir, self.resolved_chroma_dir, self.resolved_upload_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)

    @property
    def native_rerank_url(self) -> str:
        """重排只能走原生端点——兼容端点没有 /rerank（实测 404）。"""
        return f"{self.DASHSCOPE_NATIVE_URL.rstrip('/')}/services/rerank/text-rerank/text-rerank"

    @property
    def has_api_key(self) -> bool:
        return bool(self.DASHSCOPE_API_KEY and not self.DASHSCOPE_API_KEY.startswith("sk-ws-xxx"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
