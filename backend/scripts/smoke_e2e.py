"""端到端实弹自检:真 embedding + 真检索 + 真重排 + 真 LLM,走完整条 SSE 链路。

**它和 `tests/` 是两件事,不要合并。**

`tests/` 里所有 Provider 都是 mock(见 conftest 的兜底断言),因为单元测试
不该联网;代价是**「接上真实服务之后还成不成立」这件事没有任何测试覆盖**。
本脚本补的正是这一段:

  * 真 embedding 写进真 Chroma,检索能不能召回
  * 真 rerank 的分数落在什么量级,阈值截断会不会把结果全砍掉
  * 真 LLM 在**真实提示词**下会不会按 `[1]` 的格式标注引用
    —— mock 是照着我们写的格式回话的,它证明不了模型也这么干
  * 追问「它的库存还有多少」时,真模型会不会把代词补全成一个能独立检索的
    问题——**这一条 mock 原理上验不了**,它只会照抄提示词里的例子
  * usage 有没有真的流到 `done` 事件里

它不该进 CI:联网、花钱、结果依赖模型当下的输出。它是**发版前手动跑一次**
的验收入口,以及作品集里「这条链路对着真实平台跑通过」的证据。

用法:
    python backend/scripts/smoke_e2e.py
    python backend/scripts/smoke_e2e.py --question "RX9070 的库存是多少"
    python backend/scripts/smoke_e2e.py --followup "那它的价格呢"
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# 语料刻意做成电商场景的样子:同一份表里有能直接回答的问题,
# 也有表里没有的问题(用来验「不知道就说不知道」)。
CSV = (
    "商品名称,类目,价格,库存\n"
    "RX9070 显卡,显卡,4299,12\n"
    "27寸显示器,显示器,1299,30\n"
).encode()

KB_NAME = "实弹自检库"

# 表里**没有**的信息。用来验模型不会撒谎——这是整个产品存在的前提。
UNANSWERABLE = "这款显卡的保修期是几年"

# 追问。**刻意不带商品名**:「它」指谁只能从上一轮推出来。若模型没做
# 指代消解,这句话直接拿去检索会召回到所有含「库存」的片段。
FOLLOWUP = "它的库存还有多少"

# 改写成功的判据。不判长度:提示词明确要求「已经完整的句子原样返回」,
# 所以长一点短一点都不是信号;而代词被补全成具体商品名是。
# 与 CSV 里的商品名保持一致。
REWRITE_MUST_MENTION = ("RX9070", "显卡")


class _Reporter:
    def __init__(self) -> None:
        self.failed: list[str] = []

    def check(self, name: str, ok: bool, detail: str = "") -> bool:
        mark = "OK  " if ok else "FAIL"
        line = f"  [{mark}] {name}"
        if detail:
            line += f"  — {detail}"
        print(line)
        if not ok:
            self.failed.append(name)
        return ok


def parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    for block in body.replace("\r\n", "\n").split("\n\n"):
        name = data = None
        for line in block.split("\n"):
            if line.startswith(":"):
                continue
            key, _, value = line.partition(":")
            value = value.lstrip(" ")
            if key == "event":
                name = value
            elif key == "data":
                data = value
        if name and data is not None:
            events.append((name, json.loads(data)))
    return events


def main() -> int:
    parser = argparse.ArgumentParser(description="端到端实弹自检")
    parser.add_argument("--question", default="RX9070 显卡多少钱")
    parser.add_argument("--unanswerable", default=UNANSWERABLE)
    parser.add_argument("--followup", default=FOLLOWUP)
    args = parser.parse_args()

    import app.core.config as config_module
    import app.core.deps as deps_module
    import app.db.base as db_base_module
    import app.main as main_module
    from app.core.config import Settings
    from app.db.base import Database
    from app.main import app
    from fastapi.testclient import TestClient

    # 真 key 与真 Provider 都来自 .env;只把落盘位置换成临时目录,
    # 免得自检把开发库和 data/chroma 污染成「跑过一次就有的状态」。
    tmp = Path(tempfile.mkdtemp(prefix="smoke-e2e-"))
    settings = Settings(
        DATA_DIR=tmp / "data",
        CHROMA_DIR=tmp / "data" / "chroma",
        UPLOAD_DIR=tmp / "data" / "uploads",
        DATABASE_URL=f"sqlite:///{(tmp / 'data' / 'app.db').as_posix()}",
        JWT_SECRET="smoke-e2e-only-not-a-secret",
        ARGON2_TIME_COST=1,
        ARGON2_MEMORY_COST=8192,
    )

    print("\n  端到端实弹自检")
    print("  " + "─" * 62)
    print(f"  LLM        {settings.LLM_MODEL}  (backend={settings.LLM_BACKEND})")
    print(f"  Embedding  {settings.EMBED_MODEL}  (backend={settings.EMBED_BACKEND})")
    print(f"  Rerank     {settings.RERANK_MODEL}  (enabled={settings.RERANK_ENABLED})")
    print(f"  临时数据   {tmp}")
    print("  " + "─" * 62)

    # 没有真 key 就没有实弹可言,直接说清楚而不是让它跑一半崩在 401 上。
    if not settings.has_api_key:
        print("\n  .env 里没有可用的 DASHSCOPE_API_KEY,实弹自检无从谈起。\n")
        return 2
    if settings.LLM_BACKEND == "mock" or settings.EMBED_BACKEND == "mock":
        print("\n  当前配置走的是 mock 后端,这不是实弹自检。")
        print("  请在 .env 里把 EMBED_BACKEND/LLM_BACKEND 设成 dashscope。\n")
        return 2

    settings.ensure_dirs()
    test_db = Database(settings)
    db_base_module.database = test_db
    deps_module.database = test_db
    main_module.settings = settings
    config_module.settings = settings
    config_module.get_settings = lambda: settings
    deps_module.get_settings = lambda: settings
    app.dependency_overrides[deps_module.get_settings_dep] = lambda: settings

    report = _Reporter()

    with TestClient(app) as client:
        # ---- 登录取 token ----
        login = client.post(
            "/api/auth/login",
            json={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD},
        )
        # **成功的响应体不能打,失败的可以。**
        #
        # 这个脚本的输出会被贴进作品集材料里。而**登录成功的响应体里是一枚 JWT** ——
        # 哪怕它是用一次性密钥签给一个临时库的、真拿去用也没有任何权限，
        # 把 token 打进截图和日志也是个坏习惯，不该出现在演示脚本里当范例。
        # 所以成功分支只报状态码；下面失败分支打的 `login.text` 是错误 JSON、
        # 不含 token。
        #
        # （原先这句注释写的是「失败信息只回状态码,不回响应体」—— 说的正好是
        #  相反的事:状态码那条是给成功分支的,而失败分支确实打了响应体。
        #  行为一直是对的,错的是注释。）
        if not report.check("管理员登录", login.status_code == 200, f"HTTP {login.status_code}"):
            print(f"        {login.text[:200]}")
            return 1
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}

        # ---- 建库 + 上传(这一步会真的调 embedding 写向量)----
        kb = client.post("/api/kb", json={"name": KB_NAME}, headers=headers)
        if not report.check("建知识库", kb.status_code == 201, kb.text[:120]):
            return 1
        kb_id = kb.json()["id"]

        up = client.post(
            f"/api/kb/{kb_id}/documents",
            files={"files": ("实弹报价.csv", io.BytesIO(CSV), "text/csv")},
            headers=headers,
        )
        if not report.check("上传文档", up.status_code == 202, up.text[:120]):
            return 1

        doc = client.get(f"/api/kb/{kb_id}/documents/{up.json()[0]['id']}", headers=headers).json()
        ready = report.check(
            "文档已就绪(真 embedding 已写入向量库)",
            doc["status"] == "ready",
            f"status={doc['status']} chunks={doc.get('chunk_count')} err={doc.get('error')}",
        )
        if not ready:
            return 1

        # ---- 主问题:走完整条 SSE ----
        print(f"\n  提问:{args.question}")
        r = client.post(
            "/api/chat/stream",
            json={"question": args.question, "kb_ids": [kb_id]},
            headers=headers,
        )
        if not report.check("SSE 接口返回 200", r.status_code == 200, r.text[:200]):
            return 1
        report.check(
            "Content-Type 是 event-stream",
            r.headers["content-type"].startswith("text/event-stream"),
            r.headers["content-type"],
        )

        events = parse_sse(r.text)
        names = [n for n, _ in events]
        sources = next((p["sources"] for n, p in events if n == "sources"), [])
        done = next((p for n, p in events if n == "done"), None)

        report.check("事件以 sources 开头", names and names[0] == "sources", ",".join(names[:4]))
        report.check("事件以 done 结尾", names and names[-1] == "done")
        report.check("有 error 事件" if "error" in names else "无 error 事件", "error" not in names)
        report.check("检索召回了资料", bool(sources), f"{len(sources)} 条")
        report.check("done 存在", done is not None)
        if done is None:
            return 1

        tokens = "".join(p["text"] for n, p in events if n == "token")
        report.check("流出了正文", bool(tokens), f"{len(tokens)} 字")
        report.check("token 拼接 == done.answer", tokens == done["answer"])
        report.check("grounded=True", done["grounded"] is True)
        report.check(
            "模型给出了引用编号",
            bool(done["cited"]),
            f"cited={done['cited']}",
        )
        report.check(
            "引用编号在来源范围内",
            all(1 <= i <= len(sources) for i in done["cited"]),
            f"max={len(sources)}",
        )
        # usage 曾经整条链路都是 None(网关不够回声就没人上报)。
        # 这条断言盯的是「百炼这一版到底给不给」。
        report.check(
            "usage 上报了 token 数",
            done["usage"] is not None and done["usage"]["total_tokens"] > 0,
            str(done["usage"]),
        )
        # **prompt 缓存的读数要单独盯一条。** 它是 `usage` 里唯一一个可以合法
        # 为 `None` 的字段,而那意味着「平台没说」——与「命中 0 条」含义相反。
        # 断的是**报没报**,不是报了几:命中与否取决于前缀缓存,而冷启动本来就
        # 该是 0。这一条红了说明平台改了回包形状,`docs/PROMPT-CACHE.md` 的
        # 全部结论随之作废。
        cached = (done["usage"] or {}).get("cached_tokens")
        report.check(
            "平台回了 prompt 缓存读数",
            cached is not None,
            f"cached_tokens={cached}(0 是正常的:这一轮前缀是冷的)",
        )

        # ---- 表里没有的问题:必须说不知道 ----
        print(f"\n  追问(资料里没有):{args.unanswerable}")
        r2 = client.post(
            "/api/chat/stream",
            json={"question": args.unanswerable, "kb_ids": [kb_id]},
            headers=headers,
        )
        events2 = parse_sse(r2.text)
        done2 = next((p for n, p in events2 if n == "done"), None)
        if done2 is not None:
            answer2 = done2["answer"]
            print(f"\n  模型回答:{answer2[:200]}")
            # 不能断言具体措辞,只能断言「没有编出一个年份」。
            # 编造通常表现为蹦出 1/2/3 年这类具体数字。
            fabricated = any(w in answer2 for w in ("一年", "两年", "三年", "1年", "2年", "3年"))
            report.check("资料没有时不编造保修期", not fabricated)
        else:
            report.check("资料没有时也有 done", False)

        # ---- 多轮:会话、指代消解、历史落库 ----
        #
        # 这一段里**只有真模型能验的是改写**:`MockLLM` 是照着我们写的提示词
        # 回话的,它「正确地」补全代词只能说明提示词里的例子被抄了一遍,
        # 证明不了真模型会照着做。所以这里既断言,也把改写结果原样打出来
        # 给人眼看——这是多轮链路里唯一一次能看见模型的判断。
        print(f"\n  多轮会话:{args.question}")
        sess = client.post("/api/sessions", json={"kb_ids": [kb_id]}, headers=headers)
        if not report.check("建会话", sess.status_code == 201, sess.text[:120]):
            return 1
        sid = sess.json()["id"]

        r3 = client.post(
            "/api/chat/stream",
            json={"question": args.question, "session_id": sid},
            headers=headers,
        )
        d3 = next((p for n, p in parse_sse(r3.text) if n == "done"), None)
        if not report.check("第一轮有 done", d3 is not None):
            return 1
        report.check(
            "第一轮没被改写(没有历史可指代)",
            d3["search_query"] == args.question,
            f"search_query={d3['search_query'][:60]}",
        )

        # 追问一句**不带商品名**的话:「它」指谁只能从历史里推。
        # 原话直接拿去检索必然召偏——这正是指代消解存在的理由。
        print(f"\n  追问(只说「它」):{args.followup}")
        r4 = client.post(
            "/api/chat/stream",
            json={"question": args.followup, "session_id": sid},
            headers=headers,
        )
        d4 = next((p for n, p in parse_sse(r4.text) if n == "done"), None)
        if not report.check("追问有 done", d4 is not None):
            return 1
        report.check("窗口带上了上一轮", d4["history_messages"] >= 2, f"{d4['history_messages']} 条")
        # 判据是「代词被补全了」,不是「句子变长了」:提示词明确要求已经完整
        # 的句子原样返回,所以长度不是可靠信号,而商品名出现与否是。
        report.check(
            "追问被改写成能独立检索的问题",
            any(name in d4["search_query"] for name in REWRITE_MUST_MENTION),
            f"{args.followup}  →  {d4['search_query'][:80]}",
        )
        report.check("追问仍然检索到了资料", d4["grounded"] is True)

        # ---- 历史落库与找回(需求 ③④)----
        msgs = client.get(f"/api/sessions/{sid}/messages", headers=headers).json()
        report.check("两轮问答都落库了", msgs["total"] == 4, f"total={msgs['total']}")
        roles = [m["role"] for m in msgs["items"]]
        report.check("历史顺序是 提问/回答 交替", roles == ["user", "assistant"] * 2, str(roles))
        # **历史里的回答要带着当时的来源快照。** 只存一句正文的话,翻回
        # 历史时看不到「当时是依据哪几条」，引用面板那一栏就是空的。
        report.check(
            "历史里的回答带着来源快照",
            all(m["sources"] for m in msgs["items"] if m["role"] == "assistant"),
            f"{'.'.join(str(len(m['sources'])) for m in msgs['items'] if m['role'] == 'assistant')} 条",
        )

        listed = client.get("/api/sessions", headers=headers).json()
        mine = next((s for s in listed["items"] if s["id"] == sid), None)
        report.check("会话列表里能找到它", mine is not None)
        report.check(
            "标题由第一问自动生成",
            mine is not None and args.question.strip() in mine["title"],
            f"title={(mine or {}).get('title')}",
        )

        # **需求 4 的原话是「重新登录后能找回」**,所以换个 token 再查一遍
        # ——同一个 token 查得到只证明「这次会话没断」。
        again = client.post(
            "/api/auth/login",
            json={"username": settings.ADMIN_USERNAME, "password": settings.ADMIN_PASSWORD},
        )
        h2 = {"Authorization": f"Bearer {again.json()['access_token']}"}
        back = client.get(f"/api/sessions/{sid}/messages", headers=h2)
        report.check("重新登录后能找回这段对话", back.json()["total"] == 4)

        # ---- 让人眼看的输出 ----

        # ---- 让人眼看的输出 ----
        print("\n  " + "─" * 62)
        print("  来源卡片")
        for s in sources:
            head = f"    [{s['index']}] {s['doc_name']} · {s['kb_name']}"
            print(head)
            score = s.get("rerank_score")
            if score is not None:
                print(f"          rerank={score:.4f}")
            print(f"          {s['content'][:100]}")
        print("\n  模型回答")
        print(f"    {done['answer']}")
        print("\n  追问的改写结果(指代消解的唯一可见处)")
        print(f"    {args.followup}  →  {d4['search_query']}")
        print("\n  阶段耗时 (ms)")
        for k, v in done["stage_timings"].items():
            print(f"    {k:<12} {v:8.2f}")
        print("  " + "─" * 62)

    if report.failed:
        print(f"\n  {len(report.failed)} 项未通过:{', '.join(report.failed)}\n")
        return 1
    print("\n  全部通过 —— 这条链路对着真实平台跑通了。\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
