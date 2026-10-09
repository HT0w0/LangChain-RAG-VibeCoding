"""平台能力自检脚本 —— 全项目的**事实来源**。

为什么值得单独写一个脚本：本项目前期调研中,关于百炼平台的若干"已知结论"
被实测证伪——最典型的一条是"OpenAI 兼容端点不支持 Embedding",实测
`/v1/embeddings` 完全正常。文档会说错、博客会过时、二手结论会互相抄错。
所以任何平台假设在使用前都必须先跑一遍这个脚本,以它的输出为准。

它验证四件事:
  1. Chat        —— 非流式 + 流式,并实测 `enable_thinking:false` 是否真的关掉思考
  2. Embedding   —— 单条 + 批量,实测维度,并**二分探测真实批量上限**
  3. Rerank      —— 原生端点,并用「相关/不相关」文档对测出区分度
  4. 模型清单    —— 拉取实际可用模型,核对配置里的模型名是否存在

用法:
    python backend/scripts/check_api.py
    python backend/scripts/check_api.py --quick      # 跳过批量上限探测,省额度
    python backend/scripts/check_api.py --no-write   # 不写 docs/API-CHECK.md
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# backend/ 加入 sys.path,使 `import app.*` 生效(脚本直接跑时没有包上下文)
BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import httpx  # noqa: E402
from app.core.config import PROJECT_ROOT, settings  # noqa: E402

# ---------------------------------------------------------------------------
# 终端输出适配
#
# Windows 默认控制台编码是 GBK(cp936),直接 print 中文/✓ 会抛
# UnicodeEncodeError —— 而且是在脚本跑到一半时抛,前面的检查结果全丢。
# 必须先把 stdout/stderr 切成 UTF-8。
# ---------------------------------------------------------------------------

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, OSError):  # pragma: no cover - 极老的 Python 或重定向流
        pass


def _can_encode(ch: str) -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        ch.encode(enc)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


# 编码不支持时退回纯 ASCII 标记,宁可朴素也不能崩
_TICK = "✓" if _can_encode("✓") else "OK"
_CROSS = "✗" if _can_encode("✗") else "XX"
_RULE = "═" if _can_encode("═") else "="
_BAR = "─" if _can_encode("─") else "-"

_USE_COLOR = sys.stdout.isatty() and _can_encode("\033")


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


OK = lambda s: _c(s, "32")  # noqa: E731
FAIL = lambda s: _c(s, "31")  # noqa: E731
WARN = lambda s: _c(s, "33")  # noqa: E731
DIM = lambda s: _c(s, "90")  # noqa: E731
BOLD = lambda s: _c(s, "1")  # noqa: E731


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    facts: dict[str, Any] = field(default_factory=dict)
    hint: str = ""


# ---------------------------------------------------------------------------
# 1. Chat
# ---------------------------------------------------------------------------

_CHAT_PROMPT = [{"role": "user", "content": "用一句话说明什么是RAG,不要展开。"}]


async def check_chat(client: httpx.AsyncClient) -> CheckResult:
    url = f"{settings.DASHSCOPE_BASE_URL.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    facts: dict[str, Any] = {}

    # --- 非流式:确认模型存在 + 拿到 usage 形状 ---
    payload = {
        "model": settings.LLM_MODEL,
        "messages": _CHAT_PROMPT,
        "max_tokens": 128,
        "temperature": 0.3,
        # 实测有效的关思考写法。若该字段不被支持,服务端通常直接忽略而非报错,
        # 所以下面的流式测试会独立验证它是否真的生效。
        "enable_thinking": settings.LLM_ENABLE_THINKING,
    }
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        return CheckResult("Chat(非流式)", False, f"网络失败: {exc}", hint="检查网络/代理设置")

    latency_ms = (time.perf_counter() - t0) * 1000
    if resp.status_code != 200:
        code, msg = _err(resp)
        return CheckResult(
            "Chat(非流式)",
            False,
            f"HTTP {resp.status_code} [{code}] {msg[:160]}",
            hint=_hint_for(code),
        )

    body = resp.json()
    msg_obj = (body.get("choices") or [{}])[0].get("message", {})
    facts["非流式延迟"] = f"{latency_ms:.0f} ms"
    facts["usage"] = body.get("usage", {})
    facts["思考字段"] = "有 reasoning_content" if msg_obj.get("reasoning_content") else "无"

    # --- 流式:测首 token 时延(TTFT),这是 SSE 体验的关键指标 ---
    stream_payload = {**payload, "stream": True}
    ttft_ms: float | None = None
    first_text = ""
    reasoning_seen = False
    chunk_count = 0
    t0 = time.perf_counter()
    try:
        async with client.stream("POST", url, headers=headers, json=stream_payload) as sresp:
            if sresp.status_code != 200:
                raw = await sresp.aread()
                code, m = _err_raw(sresp.status_code, raw)
                return CheckResult(
                    "Chat(流式)",
                    False,
                    f"HTTP {sresp.status_code} [{code}] {m[:160]}",
                    facts=facts,
                    hint=_hint_for(code),
                )
            async for line in sresp.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    delta = json.loads(data)["choices"][0].get("delta", {})
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta.get("reasoning_content"):
                    reasoning_seen = True
                if delta.get("content"):
                    if ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t0) * 1000
                    first_text += delta["content"]
                    chunk_count += 1
    except httpx.RequestError as exc:
        return CheckResult("Chat(流式)", False, f"网络失败: {exc}", facts=facts)

    facts["首token时延"] = f"{ttft_ms:.0f} ms" if ttft_ms else "未测到"
    facts["流式块数"] = chunk_count
    facts["流式见到思考内容"] = reasoning_seen
    facts["输出样例"] = first_text[:60]
    facts["reasoning_content 残留"] = reasoning_seen

    # 关键判定:配置要求关思考,流式里就不该出现 reasoning_content
    if settings.LLM_ENABLE_THINKING is False and reasoning_seen:
        return CheckResult(
            "Chat(流式)",
            False,
            "已配置 enable_thinking=false,但流里仍出现 reasoning_content",
            facts=facts,
            hint='改用 {"thinking": {"type": "disabled"}} 再试',
        )

    return CheckResult("Chat(流式)", True, f"TTFT {facts['首token时延']}", facts=facts)


# ---------------------------------------------------------------------------
# 2. Embedding
# ---------------------------------------------------------------------------


async def check_embedding(client: httpx.AsyncClient, *, quick: bool) -> CheckResult:
    url = f"{settings.DASHSCOPE_BASE_URL.rstrip('/')}/embeddings"
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    facts: dict[str, Any] = {}

    async def embed(texts: list[str]) -> tuple[int, Any]:
        t0 = time.perf_counter()
        r = await client.post(
            url, headers=headers, json={"model": settings.EMBED_MODEL, "input": texts}
        )
        return int((time.perf_counter() - t0) * 1000), r

    # --- 单条 ---
    try:
        ms, resp = await embed(["这是一条用于自检的中文测试文本。"])
    except httpx.RequestError as exc:
        return CheckResult("Embedding(单条)", False, f"网络失败: {exc}")

    if resp.status_code != 200:
        code, msg = _err(resp)
        return CheckResult(
            "Embedding(单条)",
            False,
            f"HTTP {resp.status_code} [{code}] {msg[:160]}",
            hint=_hint_for(code),
        )

    b = resp.json()
    vec = b["data"][0]["embedding"]
    dim = len(vec)
    facts["维度"] = dim
    facts["单条延迟"] = f"{ms} ms"
    facts["usage"] = b.get("usage", {})

    if dim != settings.EMBED_DIM:
        facts["维度告警"] = f"配置 EMBED_DIM={settings.EMBED_DIM},实测 {dim}"

    # --- 批量 ---
    n = settings.EMBED_BATCH_SIZE
    batch_texts = [f"商品编号{i}的售后政策说明,用于批量自检。" for i in range(n)]
    try:
        ms_b, resp_b = await embed(batch_texts)
    except httpx.RequestError as exc:
        return CheckResult("Embedding(批量)", False, f"网络失败: {exc}", facts=facts)

    if resp_b.status_code != 200:
        code, msg = _err(resp_b)
        return CheckResult(
            "Embedding(批量)",
            False,
            f"批 {n} 条失败 HTTP {resp_b.status_code} [{code}] {msg[:120]}",
            facts=facts,
            hint="批量不可用:入库时必须逐条调用",
        )

    data = resp_b.json()["data"]
    facts["批量"] = f"{n} 条 / {ms_b} ms"
    correct_order = all(d.get("index") == i for i, d in enumerate(sorted(data, key=lambda x: x["index"])))
    facts["返回顺序可信"] = correct_order

    # --- 二分探测真实上限 ---
    if not quick:
        limit, tried = await _probe_batch_limit(client, url, headers, start=n)
        facts["批量上限探测"] = f"最大成功 {limit}(尝试: {', '.join(map(str, tried))})"
    else:
        facts["批量上限探测"] = "已跳过(--quick)"

    warn = facts.get("维度告警", "")
    return CheckResult(
        "Embedding(批量)",
        True,
        f"{dim}维 · 批{n} OK" + (f" · {warn}" if warn else ""),
        facts=facts,
        hint="把探测到的批量上限回填 .env 的 EMBED_BATCH_SIZE" if not quick else "",
    )


async def _probe_batch_limit(
    client: httpx.AsyncClient, url: str, headers: dict[str, str], *, start: int
) -> tuple[int, list[int]]:
    """指数式上探 + 二分收敛,找出服务端真正接受的批量大小。

    刻意不用纯二分:批量上限通常是个不大的整数,指数上探几次就能框住区间,
    比二分更省请求(也就更省额度)。
    """
    async def works(k: int) -> bool:
        texts = [f"批量上限探测样本 {i}" for i in range(k)]
        try:
            r = await client.post(url, headers=headers, json={"model": settings.EMBED_MODEL, "input": texts})
        except httpx.RequestError:
            return False
        return r.status_code == 200

    tried: list[int] = []
    lo = start      # 已知可用
    hi = start + 1  # 未知

    # 阶段一:指数上探,快速框出「可用/不可用」的区间。
    # 每探一次都消耗额度,所以步长放大,不做逐点试。
    for candidate in (start * 2, start * 4, start * 8, 128, 256):
        if candidate <= lo:
            continue
        tried.append(candidate)
        if await works(candidate):
            lo = candidate
            hi = candidate + 1
        else:
            hi = candidate
            break
    else:
        # 全部通过,说明上限高于 256,不再继续探(收益低、额度贵)
        return lo, tried

    # 阶段二:在 (lo, hi) 内二分。区间通常只有几十,6 次左右收敛,
    # 比逐点扫省一个量级的额度。
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        tried.append(mid)
        if await works(mid):
            lo = mid
        else:
            hi = mid
    return lo, tried


# ---------------------------------------------------------------------------
# 3. Rerank
# ---------------------------------------------------------------------------

_RERANK_QUERY = "这款蓝牙耳机的续航时间是多久?"
_RERANK_DOCS = [
    "该蓝牙耳机采用 40mAh 电池,单次充电续航 8 小时,配合充电仓可达 32 小时。",  # 相关
    "本店支持 7 天无理由退货,商品需保持包装完整。",  # 不相关
    "手机壳采用液态硅胶材质,手感细腻防滑。",  # 不相关
]


async def check_rerank(client: httpx.AsyncClient) -> CheckResult:
    if not settings.RERANK_ENABLED:
        return CheckResult("Rerank", True, "配置为关闭,跳过")

    url = settings.native_rerank_url
    headers = {
        "Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.RERANK_MODEL,
        "input": {"query": _RERANK_QUERY, "documents": _RERANK_DOCS},
        "parameters": {"top_n": len(_RERANK_DOCS), "return_documents": False},
    }

    t0 = time.perf_counter()
    try:
        resp = await client.post(url, headers=headers, json=payload)
    except httpx.RequestError as exc:
        return CheckResult("Rerank", False, f"网络失败: {exc}")

    ms = int((time.perf_counter() - t0) * 1000)
    if resp.status_code != 200:
        code, msg = _err(resp)
        return CheckResult(
            "Rerank",
            False,
            f"HTTP {resp.status_code} [{code}] {msg[:160]}",
            hint=_hint_for(code),
        )

    body = resp.json()
    results = (body.get("output") or {}).get("results") or body.get("results") or []
    if not results:
        return CheckResult("Rerank", False, f"响应无 results: {list(body)[:5]}")

    scores = {int(r["index"]): float(r.get("relevance_score", 0)) for r in results}
    rel = scores.get(0, 0.0)
    irr = [scores.get(1, 0.0), scores.get(2, 0.0)]
    gap = rel - max(irr) if irr else 0.0
    ranked_correctly = max(scores, key=lambda k: scores[k]) == 0

    facts = {
        "延迟": f"{ms} ms",
        "相关文档得分": round(rel, 4),
        "不相关文档得分": [round(x, 4) for x in irr],
        "区分度(相关-最高不相关)": round(gap, 4),
        "排序正确": ranked_correctly,
    }

    if not ranked_correctly:
        return CheckResult("Rerank", False, f"排序错误,得分: {facts}", facts=facts)
    if gap < 0.15:
        return CheckResult(
            "Rerank",
            True,
            f"可用但区分度偏低({gap:.3f})",
            facts=facts,
            hint="区分度小则阈值截断不可靠,建议只用排序不用绝对值",
        )
    return CheckResult("Rerank", True, f"区分度 {gap:.3f}", facts=facts)


# ---------------------------------------------------------------------------
# 4. 模型清单
# ---------------------------------------------------------------------------


async def check_models(client: httpx.AsyncClient) -> CheckResult:
    url = f"{settings.DASHSCOPE_BASE_URL.rstrip('/')}/models"
    headers = {"Authorization": f"Bearer {settings.DASHSCOPE_API_KEY}"}
    try:
        resp = await client.get(url, headers=headers, timeout=30.0)
    except httpx.RequestError as exc:
        return CheckResult("模型清单", False, f"网络失败: {exc}")

    if resp.status_code != 200:
        code, msg = _err(resp)
        return CheckResult("模型清单", False, f"HTTP {resp.status_code} [{code}] {msg[:120]}")

    ids = sorted(m.get("id", "") for m in (resp.json().get("data") or []))
    configured = [settings.LLM_MODEL, settings.EMBED_MODEL, settings.RERANK_MODEL]
    missing = [m for m in configured if m not in ids]

    facts = {"可用模型数": len(ids), "配置的模型缺失": missing or "无"}
    if missing:
        return CheckResult(
            "模型清单",
            False,
            f"{len(ids)} 个模型,但配置的 {missing} 不在列表中",
            facts=facts,
            hint="模型名不存在。以实际列表为准,不要照抄文档里的名字",
        )
    return CheckResult("模型清单", True, f"{len(ids)} 个可用模型", facts=facts)


# ---------------------------------------------------------------------------
# 辅助
# ---------------------------------------------------------------------------


def _err(resp: httpx.Response) -> tuple[str, str]:
    try:
        body = resp.json()
    except Exception:  # noqa: BLE001
        return "", resp.text[:200]
    e = body.get("error")
    if isinstance(e, dict):
        return str(e.get("code", "")), str(e.get("message", ""))
    if isinstance(e, str):
        return "", e
    return str(body.get("code", "")), str(body.get("message", ""))


def _err_raw(status: int, raw: bytes) -> tuple[str, str]:  # noqa: ARG001
    try:
        body = json.loads(raw)
    except Exception:  # noqa: BLE001
        return "", raw.decode("utf-8", "replace")[:200]
    e = body.get("error")
    if isinstance(e, dict):
        return str(e.get("code", "")), str(e.get("message", ""))
    return "", str(body)[:200]


def _hint_for(code: str) -> str:
    c = (code or "").lower()
    if c in {"invalid_api_key", "invalidapikey"}:
        return "API Key 无效。去百炼控制台确认 Key 有效且与工作空间匹配"
    if c in {"arrearage", "insufficient_quota"}:
        return "额度耗尽或欠费。注意欠费返回的是 400 而非 401,不要误判为认证问题"
    if "model" in c and "not" in c:
        return "模型名不存在。跑一次本脚本的模型清单检查,以实际列表为准"
    return ""


def _preflight() -> str | None:
    if not settings.DASHSCOPE_API_KEY:
        return "未配置 DASHSCOPE_API_KEY。请在项目根目录的 .env 中填写。"
    if settings.DASHSCOPE_API_KEY.startswith("sk-ws-xxx"):
        return "DASHSCOPE_API_KEY 仍是 .env.example 里的占位符,请填真实值。"
    return None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


async def main(*, quick: bool, write: bool) -> int:
    print()
    print(BOLD(_RULE * 68))
    print(BOLD("  百炼平台能力自检 —— 全项目的事实来源"))
    print(BOLD(_RULE * 68))
    print(f"  兼容端点  {DIM(settings.DASHSCOPE_BASE_URL)}")
    print(f"  原生端点  {DIM(settings.DASHSCOPE_NATIVE_URL)}")
    print(f"  Chat      {settings.LLM_MODEL}  (enable_thinking={settings.LLM_ENABLE_THINKING})")
    print(f"  Embedding {settings.EMBED_MODEL}")
    print(f"  Rerank    {settings.RERANK_MODEL}  (enabled={settings.RERANK_ENABLED})")
    print()

    if (problem := _preflight()) is not None:
        print(FAIL(f"  {_CROSS} {problem}"))
        print()
        return 2

    results: list[CheckResult] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(120.0, connect=15.0)) as client:
        # 存的是**工厂**而非协程对象。提前创建协程的话,一旦前面的检查抛异常,
        # 后面的协程就再也没人 await,Python 会刷一屏 "coroutine was never awaited",
        # 把真正的报错淹掉。
        checks: list[tuple[str, Callable[[], Awaitable[CheckResult]]]] = [
            ("Chat(非流式)", lambda: check_chat(client)),
            ("Embedding", lambda: check_embedding(client, quick=quick)),
            ("Rerank", lambda: check_rerank(client)),
            ("模型清单", lambda: check_models(client)),
        ]
        for label, factory in checks:
            print(f"  检测 {label} …", end="", flush=True)
            try:
                r = await factory()
            except Exception as exc:  # noqa: BLE001 - 自检脚本不能因单个检查崩溃
                r = CheckResult(label, False, f"未捕获异常: {type(exc).__name__}: {exc}")
            results.append(r)
            print("\r" + " " * 40 + "\r", end="")
            mark = OK(_TICK) if r.ok else FAIL(_CROSS)
            print(f"  {mark} {r.name:<16} {r.detail}")
            if r.hint:
                print(f"      {DIM('→ ' + r.hint)}")

    # ---- 详细事实 ----
    print()
    print(BOLD("  实测详情"))
    print(DIM("  " + _BAR * 64))
    for r in results:
        if not r.facts:
            continue
        print(f"  {r.name}")
        for k, v in r.facts.items():
            print(f"      {k:<22} {v}")
    print()

    passed = sum(1 for r in results if r.ok)
    total = len(results)
    summary = f"  {passed}/{total} 项通过"
    print(OK(summary) if passed == total else WARN(summary))
    print()

    if write:
        path = _write_report(results)
        print(DIM(f"  报告已写入 {path.relative_to(PROJECT_ROOT)}"))
        print()

    return 0 if passed == total else 1


def _write_report(results: list[CheckResult]) -> Path:
    """把实测结果落成 markdown,供 README 引用。"""
    from datetime import datetime

    docs = PROJECT_ROOT / "docs"
    docs.mkdir(parents=True, exist_ok=True)
    path = docs / "API-CHECK.md"

    lines = [
        "# 百炼平台能力实测报告",
        "",
        f"> 由 `backend/scripts/check_api.py` 于 {datetime.now():%Y-%m-%d %H:%M:%S} 自动生成。",
        "> 手工编辑会被下次运行覆盖。",
        "",
        "本文件是全项目的**事实来源**:任何平台假设在使用前都应以此处输出为准。",
        "",
        "## 环境",
        "",
        "| 项 | 值 |",
        "|---|---|",
        f"| 兼容端点 | `{settings.DASHSCOPE_BASE_URL}` |",
        f"| 原生端点 | `{settings.DASHSCOPE_NATIVE_URL}` |",
        f"| Chat 模型 | `{settings.LLM_MODEL}` (enable_thinking={settings.LLM_ENABLE_THINKING}) |",
        f"| Embedding 模型 | `{settings.EMBED_MODEL}` |",
        f"| Rerank 模型 | `{settings.RERANK_MODEL}` |",
        "",
        "## 检查结果",
        "",
        "| 检查项 | 结果 | 说明 |",
        "|---|---|---|",
    ]
    for r in results:
        mark = "✅" if r.ok else "❌"
        lines.append(f"| {r.name} | {mark} | {r.detail} |")

    lines += ["", "## 实测详情", ""]
    for r in results:
        if not r.facts:
            continue
        lines.append(f"### {r.name}")
        lines.append("")
        lines.append("| 项 | 值 |")
        lines.append("|---|---|")
        for k, v in r.facts.items():
            lines.append(f"| {k} | `{v}` |")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="百炼平台能力自检")
    ap.add_argument("--quick", action="store_true", help="跳过 embedding 批量上限探测(省额度)")
    ap.add_argument("--no-write", action="store_true", help="不写 docs/API-CHECK.md")
    args = ap.parse_args()

    try:
        sys.exit(asyncio.run(main(quick=args.quick, write=not args.no_write)))
    except KeyboardInterrupt:
        print("\n  已中断")
        sys.exit(130)
