"""一键启动:检查环境 → 写入示例数据 → 起后端和前端 → 打开浏览器。

**这个脚本存在的理由是「启动失败时说的话」。** 直接敲两条命令当然也能启动,
但那样失败时用户看到的是 `ModuleNotFoundError`、`ECONNREFUSED`、
或者一个一闪而过的窗口——它们都指向症状,而真正的原因(没建 venv、
没装 node_modules、.env 里没填 key、端口被别的进程占着)得自己去猜。
这个脚本把「缺什么、怎么补」直接印出来。

三条设计上的取舍:

1. **两个服务都跑在这一个控制台里**,按 `[后端]` / `[前端]` 前缀区分。
   分成两个窗口更"干净",但那样 Ctrl+C 只能停掉其中一个,用户得记着
   关另一个——而忘了关的那个会一直占着端口,下次启动就报"地址被占用"。
2. **默认不开 `--reload`。** uvicorn 的热重载会多起一个监督进程,
   Ctrl+C 时容易留下孤儿;而这个脚本的定位是「把它跑起来用」,
   不是「改后端代码」。要热重载加 `--reload`。
3. **每次启动都调一次 `seed_data.py`。** 它自己是幂等的(知识库已存在
   就一秒返回),这样就不用在这里再抄一份「怎么判断数据已就绪」——
   抄的那一份迟早和它漂开。

用法:
    python backend/scripts/dev.py              # 或不带参数跑 start.bat
    python backend/scripts/dev.py --reload     # 后端热重载
    python backend/scripts/dev.py --check      # 只检查环境,不启动
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"

#: 控制台编码 + **行缓冲**。
#:
#: 编码:Windows 默认是 GBK(cp936),中文输出会抛 UnicodeEncodeError,
#: 而且是在跑到一半的时候抛,前面的检查结果全丢。
#:
#: 行缓冲:stdout 接管道/文件时 Python 默认是块缓冲,而这个脚本自己是
#: **常驻**的——于是它自己那几行(横幅、「正在启动」、「打开:…」)会一直
#: 卡在缓冲区里,而子进程转发过来的日志却立刻出现。症状是屏幕上只有
#: 一堆 `[后端]` / `[前端]` 却看不到该往哪儿打开浏览器。接控制台时
#: 本来就是行缓冲所以看不出来,重定向一下才暴露。
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(  # type: ignore[union-attr]
            encoding="utf-8", errors="replace", line_buffering=True
        )
    except (AttributeError, OSError):  # pragma: no cover
        pass

BACKEND_URL = "http://127.0.0.1:8000"
FRONTEND_URL = "http://localhost:5173"


class Problem(Exception):
    """环境不满足启动条件。消息要写清**怎么补**,不只是缺什么。"""


# ---------------------------------------------------------------------------
# 检查
# ---------------------------------------------------------------------------


def _port_holder(port: int) -> str | None:
    """谁占着这个端口?返回它的 PID,没人占就 None。

    不去自动杀掉占用者:**这台机器上跑着的可能是用户自己的别的东西**,
    按进程名或端口去 kill 会波及与这个项目无关的进程。只说清楚是谁,
    由用户决定。

    **`netstat` 的输出必须用字节读、按 `errors="replace"` 解,不能用
    `text=True`。** `text=True` 走的是「首选编码」,而 `start.bat` 为了
    让中文不乱码传了 `-X utf8`——于是 Python 会拿 UTF-8 去解 netstat 的
    GBK 输出,读取线程直接抛 `UnicodeDecodeError`,而那个异常发生在
    子线程里,主线程只看到一个 `stdout=None`。症状是启动器报
    `'NoneType' object has no attribute 'splitlines'`——一个和编码
    毫无关系的错。
    """
    try:
        completed = subprocess.run(  # noqa: S603 - 参数是写死的
            ["netstat", "-ano"],
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        # netstat 不存在或超时都不该让启动失败——这只是个更友好的报错,
        # 真正的冲突由服务自己发现。
        return None

    # 只需要看 `LISTENING` 那一行里的端口和 PID,它们都是 ASCII;
    # 中文只出现在列标题里,解坏了也不影响判断。
    out = (completed.stdout or b"").decode("utf-8", errors="replace")

    for line in out.splitlines():
        if "LISTENING" not in line or f":{port} " not in line:
            continue
        parts = line.split()
        if parts:
            return parts[-1]
    return None


def check_environment() -> list[str]:
    """返回问题清单。空列表表示可以启动。

    **端口也查,`--check` 也不例外。** 想省掉它的话会得到一个反直觉的结果:
    用户跑 `--check` 就是要知道「现在能不能起」,而「8000 已经被占着」
    正是最该被告知的一条——那通常意味着上一个实例还在跑。
    """
    problems: list[str] = []

    if not (PROJECT_ROOT / ".env").exists():
        problems.append(
            "没有 .env。先复制一份并填入百炼的 API Key:\n"
            "      copy .env.example .env        (然后把 DASHSCOPE_API_KEY 填进去)"
        )

    if not (FRONTEND_DIR / "package.json").exists():
        problems.append(f"找不到前端目录:{FRONTEND_DIR}")
    elif not (FRONTEND_DIR / "node_modules").exists():
        # `elif`:两条同时命中会打出两句互相矛盾的「怎么办」。
        problems.append(
            "前端依赖没装:\n"
            "      cd frontend && npm install"
        )
    elif not (FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js").exists():
        # 装了一半、或者是个旧版本的目录结构。分开报,是因为「没装」和
        # 「装坏了」的下一步不一样——后者要先删 node_modules 再装。
        problems.append(
            "前端依赖不完整(找不到 vite 的入口)。试试重装:\n"
            "      cd frontend && rmdir /s /q node_modules && npm install"
        )

    if shutil.which("node") is None:
        # 单独查 node:前端那条命令是 `node vite.js`,没有 node 的话
        # 报出来的是 `FileNotFoundError: [WinError 2] 系统找不到指定的文件`,
        # 而那个错**不会说找不到的是什么**,看起来像 vite 没装。
        problems.append(
            "PATH 上找不到 node。装一个 Node.js 18+(https://nodejs.org/),"
            "装完重开这个窗口。"
        )

    for port, what in ((8000, "后端"), (5173, "前端")):
        holder = _port_holder(port)
        if holder:
            problems.append(
                f"{what}端口 {port} 已被占用(PID {holder})。\n"
                f"      如果这个项目已经在跑,直接开 {FRONTEND_URL} 就行;\n"
                f"      否则先结束那个进程再启动。"
            )

    return problems


# ---------------------------------------------------------------------------
# 启动与收尾
# ---------------------------------------------------------------------------


class Service:
    """一个子进程 + 它输出的一行行日志。"""

    def __init__(self, name: str, argv: list[str], cwd: Path) -> None:
        self.name = name
        self._proc = subprocess.Popen(
            argv,
            cwd=str(cwd),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            # **独立进程组。** Ctrl+C 时信号会发给整个前台进程组,父进程
            # 先死掉、子进程变成孤儿继续占着端口——下一次启动就报
            # 「地址被占用」,而用户完全不知道是谁占的。
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if _is_windows() else 0,
        )
        self._pump = threading.Thread(target=self._forward, daemon=True)
        self._pump.start()

    def _forward(self) -> None:
        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            # 去掉行尾换行再自己补,避免出现空行加倍。
            print(f"[{self.name}] {line.rstrip()}")

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    def stop(self) -> None:
        if not self.alive:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            # 温和信号没用才上强的。**按确切的 PID 杀,不用 taskkill /IM**
            # ——那会按进程名波及这台机器上所有同名进程。
            self._proc.kill()


def _is_windows() -> bool:
    return sys.platform.startswith("win")


def wait_ready(url: str, *, timeout: float, name: str, service: Service) -> bool:
    """轮询直到响应,或者服务已经死了。"""
    import urllib.error
    import urllib.request

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not service.alive:
            print(f"\n  [{name}] 进程已退出,没能起来。上面的日志里有原因。\n")
            return False
        try:
            with urllib.request.urlopen(url, timeout=2):  # noqa: S310 - 只有 localhost
                return True
        except (urllib.error.URLError, OSError, TimeoutError):
            time.sleep(0.4)
    print(f"\n  [{name}] 等了 {timeout:.0f} 秒还没就绪。\n")
    return False


def run_seed() -> None:
    """写入示例数据。**每次都调,幂等由脚本自己保证。**"""
    print("  检查示例数据……")
    result = subprocess.run(
        [sys.executable, str(BACKEND_DIR / "scripts" / "seed_data.py")],
        cwd=str(PROJECT_ROOT),
        check=False,
    )
    if result.returncode != 0:
        # 种子失败不该拦住启动:知识库可以之后在界面上建,
        # 而一个能打开的空应用比一个打不开的应用有用。
        print("\n  ⚠ 示例数据写入失败(应用仍会启动)。")
        print("     常见原因:API Key 没填、或额度用尽。\n")


def main(*, reload: bool, check_only: bool, open_browser: bool) -> int:
    print()
    print("=" * 68)
    print("  电商知识库 RAG 问答系统")
    print("=" * 68)

    problems = check_environment()
    if problems:
        print(f"\n  还不能启动,有 {len(problems)} 项要先处理:\n")
        for index, problem in enumerate(problems, start=1):
            print(f"  {index}. {problem}\n")
        return 1

    print("  环境检查通过。\n")
    if check_only:
        return 0

    run_seed()

    python = sys.executable
    backend = Service(
        "后端",
        [python, "-m", "uvicorn", "app.main:app", "--app-dir", "backend",
         "--port", "8000", *(["--reload"] if reload else [])],
        PROJECT_ROOT,
    )
    # **不能用 `npx vite`。** Windows 上 `npx` 是 `npx.cmd`,而
    # `CreateProcess` 不能直接执行批处理文件——`Popen(["npx", ...])` 会抛
    # `FileNotFoundError: [WinError 2] 系统找不到指定的文件`,而那个错误
    # 不会说是找不到 `npx`。`node` 是真的 .exe,直接跑 vite 的入口,
    # 这也正是 npm 底下在做的事。
    node = shutil.which("node")
    assert node is not None  # check_environment 已经把这条挡在前面了
    frontend = Service(
        "前端",
        [node, str(FRONTEND_DIR / "node_modules" / "vite" / "bin" / "vite.js"),
         "--port", "5173", "--strictPort"],
        FRONTEND_DIR,
    )
    services = [backend, frontend]

    try:
        print("  正在启动……\n")
        ok = wait_ready(f"{BACKEND_URL}/api/health", timeout=60, name="后端", service=backend)
        ok = wait_ready(FRONTEND_URL, timeout=90, name="前端", service=frontend) and ok
        if not ok:
            return 1

        print()
        print("  " + "-" * 64)
        print(f"  打开:{FRONTEND_URL}")
        print(f"  接口文档:{BACKEND_URL}/docs")
        print("  登录:admin / 123456")
        print()
        print("  按 Ctrl+C 停止(后端和前端会一起停)。")
        print("  " + "-" * 64)
        print()

        if open_browser:
            webbrowser.open(FRONTEND_URL)

        # 主线程就在这里等着。子进程的输出由各自的线程转发。
        while all(service.alive for service in services):
            time.sleep(0.5)

        dead = [s.name for s in services if not s.alive]
        print(f"\n  {', '.join(dead)} 退出了。上面的日志里有原因。\n")
        return 1
    except KeyboardInterrupt:
        print("\n  正在停止……")
        return 0
    finally:
        for service in services:
            with contextlib.suppress(Exception):
                service.stop()
        print("  已停止。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="一键启动开发环境")
    parser.add_argument("--reload", action="store_true", help="后端代码改动后自动重启")
    parser.add_argument("--check", action="store_true", help="只检查环境,不启动")
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args()
    raise SystemExit(
        main(reload=args.reload, check_only=args.check, open_browser=not args.no_browser)
    )
