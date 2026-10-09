"""日志与请求追踪。

每个请求分配一个 `trace_id`,贯穿所有日志与 `request_metrics` 表。没有它,
排查「这次问答为什么慢了」只能靠时间戳猜——而 RAG 链路上有改写/检索/重排/
生成四个阶段,少了 trace_id 根本对不上号。

JSON 输出给生产(便于采集),人类可读给开发。
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

_trace_id: ContextVar[str] = ContextVar("trace_id", default="-")


def new_trace_id() -> str:
    tid = uuid.uuid4().hex[:16]
    _trace_id.set(tid)
    return tid


def get_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(tid: str) -> None:
    _trace_id.set(tid)


class _TraceFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.trace_id = _trace_id.get()
        return True


class _HumanFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[90m",
        "INFO": "\033[36m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[41m",
    }
    RESET = "\033[0m"

    def __init__(self, *, color: bool) -> None:
        super().__init__()
        self._color = color

    def format(self, record: logging.LogRecord) -> str:
        level = record.levelname
        if self._color:
            level = f"{self.COLORS.get(record.levelname, '')}{level:<8}{self.RESET}"
        else:
            level = f"{level:<8}"
        ts = time.strftime("%H:%M:%S", time.localtime(record.created))
        head = f"{ts} {level} {record.trace_id} {record.name}"
        msg = record.getMessage()
        if record.exc_info:
            msg += "\n" + self.formatException(record.exc_info)
        return f"{head} | {msg}"


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "trace_id": record.trace_id,
            "msg": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            payload.update(extra)
        return json.dumps(payload, ensure_ascii=False)


def _force_utf8_streams() -> None:
    """把 stdout/stderr 切成 UTF-8。

    Windows 默认控制台/重定向编码是 GBK(cp936),中文日志会变成乱码,
    更糟的是遇到 GBK 编不出的字符会直接抛 UnicodeEncodeError——
    在一个 try/except 里打日志导致业务逻辑被异常打断,是很难查的一类 bug。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError, ValueError):  # pragma: no cover
            pass


def setup_logging(*, debug: bool = True, json_output: bool = False, level: str | None = None) -> None:
    _force_utf8_streams()

    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    handler.addFilter(_TraceFilter())
    handler.setFormatter(
        _JsonFormatter() if json_output else _HumanFormatter(color=sys.stdout.isatty())
    )

    root.addHandler(handler)
    root.setLevel(level or ("DEBUG" if debug else "INFO"))

    # 第三方库的 INFO 噪音太大,压到 WARNING。
    # httpx 会把每个请求打出来,问答高峰期刷屏;chromadb 的遥测日志同理。
    #
    # aiosqlite / sqlalchemy.engine **必须在列表里**:它们把每一条 SQL 都打成两条
    # DEBUG(executing / completed)。入库一篇文档就是几百条,开着 DEBUG 时
    # 应用自己的日志会被彻底淹没——排障时最需要的那两行反而找不到了。
    for noisy in (
        "httpx",
        "httpcore",
        "urllib3",
        "chromadb",
        "sentence_transformers",
        "jieba",
        "aiosqlite",
        "sqlalchemy.engine",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # 注意 chromadb 的匿名遥测要靠环境变量在 import 前关闭(见 main.py 顶部),
    # 日志级别压不掉它。


def log_extra(**fields: Any) -> dict[str, Any]:
    """用法: logger.info("检索完成", extra=log_extra(chunks=20, ms=42))"""
    return {"extra_fields": fields}
