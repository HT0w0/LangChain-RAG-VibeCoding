@echo off
REM ---------------------------------------------------------------------------
REM  One-click launcher (Windows). Double-click this file.
REM
REM  ASCII ONLY. Do not put Chinese (or any non-ASCII text) in this file,
REM  not even inside REM comments. cmd.exe reads batch files byte-wise and
REM  loses sync on multi-byte encodings, and the failure mode is brutal:
REM  the tail of a comment gets executed as a command, so you get
REM  "not recognized as an internal or external command" pointing at a
REM  line that is clearly a comment. All the actual logic and all the
REM  Chinese output live in backend\scripts\dev.py, where Python controls
REM  the encoding.
REM
REM  Arguments are passed through as-is:
REM      start.bat --check          check the environment only
REM      start.bat --reload         auto-reload the backend on code changes
REM      start.bat --no-browser     do not open the browser
REM ---------------------------------------------------------------------------

setlocal
cd /d "%~dp0"

REM dev.py writes UTF-8; switch the console to match or Chinese turns to mojibake.
chcp 65001 >nul

if not exist ".venv\Scripts\python.exe" (
    echo.
    echo   [ERROR] Python venv not found: .venv\Scripts\python.exe
    echo.
    echo   Run these first:
    echo.
    echo       python -m venv .venv
    echo       .venv\Scripts\activate
    echo       pip install -e ".[dev]" -i https://pypi.tuna.tsinghua.edu.cn/simple
    echo       copy .env.example .env
    echo.
    echo   Then fill DASHSCOPE_API_KEY in .env and run this file again.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -X utf8 "backend\scripts\dev.py" %*
set "EXITCODE=%ERRORLEVEL%"

REM Pause on failure so the window does not vanish before you can read it.
if not "%EXITCODE%"=="0" (
    echo.
    echo   Exit code: %EXITCODE%
    pause
)

exit /b %EXITCODE%
