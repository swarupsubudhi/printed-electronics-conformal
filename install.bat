@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  nScryptConformal v2  —  install.bat
REM  Run this ONCE to create a local virtual environment and install
REM  all required packages.  After this, JobCommand.vbs / launch.bat
REM  will use .venv\Scripts\pythonw.exe automatically.
REM
REM  Usage: double-click install.bat  (a console window will appear briefly)
REM ─────────────────────────────────────────────────────────────────────────

cd /d "%~dp0"

echo.
echo  nScryptConformal v2 — Setup
echo  ════════════════════════════
echo.

REM ── Check Python is available ─────────────────────────────────────────────
where python >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo  ERROR: python not found on PATH.
    echo  Download and install Python 3.11+ from https://python.org
    echo  and make sure "Add Python to PATH" is checked during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('python --version') do echo  Found: %%v
echo.

REM ── Create virtual environment ────────────────────────────────────────────
if exist ".venv\" (
    echo  Virtual environment already exists at .venv\
    echo  Delete .venv\ and re-run this script to rebuild it.
) else (
    echo  Creating virtual environment in .venv\ ...
    python -m venv .venv
    if %ERRORLEVEL% NEQ 0 (
        echo  ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo  Done.
)
echo.

REM ── Upgrade pip ───────────────────────────────────────────────────────────
echo  Upgrading pip...
.venv\Scripts\python -m pip install --upgrade pip --quiet
echo.

REM ── Install required packages ─────────────────────────────────────────────
echo  Installing required packages...
.venv\Scripts\pip install ^
    customtkinter ^
    numpy ^
    scipy ^
    matplotlib ^
    pandas ^
    openpyxl ^
    --quiet

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo  ERROR: Package installation failed.
    echo  Check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo  ════════════════════════════
echo  Setup complete.
echo  Double-click JobCommand.vbs to launch nScryptConformal v2.
echo  ════════════════════════════
echo.
pause
