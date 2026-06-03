@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM  nScryptConformal v2  —  launch.bat
REM  Called silently by JobCommand.vbs.
REM
REM  Priority order for Python:
REM    1. .venv\ in the same folder as this batch file (virtual environment)
REM    2. pythonw on the system PATH
REM    3. Hard-coded fallback (edit PYTHON_FALLBACK below if needed)
REM
REM  Errors are written to launch_error.log in the same folder.
REM  No console window is shown to the user.
REM ─────────────────────────────────────────────────────────────────────────

REM Change to the directory containing this batch file so all relative paths work
cd /d "%~dp0"

REM ── 1. Try local virtual environment (.venv) ──────────────────────────────
if exist "C:\Users\swaru\AppData\Local\Programs\Python\Python313\python.exe" (
    set PYTHON="C:\Users\swaru\AppData\Local\Programs\Python\Python313\python.exe"
    goto :launch
)

REM ── 2. Try pythonw on the system PATH ────────────────────────────────────
where pythonw >nul 2>&1
if %ERRORLEVEL% == 0 (
    set PYTHON=pythonw
    goto :launch
)

REM ── 3. Hard-coded fallback path ───────────────────────────────────────────
REM  Uncomment and edit this line if Python is not on PATH:
REM  set PYTHON="C:\Python311\pythonw.exe"
REM  if defined PYTHON goto :launch

REM ── No Python found — log error ───────────────────────────────────────────
echo [%date% %time%] ERROR: pythonw not found. > launch_error.log
echo Install Python 3.11+ and ensure it is on the system PATH, >> launch_error.log
echo or create a virtual environment in .venv\ by running: >> launch_error.log
echo     python -m venv .venv >> launch_error.log
echo     .venv\Scripts\pip install customtkinter numpy scipy matplotlib pandas openpyxl >> launch_error.log
REM Show an error dialog using PowerShell (no console window)
powershell -WindowStyle Hidden -Command ^
  "Add-Type -AssemblyName PresentationFramework; ^
   [System.Windows.MessageBox]::Show('Python not found.`n`nInstall Python 3.11+ or create a .venv virtual environment.`nSee launch_error.log for details.', ^
   'nScryptConformal v2', 'OK', 'Error')"
goto :eof

REM ── Launch ─────────────────────────────────────────────────────────────────
:launch
REM Add backend, ui, windows subfolders to PYTHONPATH so imports resolve cleanly
set PYTHONPATH=%~dp0backend;%~dp0ui;%~dp0windows;%PYTHONPATH%

REM Launch the application (pythonw suppresses the console window)
%PYTHON% "%~dp0main.pyw" >> launch_error.log 2>&1

REM If Python exited with a non-zero code, the error is already in the log.
REM Nothing else to do — the app manages its own error dialogs.
