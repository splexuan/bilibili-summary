@echo off
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"

:: Check venv
if not exist .venv\Scripts\python.exe (
    echo [ERROR] .venv not found, please run: python -m venv .venv
    pause
    exit /b 1
)

call .venv\Scripts\activate.bat
set PYTHON=.venv\Scripts\python.exe

echo ========================================
echo   Bili Video Summary Tool
echo   Python: .venv virtual environment
echo ========================================
echo.
echo Starting server...  http://localhost:3195
echo.

%PYTHON% -B app.py
pause
