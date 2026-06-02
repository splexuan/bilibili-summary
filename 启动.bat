@echo off
set PYTHONIOENCODING=utf-8
cd /d "%~dp0"
echo ========================================
echo   Bili Video Summary Tool
echo ========================================
echo.
echo Starting server...
echo URL: http://localhost:3195
echo.

if exist "py310\python.exe" (
    py310\python.exe -B app.py
) else (
    python -B app.py
)
pause
