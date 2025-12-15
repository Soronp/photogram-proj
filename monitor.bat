@echo off
chcp 65001 > nul
title MARK-2 Pipeline Monitor
cd /d "%~dp0"

echo ========================================
echo    MARK-2 PIPELINE MONITOR
echo ========================================
echo.

echo Checking virtual environment...
if exist "venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
    echo.
    echo Python info:
    python --version
    echo.
) else (
    echo Virtual environment not found.
    echo Using system Python...
    echo.
)

echo Starting pipeline monitor...
echo.

REM Run the monitor
python monitor.py

REM Check exit code
if errorlevel 1 (
    echo.
    echo Pipeline failed with error code: %errorlevel%
    pause
)

exit