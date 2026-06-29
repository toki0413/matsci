@echo off
REM Huginn Agent portable launcher
REM Usage: drag this file to any folder or add to PATH

set "SCRIPT_DIR=%~dp0"
set "PYTHON=%SCRIPT_DIR%.venv\Scripts\python.exe"

if not exist "%PYTHON%" (
    REM Try system Python
    where python >nul 2>nul
    if %errorlevel% == 0 (
        set "PYTHON=python"
    ) else (
        echo [ERROR] Python not found. Please install Python 3.11+ or run install.bat first.
        pause
        exit /b 1
    )
)

REM Run huginn-agent with all arguments passed through
%PYTHON% -m huginn.cli %*
