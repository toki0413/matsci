@echo off
REM Huginn Agent - One-click installer for Windows
REM Requires Python 3.11+ (https://python.org) or uv (https://docs.astral.sh/uv)

echo ==========================================
echo   Huginn Agent Installer
echo   Material Science AI Agent Harness
echo ==========================================
echo.

set "SCRIPT_DIR=%~dp0"
set "WHEEL=%SCRIPT_DIR%dist\huginn_agent-0.1.0-py3-none-any.whl"

if not exist "%WHEEL%" (
    echo [ERROR] Wheel not found: %WHEEL%
    echo Run 'uv build' first to generate the wheel.
    pause
    exit /b 1
)

REM Check for uv (preferred, faster)
where uv >nul 2>nul
if %errorlevel% == 0 (
    echo [OK] uv found, using uv for installation...
    uv pip install --force-reinstall "%WHEEL%"
    goto :done
)

REM Check for pip
where pip >nul 2>nul
if %errorlevel% == 0 (
    echo [OK] pip found, using pip for installation...
    pip install --force-reinstall "%WHEEL%"
    goto :done
)

echo [ERROR] Neither uv nor pip found.
echo Please install Python 3.11+ from https://python.org
echo Or install uv from https://docs.astral.sh/uv
echo.
pause
exit /b 1

:done
echo.
echo ==========================================
echo   Installation complete!
echo.
echo   Usage:
echo     huginn-agent --help
echo     huginn-agent coder "fix bug"
echo     huginn-agent serve
echo     huginn-agent chat
echo.
echo   Run 'huginn-agent configure' for first-time setup.
echo ==========================================
pause
