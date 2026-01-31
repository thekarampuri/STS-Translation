@echo off
TITLE STS-Translation Server

echo ==================================================
echo Starting Bhasha Setu STS Translation Server...
echo ==================================================

:: Set the specific environment python path
set "ENV_PYTHON=C:\Users\Admin\miniconda3\envs\indic-tts\python.exe"

:: Check if the specific environment python exists
if exist "%ENV_PYTHON%" (
    echo Found configured Python environment: %ENV_PYTHON%
    set "PYTHON_EXE=%ENV_PYTHON%"
) else (
    echo Configured Python environment not found at: %ENV_PYTHON%
    echo Falling back to system 'python'...
    set "PYTHON_EXE=python"
)

echo.
echo Using Python: %PYTHON_EXE%
echo Working Directory: %CD%
echo.

:: Run the application
"%PYTHON_EXE%" app.py

:: Check exit code
if %errorlevel% neq 0 (
    echo.
    echo ==================================================
    echo Server stopped with error code %errorlevel%.
    echo ==================================================
    pause
) else (
    echo Server stopped normally.
)
