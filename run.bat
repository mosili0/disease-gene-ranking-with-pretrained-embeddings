@echo off
setlocal
set "PROJ=%~dp0"
if defined BIOFM_PYTHON (
    set "TASK_PYTHON=%BIOFM_PYTHON%"
) else if exist "%PROJ%.venv\Scripts\python.exe" (
    set "TASK_PYTHON=%PROJ%.venv\Scripts\python.exe"
) else (
    set "TASK_PYTHON=python"
)
"%TASK_PYTHON%" -c "import numpy,pandas,scipy,matplotlib,torch" >nul 2>&1
if errorlevel 1 (
    echo [error] Python dependencies are missing. Follow README.md environment setup.
    echo         Or set BIOFM_PYTHON to the Python executable in your existing environment.
    exit /b 1
)
"%TASK_PYTHON%" "%PROJ%main.py" %*
set "TASK_EXIT=%ERRORLEVEL%"
exit /b %TASK_EXIT%
