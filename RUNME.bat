@echo off
setlocal
set "TARGET_VERSION=Python 3.11.9"
set "VENV_PATH=.venv"
for /f "delims=" %%i in ('python --version 2^>^&1') do set "INSTALLED_VERSION=%%i"
if "%INSTALLED_VERSION%" NEQ "%TARGET_VERSION%" (
    echo "This project requires Python %TARGET_VERSION%"
    exit /b 1
)
if exist "%VENV_PATH%\Scripts\python.exe" (
    echo [INFO] Venv found and looks valid.
) else (
    echo [WARNING] Venv is missing or corrupted. Recreating...
    if exist "%VENV_PATH%" rmdir /s /q "%VENV_PATH%"
    python -m venv %VENV_PATH%
    if not exist "%VENV_PATH%\Scripts\python.exe" (
        echo [FATAL] Python failed to create a functional venv.
        echo Check permissions or antivirus settings.
        pause
        exit /b 1
    )
)
echo [INFO] Activating environment...
call %VENV_PATH%\Scripts\activate

python -m pip install tesserocr-2.9.1-cp311-cp311-win_amd64.whl
if %errorlevel% neq 0 (echo [ERROR] Failed to install tesserocr && exit /b 1)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
if %errorlevel% neq 0 (echo [ERROR] Failed to install torch && exit /b 1)
python -m pip install -r requirements.txt
if %errorlevel% neq 0 (exit /b 1)

python -m pip show en_core_web_md >nul 2>&1
if %errorlevel% neq 0 (python -m spacy download en_core_web_md)

python -m pip show ru_core_news_lg >nul 2>&1
if %errorlevel% neq 0 (python -m spacy download ru_core_news_lg)

python -m pip install -e .

echo [SUCCESS] Setup completed.
endlocal
pause
