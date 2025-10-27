@echo off
setlocal enabledelayedexpansion

:: ======================================
:: CONFIGURATION
:: ======================================
set "CONDA_DIR=%USERPROFILE%\miniconda3"
set "ENV_NAME=myenv"
set "REQ_FILE=requirements.txt"
set "UI_FILE=ui.py"
set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"

:: ======================================
:: 1. Install Miniconda if not installed
:: ======================================
if not exist "%CONDA_DIR%\Scripts\conda.exe" (
    echo [INFO] Miniconda not found. Installing...
    curl -L "%MINICONDA_URL%" -o "%TEMP%\miniconda.exe"
    start /wait "" "%TEMP%\miniconda.exe" /S /D=%CONDA_DIR%
    del "%TEMP%\miniconda.exe"
) else (
    echo [OK] Miniconda already installed at %CONDA_DIR%
)

:: ======================================
:: 2. Initialize Conda
:: ======================================
call "%CONDA_DIR%\Scripts\activate.bat"
call conda init cmd.exe >nul 2>&1

:: ======================================
:: 3. Create or reuse environment
:: ======================================
echo [INFO] Checking environment "%ENV_NAME%"...
for /f "tokens=1" %%i in ('conda env list ^| findstr "%ENV_NAME%"') do (
    set "ENV_FOUND=1"
)
if defined ENV_FOUND (
    echo [OK] Environment "%ENV_NAME%" already exists. Skipping creation.
) else (
    if exist "%REQ_FILE%" (
        echo [INFO] Creating new environment "%ENV_NAME%" and installing from %REQ_FILE%...
        call conda create -y -n %ENV_NAME% python=3.10
        call conda activate %ENV_NAME%
        pip install -r "%REQ_FILE%"
    ) else if exist "environment.yml" (
        echo [INFO] Creating environment from environment.yml...
        call conda env create -f environment.yml
    ) else (
        echo [WARN] No requirements.txt or environment.yml found. Creating empty environment.
        call conda create -y -n %ENV_NAME% python=3.10
    )
)

:: ======================================
:: 4. Activate environment
:: ======================================
call conda activate %ENV_NAME%

:: ======================================
:: 5. Run UI script
:: ======================================
if exist "%UI_FILE%" (
    echo [INFO] Running %UI_FILE%...
    python "%UI_FILE%"
) else (
    echo [ERROR] %UI_FILE% not found in current directory!
    exit /b 1
)

endlocal
pause
