@echo off
title Install LanceDB for OfflineRAG
color 0A

set PYTHON_DIR=%~dp0python
set PYTHON_EXE=%PYTHON_DIR%\python.exe
set PTH_FILE=%PYTHON_DIR%\python311._pth

echo ========================================================
echo Checking Portable Python environment...
echo ========================================================
if not exist "%PYTHON_EXE%" (
    echo ERROR: python.exe not found at %PYTHON_EXE%
    echo Please run this script from the root OfflineRAG folder.
    pause
    exit /b 1
)

echo Enabling site-packages in portable Python (modifying ._pth)...
powershell -Command "(Get-Content '%PTH_FILE%') -replace '^#import site', 'import site' | Set-Content '%PTH_FILE%'"

echo.
echo ========================================================
echo Checking for pip...
echo ========================================================
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Pip not found in portable python. Downloading get-pip.py...
    powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
    echo Installing pip...
    "%PYTHON_EXE%" get-pip.py
    del get-pip.py
) else (
    echo Pip is already installed.
)

echo.
echo ========================================================
echo Installing LanceDB (this may take a minute)...
echo ========================================================
"%PYTHON_EXE%" -m pip install lancedb numpy

echo.
echo ========================================================
echo Installation complete! 
echo app.py will now automatically detect and use LanceDB.
echo ========================================================
pause