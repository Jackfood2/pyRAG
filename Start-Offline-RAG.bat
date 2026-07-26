@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime\llama-server.exe" goto :missing
if not exist "python\python.exe" goto :missing
rem  Models are now started BY app.py and downloaded from the UI.
rem  (install_models.bat can still pre-fetch the two defaults if you want.)
start "Offline RAG" /b "python\python.exe" app.py
timeout /t 3 /nobreak >nul
start "" http://127.0.0.1:8765
exit /b 0

:missing
echo Offline RAG is incomplete. Keep the runtime and python folders with this launcher.
pause
exit /b 1