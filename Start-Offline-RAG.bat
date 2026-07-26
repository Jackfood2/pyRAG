@echo off
setlocal
cd /d "%~dp0"
if not exist "runtime\llama-server.exe" goto :missing
if not exist "python\python.exe" goto :missing
if not exist "models\nomic-embed-text-v1.5.Q4_K_M.gguf" goto :missing
if not exist "models\bge-reranker-v2-m3-Q4_K_M.gguf" goto :missing
set "RAG_THREADS=%NUMBER_OF_PROCESSORS%"
if "%RAG_THREADS%"=="" set "RAG_THREADS=4"
start "Offline RAG embeddings" /b "runtime\llama-server.exe" -m "models\nomic-embed-text-v1.5.Q4_K_M.gguf" --embedding --host 127.0.0.1 --port 8787 -t %RAG_THREADS%
start "Offline RAG reranker" /b "runtime\llama-server.exe" -m "models\bge-reranker-v2-m3-Q4_K_M.gguf" --reranking --host 127.0.0.1 --port 8788 -t %RAG_THREADS%
start "Offline RAG server" /b "python\python.exe" app.py
timeout /t 4 /nobreak >nul
start "Offline RAG" http://127.0.0.1:8765
exit /b 0

:missing
echo Offline RAG is incomplete. Keep the runtime, python, and models folders with this launcher.
pause
exit /b 1
