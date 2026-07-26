@echo off
setlocal

:: Set the target directory to a "models" folder in the same directory as this script
set "SCRIPT_DIR=%~dp0"
set "MODELS_DIR=%SCRIPT_DIR%models"

:: Create the models directory if it doesn't exist
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"

echo =========================================================
echo  Downloading RAG Models to: %MODELS_DIR%
echo =========================================================
echo.

:: ---------------------------------------------------------
:: 1. Download nomic-embed-text-v1.5.Q4_K_M.gguf
:: ---------------------------------------------------------
set "EMBED_FILE=nomic-embed-text-v1.5.Q4_K_M.gguf"
set "EMBED_URL=https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf"

if exist "%MODELS_DIR%\%EMBED_FILE%" (
    echo [1/2] %EMBED_FILE% already exists. Skipping.
) else (
    echo [1/2] Downloading %EMBED_FILE% ...
    curl -L -f -o "%MODELS_DIR%\%EMBED_FILE%" "%EMBED_URL%"
    if errorlevel 1 (
        echo [ERROR] Failed to download %EMBED_FILE%.
        del "%MODELS_DIR%\%EMBED_FILE%" 2>nul
        pause
        exit /b 1
    )
    echo Download complete.
)
echo.

:: ---------------------------------------------------------
:: 2. Download bge-reranker-v2-m3-Q4_K_M.gguf
:: ---------------------------------------------------------
set "RERANK_FILE=bge-reranker-v2-m3-Q4_K_M.gguf"

if exist "%MODELS_DIR%\%RERANK_FILE%" (
    echo [2/2] %RERANK_FILE% already exists. Skipping.
    goto :finish
)

echo [2/2] Downloading %RERANK_FILE% ...

:: Using verified, public, ungated mirrors
set "RERANK_URL_1=https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf"
set "RERANK_URL_2=https://huggingface.co/puppyM/bge-reranker-v2-m3-Q4_K_M-GGUF/resolve/main/bge-reranker-v2-m3-q4_k_m.gguf"

echo Trying source 1 (gpustack)...
:: Download directly to the final filename, skipping the temp file method entirely
curl -L -f -o "%MODELS_DIR%\%RERANK_FILE%" "%RERANK_URL_1%"
if errorlevel 1 (
    echo Source 1 failed. Trying source 2 (puppyM)...
    del "%MODELS_DIR%\%RERANK_FILE%" 2>nul
    curl -L -f -o "%MODELS_DIR%\%RERANK_FILE%" "%RERANK_URL_2%"
)

if errorlevel 1 (
    echo [ERROR] Failed to download %RERANK_FILE% from all sources.
    del "%MODELS_DIR%\%RERANK_FILE%" 2>nul
    pause
    exit /b 1
)

echo Download complete.

:finish
echo.
echo =========================================================
echo  All models downloaded successfully to:
echo  %MODELS_DIR%
echo =========================================================
pause
exit /b 0