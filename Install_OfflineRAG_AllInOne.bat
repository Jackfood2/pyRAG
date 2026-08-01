@echo off
setlocal EnableExtensions DisableDelayedExpansion
title OfflineRAG Complete One-Click Installer
color 0A

set "ROOT=%~dp0"
set "PYTHON_DIR=%ROOT%python"
set "PYTHON_EXE=%PYTHON_DIR%\python.exe"
set "PTH_FILE=%PYTHON_DIR%\python311._pth"
set "MODELS_DIR=%ROOT%models"
set "LOG_DIR=%ROOT%data\logs"
set "LOG_FILE=%LOG_DIR%\complete-setup.log"
set "GET_PIP=%TEMP%\offlinerag-get-pip-%RANDOM%-%RANDOM%.py"
set "CURL=curl.exe"

set "EMBED_FILE=nomic-embed-text-v1.5.Q4_K_M.gguf"
set "EMBED_URL=https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF/resolve/main/nomic-embed-text-v1.5.Q4_K_M.gguf?download=true"
set "RERANK_FILE=bge-reranker-v2-m3-Q4_K_M.gguf"
set "RERANK_URL_1=https://huggingface.co/gpustack/bge-reranker-v2-m3-GGUF/resolve/main/bge-reranker-v2-m3-Q4_K_M.gguf?download=true"
set "RERANK_URL_2=https://huggingface.co/puppyM/bge-reranker-v2-m3-Q4_K_M-GGUF/resolve/main/bge-reranker-v2-m3-q4_k_m.gguf?download=true"

if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1
call :log ========================================================
call :log OfflineRAG Complete One-Click Installer started
call :log ========================================================

cls
echo ========================================================
echo  OfflineRAG Complete One-Click Installer
echo ========================================================
echo.
echo This single file installs everything needed for a new copy:
echo.
echo   - pip for the bundled portable Python, if missing
echo   - LanceDB and NumPy
echo   - PDF, XLSX, XLS, DOCX, MSG and PST dependencies
echo   - Nomic embedding model
echo   - BGE reranker model
echo.
echo It does NOT require Install_LanceDB.bat,
echo Install_Dependencies.bat, or Install_Models.bat.
echo.
echo You will now be asked whether this computer should use GPU acceleration.
echo.

if not exist "%PYTHON_EXE%" call :fail "Portable Python was not found at %PYTHON_EXE%"
if not exist "%PTH_FILE%" call :fail "Portable Python configuration was not found at %PTH_FILE%"
where curl.exe >nul 2>&1
if errorlevel 1 call :fail "curl.exe was not found. A supported Windows 10 or Windows 11 installation is required."

choice /C YN /N /M "Will this computer use GPU acceleration? [Y/N]: "
if errorlevel 2 (
    set "INSTALL_GPU=0"
    call :log User selected CPU-only installation.
) else (
    set "INSTALL_GPU=1"
    call :log User selected dependency, model, and GPU installation.
)

call :section "Step 1 of 6 - Enable portable Python site-packages"
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
  "$p='%PTH_FILE%'; $lines=Get-Content -LiteralPath $p; if($lines -notcontains 'import site'){ $lines=$lines -replace '^\s*#\s*import site\s*$','import site'; if($lines -notcontains 'import site'){ $lines += 'import site' }; Set-Content -LiteralPath $p -Value $lines -Encoding ASCII }"
if errorlevel 1 call :fail "Could not enable site-packages in python311._pth"
call :log Portable Python site-packages enabled.

call :section "Step 2 of 6 - Install pip if required"
"%PYTHON_EXE%" -m pip --version >nul 2>&1
if errorlevel 1 (
    echo Downloading the official get-pip.py bootstrap...
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command ^
      "$ProgressPreference='SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol=[Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%GET_PIP%' -TimeoutSec 300"
    if errorlevel 1 call :fail "Could not download get-pip.py"
    "%PYTHON_EXE%" "%GET_PIP%" --disable-pip-version-check
    if errorlevel 1 call :fail "pip installation failed"
    del /q "%GET_PIP%" >nul 2>&1
) else (
    echo pip is already available.
)
"%PYTHON_EXE%" -m pip --version
if errorlevel 1 call :fail "pip verification failed"
call :log pip is available.

call :section "Step 3 of 6 - Install LanceDB and all document dependencies"
echo Installing packages. This may take several minutes...
"%PYTHON_EXE%" -m pip install --upgrade --disable-pip-version-check ^
  numpy lancedb openpyxl defusedxml pymupdf pypdf xlrd extract-msg python-docx rapidfuzz pywin32
if errorlevel 1 call :fail "One or more Python packages failed to install"

"%PYTHON_EXE%" -c "import numpy,lancedb,openpyxl,defusedxml,pymupdf,pypdf,xlrd,extract_msg,docx,rapidfuzz,win32com.client,pythoncom; print('All Python dependencies imported successfully.')"
if errorlevel 1 call :fail "A package was installed but could not be imported"
call :log LanceDB and all document dependencies installed and verified.

call :section "Step 4 of 6 - Download embedding and reranker models"
if not exist "%MODELS_DIR%" mkdir "%MODELS_DIR%"
if errorlevel 1 call :fail "Could not create %MODELS_DIR%"

call :download "%EMBED_FILE%" "%EMBED_URL%" ""
if errorlevel 1 call :fail "Failed to install %EMBED_FILE%"
call :download "%RERANK_FILE%" "%RERANK_URL_1%" "%RERANK_URL_2%"
if errorlevel 1 call :fail "Failed to install %RERANK_FILE%"
call :log Both GGUF models downloaded and validated.

call :section "Step 5 of 6 - Final installation verification"
"%PYTHON_EXE%" -c "import lancedb,numpy,openpyxl,pymupdf,pypdf,xlrd,extract_msg,docx,rapidfuzz,win32com.client,pythoncom; print('Python components: OK')"
if errorlevel 1 call :fail "Final Python verification failed"
call :validate_gguf "%MODELS_DIR%\%EMBED_FILE%"
if errorlevel 1 call :fail "Embedding model failed final validation"
call :validate_gguf "%MODELS_DIR%\%RERANK_FILE%"
if errorlevel 1 call :fail "Reranker model failed final validation"

if exist "%GET_PIP%" del /q "%GET_PIP%" >nul 2>&1
call :log Core installation passed final verification.

call :section "Step 6 of 6 - Optional GPU runtime"
if "%INSTALL_GPU%"=="1" (
    echo Detecting the GPU and installing the appropriate llama.cpp runtime...
    powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText('%~f0'); $m='#===OFFLINERAG_EMBEDDED_GPU_PS==='; $i=$s.IndexOf($m); if($i -lt 0){Write-Host 'Embedded GPU installer is missing.' -ForegroundColor Red; exit 2}; & ([scriptblock]::Create($s.Substring($i))) '%ROOT%'"
    if errorlevel 1 call :fail "GPU runtime installation failed. The CPU runtime was preserved or restored."
    call :log GPU runtime installation completed successfully.
) else (
    echo CPU-only selected. GPU runtime installation skipped.
    call :log GPU runtime installation skipped by user.
)

call :log Complete installation passed final verification.

echo.
echo ========================================================
echo  OfflineRAG installation completed successfully.
echo ========================================================
echo.
echo Installed and verified:
echo   [OK] pip
echo   [OK] LanceDB and NumPy
echo   [OK] PDF support: PyMuPDF and pypdf
echo   [OK] Excel support: openpyxl and xlrd
echo   [OK] DOCX, MSG and PST dependencies
echo   [OK] Embedding model
echo   [OK] Reranker model
echo.
if "%INSTALL_GPU%"=="1" (
    echo   [OK] GPU runtime selected and installed
) else (
    echo   [OK] CPU-only runtime retained
)
echo.
echo Start OfflineRAG and build the document index.
echo.
echo Log: %LOG_FILE%
pause
endlocal & exit /b 0

:download
set "FILE=%~1"
set "URL1=%~2"
set "URL2=%~3"
set "DEST=%MODELS_DIR%\%FILE%"
set "PART=%DEST%.part"

if exist "%DEST%" (
    call :validate_gguf "%DEST%"
    if not errorlevel 1 (
        echo [OK] %FILE% is already installed and valid.
        call :log Existing valid model retained: %FILE%
        exit /b 0
    )
    echo [WARN] Existing %FILE% is invalid and will be replaced.
    del /q "%DEST%" >nul 2>&1
)

echo Downloading %FILE%...
"%CURL%" --location --fail --show-error --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 30 --continue-at - --output "%PART%" "%URL1%"
if errorlevel 1 (
    if "%URL2%"=="" (
        del /q "%PART%" >nul 2>&1
        exit /b 1
    )
    echo Primary source failed. Trying the fallback source...
    del /q "%PART%" >nul 2>&1
    "%CURL%" --location --fail --show-error --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 30 --continue-at - --output "%PART%" "%URL2%"
    if errorlevel 1 (
        del /q "%PART%" >nul 2>&1
        exit /b 1
    )
)

call :validate_gguf "%PART%"
if errorlevel 1 (
    del /q "%PART%" >nul 2>&1
    exit /b 1
)
move /y "%PART%" "%DEST%" >nul
if errorlevel 1 exit /b 1
echo [OK] %FILE%
exit /b 0

:validate_gguf
if not exist "%~1" exit /b 1
for %%A in ("%~1") do if %%~zA LSS 1048576 exit /b 1
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -Command "$p='%~1'; $s=[IO.File]::OpenRead($p); try{$b=New-Object byte[] 4; if($s.Read($b,0,4)-ne 4){exit 1}; if([Text.Encoding]::ASCII.GetString($b)-ne 'GGUF'){exit 1}} finally {$s.Dispose()}"
exit /b %ERRORLEVEL%

:section
echo.
echo ========================================================
echo %~1
echo ========================================================
call :log %~1
exit /b 0

:log
echo [%date% %time%] %*>>"%LOG_FILE%"
exit /b 0

:fail
echo.
echo ========================================================
echo  INSTALLATION FAILED
echo ========================================================
echo %~1
call :log ERROR: %~1
if exist "%GET_PIP%" del /q "%GET_PIP%" >nul 2>&1
echo.
echo Review the message above and this log:
echo %LOG_FILE%
pause
endlocal & exit /b 1

#===OFFLINERAG_EMBEDDED_GPU_PS===
param([string]$InstallRoot)
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = if ($InstallRoot) { $InstallRoot.TrimEnd('\','/') } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$rt = Join-Path $root 'runtime'
$bak = Join-Path $root 'runtime_cpu_backup'
$stage = Join-Path $root ('runtime_gpu_stage_' + [guid]::NewGuid().ToString('N'))
$old = Join-Path $root ('runtime_previous_' + [guid]::NewGuid().ToString('N'))
$zip = Join-Path $env:TEMP ('offlinerag_gpu_' + [guid]::NewGuid().ToString('N') + '.zip')
$cudartZip = Join-Path $env:TEMP ('offlinerag_cudart_' + [guid]::NewGuid().ToString('N') + '.zip')
$ex = Join-Path $env:TEMP ('offlinerag_gpu_ex_' + [guid]::NewGuid().ToString('N'))

function Say([string]$Message,[ConsoleColor]$Color='White') { Write-Host $Message -ForegroundColor $Color }
function Pause-Installer { }
function Remove-Safe([string]$Path) { if ($Path -and (Test-Path -LiteralPath $Path)) { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue } }
function Fail([string]$Message,[int]$Code=1) { Say ''; Say ('ERROR: ' + $Message) Red; Pause-Installer; exit $Code }

try {
    Say ''
    Say '========================================================' Cyan
    Say ' OfflineRAG - GPU runtime installer (llama.cpp)' Cyan
    Say '========================================================' Cyan
    Say "App folder : $root"
    Say "Runtime    : $rt"
    Say ''

    if (-not (Test-Path -LiteralPath $rt -PathType Container)) { Fail "runtime folder not found at $rt" }
    if (-not (Test-Path -LiteralPath (Join-Path $rt 'llama-server.exe') -PathType Leaf)) { Fail 'runtime\llama-server.exe is missing.' }

    $adapterNames = ''
    $hasRealGpu = $false
    try {
        $adapters = @(Get-CimInstance Win32_VideoController -ErrorAction Stop | Where-Object {
            $_.Name -and $_.Name -notmatch 'Microsoft Basic Display|Remote Display|Virtual' -and (!$_.Status -or $_.Status -eq 'OK')
        })
        if ($adapters.Count) {
            $hasRealGpu = $true
            $adapterNames = ($adapters | ForEach-Object Name) -join ' | '
        }
    } catch { Say "GPU enumeration warning: $($_.Exception.Message)" Yellow }

    $nvOk = $false
    $nvCudaMajor = 0
    try {
        $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction Stop
        $output = & $nvidiaSmi.Source 2>$null
        if ($LASTEXITCODE -eq 0 -and $output) {
            $nvOk = $true
            $text = $output | Out-String
            if ($text -match 'CUDA Version:\s*(\d+)(?:\.(\d+))?') { $nvCudaMajor = [int]$Matches[1] }
        }
    } catch {}

    $mode = if ($nvOk -and $nvCudaMajor -ge 12) { 'cuda' } elseif ($hasRealGpu) { 'vulkan' } else { $null }
    Say "Display adapter(s): $(if($adapterNames){$adapterNames}else{'(none / basic display only)'})"
    if ($nvOk) { Say "NVIDIA            : driver OK; reports CUDA $nvCudaMajor.x compatibility" Green }
    if (-not $mode) {
        Say ''
        Say 'No compatible GPU was detected. The CPU runtime was not changed.' Yellow
        Pause-Installer
        exit 0
    }

    $cuPreferred = if ($nvCudaMajor -ge 13) { 13 } else { 12 }
    Say "Chosen backend    : $mode$(if($mode -eq 'cuda'){" (prefer CUDA $cuPreferred.x)"}else{' (NVIDIA / AMD / Intel)'})" Green
    Say ''
    Say 'GPU installation was selected in the main setup.' Green

    Say 'Querying the latest llama.cpp release on GitHub...' Gray
    $headers = @{ 'User-Agent'='OfflineRAG-GPU-Installer'; 'Accept'='application/vnd.github+json' }
    $release = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' -Headers $headers -TimeoutSec 45
    $assets = @($release.assets | Where-Object { $_.name -match '\.zip$' -and $_.name -match 'win' -and $_.name -match 'x64' })
    if (-not $assets.Count) { Fail 'The latest release has no Windows x64 ZIP assets.' }
    Say "Latest release    : $($release.tag_name)"

    function Pick-Asset([string[]]$Patterns) {
        foreach ($pattern in $Patterns) {
            $match = $assets | Where-Object { $_.name -match $pattern } | Sort-Object name | Select-Object -First 1
            if ($match) { return $match }
        }
        return $null
    }

    if ($mode -eq 'cuda') {
        $asset = Pick-Asset @("win-cuda-cu$cuPreferred.*x64\.zip$", "win-cuda.*cu$cuPreferred.*x64\.zip$")
        if (-not $asset) {
            $other = if ($cuPreferred -eq 12) { 13 } else { 12 }
            $asset = Pick-Asset @("win-cuda-cu$other.*x64\.zip$", "win-cuda.*cu$other.*x64\.zip$")
        }
        if (-not $asset) { $asset = Pick-Asset @('win-cuda.*x64\.zip$') }
    } else {
        $asset = Pick-Asset @('win-vulkan.*x64\.zip$', 'win.*vulkan.*x64\.zip$')
    }
    if (-not $asset) {
        Say 'Matching assets found in the release:' Gray
        $assets | ForEach-Object { Say ('  ' + $_.name) Gray }
        Fail "No matching $mode asset was found. llama.cpp may have changed its asset names."
    }
    Say "Selected asset    : $($asset.name) ($([math]::Round($asset.size / 1MB, 1)) MB)" Green

    # Recent releases package CUDA runtime DLLs separately. Match the CUDA
    # major/minor encoded in the selected llama archive and overlay both.
    $cudartAsset = $null
    if ($mode -eq 'cuda') {
        $cudaLabel = $null
        if ($asset.name -match 'cuda[-_](?:cu)?(\d+(?:\.\d+)?)') { $cudaLabel = $Matches[1] }
        if ($cudaLabel) {
            $escaped = [regex]::Escape($cudaLabel)
            $cudartAsset = $release.assets | Where-Object {
                $_.name -match "^cudart-.*win-cuda-$escaped-x64\.zip$"
            } | Select-Object -First 1
        }
        if (-not $cudartAsset) {
            $cudartAsset = $release.assets | Where-Object {
                $_.name -match '^cudart-.*win-cuda-.*-x64\.zip$'
            } | Sort-Object name -Descending | Select-Object -First 1
        }
        if ($cudartAsset) { Say "CUDA runtime asset: $($cudartAsset.name)" Green }
        else { Say 'No companion CUDA-runtime archive was published; continuing with the main archive.' Yellow }
    }

    Say 'Downloading...' Gray
    Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -Headers $headers -TimeoutSec 1800
    if (-not (Test-Path -LiteralPath $zip) -or (Get-Item -LiteralPath $zip).Length -lt 1MB) { Fail 'The downloaded archive is missing or unexpectedly small.' }
    $magic = [IO.File]::ReadAllBytes($zip)[0..1]
    if ($magic[0] -ne 0x50 -or $magic[1] -ne 0x4B) { Fail 'The downloaded file is not a ZIP archive.' }
    if ($cudartAsset) {
        Invoke-WebRequest -Uri $cudartAsset.browser_download_url -OutFile $cudartZip -Headers $headers -TimeoutSec 1800
        if (-not (Test-Path -LiteralPath $cudartZip) -or (Get-Item -LiteralPath $cudartZip).Length -lt 1MB) { Fail 'The CUDA runtime archive is missing or unexpectedly small.' }
    }

    Say 'Extracting to a staging folder...' Gray
    New-Item -ItemType Directory -Path $ex -Force | Out-Null
    Expand-Archive -LiteralPath $zip -DestinationPath $ex -Force
    if ($cudartAsset) { Expand-Archive -LiteralPath $cudartZip -DestinationPath $ex -Force }
    $server = Get-ChildItem -LiteralPath $ex -Recurse -File -Filter 'llama-server.exe' | Select-Object -First 1
    if (-not $server) { Fail 'The archive does not contain llama-server.exe.' }
    $sourceDir = $server.Directory.FullName
    $expectedPattern = if ($mode -eq 'cuda') { 'ggml-cuda*.dll' } else { 'ggml-vulkan*.dll' }
    if (-not (Get-ChildItem -LiteralPath $sourceDir -File -Filter $expectedPattern -ErrorAction SilentlyContinue)) {
        Fail "The selected archive does not contain $expectedPattern alongside llama-server.exe."
    }

    New-Item -ItemType Directory -Path $stage -Force | Out-Null
    Get-ChildItem -LiteralPath $rt -Force | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $stage -Recurse -Force }
    Get-ChildItem -LiteralPath $sourceDir -File | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $stage -Force }
    if ($cudartAsset) {
        Get-ChildItem -LiteralPath $ex -Recurse -File | Where-Object {
            $_.Name -match '^(cudart|cublas|cublasLt|nvrtc|nvJitLink).*\.dll$'
        } | ForEach-Object { Copy-Item -LiteralPath $_.FullName -Destination $stage -Force }
    }
    if (-not (Test-Path -LiteralPath (Join-Path $stage 'llama-server.exe')) -or
        -not (Get-ChildItem -LiteralPath $stage -File -Filter $expectedPattern -ErrorAction SilentlyContinue)) {
        Fail 'Staged runtime verification failed.'
    }

    if (-not (Test-Path -LiteralPath $bak)) {
        Say 'Creating the one-time CPU fallback: runtime_cpu_backup...' Gray
        Copy-Item -LiteralPath $rt -Destination $bak -Recurse -Force
    } else {
        Say 'Keeping the existing runtime_cpu_backup unchanged.' Gray
    }

    Say 'Activating the staged GPU runtime atomically...' Gray
    Move-Item -LiteralPath $rt -Destination $old
    try {
        Move-Item -LiteralPath $stage -Destination $rt
    } catch {
        if (Test-Path -LiteralPath $old) { Move-Item -LiteralPath $old -Destination $rt }
        throw
    }
    Remove-Safe $old

    Say ''
    Say '========================================================' Cyan
    Say ' GPU runtime installed successfully.' Green
    Say " Backend: $mode; release: $($release.tag_name)" Green
    Say '========================================================' Cyan
    Say 'Restart OfflineRAG. GPU offload can remain enabled with 99 layers.' White
    Say 'To restore CPU later, replace runtime with runtime_cpu_backup.' Gray
    Pause-Installer
    exit 0
} catch {
    if ((Test-Path -LiteralPath $old) -and -not (Test-Path -LiteralPath $rt)) {
        Move-Item -LiteralPath $old -Destination $rt -ErrorAction SilentlyContinue
    }
    Fail $_.Exception.Message
} finally {
    Remove-Safe $zip
    Remove-Safe $cudartZip
    Remove-Safe $ex
    Remove-Safe $stage
    if ((Test-Path -LiteralPath $old) -and (Test-Path -LiteralPath $rt)) { Remove-Safe $old }
}
