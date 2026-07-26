@echo off
title Install GPU runtime for OfflineRAG
color 0A
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=[IO.File]::ReadAllText('%~f0'); $m='#===RAG_GPU'+'_PS==='; $i=$s.IndexOf($m); if($i -lt 0){Write-Host 'Installer file is corrupt.'; exit 2}; & ([scriptblock]::Create($s.Substring($i))) '%~dp0'"
exit /b %ERRORLEVEL%

#===RAG_GPU_PS===
$ErrorActionPreference = 'Stop'
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$root = if ($args -and $args[0]) { ([string]$args[0]).TrimEnd('\','/') } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$rt   = Join-Path $root 'runtime'
$bak  = Join-Path $root 'runtime_cpu_backup'
function Say($m,$c='White'){ Write-Host $m -ForegroundColor $c }

Say ''
Say '========================================================' Cyan
Say ' OfflineRAG  -  GPU runtime installer (llama.cpp)' Cyan
Say '========================================================' Cyan
Say "App folder : $root"
Say "Runtime    : $rt"
Say ''

# ---- 1. detect a GPU ------------------------------------------------------
$adapterNames = ''; $hasRealGpu = $false
try {
  $ad = Get-CimInstance Win32_VideoController -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -and ($_.Name -notmatch 'Microsoft Basic Display') -and ($_.Status -eq 'OK') }
  if ($ad) { $hasRealGpu = $true; $adapterNames = ($ad | ForEach-Object { $_.Name }) -join ' | ' }
} catch {}

$nvOk = $false; $nvCuda = 0
try {
  $out = & nvidia-smi 2>$null
  if ($LASTEXITCODE -eq 0 -and $out) {
    $nvOk = $true
    $txt  = ($out | Out-String)
    if ($txt -match 'CUDA Version:\s*(\d+)') { $nvCuda = [int]$Matches[1] }
  }
} catch {}

$mode = $null
if     ($nvOk -and $nvCuda -ge 12) { $mode = 'cuda' }
elseif ($hasRealGpu)               { $mode = 'vulkan' }

Say "Display adapter(s): $(if($adapterNames){$adapterNames}else{'(none / basic display only)'})"
if ($nvOk) { Say "NVIDIA            : driver OK, supports CUDA $nvCuda.x" Green }

if (-not $mode) {
  Say ''
  Say 'No compatible GPU found - nothing will be installed.' Yellow
  Say 'The app keeps using the CPU runtime (no files were changed).' Gray
  Read-Host 'Press Enter to close'; exit 0
}

$cuPref = if ($nvCuda -ge 13) { 13 } else { 12 }
Say "Chosen backend    : $mode$(if($mode -eq 'cuda'){" (CUDA $cuPref.x)"}else{' (universal: NVIDIA / AMD / Intel)'})" Green
Say ''
$ans = Read-Host "Download the $mode llama.cpp build and replace runtime\ ?  [Y/N]"
if ($ans -notmatch '^[Yy]') { Say 'Aborted - no changes made.' Yellow; exit 0 }
if (-not (Test-Path $rt))   { Say "ERROR: runtime folder not found at $rt" Red; Read-Host 'Press Enter'; exit 1 }

# ---- 2. back up the current CPU runtime (once) ----------------------------
if (-not (Test-Path $bak)) {
  Say 'Backing up current runtime -> runtime_cpu_backup ...' Gray
  Copy-Item -Path $rt -Destination $bak -Recurse -Force
} else {
  Say 'runtime_cpu_backup already exists - keeping it as your CPU fallback.' Gray
}

# ---- 3. pick the right asset from the latest GitHub release ---------------
Say 'Querying latest llama.cpp release on GitHub ...' Gray
try {
  $rel = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases/latest' `
                           -Headers @{ 'User-Agent'='OfflineRAG-GPU-Installer' }
} catch { Say "ERROR querying GitHub: $($_.Exception.Message)" Red; Read-Host 'Press Enter'; exit 1 }
$assets = @($rel.assets)
Say "Latest release    : $($rel.tag_name)  ($($assets.Count) assets)"

function Pick($patterns){
  foreach ($p in $patterns) {
    $a = $assets | Where-Object { $_.name -match $p } | Select-Object -First 1
    if ($a) { return $a }
  }
  return $null
}
$asset = $null
if ($mode -eq 'cuda') {
  $asset = Pick @("win-cuda-cu$cuPref.*x64\.zip$", "win-cuda-cu$cuPref")
  if (-not $asset) { $o = if($cuPref -eq 12){13}else{12}; $asset = Pick @("win-cuda-cu$o.*x64\.zip$","win-cuda-cu$o") }
  if (-not $asset) { $asset = Pick @("win-cuda.*x64\.zip$", "win-cuda") }
} else {
  $asset = Pick @("win-vulkan.*x64\.zip$", "win-vulkan")
}
if (-not $asset) {
  Say "ERROR: no matching $mode Windows x64 asset in this release." Red
  Say 'Available assets:'; $assets | ForEach-Object { Say ("   " + $_.name) Gray }
  Read-Host 'Press Enter'; exit 1
}
Say "Selected asset    : $($asset.name)  ($([math]::Round(($asset.size/1MB),1)) MB)" Green

# ---- 4. download ----------------------------------------------------------
$zip = Join-Path $env:TEMP ("rag_gpu_" + [guid]::NewGuid().ToString('N') + '.zip')
Say 'Downloading ... (one-time, please wait)' Gray
$ProgressPreference = 'SilentlyContinue'
try { Invoke-WebRequest -Uri $asset.browser_download_url -OutFile $zip -UserAgent 'OfflineRAG-GPU-Installer' }
catch { Say "ERROR downloading: $($_.Exception.Message)" Red; Read-Host 'Press Enter'; exit 1 }

# ---- 5. extract + overlay runtime\ ---------------------------------------
$ex = Join-Path $env:TEMP ("rag_gpu_ex_" + [guid]::NewGuid().ToString('N'))
Say 'Extracting ...' Gray
try { Expand-Archive -Path $zip -DestinationPath $ex -Force }
catch { Say "ERROR extracting: $($_.Exception.Message)" Red; Read-Host 'Press Enter'; exit 1 }
$server = Get-ChildItem -Path $ex -Recurse -Filter 'llama-server.exe' -ErrorAction SilentlyContinue | Select-Object -First 1
if (-not $server) { Say 'ERROR: archive has no llama-server.exe (unexpected layout).' Red; Read-Host 'Press Enter'; exit 1 }
$srcDir = $server.DirectoryName
Say "Installing GPU binaries into runtime\ (overwriting CPU copies) ..." Gray
Get-ChildItem -Path $srcDir -File | ForEach-Object { Copy-Item -Path $_.FullName -Destination (Join-Path $rt $_.Name) -Force }

# ---- 6. verify + clean up -------------------------------------------------
$haveCuda   = (Get-ChildItem $rt -Filter 'ggml-cuda*.dll'   -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
$haveVulkan = (Get-ChildItem $rt -Filter 'ggml-vulkan*.dll' -ErrorAction SilentlyContinue | Measure-Object).Count -gt 0
$haveServer = Test-Path (Join-Path $rt 'llama-server.exe')
Remove-Item $zip -Force -ErrorAction SilentlyContinue
Remove-Item $ex  -Recurse -Force -ErrorAction SilentlyContinue

Say ''
Say '========================================================' Cyan
if ($haveServer -and ($haveCuda -or $haveVulkan)) {
  Say ' GPU runtime installed successfully.' Green
  $b = if ($haveCuda) { 'cuda' } else { 'vulkan' }
  Say " Backend DLL now present: ggml-$b.dll" Green
} else {
  Say ' WARNING: copy finished but the expected GPU DLL was not found in runtime\.' Yellow
}
Say ''
Say 'Next steps:' White
Say '  1. (Re)start OfflineRAG with Start-Offline-RAG.bat.' White
Say '  2. In Settings keep "GPU offload" ON (default) and GPU layers = 99 (all).' White
Say '  3. The app auto-detects the backend; if GPU init ever fails it falls' White
Say '     back to CPU by itself - so a wrong driver cannot brick anything.' White
Say ''
Say 'To revert to CPU: delete runtime\ then rename runtime_cpu_backup\ to runtime\.' Gray
Say ''
Read-Host 'Press Enter to close'
exit 0