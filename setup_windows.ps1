param([string]$EnginesRoot = "",[switch]$SkipEngines,[switch]$SkipDownloads)
$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($EnginesRoot)) { $EnginesRoot = Join-Path (Split-Path $ProjectRoot -Parent) "CalibrationEngines" }
$EnginesRoot = [System.IO.Path]::GetFullPath($EnginesRoot)

Write-Host "=============================================================="
Write-Host "FootballAnalysisAI - Portable Windows Setup v1"
Write-Host "Project : $ProjectRoot"
Write-Host "Engines : $EnginesRoot"
Write-Host "Mode    : CPU baseline"
Write-Host "=============================================================="

if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is not installed." }

function Get-Python310 {
    $candidate=Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"
    if (Test-Path $candidate) { return $candidate }
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $resolved=& py -3.10 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $resolved) { return $resolved.Trim() }
    }
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        & winget install --id Python.Python.3.10 -e --accept-package-agreements --accept-source-agreements
        if (Test-Path $candidate) { return $candidate }
    }
    throw "Python 3.10 is required."
}

$Py310=Get-Python310
$MainPy=Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $MainPy)) {
    Write-Host "[CREATE] main .venv"
    & $Py310 -m venv (Join-Path $ProjectRoot ".venv")
}
& $MainPy -m pip install --upgrade pip
& $MainPy -m pip install "torch==1.13.1+cpu" "torchvision==0.14.1+cpu" --extra-index-url "https://download.pytorch.org/whl/cpu" --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
if ($LASTEXITCODE -ne 0) { throw "Main PyTorch install failed." }
& $MainPy -m pip install -r (Join-Path $ProjectRoot "requirements\main-windows-cpu.txt")
if ($LASTEXITCODE -ne 0) { throw "Main requirements install failed." }

if (-not $SkipEngines) {
    $py39=Join-Path $env:LOCALAPPDATA "Programs\Python\Python39\python.exe"
    if (-not (Test-Path $py39) -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        & winget install --id Python.Python.3.9 -e --accept-package-agreements --accept-source-agreements
    }
    & (Join-Path $ProjectRoot "scripts\install_engines.ps1") -EnginesRoot $EnginesRoot -SkipDownloads:$SkipDownloads
}

$config=Join-Path $ProjectRoot "configs\local.yaml"
if (-not (Test-Path $config)) {
    Copy-Item (Join-Path $ProjectRoot "configs\local.example.yaml") $config
    Write-Host "[CREATE] configs\local.yaml"
}

Write-Host "[CHECK] installation"
& $MainPy (Join-Path $ProjectRoot "scripts\health_check.py") --engines-root $EnginesRoot
$exit=$LASTEXITCODE
if ($exit -eq 0) {
    Write-Host "=============================================================="
    Write-Host "READY - Portable CPU baseline installed."
    Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
    Write-Host "=============================================================="
} else {
    Write-Host "Setup finished with health-check failures."
    exit $exit
}
