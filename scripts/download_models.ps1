param([string]$EnginesRoot="")

$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

if ([string]::IsNullOrWhiteSpace($EnginesRoot)) {
    $EnginesRoot = Join-Path (Split-Path $ProjectRoot -Parent) "CalibrationEngines"
}

$PnL = Join-Path $EnginesRoot "PnLCalib"
$TV  = Join-Path $EnginesRoot "tvcalib"

function GetFile($Uri, $Out, $Label) {
    if (Test-Path $Out) {
        $l = (Get-Item $Out).Length
        if ($l -gt 0) {
            Write-Host "[OK] $Label ($l bytes)"
            return
        }
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $Out -Parent) | Out-Null

    Write-Host "[DOWNLOAD] $Label"
    Invoke-WebRequest -Uri $Uri -OutFile $Out
}

function Test-ModelFile($Path, $ExpectedBytes, $ExpectedSha256) {
    if (-not (Test-Path $Path)) {
        return $false
    }

    $item = Get-Item $Path

    if ($item.Length -ne [int64]$ExpectedBytes) {
        return $false
    }

    $hash = (Get-FileHash $Path -Algorithm SHA256).Hash

    return $hash.Equals(
        [string]$ExpectedSha256,
        [System.StringComparison]::OrdinalIgnoreCase
    )
}

function Install-FootballModel($Model, $PythonExe) {
    $modelsDir = Join-Path $ProjectRoot "models"
    New-Item -ItemType Directory -Force -Path $modelsDir | Out-Null

    $out = Join-Path $modelsDir $Model.name

    if (Test-ModelFile $out $Model.bytes $Model.sha256) {
        Write-Host "[OK] $($Model.name) verified"
        return
    }

    if (Test-Path $out) {
        Write-Host "[WARN] Invalid model file removed: $($Model.name)"
        Remove-Item $out -Force
    }

    Write-Host "[DOWNLOAD] $($Model.name)"

    & $PythonExe -m gdown $Model.gdrive_id -O $out

    if ($LASTEXITCODE -ne 0) {
        throw "Model download failed: $($Model.name)"
    }

    if (-not (Test-ModelFile $out $Model.bytes $Model.sha256)) {
        throw "Model integrity check failed: $($Model.name)"
    }

    Write-Host "[OK] $($Model.name) downloaded and SHA256 verified"
}

# ----------------------------------------------------------------------
# Calibration models
# ----------------------------------------------------------------------

GetFile `
    "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp" `
    (Join-Path $PnL "weights\SV_kp") `
    "PnLCalib SV_kp"

GetFile `
    "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines" `
    (Join-Path $PnL "weights\SV_lines") `
    "PnLCalib SV_lines"

GetFile `
    "https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt" `
    (Join-Path $TV "data\segment_localization\train_59.pt") `
    "TVCalib train_59.pt"

# ----------------------------------------------------------------------
# Football detection models
# ----------------------------------------------------------------------

$manifestPath = Join-Path $ProjectRoot "configs\model_manifest.json"

if (-not (Test-Path $manifestPath)) {
    throw "Model manifest missing: $manifestPath"
}

$mainPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $mainPython)) {
    throw "Main Python environment missing: $mainPython"
}

$manifest = Get-Content $manifestPath -Raw | ConvertFrom-Json

foreach ($model in $manifest.models) {
    if ($model.required) {
        Install-FootballModel $model $mainPython
    }
}

Write-Host "=============================================================="
Write-Host "MODEL DOWNLOAD / VERIFICATION COMPLETE"
Write-Host "=============================================================="
