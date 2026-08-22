param(
    [string]$EnginesRoot = "",
    [ValidateSet("auto","cpu","cuda")]
    [string]$Profile = "auto",
    [switch]$SkipEngines,
    [switch]$SkipDownloads
)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot

if ([string]::IsNullOrWhiteSpace($EnginesRoot)) {
    $EnginesRoot = Join-Path (Split-Path $ProjectRoot -Parent) "CalibrationEngines"
}

$EnginesRoot = [System.IO.Path]::GetFullPath($EnginesRoot)

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is not installed."
}

function Get-Python310 {
    $candidate = Join-Path $env:LOCALAPPDATA "Programs\Python\Python310\python.exe"

    if (Test-Path $candidate) {
        return $candidate
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        $resolved = & py -3.10 -c "import sys; print(sys.executable)" 2>$null

        if ($LASTEXITCODE -eq 0 -and $resolved) {
            return $resolved.Trim()
        }
    }

    if (Get-Command winget -ErrorAction SilentlyContinue) {
        Write-Host "[INSTALL] Python 3.10"

        & winget install `
            --id Python.Python.3.10 `
            -e `
            --accept-package-agreements `
            --accept-source-agreements

        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Python 3.10 is required."
}

function Test-NvidiaGpu {
    if (-not (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
        return $false
    }

    & nvidia-smi --query-gpu=name --format=csv,noheader 2>$null | Out-Null

    return ($LASTEXITCODE -eq 0)
}

function Ensure-Venv {
    param(
        [string]$BasePython,
        [string]$VenvRoot,
        [string]$ExpectedVersion
    )

    $pythonExe = Join-Path $VenvRoot "Scripts\python.exe"
    $healthy = $false

    if (Test-Path $pythonExe) {
        try {
            $actual = & $pythonExe -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            $pythonOk = (
                $LASTEXITCODE -eq 0 -and
                $actual -and
                $actual.Trim() -eq $ExpectedVersion
            )

            & $pythonExe -m pip --version *> $null
            $pipOk = ($LASTEXITCODE -eq 0)

            $healthy = ($pythonOk -and $pipOk)
        }
        catch {
            $healthy = $false
        }
    }

    if (-not $healthy) {
        if (Test-Path $VenvRoot) {
            Write-Host "[REBUILD] invalid main virtual environment: $VenvRoot"
            Remove-Item $VenvRoot -Recurse -Force
        }
        else {
            Write-Host "[CREATE] main virtual environment: $VenvRoot"
        }

        & $BasePython -m venv $VenvRoot

        if ($LASTEXITCODE -ne 0 -or -not (Test-Path $pythonExe)) {
            throw "Main virtual environment creation failed."
        }

        & $pythonExe -m pip --version *> $null

        if ($LASTEXITCODE -ne 0) {
            throw "pip is unavailable in main virtual environment."
        }
    }
    else {
        Write-Host "[OK] main virtual environment | Python $ExpectedVersion | pip OK"
    }

    return $pythonExe
}

function Get-TorchState {
    param([string]$PythonExe)

    try {
        $lines = @(
            & $PythonExe -c "import torch,torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.version.cuda or 'NONE'); print('1' if torch.cuda.is_available() else '0')" 2>$null
        )

        if ($LASTEXITCODE -ne 0 -or $lines.Count -lt 4) {
            return $null
        }

        return [PSCustomObject]@{
            Torch       = $lines[0].Trim()
            TorchVision = $lines[1].Trim()
            CudaBuild   = $lines[2].Trim()
            CudaReady   = ($lines[3].Trim() -eq "1")
        }
    }
    catch {
        return $null
    }
}

function Install-CpuTorch {
    param([string]$PythonExe)

    Write-Host "[INSTALL] PyTorch CPU profile"

    & $PythonExe -m pip install `
        "torch==1.13.1+cpu" `
        "torchvision==0.14.1+cpu" `
        --extra-index-url "https://download.pytorch.org/whl/cpu" `
        --trusted-host download.pytorch.org `
        --trusted-host download-r2.pytorch.org

    if ($LASTEXITCODE -ne 0) {
        throw "CPU PyTorch install failed."
    }
}

function Install-CudaTorch {
    param([string]$PythonExe)

    Write-Host "[INSTALL] PyTorch CUDA 11.6 profile"

    & $PythonExe -m pip install typing_extensions

    if ($LASTEXITCODE -ne 0) {
        throw "typing_extensions install failed."
    }

    $wheelRoot = Join-Path $env:TEMP "FootballAnalysisAI\wheels"
    New-Item -ItemType Directory -Force -Path $wheelRoot | Out-Null

    $torchWheel = Join-Path $wheelRoot "torch-1.13.1+cu116-cp310-cp310-win_amd64.whl"
    $visionWheel = Join-Path $wheelRoot "torchvision-0.14.1+cu116-cp310-cp310-win_amd64.whl"

    $torchUrl = "https://download.pytorch.org/whl/cu116/torch-1.13.1%2Bcu116-cp310-cp310-win_amd64.whl"
    $visionUrl = "https://download.pytorch.org/whl/cu116/torchvision-0.14.1%2Bcu116-cp310-cp310-win_amd64.whl"

    $cacheInstalled = $false

    if ((Test-Path $torchWheel) -and (Test-Path $visionWheel)) {
        Write-Host "[CACHE] Testing existing CUDA wheels"

        & $PythonExe -m pip install `
            --force-reinstall `
            --no-deps `
            $torchWheel `
            $visionWheel

        if ($LASTEXITCODE -eq 0) {
            $cacheInstalled = $true
        }
        else {
            Write-Host "[WARN] Invalid cached CUDA wheels. Re-downloading."
            Remove-Item $torchWheel,$visionWheel -Force -ErrorAction SilentlyContinue
        }
    }

    if (-not $cacheInstalled) {
        if (Get-Command curl.exe -ErrorAction SilentlyContinue) {
            Write-Host "[DOWNLOAD] CUDA torch wheel"

            & curl.exe `
                -L `
                --retry 20 `
                --retry-delay 5 `
                --retry-all-errors `
                -C - `
                -o $torchWheel `
                $torchUrl

            if ($LASTEXITCODE -ne 0) {
                throw "CUDA torch wheel download failed."
            }

            Write-Host "[DOWNLOAD] CUDA torchvision wheel"

            & curl.exe `
                -L `
                --retry 20 `
                --retry-delay 5 `
                --retry-all-errors `
                -C - `
                -o $visionWheel `
                $visionUrl

            if ($LASTEXITCODE -ne 0) {
                throw "CUDA torchvision wheel download failed."
            }

            & $PythonExe -m pip install `
                --force-reinstall `
                --no-deps `
                $torchWheel `
                $visionWheel
        }
        else {
            & $PythonExe -m pip install `
                "torch==1.13.1+cu116" `
                "torchvision==0.14.1+cu116" `
                --extra-index-url "https://download.pytorch.org/whl/cu116" `
                --trusted-host download.pytorch.org `
                --trusted-host download-r2.pytorch.org
        }

        if ($LASTEXITCODE -ne 0) {
            throw "CUDA PyTorch install failed."
        }
    }
}

$ResolvedProfile = $Profile
$NvidiaAvailable = Test-NvidiaGpu

if ($Profile -eq "auto") {
    if ($NvidiaAvailable) {
        $ResolvedProfile = "cuda"
    }
    else {
        $ResolvedProfile = "cpu"
    }
}

if ($ResolvedProfile -eq "cuda" -and -not $NvidiaAvailable) {
    throw "CUDA profile requested but no working NVIDIA GPU/driver was detected."
}

Write-Host "=============================================================="
Write-Host "FootballAnalysisAI - Portable Windows Setup V2"
Write-Host "Project : $ProjectRoot"
Write-Host "Engines : $EnginesRoot"
Write-Host "Profile : $ResolvedProfile"
Write-Host "NVIDIA  : $NvidiaAvailable"
Write-Host "=============================================================="

$Py310 = Get-Python310

$MainVenv = Join-Path $ProjectRoot ".venv"
$MainPy = Ensure-Venv `
    -BasePython $Py310 `
    -VenvRoot $MainVenv `
    -ExpectedVersion "3.10"

& $MainPy -m pip install --upgrade pip

if ($LASTEXITCODE -ne 0) {
    throw "Main pip upgrade failed."
}

$torchState = Get-TorchState $MainPy

if ($ResolvedProfile -eq "cuda") {
    $cudaCorrect = (
        $null -ne $torchState -and
        $torchState.Torch -eq "1.13.1+cu116" -and
        $torchState.TorchVision -eq "0.14.1+cu116" -and
        $torchState.CudaReady
    )

    if ($cudaCorrect) {
        Write-Host "[OK] CUDA PyTorch already installed | torch=$($torchState.Torch)"
    }
    else {
        Install-CudaTorch $MainPy

        $torchState = Get-TorchState $MainPy

        if (
            $null -eq $torchState -or
            $torchState.Torch -ne "1.13.1+cu116" -or
            -not $torchState.CudaReady
        ) {
            if ($Profile -eq "auto") {
                Write-Host "[WARN] CUDA verification failed. Falling back to CPU profile."
                Install-CpuTorch $MainPy
                $ResolvedProfile = "cpu"
            }
            else {
                throw "CUDA PyTorch installed but CUDA verification failed."
            }
        }
    }
}
else {
    $cpuCorrect = (
        $null -ne $torchState -and
        $torchState.Torch -eq "1.13.1+cpu" -and
        $torchState.TorchVision -eq "0.14.1+cpu"
    )

    if ($cpuCorrect) {
        Write-Host "[OK] CPU PyTorch already installed | torch=$($torchState.Torch)"
    }
    else {
        Install-CpuTorch $MainPy
    }
}

& $MainPy -m pip install -r (Join-Path $ProjectRoot "requirements\main-windows-cpu.txt")

if ($LASTEXITCODE -ne 0) {
    throw "Main requirements install failed."
}

if (-not $SkipEngines) {
    $py39 = Join-Path $env:LOCALAPPDATA "Programs\Python\Python39\python.exe"

    if (-not (Test-Path $py39) -and (Get-Command winget -ErrorAction SilentlyContinue)) {
        Write-Host "[INSTALL] Python 3.9"

        & winget install `
            --id Python.Python.3.9 `
            -e `
            --accept-package-agreements `
            --accept-source-agreements
    }

    & (Join-Path $ProjectRoot "scripts\install_engines.ps1") `
        -EnginesRoot $EnginesRoot `
        -SkipDownloads:$SkipDownloads

    if ($LASTEXITCODE -ne 0) {
        throw "Calibration engine setup failed."
    }
}

$config = Join-Path $ProjectRoot "configs\local.yaml"

if (-not (Test-Path $config)) {
    Copy-Item `
        (Join-Path $ProjectRoot "configs\local.example.yaml") `
        $config

    Write-Host "[CREATE] configs\local.yaml"
}

Write-Host "[CHECK] installation"

& $MainPy `
    (Join-Path $ProjectRoot "scripts\health_check.py") `
    --engines-root $EnginesRoot

$exit = $LASTEXITCODE

if ($exit -eq 0) {
    Write-Host "=============================================================="
    Write-Host "READY - Portable Windows Setup V2 installed."
    Write-Host "Profile : $ResolvedProfile"
    Write-Host "Activate: .\.venv\Scripts\Activate.ps1"
    Write-Host "=============================================================="
}
else {
    Write-Host "Setup finished with health-check failures."
    exit $exit
}
