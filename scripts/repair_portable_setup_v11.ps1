param()

$ErrorActionPreference = "Stop"
$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$MainPy = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (-not (Test-Path $MainPy)) {
    throw "Main .venv not found: $MainPy"
}

Write-Host "[FIX] Pinning Supervision 0.25.0 for the project's ByteTrack API..."
& $MainPy -m pip install --upgrade --force-reinstall "supervision==0.25.0"
if ($LASTEXITCODE -ne 0) {
    throw "Supervision installation failed."
}

Write-Host ""
Write-Host "[CHECK]"
& $MainPy (Join-Path $ProjectRoot "scripts\health_check.py")
exit $LASTEXITCODE
