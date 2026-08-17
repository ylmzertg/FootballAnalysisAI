param([string]$EnginesRoot="")
$ErrorActionPreference="Stop"
$ProjectRoot=(Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if ([string]::IsNullOrWhiteSpace($EnginesRoot)) {$EnginesRoot=Join-Path (Split-Path $ProjectRoot -Parent) "CalibrationEngines"}
$PnL=Join-Path $EnginesRoot "PnLCalib"
$TV=Join-Path $EnginesRoot "tvcalib"

function GetFile($Uri,$Out,$Label){
  if(Test-Path $Out){$l=(Get-Item $Out).Length;if($l -gt 0){Write-Host "[OK] $Label ($l bytes)";return}}
  New-Item -ItemType Directory -Force -Path (Split-Path $Out -Parent)|Out-Null
  Write-Host "[DOWNLOAD] $Label"
  Invoke-WebRequest -Uri $Uri -OutFile $Out
}

GetFile "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp" (Join-Path $PnL "weights\SV_kp") "PnLCalib SV_kp"
GetFile "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines" (Join-Path $PnL "weights\SV_lines") "PnLCalib SV_lines"
GetFile "https://tib.eu/cloud/s/x68XnTcZmsY4Jpg/download/train_59.pt" (Join-Path $TV "data\segment_localization\train_59.pt") "TVCalib train_59.pt"
