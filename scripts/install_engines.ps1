param([string]$EnginesRoot="",[switch]$SkipDownloads)
$ErrorActionPreference="Stop"
$ProjectRoot=(Resolve-Path (Join-Path $PSScriptRoot "..")).Path
if([string]::IsNullOrWhiteSpace($EnginesRoot)){$EnginesRoot=Join-Path (Split-Path $ProjectRoot -Parent) "CalibrationEngines"}
New-Item -ItemType Directory -Force -Path $EnginesRoot|Out-Null

function Py($v){
  $compact=$v.Replace(".","")
  $p=Join-Path $env:LOCALAPPDATA "Programs\Python\Python$compact\python.exe"
  if(Test-Path $p){return $p}
  $x=& py "-$v" -c "import sys;print(sys.executable)" 2>$null
  if($LASTEXITCODE -eq 0 -and $x){return $x.Trim()}
  throw "Python $v not found."
}
function Repo($path,$url,[switch]$Sub){
  if(Test-Path (Join-Path $path ".git")){if($Sub){git -C $path submodule update --init --recursive};return}
  if($Sub){git clone --recurse-submodules $url $path}else{git clone $url $path}
  if($LASTEXITCODE -ne 0){throw "git clone failed: $url"}
}
function Venv($py,$root){
  $venvRoot=Join-Path $root ".venv"
  $v=Join-Path $venvRoot "Scripts\python.exe"

  $expected=& $py -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"
  if($LASTEXITCODE -ne 0 -or -not $expected){
    throw "Could not determine Python version: $py"
  }
  $expected=$expected.Trim()

  $healthy=$false

  if(Test-Path $v){
    try{
      $actual=& $v -c "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
      $pythonOk=($LASTEXITCODE -eq 0 -and $actual -and $actual.Trim() -eq $expected)

      & $v -m pip --version *> $null
      $pipOk=($LASTEXITCODE -eq 0)

      $healthy=($pythonOk -and $pipOk)
    }
    catch{
      $healthy=$false
    }
  }

  if(-not $healthy){
    if(Test-Path $venvRoot){
      Write-Host "[REBUILD] invalid virtual environment: $venvRoot"
      Remove-Item $venvRoot -Recurse -Force
    }
    else{
      Write-Host "[CREATE] virtual environment: $venvRoot"
    }

    & $py -m venv $venvRoot

    if($LASTEXITCODE -ne 0 -or -not(Test-Path $v)){
      throw "Virtual environment creation failed: $venvRoot"
    }

    & $v -m pip --version *> $null
    if($LASTEXITCODE -ne 0){
      throw "pip is unavailable after venv creation: $venvRoot"
    }
  }
  else{
    Write-Host "[OK] virtual environment: $venvRoot | Python $expected | pip OK"
  }

  return $v
}

$py310=Py "3.10"; $py39=Py "3.9"
$pnl=Join-Path $EnginesRoot "PnLCalib"; $tv=Join-Path $EnginesRoot "tvcalib"
Repo $pnl "https://github.com/mguti97/PnLCalib.git"
Repo $tv "https://github.com/MM4SPA/tvcalib.git" -Sub

$pnlpy=Venv $py310 $pnl
& $pnlpy -m pip install --upgrade pip
& $pnlpy -m pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
& $pnlpy -m pip install -r (Join-Path $ProjectRoot "requirements\pnlcalib-windows-cpu.txt")
if($LASTEXITCODE -ne 0){throw "PnLCalib install failed"}

$tvpy=Venv $py39 $tv
& $tvpy -m pip install pip==24.0 setuptools wheel
& $tvpy -m pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cpu --trusted-host download.pytorch.org --trusted-host download-r2.pytorch.org
& $tvpy -m pip install -r (Join-Path $ProjectRoot "requirements\tvcalib-windows-cpu.txt")
if($LASTEXITCODE -ne 0){throw "TVCalib dependency install failed"}

# Critical: remove all competing OpenCV distributions and stale cv2 files.
& $tvpy -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless
$site=& $tvpy -c "import site;print(site.getsitepackages()[0])"
Remove-Item (Join-Path $site "cv2") -Recurse -Force -ErrorAction SilentlyContinue
& $tvpy -m pip install --no-cache-dir --force-reinstall numpy==1.19.5 opencv-python-headless==4.5.5.62
if($LASTEXITCODE -ne 0){throw "TVCalib OpenCV repair failed"}

if(-not $SkipDownloads){& (Join-Path $ProjectRoot "scripts\download_models.ps1") -EnginesRoot $EnginesRoot}
