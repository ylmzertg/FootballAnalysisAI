# Portable Windows Setup v1.2

Fresh Windows machine:

```powershell
git clone https://github.com/ylmzertg/FootballAnalysisAI.git
cd FootballAnalysisAI
Set-ExecutionPolicy -Scope Process Bypass
.\setup_windows.ps1
```

Validated CPU baseline:
- Main: Python 3.10.11 / torch 1.13.1+cpu / NumPy 1.26.4 / OpenCV 4.11 / supervision 0.25.0
- PnLCalib: Python 3.10 / torch 2.3.1+cpu
- TVCalib: Python 3.9 / torch 1.11.0+cpu / torchvision 0.12.0+cpu / NumPy 1.19.5 / OpenCV headless 4.5.5.62

TVCalib setup explicitly removes all competing OpenCV packages and stale `cv2`
directories before installing the legacy headless build.
