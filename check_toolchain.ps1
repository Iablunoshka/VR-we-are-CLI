$ErrorActionPreference = "Stop"

. (Join-Path $PSScriptRoot "dev_env.ps1")

Write-Host "`n--- Executables ---"
Get-Command python, ffmpeg, cl, cmake, nvcc | Select-Object Name, Source

Write-Host "`n--- Versions ---"
python --version
ffmpeg -hide_banner -version | Select-Object -First 1
cmd.exe /d /c "cl 2>&1" | Select-Object -First 1
cmake --version | Select-Object -First 1
nvcc --version | Select-Object -Last 1

Write-Host "`n--- Python GPU stack ---"
python -c "import sys, torch, cupy, PyNvVideoCodec as nvc; print('executable:', sys.executable); print('torch:', torch.__version__); print('torch CUDA:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0)); print('cupy:', cupy.__version__); print('PyNvVideoCodec:', nvc.__version__)"

