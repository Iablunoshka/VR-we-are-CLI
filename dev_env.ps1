$ErrorActionPreference = "Stop"

$projectRoot = "F:\VR-we-are-CLI_GPU"
$projectVenv = "D:\Programs\Pycharm_projects\CLI\venv"
$ffmpegBin = "C:\Users\Suprim\AppData\Local\Microsoft\WinGet\Links"
$cudaRoot = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
$pynvcCudaRuntime = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin"
$vsDevCmd = "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"

$requiredPaths = @(
    (Join-Path $projectVenv "Scripts\python.exe"),
    (Join-Path $ffmpegBin "ffmpeg.exe"),
    (Join-Path $cudaRoot "bin\nvcc.exe"),
    (Join-Path $pynvcCudaRuntime "cudart64_12.dll"),
    $vsDevCmd
)

foreach ($requiredPath in $requiredPaths) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required development tool not found: $requiredPath"
    }
}

# VsDevCmd is a batch file. Import the environment it creates into this
# PowerShell process so cl.exe, INCLUDE, LIB and Windows SDK paths persist.
$devEnvironment = & $env:ComSpec /d /s /c (
    'call "{0}" -no_logo -arch=x64 -host_arch=x64 >nul && set' -f $vsDevCmd
)

if ($LASTEXITCODE -ne 0) {
    throw "VsDevCmd failed with exit code $LASTEXITCODE"
}

foreach ($line in $devEnvironment) {
    if ($line -match '^([^=]+)=(.*)$') {
        Set-Item -Path ("Env:" + $matches[1]) -Value $matches[2]
    }
}

$env:PROJECT_ROOT = $projectRoot
$env:PROJECT_VENV = $projectVenv
$env:VIRTUAL_ENV = $projectVenv
$env:FFMPEG_BIN = $ffmpegBin
$env:CUDA_ROOT = $cudaRoot
$env:CUDA_PATH = $cudaRoot
$env:PYNVC_CUDA_RUNTIME = $pynvcCudaRuntime
$env:PYTHONHOME = $null
$env:Path = @(
    (Join-Path $projectVenv "Scripts"),
    $ffmpegBin,
    (Join-Path $cudaRoot "bin\x64"),
    (Join-Path $cudaRoot "bin"),
    $pynvcCudaRuntime,
    $env:Path
) -join ";"

Set-Location -LiteralPath $projectRoot

Write-Host "Development environment ready: $projectRoot"
Write-Host "Python: $(Join-Path $projectVenv 'Scripts\python.exe')"
Write-Host "FFmpeg: $(Join-Path $ffmpegBin 'ffmpeg.exe')"
Write-Host "MSVC/CUDA: x64 / CUDA 13.3"
Write-Host "PyNvVideoCodec runtime: CUDA 12.1"
