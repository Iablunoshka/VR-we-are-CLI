@echo off

set "PROJECT_ROOT=F:\VR-we-are-CLI_GPU"
set "PROJECT_VENV=D:\Programs\Pycharm_projects\CLI\venv"
set "FFMPEG_BIN=C:\Users\Suprim\AppData\Local\Microsoft\WinGet\Links"
set "CUDA_ROOT=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3"
set "VSDEVCMD=C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"

if not exist "%PROJECT_VENV%\Scripts\python.exe" goto :missing_python
if not exist "%FFMPEG_BIN%\ffmpeg.exe" goto :missing_ffmpeg
if not exist "%VSDEVCMD%" goto :missing_vs

call "%VSDEVCMD%" -no_logo -arch=x64 -host_arch=x64
if errorlevel 1 exit /b %errorlevel%

set "VIRTUAL_ENV=%PROJECT_VENV%"
set "PATH=%PROJECT_VENV%\Scripts;%FFMPEG_BIN%;%CUDA_ROOT%\bin\x64;%CUDA_ROOT%\bin;%PYNVC_CUDA_RUNTIME%;%PATH%"
set "PYTHONHOME="
cd /d "%PROJECT_ROOT%"

echo Development environment ready: %PROJECT_ROOT%
echo Python: %PROJECT_VENV%\Scripts\python.exe
echo FFmpeg: %FFMPEG_BIN%\ffmpeg.exe
echo MSVC/CUDA: x64 / CUDA 13.3
goto :eof

:missing_python
echo ERROR: project Python not found: %PROJECT_VENV%\Scripts\python.exe
exit /b 1

:missing_ffmpeg
echo ERROR: FFmpeg not found: %FFMPEG_BIN%\ffmpeg.exe
exit /b 1

:missing_vs
echo ERROR: Visual Studio developer environment not found: %VSDEVCMD%
exit /b 1
