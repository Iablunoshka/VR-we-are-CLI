# Local development environment

This project uses tools installed outside the repository. Do not infer that the
environment is broken when a sandboxed process cannot execute them.

Canonical paths:

- Project Python: `D:\Programs\Pycharm_projects\CLI\venv\Scripts\python.exe`
- FFmpeg: `C:\Users\Suprim\AppData\Local\Microsoft\WinGet\Links\ffmpeg.exe`
- Visual Studio environment: `C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat`
- CUDA: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3`
- PyNvVideoCodec CUDA runtime: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\bin`
- CMake: `C:\Program Files\CMake\bin\cmake.exe`

Before development commands:

- PowerShell: dot-source `F:\VR-we-are-CLI_GPU\dev_env.ps1`
- cmd.exe: call `F:\VR-we-are-CLI_GPU\dev_env.cmd`
- Full validation: `powershell -ExecutionPolicy Bypass -File F:\VR-we-are-CLI_GPU\check_toolchain.ps1`

Codex command execution is sandboxed separately from the user's interactive
terminal. Commands that execute the project Python, FFmpeg, MSVC, CUDA, or their
child processes may require a scoped escalation even though the project and
tools are correctly installed. Request that escalation instead of changing or
recreating the venv.

Keep both CUDA DLL directories in PATH, in this order: `CUDA\v13.3\bin\x64`
and `CUDA\v13.3\bin`. PyNvVideoCodec 2.1 requires DLLs from the `bin\x64`
directory even though its own environment bootstrap only probes `bin`.
The venv's `sitecustomize.py` registers the CUDA 12.1 DLL directory so
PyNvVideoCodec can resolve `cudart64_12.dll`; do not change the primary
`CUDA_PATH` away from CUDA 13.3.
