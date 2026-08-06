# Windows GPU environment (Python 3.12)

This file records the tested environment for the RTX 5090 development build.
Do not delete the working Python 3.10 venv until the Python 3.12 environment
passes the smoke test and a complete video conversion.

## Install manually

Install these system components before creating the venv:

1. Python 3.12 x64 from python.org. Enable the Python launcher (`py.exe`).
2. NVIDIA display driver with NVDEC/NVENC support.
3. NVIDIA CUDA Toolkit 13.3. This remains the primary development toolkit and
   supplies `nvcc`, NVRTC and the CUDA 13 libraries used by CuPy.
4. NVIDIA CUDA Toolkit 12.1 runtime side-by-side with 13.3. PyNvVideoCodec
   2.1.0 requires `cudart64_12.dll`. The application registers its DLL folder
   automatically; `CUDA_PATH` must continue to point to CUDA 13.3.
5. Visual Studio Build Tools 2026 with:
   - MSVC x64/x86 C++ build tools;
   - Windows 11 SDK;
   - C++ CMake tools for Windows.
6. CMake 4.3 or newer.
7. FFmpeg 8.1 full build with `nvdec`, `nvenc`, `cuvid` and `ffnvcodec`.

The paths currently tested on this workstation are recorded in `AGENTS.md`.

## Create a Python 3.12 venv

Create a new directory instead of overwriting the working Python 3.10 venv:

```bat
py -3.12 -m venv D:\Programs\Pycharm_projects\CLI\venv312
call D:\Programs\Pycharm_projects\CLI\venv312\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
```

Confirm that the selected interpreter is correct:

```bat
python --version
python -c "import sys; print(sys.executable)"
```

## Install the CUDA build of PyTorch manually

PyTorch CUDA wheels use a dedicated package index and are therefore not pinned
inside the general `requirements.txt` file:

```bat
python -m pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu130
```

Install the remaining pinned application and native-extension dependencies:

```bat
cd /d F:\VR-we-are-CLI_GPU
python -m pip install -r requirements.txt
python -m pip check
```

`triton-windows==3.5.1.post24` is pinned because the pipeline uses
`torch.compile()` on Windows. `pybind11==3.0.4` is included for the planned
native NVIDIA Video Codec SDK bridge.

## Required DLL relationship

The environment intentionally contains two CUDA generations:

- CUDA 13.3: primary compiler/runtime toolchain for CuPy and native builds;
- CUDA 12.1: compatibility runtime for PyNvVideoCodec's `cudart64_12.dll`.

Do not replace `cudart64_12.dll` with a renamed CUDA 13 DLL and do not copy the
DLL into the repository. `pipeline_core.py` registers the real CUDA 12.1 DLL
directory with `os.add_dll_directory()` before importing PyNvVideoCodec.

## Smoke test

After adapting `PROJECT_VENV` in `dev_env.cmd` and `dev_env.ps1` to `venv312`,
run:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File F:\VR-we-are-CLI_GPU\check_toolchain.ps1
```

Expected key results:

- Python 3.12.x;
- Torch 2.9.0+cu130;
- CUDA available: True;
- NVIDIA GeForce RTX 5090;
- CuPy 14.1.1;
- PyNvVideoCodec 2.1.0;
- FFmpeg, cl.exe, CMake and nvcc all resolved.

Then run a short conversion before replacing the old venv:

```bat
python main.py -i F:\test_media\4k_10kframes.mp4 -o F:\test_media\py312-smoke.mp4 --debug -m depth-anything/Depth-Anything-V2-Small-hf -pre 1 --processors 1 -b 19 -c hevc_nvenc
```

