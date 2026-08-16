# GPU runtime environment (Python 3.12)

The tested runtime uses Python 3.12 x64 and supports NVIDIA GPUs from Turing
(SM 7.5) through consumer Blackwell (SM 12.0).

## System requirements

- Windows x64 or Linux x86_64;
- Python 3.12 x64;
- recent NVIDIA display driver exposing CUDA Driver API 13.0 or newer;
- FFmpeg and FFprobe available in `PATH`;
- a modified PyNvVideoCodec 2.2.0 wheel for Python 3.12 and the host platform.

CUDA Toolkit, Visual Studio, CMake and Video Codec SDK are build dependencies
only. They are not required to run the packaged application.

## Create the environment

Windows:

```bat
py -3.12 -m venv venv
venv\Scripts\activate
python setup_env.py --check-only
python setup_env.py
```

Linux:

```bash
python3.12 -m venv venv
source venv/bin/activate
python setup_env.py --check-only --pynv-wheel /path/to/custom-linux-wheel.whl
python setup_env.py --pynv-wheel /path/to/custom-linux-wheel.whl
```

The Windows alpha wheel is discovered automatically in `wheels/`. Linux support
is connected to the same setup flow, but its modified wheel must be built and
provided separately.

## Pinned CUDA stack

- PyTorch 2.9.0 + CUDA 13.0;
- TorchVision 0.24.0 + CUDA 13.0;
- CuPy 14.1.1 with CUDA component wheels;
- Triton Windows 3.5.1.post24 on Windows;
- modified PyNvVideoCodec 2.2.0.

Torch, CuPy and PyNvVideoCodec cover SM 7.5 through SM 12.0. Triton 3.5 requires
SM 8.0 or newer, so `torch.compile` is disabled on SM 7.5 while CUDA eager
inference remains enabled.

## Diagnostics

Print installation commands without modifying the environment:

```bash
python setup_env.py --dry-run
```

After installation, setup verifies Torch CUDA execution, Torch compilation on
SM 8.0+, a CuPy CUDA operation, PyNvVideoCodec import and `pip check`.
