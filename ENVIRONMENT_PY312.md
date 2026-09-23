# GPU runtime environment (Python 3.12)

The tested runtime uses Python 3.12 x64 and supports NVIDIA GPUs from Turing
(SM 7.5) through consumer Blackwell (SM 12.0).

## System requirements

- Windows x64 or Linux x86_64;
- Python 3.12 x64;
- recent NVIDIA display driver exposing CUDA Driver API 13.0 or newer;
- FFmpeg and FFprobe available in `PATH`;
- the matching production PyNvVideoCodec 2.2.0 wheel in `wheels/`.

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
python setup_env.py --check-only
python setup_env.py
```

The installer selects and verifies the Windows or Linux production wheel from
`wheels/`. `--pynv-wheel PATH` is available when the same production artifact
is stored outside that directory.

## Pinned CUDA stack

- PyTorch 2.9.0 + CUDA 13.0;
- TorchVision 0.24.0 + CUDA 13.0;
- CuPy 14.1.1; Linux deliberately does not install CuPy's `ctk` extra, because
  that would replace PyTorch's pinned CUDA 13.0 component wheels;
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
