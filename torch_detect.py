import re
import shutil
import subprocess
import sys


TORCH_VERSION = "2.9.0"
TORCHVISION_VERSION = "0.24.0"
TORCH_INDEX = "cu130"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu130"
MIN_SM = 75
MAX_SM = 120
MIN_DRIVER_CUDA = (13, 0)


class DetectionError(RuntimeError):
    pass


def _run_nvidia_smi(*args):
    executable = shutil.which("nvidia-smi")
    if executable is None:
        raise DetectionError("nvidia-smi was not found. Install or update the NVIDIA driver.")

    try:
        return subprocess.check_output(
            [executable, *args],
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise DetectionError(f"nvidia-smi failed: {exc}") from exc


def _version_tuple(value):
    parts = re.findall(r"\d+", value)
    if len(parts) < 2:
        raise DetectionError(f"Could not parse version: {value}")
    return int(parts[0]), int(parts[1])


def sm_from_cc(value):
    text = str(value).strip()
    match = re.fullmatch(r"(\d+)\.(\d+)", text)
    if not match:
        raise DetectionError(f"Could not parse compute capability: {value}")
    return int(match.group(1)) * 10 + int(match.group(2))


def get_gpu_info():
    output = _run_nvidia_smi(
        "--query-gpu=index,name,compute_cap,driver_version",
        "--format=csv,noheader,nounits",
    )
    first_line = output.splitlines()[0]
    fields = [field.strip() for field in first_line.split(",", 3)]
    if len(fields) != 4:
        raise DetectionError(f"Unexpected nvidia-smi GPU output: {first_line}")

    index, name, compute_capability, driver_version = fields
    return {
        "gpu_index": int(index),
        "gpu_name": name,
        "compute_capability": compute_capability,
        "sm": sm_from_cc(compute_capability),
        "driver_version": driver_version,
    }


def get_driver_cuda_version():
    output = _run_nvidia_smi()
    match = re.search(r"CUDA (?:UMD )?Version:\s*([\d.]+)", output)
    if not match:
        raise DetectionError("nvidia-smi did not report the supported CUDA version.")
    return match.group(1)


def choose_torch(sm):
    if sm < MIN_SM or sm > MAX_SM:
        raise DetectionError(
            f"Unsupported GPU architecture SM {sm}. "
            f"This build supports SM {MIN_SM} through SM {MAX_SM}."
        )

    return {
        "version": f"{TORCH_VERSION}+{TORCH_INDEX}",
        "index": TORCH_INDEX,
        "min_sm": MIN_SM,
        "max_sm": MAX_SM,
        "compile_supported": sm >= 80,
        "pip_args": [
            "install",
            f"torch=={TORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            "--index-url",
            TORCH_INDEX_URL,
        ],
    }


def detect_environment():
    gpu = get_gpu_info()
    cuda_version = get_driver_cuda_version()
    if _version_tuple(cuda_version) < MIN_DRIVER_CUDA:
        raise DetectionError(
            f"The NVIDIA driver exposes CUDA {cuda_version}; CUDA 13.0 or newer is required. "
            "Install the latest NVIDIA driver."
        )

    choice = choose_torch(gpu["sm"])
    return {**gpu, "driver_cuda": cuda_version, "torch": choice}


def main():
    try:
        result = detect_environment()
    except DetectionError as exc:
        print(f"Environment detection failed: {exc}")
        return None

    print(f"GPU: {result['gpu_name']}")
    print(
        f"Compute capability: {result['compute_capability']} "
        f"(SM {result['sm']})"
    )
    print(f"NVIDIA driver: {result['driver_version']}")
    print(f"Driver CUDA API: {result['driver_cuda']}")
    print(f"PyTorch build: {result['torch']['version']}")
    if result["torch"]["compile_supported"]:
        print("torch.compile: enabled")
    else:
        print("torch.compile: disabled on SM 7.5; CUDA eager inference remains enabled")
    return result


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
