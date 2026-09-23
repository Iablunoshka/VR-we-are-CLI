import argparse
import hashlib
import importlib
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch_detect


CUPY_WINDOWS_REQUIREMENT = "cupy-cuda13x[ctk]==14.1.1"
CUPY_LINUX_REQUIREMENT = "cupy-cuda13x==14.1.1"
TRITON_WINDOWS_REQUIREMENT = "triton-windows==3.5.1.post24"
PYNV_VERSION = "2.2.0"
PYNV_WHEEL_SHA256 = {
    "Windows": "74112459fe31eeeb2373d124bac29da1eb0900ed953e0e02b71ab9fc66dd897f",
    "Linux": "2e3f1252ef3d3d8c5eb070430e0d3238cfc8438bf8e0f11f5eaf427eb431d158",
}
SUPPORTED_SYSTEMS = {"Windows", "Linux"}
REQUIRED_PROJECT_FILES = (
    "main.py",
    "presets.json",
    "requirements.txt",
    "torch_detect.py",
    "video_mux.py",
)


class SetupError(RuntimeError):
    pass


def run_command(command, *, dry_run=False):
    command = [str(part) for part in command]
    print("  " + subprocess.list2cmdline(command))
    if dry_run:
        return
    try:
        subprocess.run(command, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupError(f"Command failed: {exc}") from exc


def pip_install(arguments, *, dry_run=False):
    run_command([sys.executable, "-m", "pip", *arguments], dry_run=dry_run)


def check_python():
    if sys.version_info[:2] != (3, 12):
        raise SetupError(
            f"Python 3.12 x64 is required; current version is {platform.python_version()}."
        )
    if sys.maxsize <= 2**32:
        raise SetupError("A 64-bit Python installation is required.")
    if sys.prefix == sys.base_prefix:
        raise SetupError("Run setup_env.py from an activated virtual environment.")


def check_platform():
    system = platform.system()
    if system not in SUPPORTED_SYSTEMS:
        raise SetupError(f"Unsupported operating system: {system}")

    machine = platform.machine().lower()
    if machine not in {"amd64", "x86_64"}:
        raise SetupError(f"Unsupported CPU architecture: {platform.machine()}")
    return system


def check_project_files(script_dir):
    missing = [name for name in REQUIRED_PROJECT_FILES if not (script_dir / name).is_file()]
    if missing:
        raise SetupError("Missing project files: " + ", ".join(missing))


def check_program(name, version_argument="-version"):
    executable = shutil.which(name)
    if executable is None:
        raise SetupError(f"{name} was not found in PATH.")

    try:
        result = subprocess.run(
            [executable, version_argument],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SetupError(f"{name} exists but could not be executed: {exc}") from exc

    first_line = (result.stdout or result.stderr).splitlines()[0]
    print(f"{name}: {first_line}")
    return Path(executable)


def wheel_matches_platform(path, system):
    name = path.name.lower()
    if f"pynvvideocodec-{PYNV_VERSION}-cp312-cp312" not in name:
        return False
    if system == "Windows":
        return name.endswith("-win_amd64.whl")
    return "linux_x86_64.whl" in name



def verify_pynv_wheel(wheel, system):
    digest = hashlib.sha256()
    with wheel.open("rb") as wheel_file:
        for chunk in iter(lambda: wheel_file.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != PYNV_WHEEL_SHA256[system]:
        raise SetupError(
            f"The {system} PyNvVideoCodec wheel does not match the tested production build."
        )


def find_pynv_wheel(script_dir, system, explicit_path=None):
    if explicit_path:
        wheel = Path(explicit_path).expanduser().resolve()
        if not wheel.is_file():
            raise SetupError(f"PyNvVideoCodec wheel not found: {wheel}")
        if not wheel_matches_platform(wheel, system):
            raise SetupError(
                f"PyNvVideoCodec wheel is not compatible with Python 3.12/{system}: "
                f"{wheel.name}"
            )
        return wheel

    candidates = []
    for directory in (script_dir / "wheels", script_dir / "dist_production", script_dir):
        if directory.is_dir():
            candidates.extend(directory.glob("pynvvideocodec-*.whl"))

    compatible = sorted(path for path in candidates if wheel_matches_platform(path, system))
    if not compatible:
        platform_hint = "win_amd64" if system == "Windows" else "linux_x86_64"
        raise SetupError(
            "A modified PyNvVideoCodec 2.2.0 cp312 wheel is required. "
            f"Place the {platform_hint} wheel in the project 'wheels' directory "
            "or pass --pynv-wheel PATH."
        )
    return compatible[-1].resolve()


def preflight(pynv_wheel=None):
    script_dir = Path(__file__).resolve().parent
    check_python()
    system = check_platform()
    check_project_files(script_dir)
    check_program("ffmpeg")
    check_program("ffprobe")

    try:
        gpu = torch_detect.detect_environment()
    except torch_detect.DetectionError as exc:
        raise SetupError(str(exc)) from exc

    wheel = find_pynv_wheel(script_dir, system, pynv_wheel)
    verify_pynv_wheel(wheel, system)

    print(f"Python: {platform.python_version()} ({sys.executable})")
    print(f"Platform: {system} {platform.machine()}")
    print(f"GPU: {gpu['gpu_name']} (SM {gpu['sm']})")
    print(f"NVIDIA driver: {gpu['driver_version']} / CUDA API {gpu['driver_cuda']}")
    print(f"PyNvVideoCodec wheel: {wheel}")
    return {
        "script_dir": script_dir,
        "system": system,
        "gpu": gpu,
        "pynv_wheel": wheel,
    }


def install_cuda_stack(config, *, dry_run=False):
    print("\nInstalling the pinned CUDA stack...")
    pip_install(config["gpu"]["torch"]["pip_args"], dry_run=dry_run)
    cupy_requirement = (
        CUPY_WINDOWS_REQUIREMENT
        if config["system"] == "Windows"
        else CUPY_LINUX_REQUIREMENT
    )
    pip_install(["install", cupy_requirement], dry_run=dry_run)
    run_command([sys.executable, "-m", "pip", "check"], dry_run=dry_run)
    if config["system"] == "Windows":
        pip_install(["install", TRITON_WINDOWS_REQUIREMENT], dry_run=dry_run)
    pip_install(
        ["install", "--no-deps", str(config["pynv_wheel"])],
        dry_run=dry_run,
    )


def install_runtime_requirements(config, *, dry_run=False):
    print("\nInstalling application dependencies...")
    pip_install(
        ["install", "-r", str(config["script_dir"] / "requirements.txt")],
        dry_run=dry_run,
    )


def install_gui(config, *, dry_run=False):
    gui_dir = config["script_dir"] / "GUI"
    if not (gui_dir / "pyproject.toml").is_file():
        raise SetupError(f"GUI package not found: {gui_dir}")
    pip_install(["install", "-e", f"{gui_dir}[pyside6]"], dry_run=dry_run)


def verify_environment(expected_sm):
    import torch
    import cupy as cp
    import PyNvVideoCodec as nvc

    if not torch.cuda.is_available():
        raise SetupError("PyTorch was installed, but CUDA is unavailable.")

    torch_cc = torch.cuda.get_device_capability(0)
    detected_sm = torch_cc[0] * 10 + torch_cc[1]
    if detected_sm != expected_sm:
        raise SetupError(f"GPU changed during setup: expected SM {expected_sm}, got SM {detected_sm}.")

    tensor = torch.ones(1, device="cuda")
    if expected_sm >= 80:
        compiled = torch.compile(lambda value: value + 1, backend="inductor")
        tensor = compiled(tensor)
    else:
        tensor = tensor + 1
    torch.cuda.synchronize()
    if tensor.item() != 2:
        raise SetupError("PyTorch CUDA smoke test returned an invalid result.")

    cupy_value = cp.arange(4, dtype=cp.float32).sum()
    if float(cupy_value.get()) != 6.0:
        raise SetupError("CuPy CUDA smoke test returned an invalid result.")

    print(f"\nPyTorch: {torch.__version__}; CUDA architectures: {torch.cuda.get_arch_list()}")
    print(f"CuPy: {cp.__version__}; device CC: {cp.cuda.Device(0).compute_capability}")
    print(f"PyNvVideoCodec: {nvc.__version__} ({nvc.__file__})")
    if expected_sm < 80:
        print("torch.compile was intentionally skipped on SM 7.5.")


def parse_args():
    parser = argparse.ArgumentParser(description="Install the VR We Are GPU environment.")
    parser.add_argument("--gui", action="store_true", help="Install the optional GUI package.")
    parser.add_argument(
        "--pynv-wheel",
        help="Path to the modified PyNvVideoCodec 2.2.0 wheel for this platform.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run preflight checks without installing packages.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print installation commands without executing them.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        print("VR We Are GPU environment setup\n")
        config = preflight(args.pynv_wheel)
        if args.check_only:
            print("\nPreflight checks passed.")
            return 0

        print("\nUpdating Python packaging tools...")
        pip_install(["install", "--upgrade", "pip", "setuptools", "wheel"], dry_run=args.dry_run)
        install_cuda_stack(config, dry_run=args.dry_run)
        install_runtime_requirements(config, dry_run=args.dry_run)
        if args.gui:
            install_gui(config, dry_run=args.dry_run)

        if args.dry_run:
            print("\nDry run completed; no packages were installed.")
            return 0

        importlib.invalidate_caches()
        verify_environment(config["gpu"]["sm"])
        run_command([sys.executable, "-m", "pip", "check"])
        print("\nSetup completed successfully.")
        return 0
    except SetupError as exc:
        print(f"\nSetup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
