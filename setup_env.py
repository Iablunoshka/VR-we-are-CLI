import argparse
import importlib
import os
import shutil
import subprocess
import sys


def install_torch_auto():
    """
    Install PyTorch using install command from torch_detect.py.
    """

    try:
        import torch_detect
    except Exception as e:
        print(f"Error: cannot import torch_detect.py: {e}")
        print("Make sure torch_detect.py is in the same folder.")
        return False

    print("\nDetecting best PyTorch build for your GPU...")
    choice = torch_detect.main()

    # CPU fallback
    if not choice:
        print("Installing CPU version of PyTorch...")
        cpu_cmd = f'"{sys.executable}" -m pip install torch --index-url https://download.pytorch.org/whl/cpu'

        try:
            subprocess.check_call(cpu_cmd, shell=True)
            return True
        except subprocess.CalledProcessError as e:
            print(f"CPU installation failed: {e}")
            return False

    try:
        subprocess.check_call(choice["cmd"], shell=True)
        print("PyTorch installation completed!")
        return True

    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")
        print("Falling back to CPU version...")

        cpu_cmd = f'"{sys.executable}" -m pip install torch  --index-url https://download.pytorch.org/whl/cpu'
        subprocess.check_call(cpu_cmd, shell=True)
        return True

def test_torch():
    """Verify that PyTorch is installed and CUDA works."""
    try:
        import torch
        print("\nPyTorch installed!")
        print(f"Version PyTorch: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"CUDA is available: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("CUDA is not available, in use CPU.")
    except ImportError:
        print("\nError: PyTorch is not installed or imported!")
        

def test_ffmpeg():
    """Check if FFmpeg is installed and functional."""
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        try:
            result = subprocess.run([ffmpeg_path, "-version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("FFmpeg is installed and working!")
                print(result.stdout.splitlines()[0])
            else:
                print("FFmpeg exists but failed to run.")
                print(result.stderr)
        except Exception as e:
            print(f"Error running FFmpeg: {e}")
    else:
        print(
            "FFmpeg not found.\n\n"
            "On Windows, you can install it via Chocolatey:\n"
            "  choco install ffmpeg\n"
            "More info: https://chocolatey.org/install\n\n"
            "On Ubuntu/Debian:\n"
            "  sudo apt install ffmpeg\n\n"
            "There is also a guide by this user:\n"
            "  https://github.com/aaatipamula/ffmpeg-install"
        )
        
def install_requirements_and_refresh():
    """
    Install packages from requirements.txt located in the same folder as this script
    and refresh Python's import system.
    """
    print("Upgrading pip...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        print("pip upgraded successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error upgrading pip: {e}")
        return False

    script_dir = os.path.dirname(os.path.abspath(__file__))
    req_file = os.path.join(script_dir, "requirements.txt")

    if not os.path.exists(req_file):
        print(f"requirements.txt not found in {script_dir}")
        return False

    print(f"Installing packages from {req_file} ...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", req_file])
        print("All packages from requirements.txt installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing requirements: {e}")
        return False

    # refresh
    importlib.invalidate_caches()
    if hasattr(sys, 'path_importer_cache'):
        sys.path_importer_cache.clear()

    return True


def install_gui():
    """
    Install the optional GUI package and its PySide6 Essentials dependency.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    gui_dir = os.path.join(script_dir, "GUI")
    gui_project = os.path.join(gui_dir, "pyproject.toml")

    if not os.path.exists(gui_project):
        print(f"GUI package not found in {gui_dir}")
        return False

    print(f"Installing GUI from {gui_dir} ...")
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-e", f"{gui_dir}[pyside6]"]
        )
        print("GUI installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error installing GUI: {e}")
        return False

    importlib.invalidate_caches()
    if hasattr(sys, 'path_importer_cache'):
        sys.path_importer_cache.clear()

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Install the VR We Are environment.")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Install the optional PySide6 GUI in the active environment.",
    )
    args = parser.parse_args()

    install_requirements_and_refresh()
    install_torch_auto()
    test_torch()
    test_ffmpeg()

    if args.gui and not install_gui():
        print("\nSetup completed, but the GUI installation failed.")
        raise SystemExit(1)

    print("\nSetup complete! You're ready to run the pipeline.")
