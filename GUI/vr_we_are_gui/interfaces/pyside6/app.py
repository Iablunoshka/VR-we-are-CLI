"""PySide6 application bootstrap."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from PySide6.QtWidgets import QApplication, QMessageBox

from ...adapters import SubprocessRunner
from ...application import ConversionController
from ...core import CliTarget
from .main_window import MainWindow
from .theme import configure_appearance


def locate_cli_target() -> CliTarget:
    gui_root = Path(__file__).resolve().parents[3]
    default_cli_root = gui_root.parent
    cli_root = Path(os.environ.get("VR_WE_ARE_CLI_ROOT", default_cli_root)).resolve()
    python_executable = Path(os.environ.get("VR_WE_ARE_PYTHON", sys.executable)).resolve()
    return CliTarget(python_executable, cli_root / "main.py")


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("VR We Are")
    app.setOrganizationName("VR We Are")
    configure_appearance(app)

    target = locate_cli_target()
    if not target.main_script.is_file():
        QMessageBox.critical(None, "CLI not found", f"Could not find the CLI entry point:\n{target.main_script}")
        return 2

    controller = ConversionController(target, SubprocessRunner())
    window = MainWindow(controller)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
