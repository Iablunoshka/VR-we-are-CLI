"""Qt appearance setup that preserves system palette changes."""

from PySide6.QtWidgets import QApplication


def configure_appearance(app: QApplication) -> None:
    """Use Qt's cross-platform style without overriding its color palette."""

    app.setStyle("Fusion")
    font = app.font()
    font.setPointSize(10)
    app.setFont(font)
