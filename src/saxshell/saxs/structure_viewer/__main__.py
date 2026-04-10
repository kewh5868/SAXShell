from saxshell.saxs.structure_viewer.ui.main_window import (
    launch_structure_viewer_ui,
)

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    window = launch_structure_viewer_ui()
    window.show()
    app = QApplication.instance()
    raise SystemExit(app.exec() if app is not None else 0)
