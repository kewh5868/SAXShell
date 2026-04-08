from saxshell.saxs.electron_density_mapping.ui.main_window import (
    launch_electron_density_mapping_ui,
)

if __name__ == "__main__":
    from PySide6.QtWidgets import QApplication

    window = launch_electron_density_mapping_ui()
    window.show()
    app = QApplication.instance()
    raise SystemExit(app.exec() if app is not None else 0)
