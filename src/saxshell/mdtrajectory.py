from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from saxshell.mdtrajectory.ui.main_window import launch_mdtrajectory_app


def main() -> None:
    app = QApplication(sys.argv)
    launch_mdtrajectory_app()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()