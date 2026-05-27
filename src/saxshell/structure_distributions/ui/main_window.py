from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.bondanalysis.results import (
    BondAnalysisDistributionSeries,
    BondAnalysisPlotRequest,
    DistributionCategory,
)
from saxshell.bondanalysis.ui.plot_window import BondAnalysisPlotWindow
from saxshell.saxs.ui.branding import (
    configure_saxshell_application,
    load_saxshell_icon,
    prepare_saxshell_application_identity,
)
from saxshell.structure_distributions import (
    StructureDistributionGroup,
    StructureDistributionIndex,
    StructureDistributionLeaf,
    load_structure_distribution_index,
    project_structure_distribution_store_dir,
    validate_structure_distribution_leaves,
)

_OPEN_WINDOWS: list["StructureDistributionBrowserWindow"] = []


class StructureDistributionBrowserWindow(QMainWindow):
    """Project-level browser for cached structure distributions."""

    def __init__(
        self,
        *,
        initial_project_dir: str | Path | None = None,
        initial_store_dir: str | Path | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._project_dir = (
            None
            if initial_project_dir is None
            else Path(initial_project_dir).expanduser().resolve()
        )
        self._index: StructureDistributionIndex | None = None
        self._plot_windows: list[BondAnalysisPlotWindow] = []
        self._build_ui()
        if initial_store_dir is not None:
            self.store_dir_edit.setText(str(Path(initial_store_dir)))
        elif self._project_dir is not None:
            self.store_dir_edit.setText(
                str(
                    project_structure_distribution_store_dir(self._project_dir)
                )
            )
        self.refresh_index()

    def _build_ui(self) -> None:
        self.setWindowTitle("SAXSShell Structure Distribution Browser")
        self.setWindowIcon(load_saxshell_icon())
        self.resize(1120, 760)

        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([470, 650])
        root.addWidget(splitter)

        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        paths_group = QGroupBox("Shared Store")
        paths_layout = QVBoxLayout(paths_group)
        path_row = QHBoxLayout()
        self.store_dir_edit = QLineEdit()
        self.store_dir_edit.setPlaceholderText(
            "Select analysis/structure_distributions"
        )
        path_row.addWidget(self.store_dir_edit, stretch=1)
        browse_button = QPushButton("Browse")
        browse_button.clicked.connect(self._choose_store_dir)
        path_row.addWidget(browse_button)
        paths_layout.addLayout(path_row)

        refresh_button = QPushButton("Refresh Browser")
        refresh_button.clicked.connect(self.refresh_index)
        paths_layout.addWidget(refresh_button)
        layout.addWidget(paths_group)

        self.summary_label = QLabel()
        self.summary_label.setWordWrap(True)
        self.summary_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.summary_label)
        layout.addStretch(1)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        controls = QHBoxLayout()
        self.open_selected_button = QPushButton("Open Selected in Window")
        self.open_selected_button.clicked.connect(
            self.open_selected_plot_window
        )
        controls.addWidget(self.open_selected_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.results_tree = QTreeWidget()
        self.results_tree.setColumnCount(4)
        self.results_tree.setHeaderLabels(
            ["Distribution", "Scope", "Structures", "Values"]
        )
        self.results_tree.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.results_tree.itemSelectionChanged.connect(
            self._on_selection_changed
        )
        header = self.results_tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        layout.addWidget(self.results_tree, stretch=1)

        self.status_label = QLabel()
        self.status_label.setWordWrap(True)
        self.status_label.setFrameShape(QFrame.Shape.StyledPanel)
        layout.addWidget(self.status_label)
        return panel

    def _choose_store_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select structure distribution store directory",
            self.store_dir_edit.text().strip() or str(Path.home()),
        )
        if not selected:
            return
        self.store_dir_edit.setText(selected)
        self.refresh_index()

    def refresh_index(self) -> None:
        store_dir = self._store_dir_path()
        if store_dir is None:
            self._index = None
            self.results_tree.clear()
            self.summary_label.setText(
                "Select a structure distribution store directory."
            )
            self.status_label.setText("No store selected.")
            return

        self._index = load_structure_distribution_index(store_dir)
        self._populate_results_tree()
        self._update_summary()

    def _store_dir_path(self) -> Path | None:
        text = self.store_dir_edit.text().strip()
        if not text:
            return None
        return Path(text).expanduser().resolve()

    def _populate_results_tree(self) -> None:
        self.results_tree.clear()
        if self._index is None:
            return
        if not self._index.groups:
            self.status_label.setText(
                "No current cached structure distributions were found."
            )
            return
        current_source = None
        source_item = None
        current_category = None
        category_item = None
        for group in self._index.groups:
            if group.source_name != current_source:
                source_item = QTreeWidgetItem(
                    [group.source_name, "application", "", ""]
                )
                source_item.setFlags(
                    source_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                )
                self.results_tree.addTopLevelItem(source_item)
                current_source = group.source_name
                current_category = None
            if source_item is None:
                continue
            category_label = _category_label(group.category)
            if category_label != current_category:
                category_item = QTreeWidgetItem([category_label, "", "", ""])
                category_item.setFlags(
                    category_item.flags() & ~Qt.ItemFlag.ItemIsSelectable
                )
                source_item.addChild(category_item)
                current_category = category_label
            if category_item is not None:
                self._populate_group(category_item, group)
        self.results_tree.expandAll()
        self.status_label.setText(
            "Select one all-clusters row or one or more matching cluster rows "
            "to open a histogram window."
        )

    def _populate_group(
        self,
        parent_item: QTreeWidgetItem,
        group: StructureDistributionGroup,
    ) -> None:
        group_item = QTreeWidgetItem([group.display_label, "", "", ""])
        group_item.setFlags(group_item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
        parent_item.addChild(group_item)
        group_item.addChild(self._make_leaf_item(group.all_leaf))
        for leaf in group.cluster_leaves:
            group_item.addChild(self._make_leaf_item(leaf))

    def _make_leaf_item(
        self,
        leaf: StructureDistributionLeaf,
    ) -> QTreeWidgetItem:
        scope_kind = "all clusters" if leaf.is_all else "cluster"
        item = QTreeWidgetItem(
            [
                leaf.scope_name,
                scope_kind,
                str(leaf.structure_count),
                str(leaf.point_count),
            ]
        )
        item.setData(0, Qt.ItemDataRole.UserRole, leaf)
        item.setToolTip(
            0,
            f"{leaf.source_name} - {leaf.display_label} - "
            f"{leaf.scope_name}: {leaf.point_count} values",
        )
        return item

    def _update_summary(self) -> None:
        if self._index is None:
            return
        store_count = len(self._index.manifest_paths)
        source_text = ", ".join(self._index.source_names) or "none"
        summary = (
            f"Store: {self._index.root_dir}\n"
            f"Applications: {source_text}\n"
            f"Manifests: {store_count}; entries: {self._index.entry_count}; "
            f"distributions: {len(self._index.groups)}"
        )
        if self._index.stale_entry_count:
            summary += (
                f"\nSkipped stale entries: {self._index.stale_entry_count}"
            )
        if self._index.missing_array_count:
            summary += (
                f"\nMissing cached arrays: {self._index.missing_array_count}"
            )
        self.summary_label.setText(summary)

    def _selected_leaves(self) -> tuple[StructureDistributionLeaf, ...]:
        leaves: list[StructureDistributionLeaf] = []
        for item in self.results_tree.selectedItems():
            payload = item.data(0, Qt.ItemDataRole.UserRole)
            if isinstance(payload, StructureDistributionLeaf):
                leaves.append(payload)
        return tuple(leaves)

    def _on_selection_changed(self) -> None:
        leaves = self._selected_leaves()
        if not leaves:
            self.status_label.setText(
                "Select one cached distribution to plot, or select matching "
                "cluster rows to overlay them."
            )
            return
        try:
            validate_structure_distribution_leaves(leaves)
        except ValueError as exc:
            self.status_label.setText(str(exc))
            return
        if len(leaves) == 1:
            leaf = leaves[0]
            self.status_label.setText(
                f"Ready to open {leaf.display_label} for {leaf.scope_name}."
            )
            return
        scopes = ", ".join(
            f"{leaf.source_name}: {leaf.scope_name}" for leaf in leaves
        )
        self.status_label.setText(
            f"Ready to overlay {leaves[0].display_label}: {scopes}"
        )

    def open_selected_plot_window(self) -> None:
        leaves = self._selected_leaves()
        if not leaves:
            QMessageBox.information(
                self,
                "Structure Distributions",
                "Select one or more cached distributions first.",
            )
            return
        try:
            validate_structure_distribution_leaves(leaves)
            plot_request = self._plot_request_for_leaves(leaves)
        except Exception as exc:
            QMessageBox.warning(self, "Structure Distributions", str(exc))
            return
        self._open_plot_window_for_request(plot_request)

    def _plot_request_for_leaves(
        self,
        leaves: tuple[StructureDistributionLeaf, ...],
    ) -> BondAnalysisPlotRequest:
        first = leaves[0]
        if len(leaves) == 1:
            title = (
                f"{first.source_name}: {first.display_label} "
                f"({first.scope_name})"
            )
        else:
            title = f"{first.display_label} across selected clusters"
        series = tuple(
            BondAnalysisDistributionSeries(
                label=_series_label(leaf),
                values=leaf.values,
            )
            for leaf in leaves
        )
        return BondAnalysisPlotRequest(
            category=cast(DistributionCategory, first.category),
            display_label=first.display_label,
            title=title,
            xlabel=first.xlabel,
            series=series,
        )

    def _open_plot_window_for_request(
        self,
        plot_request: BondAnalysisPlotRequest,
    ) -> None:
        default_output_dir = self._store_dir_path() or Path.cwd()
        if self._plot_windows:
            window = self._plot_windows[0]
            window.add_plot_request(plot_request)
        else:
            window = BondAnalysisPlotWindow(
                plot_request,
                default_output_dir=default_output_dir,
                parent=self,
            )
            window.destroyed.connect(
                lambda _obj=None, win=window: self._remove_plot_window(win)
            )
            self._plot_windows.append(window)
        window.show()
        window.raise_()
        window.activateWindow()

    def _remove_plot_window(self, window: BondAnalysisPlotWindow) -> None:
        self._plot_windows = [
            existing
            for existing in self._plot_windows
            if existing is not window
        ]


def launch_structure_distribution_browser_ui(
    *,
    initial_project_dir: str | Path | None = None,
    initial_store_dir: str | Path | None = None,
) -> StructureDistributionBrowserWindow:
    app = QApplication.instance()
    if app is None:
        prepare_saxshell_application_identity()
        app = QApplication(sys.argv)
    configure_saxshell_application(app)

    window = StructureDistributionBrowserWindow(
        initial_project_dir=initial_project_dir,
        initial_store_dir=initial_store_dir,
    )
    window.show()
    window.raise_()
    _OPEN_WINDOWS.append(window)
    window.destroyed.connect(
        lambda _obj=None, win=window: _forget_open_window(win)
    )
    return window


def _forget_open_window(window: StructureDistributionBrowserWindow) -> None:
    _OPEN_WINDOWS[:] = [
        existing for existing in _OPEN_WINDOWS if existing is not window
    ]


def _category_label(category: str) -> str:
    return {
        "bond": "Bond Pairs",
        "angle": "Bond Angles",
        "coordination": "Coordination Numbers",
        "cutoff_pair": "Pair Distances",
    }.get(category, str(category).replace("_", " ").title())


def _series_label(leaf: StructureDistributionLeaf) -> str:
    if leaf.is_all:
        return f"{leaf.source_name}: all clusters"
    return f"{leaf.source_name}: {leaf.scope_name}"
