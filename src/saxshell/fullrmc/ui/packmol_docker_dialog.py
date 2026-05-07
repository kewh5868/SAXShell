from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from saxshell.fullrmc.packmol_docker import (
    DEFAULT_PACKMOL_CONTAINER_ROOT,
    PackmolDockerClient,
    PackmolDockerContainerRecord,
    PackmolDockerLink,
    docker_daemon_unavailable_hint,
)

_TREE_PATH_ROLE = Qt.ItemDataRole.UserRole
_TREE_LOADED_ROLE = Qt.ItemDataRole.UserRole + 1


class PackmolDockerLinkDialog(QDialog):
    def __init__(
        self,
        *,
        current_link: PackmolDockerLink | None = None,
        recent_presets: (
            list[PackmolDockerLink] | tuple[PackmolDockerLink, ...]
        ) = (),
        docker_client: PackmolDockerClient | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Link Packmol Docker Container")
        self.resize(920, 620)

        self._docker_client = docker_client or PackmolDockerClient()
        self._recent_presets = list(recent_presets)
        self._available_containers: list[PackmolDockerContainerRecord] = []
        self._selected_link: PackmolDockerLink | None = None
        self._validated_signature: tuple[str, str, str, str] | None = None

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(12, 12, 12, 12)
        root_layout.setSpacing(12)

        intro_label = QLabel(
            "Validate a Packmol-ready Docker container, choose the "
            "container-side project folder, and link it to the current "
            "rmcsetup workflow. You can pick from discovered Docker "
            "containers or type a container name manually. The "
            "container-side project folder must live inside the bind-mounted "
            f"root {DEFAULT_PACKMOL_CONTAINER_ROOT}."
        )
        intro_label.setWordWrap(True)
        root_layout.addWidget(intro_label)

        preset_group = QGroupBox("Recent Container Presets")
        preset_layout = QHBoxLayout(preset_group)
        self.preset_name_edit = QLineEdit()
        self.preset_name_edit.setPlaceholderText("Preset display name")
        preset_layout.addWidget(self.preset_name_edit, stretch=1)
        self.preset_combo = QTreeWidget()
        self.preset_combo.setHeaderHidden(True)
        self.preset_combo.setMaximumHeight(110)
        preset_layout.addWidget(self.preset_combo, stretch=1)
        preset_button_column = QVBoxLayout()
        self.load_preset_button = QPushButton("Load Selected Preset")
        self.load_preset_button.clicked.connect(self._load_selected_preset)
        preset_button_column.addWidget(self.load_preset_button)
        preset_button_column.addStretch(1)
        preset_layout.addLayout(preset_button_column)
        root_layout.addWidget(preset_group)

        form_group = QGroupBox("Container Settings")
        form_layout = QFormLayout(form_group)
        discovered_row = QWidget()
        discovered_layout = QHBoxLayout(discovered_row)
        discovered_layout.setContentsMargins(0, 0, 0, 0)
        discovered_layout.setSpacing(6)
        self.available_container_combo = QComboBox()
        self.available_container_combo.setMinimumContentsLength(32)
        discovered_layout.addWidget(self.available_container_combo, stretch=1)
        self.refresh_containers_button = QPushButton("Refresh List")
        self.refresh_containers_button.clicked.connect(
            self._refresh_available_containers
        )
        discovered_layout.addWidget(self.refresh_containers_button)
        self.use_available_container_button = QPushButton("Use Selected")
        self.use_available_container_button.clicked.connect(
            self._use_available_container
        )
        discovered_layout.addWidget(self.use_available_container_button)
        form_layout.addRow("Discovered containers", discovered_row)
        self.container_name_edit = QLineEdit()
        self.container_name_edit.setPlaceholderText("Docker container name")
        form_layout.addRow("Container name", self.container_name_edit)
        self.packmol_command_edit = QLineEdit("packmol")
        form_layout.addRow("Packmol command", self.packmol_command_edit)
        self.shell_command_edit = QLineEdit("sh")
        form_layout.addRow("Shell command", self.shell_command_edit)
        self.container_root_edit = QLineEdit(DEFAULT_PACKMOL_CONTAINER_ROOT)
        self.container_root_edit.setPlaceholderText(
            "/packmol_input_files/project_name"
        )
        form_layout.addRow("Container project root", self.container_root_edit)
        root_layout.addWidget(form_group)

        command_row = QHBoxLayout()
        self.test_connection_button = QPushButton("Test Container")
        self.test_connection_button.clicked.connect(self._test_connection)
        command_row.addWidget(self.test_connection_button)
        self.refresh_tree_button = QPushButton("Refresh Directory Tree")
        self.refresh_tree_button.clicked.connect(self._refresh_directory_tree)
        command_row.addWidget(self.refresh_tree_button)
        self.use_selected_directory_button = QPushButton(
            "Use Selected Directory"
        )
        self.use_selected_directory_button.clicked.connect(
            self._use_selected_directory
        )
        command_row.addWidget(self.use_selected_directory_button)
        command_row.addStretch(1)
        root_layout.addLayout(command_row)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.content_splitter.setChildrenCollapsible(False)
        root_layout.addWidget(self.content_splitter, stretch=1)

        self.directory_tree = QTreeWidget()
        self.directory_tree.setHeaderLabel("Container Directories")
        self.directory_tree.itemExpanded.connect(self._on_tree_item_expanded)
        self.directory_tree.itemSelectionChanged.connect(
            self._on_tree_selection_changed
        )
        self.content_splitter.addWidget(self.directory_tree)

        details_panel = QWidget()
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(0, 0, 0, 0)
        details_layout.setSpacing(8)
        self.selected_directory_label = QLabel("Selected directory: (none)")
        self.selected_directory_label.setWordWrap(True)
        details_layout.addWidget(self.selected_directory_label)
        self.status_box = QPlainTextEdit()
        self.status_box.setReadOnly(True)
        self.status_box.setPlainText(
            "Press Test Container to initialize Docker, verify Packmol "
            "inside the selected container, and load the directory tree."
        )
        details_layout.addWidget(self.status_box, stretch=1)
        self.content_splitter.addWidget(details_panel)
        self.content_splitter.setStretchFactor(0, 1)
        self.content_splitter.setStretchFactor(1, 1)
        self.content_splitter.setSizes([420, 420])

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.button_box.button(QDialogButtonBox.StandardButton.Ok).setText(
            "Link Container"
        )
        root_layout.addWidget(self.button_box)

        self._populate_preset_tree()
        if current_link is not None:
            self._apply_link_to_fields(current_link)
        elif self._recent_presets:
            self._apply_link_to_fields(self._recent_presets[0])
        self._refresh_available_containers(show_feedback=False)

    def selected_link(self) -> PackmolDockerLink | None:
        return self._selected_link

    def _populate_preset_tree(self) -> None:
        self.preset_combo.clear()
        for preset in self._recent_presets:
            item = QTreeWidgetItem([preset.resolved_display_name])
            item.setData(0, _TREE_PATH_ROLE, preset.to_dict())
            item.setToolTip(0, preset.summary_text())
            self.preset_combo.addTopLevelItem(item)
        has_presets = self.preset_combo.topLevelItemCount() > 0
        self.preset_combo.setEnabled(has_presets)
        self.load_preset_button.setEnabled(has_presets)
        if has_presets:
            self.preset_combo.setCurrentItem(self.preset_combo.topLevelItem(0))

    def _apply_link_to_fields(self, link: PackmolDockerLink) -> None:
        self.preset_name_edit.setText(link.resolved_display_name)
        self.container_name_edit.setText(link.container_name)
        self.packmol_command_edit.setText(link.packmol_command)
        self.shell_command_edit.setText(link.shell_command)
        self.container_root_edit.setText(link.container_project_root)
        self._validated_signature = None
        self._selected_link = None

    def _load_selected_preset(self) -> None:
        current_item = self.preset_combo.currentItem()
        if current_item is None:
            return
        payload = current_item.data(0, _TREE_PATH_ROLE)
        if not isinstance(payload, dict):
            return
        preset = PackmolDockerLink.from_dict(payload)
        if preset is None:
            return
        self._apply_link_to_fields(preset)
        self.status_box.setPlainText(
            "Preset loaded. Test the container to validate the current "
            "Docker state and refresh the directory tree."
        )
        self._sync_available_container_selection()

    def _draft_link(self) -> PackmolDockerLink:
        container_name = self.container_name_edit.text().strip()
        if not container_name:
            raise ValueError("Enter a Docker container name before linking.")
        display_name = self.preset_name_edit.text().strip() or container_name
        return PackmolDockerLink(
            display_name=display_name,
            container_name=container_name,
            packmol_command=self.packmol_command_edit.text().strip()
            or "packmol",
            shell_command=self.shell_command_edit.text().strip() or "sh",
            container_project_root=self.container_root_edit.text().strip()
            or DEFAULT_PACKMOL_CONTAINER_ROOT,
        )

    def _link_signature(
        self, link: PackmolDockerLink
    ) -> tuple[str, str, str, str]:
        return (
            link.container_name.strip(),
            link.packmol_command.strip(),
            link.shell_command.strip(),
            link.container_project_root.strip(),
        )

    def _format_docker_failure(
        self,
        exc: Exception,
        *,
        include_attached_shell_hint: bool,
    ) -> tuple[str, bool]:
        details = str(exc).strip() or "Docker command failed."
        daemon_hint = docker_daemon_unavailable_hint(details)
        sections: list[str] = []
        if daemon_hint is not None:
            sections.append(daemon_hint)
        sections.append(details)
        if include_attached_shell_hint:
            if daemon_hint is not None:
                sections.append(
                    "After Docker is running, if your container only stays "
                    "alive with an attached shell, start it manually with "
                    "`docker start -i <container_name>` before retrying."
                )
            else:
                sections.append(
                    "If your container only stays alive with an attached "
                    "shell, start it manually with `docker start -i "
                    "<container_name>` before retrying."
                )
        return "\n\n".join(sections), daemon_hint is not None

    def _test_connection(self) -> bool:
        try:
            draft = self._draft_link()
            result = self._docker_client.verify_link(draft)
        except Exception as exc:
            self._selected_link = None
            self._validated_signature = None
            formatted_message, _ = self._format_docker_failure(
                exc,
                include_attached_shell_hint=True,
            )
            self.status_box.setPlainText(
                "Docker validation failed.\n\n" f"{formatted_message}"
            )
            return False
        draft.last_verified_at = result.verified_at
        draft.container_id = result.container_id
        draft.image_name = result.image_name
        draft.packmol_command_path = result.packmol_command_path
        draft.packmol_version = result.packmol_version
        draft.container_project_root = result.container_project_root
        self.container_root_edit.setText(result.container_project_root)
        self.status_box.setPlainText(result.summary_text(draft))
        self._selected_link = draft
        self._validated_signature = self._link_signature(draft)
        self._load_directory_tree(draft, result.container_project_root)
        return True

    def _refresh_available_containers(
        self,
        *,
        show_feedback: bool = True,
    ) -> None:
        try:
            records = self._docker_client.list_containers()
        except Exception as exc:
            self._available_containers = []
            self.available_container_combo.clear()
            self.available_container_combo.setEnabled(False)
            self.use_available_container_button.setEnabled(False)
            if show_feedback:
                formatted_message, daemon_unavailable = (
                    self._format_docker_failure(
                        exc,
                        include_attached_shell_hint=False,
                    )
                )
                self.status_box.appendPlainText(
                    "\nUnable to list Docker containers.\n\n"
                    f"{formatted_message}"
                )
                if not daemon_unavailable:
                    self.status_box.appendPlainText(
                        "\nYou can still type a container name manually and "
                        "press Test Container."
                    )
            return
        self._available_containers = list(records)
        self.available_container_combo.clear()
        for record in self._available_containers:
            self.available_container_combo.addItem(
                record.summary_label,
                record,
            )
        has_records = bool(self._available_containers)
        self.available_container_combo.setEnabled(has_records)
        self.use_available_container_button.setEnabled(has_records)
        self._sync_available_container_selection()
        if not show_feedback:
            return
        if has_records:
            self.status_box.appendPlainText(
                "\nDiscovered "
                f"{len(self._available_containers)} Docker container(s). "
                "Select one to populate the container name field, then "
                "press Test Container to verify Packmol."
            )
            return
        self.status_box.appendPlainText(
            "\nNo Docker containers were found. You can still type a "
            "container name manually and press Test Container."
        )

    def _sync_available_container_selection(self) -> None:
        current_name = self.container_name_edit.text().strip()
        if not current_name:
            return
        for index, record in enumerate(self._available_containers):
            if record.name == current_name:
                self.available_container_combo.setCurrentIndex(index)
                return

    def _selected_available_container(
        self,
    ) -> PackmolDockerContainerRecord | None:
        record = self.available_container_combo.currentData()
        if isinstance(record, PackmolDockerContainerRecord):
            return record
        return None

    def _use_available_container(self) -> None:
        record = self._selected_available_container()
        if record is None:
            return
        self.container_name_edit.setText(record.name)
        self._selected_link = None
        self._validated_signature = None
        self.status_box.appendPlainText(
            "\nLoaded container name from the discovered Docker list. "
            "Press Test Container to verify Packmol in this container."
        )

    def _refresh_directory_tree(self) -> None:
        if not self._test_connection():
            return

    def _load_directory_tree(
        self,
        link: PackmolDockerLink,
        root_path: str,
    ) -> None:
        self.directory_tree.clear()
        root_item = QTreeWidgetItem([root_path])
        root_item.setData(0, _TREE_PATH_ROLE, root_path)
        root_item.setData(0, _TREE_LOADED_ROLE, False)
        root_item.addChild(QTreeWidgetItem(["Loading..."]))
        self.directory_tree.addTopLevelItem(root_item)
        root_item.setExpanded(True)
        self._populate_tree_item_children(root_item, link)
        self.directory_tree.setCurrentItem(root_item)
        self._on_tree_selection_changed()

    def _populate_tree_item_children(
        self,
        item: QTreeWidgetItem,
        link: PackmolDockerLink | None = None,
    ) -> None:
        active_link = link or self._selected_link
        if active_link is None:
            return
        if item.data(0, _TREE_LOADED_ROLE):
            return
        directory = item.data(0, _TREE_PATH_ROLE)
        if not isinstance(directory, str) or not directory:
            return
        item.takeChildren()
        try:
            entries = self._docker_client.list_directories(
                active_link, directory
            )
        except Exception as exc:
            item.addChild(QTreeWidgetItem([f"Unable to load folders: {exc}"]))
            item.setData(0, _TREE_LOADED_ROLE, True)
            return
        for entry in entries:
            child = QTreeWidgetItem([entry.name])
            child.setData(0, _TREE_PATH_ROLE, entry.path)
            child.setData(0, _TREE_LOADED_ROLE, False)
            child.addChild(QTreeWidgetItem(["Loading..."]))
            item.addChild(child)
        item.setData(0, _TREE_LOADED_ROLE, True)

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        self._populate_tree_item_children(item)

    def _on_tree_selection_changed(self) -> None:
        current_item = self.directory_tree.currentItem()
        if current_item is None:
            self.selected_directory_label.setText("Selected directory: (none)")
            return
        directory = current_item.data(0, _TREE_PATH_ROLE)
        if not isinstance(directory, str) or not directory:
            self.selected_directory_label.setText("Selected directory: (none)")
            return
        self.selected_directory_label.setText(
            f"Selected directory: {directory}"
        )

    def _use_selected_directory(self) -> None:
        current_item = self.directory_tree.currentItem()
        if current_item is None:
            return
        directory = current_item.data(0, _TREE_PATH_ROLE)
        if not isinstance(directory, str) or not directory:
            return
        self.container_root_edit.setText(directory)
        self.status_box.appendPlainText(
            "\nUpdated container project root from the selected directory."
        )

    def accept(self) -> None:
        try:
            draft = self._draft_link()
        except Exception as exc:
            QMessageBox.warning(self, "Packmol Docker link invalid", str(exc))
            return
        if self._validated_signature != self._link_signature(draft):
            if not self._test_connection():
                QMessageBox.warning(
                    self,
                    "Packmol Docker link invalid",
                    "Docker validation failed. Review the status box for "
                    "details and retry.",
                )
                return
        super().accept()
