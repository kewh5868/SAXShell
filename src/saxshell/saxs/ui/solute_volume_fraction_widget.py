from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QTextEdit,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from saxshell.fullrmc.solution_properties import (
    SolutionPropertiesSettings,
    solution_properties_mode_hint_text,
)
from saxshell.fullrmc.solution_property_presets import (
    SolutionPropertiesPreset,
    load_solution_property_presets,
    ordered_solution_property_preset_names,
    solution_property_presets_path,
)
from saxshell.saxs.solute_volume_fraction import (
    SoluteVolumeFractionEstimate,
    SoluteVolumeFractionSettings,
    calculate_solute_volume_fraction_estimate,
)

SOLUTION_MODE_ITEMS = (
    ("Masses", "mass"),
    ("Mass Percent", "mass_percent"),
    ("Molarity (per liter)", "molarity_per_liter"),
)

SOLUTE_VOLUME_FRACTION_HELP_TEXT = (
    "Solute volume fraction estimator\n\n"
    "This calculator uses the current solution-composition model to recover "
    "solute and solvent masses, then estimates the solute fraction in the "
    "measured solution volume using a SAXS-style concentration x specific-"
    "volume relation:\n\n"
    "c_solute = m_solute / V_solution\n"
    "vbar_solute ~= 1 / rho_solute\n"
    "phi_solute ~= c_solute * vbar_solute\n"
    "          = V_solute / V_solution\n\n"
    "The widget still reports additive component volumes as a diagnostic, "
    "but the main fitted fraction now uses the measured solution volume in "
    "the denominator rather than V_solute + V_solvent.\n\n"
    "Additive-volume check:\n"
    "V_solute = m_solute / rho_solute\n"
    "V_solvent = m_solvent / rho_solvent\n"
    "V_additive = V_solute + V_solvent\n\n"
    "The solution presets come from the fullrmc Solution Properties tool. "
    "Those presets populate the composition fields, but the relevant pure-"
    "component densities should still be reviewed for this estimate. In "
    "molarity mode, SAXSShell uses solution-density plus solvent-density "
    "closure, so solvent density stays active and solute density is not "
    "required.\n\n"
    "Citation link:\n"
    "Hajizadeh et al. (2018), concentration-dependent SAXS mass estimates "
    "require calibrated intensity, accurate solute concentration, and partial "
    "specific volume.\n"
    "https://www.nature.com/articles/s41598-018-25355-2"
)

SOLUTE_VOLUME_FRACTION_CITATION_URL = (
    "https://www.nature.com/articles/s41598-018-25355-2"
)


class SoluteVolumeFractionWidget(QWidget):
    estimate_calculated = Signal(object)
    estimate_failed = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._solution_presets: dict[str, SolutionPropertiesPreset] = {}
        self._updating_solution_preset_selection = False
        self._current_estimate: SoluteVolumeFractionEstimate | None = None
        self._build_ui()
        self._reload_presets()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.target_label = QLabel(
            "This calculator is not currently linked to a Prefit "
            "solute/solvent fraction parameter."
        )
        self.target_label.setWordWrap(True)
        layout.addWidget(self.target_label)

        preset_group = QGroupBox("Solution Presets")
        preset_layout = QVBoxLayout(preset_group)
        preset_row = QHBoxLayout()
        self.solution_preset_combo = QComboBox()
        preset_row.addWidget(self.solution_preset_combo, stretch=1)
        self.load_solution_preset_button = QPushButton("Load")
        self.load_solution_preset_button.clicked.connect(
            self._load_selected_solution_preset
        )
        preset_row.addWidget(self.load_solution_preset_button)
        preset_layout.addLayout(preset_row)
        preset_hint = QLabel(
            "These presets reuse the fullrmc Solution Properties inputs. "
            "The density fields relevant to the selected estimator mode "
            "remain editable below.\n"
            f"Preset file: {solution_property_presets_path()}"
        )
        preset_hint.setWordWrap(True)
        preset_layout.addWidget(preset_hint)
        layout.addWidget(preset_group)

        self.solution_mode_combo = QComboBox()
        for label, value in SOLUTION_MODE_ITEMS:
            self.solution_mode_combo.addItem(label, userData=value)
        self.solution_mode_combo.currentIndexChanged.connect(
            self._on_solution_mode_changed
        )
        self.solution_mode_combo.currentIndexChanged.connect(
            self._on_solution_settings_changed
        )

        self.solution_density_spin = self._new_float_spin(
            maximum=100.0,
            step=0.01,
            decimals=6,
            value=1.0,
        )
        self.solution_density_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        self.solute_stoich_edit = QLineEdit()
        self.solute_stoich_edit.setPlaceholderText("e.g. Cs1Pb1I3")
        self.solute_stoich_edit.textChanged.connect(
            self._on_solution_settings_changed
        )

        self.solvent_stoich_edit = QLineEdit()
        self.solvent_stoich_edit.setPlaceholderText("e.g. H2O or C3H7NO")
        self.solvent_stoich_edit.textChanged.connect(
            self._on_solution_settings_changed
        )

        self.molar_mass_solute_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=1.0,
            decimals=6,
            value=0.0,
        )
        self.molar_mass_solute_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        self.molar_mass_solvent_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=1.0,
            decimals=6,
            value=0.0,
        )
        self.molar_mass_solvent_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        self.solute_density_spin = self._new_float_spin(
            maximum=100.0,
            step=0.01,
            decimals=6,
            value=1.0,
        )
        self.solute_density_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        self.solvent_density_spin = self._new_float_spin(
            maximum=100.0,
            step=0.01,
            decimals=6,
            value=1.0,
        )
        self.solvent_density_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        fields_layout = QGridLayout()
        fields_layout.setColumnStretch(1, 1)
        fields_layout.setColumnStretch(3, 1)
        self.solution_mode_label = QLabel("Input mode")
        fields_layout.addWidget(self.solution_mode_label, 0, 0)
        fields_layout.addWidget(self.solution_mode_combo, 0, 1)
        self.solution_density_label = QLabel("Solution density (g/mL)")
        fields_layout.addWidget(self.solution_density_label, 0, 2)
        fields_layout.addWidget(self.solution_density_spin, 0, 3)
        self.solute_stoich_label = QLabel("Solute stoichiometry")
        fields_layout.addWidget(self.solute_stoich_label, 1, 0)
        fields_layout.addWidget(self.solute_stoich_edit, 1, 1)
        self.solvent_stoich_label = QLabel("Solvent stoichiometry")
        fields_layout.addWidget(self.solvent_stoich_label, 1, 2)
        fields_layout.addWidget(self.solvent_stoich_edit, 1, 3)
        self.molar_mass_solute_label = QLabel("Solute molar mass (g/mol)")
        fields_layout.addWidget(self.molar_mass_solute_label, 2, 0)
        fields_layout.addWidget(self.molar_mass_solute_spin, 2, 1)
        self.molar_mass_solvent_label = QLabel("Solvent molar mass (g/mol)")
        fields_layout.addWidget(self.molar_mass_solvent_label, 2, 2)
        fields_layout.addWidget(self.molar_mass_solvent_spin, 2, 3)
        self.solute_density_label = QLabel("Solute density (g/mL)")
        fields_layout.addWidget(self.solute_density_label, 3, 0)
        fields_layout.addWidget(self.solute_density_spin, 3, 1)
        self.solvent_density_label = QLabel("Solvent density (g/mL)")
        fields_layout.addWidget(self.solvent_density_label, 3, 2)
        fields_layout.addWidget(self.solvent_density_spin, 3, 3)
        layout.addLayout(fields_layout)
        self.solution_mode_hint_label = QLabel()
        self.solution_mode_hint_label.setWordWrap(True)
        layout.addWidget(self.solution_mode_hint_label)

        self.solution_mode_stack = QStackedWidget()

        mass_page = QWidget()
        mass_form = QFormLayout(mass_page)
        self.mass_solute_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=0.1,
            decimals=6,
            value=0.0,
        )
        self.mass_solute_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_form.addRow("Mass solute (g)", self.mass_solute_spin)
        self.mass_solvent_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=0.1,
            decimals=6,
            value=0.0,
        )
        self.mass_solvent_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_form.addRow("Mass solvent (g)", self.mass_solvent_spin)
        self.solution_mode_stack.addWidget(mass_page)

        mass_percent_page = QWidget()
        mass_percent_form = QFormLayout(mass_percent_page)
        self.mass_percent_solute_spin = self._new_float_spin(
            maximum=100.0,
            step=0.1,
            decimals=4,
            value=0.0,
        )
        self.mass_percent_solute_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_percent_form.addRow(
            "Mass percent solute (%)",
            self.mass_percent_solute_spin,
        )
        self.total_mass_solution_spin = self._new_float_spin(
            maximum=1_000_000.0,
            step=0.1,
            decimals=6,
            value=0.0,
        )
        self.total_mass_solution_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        mass_percent_form.addRow(
            "Total solution mass (g)",
            self.total_mass_solution_spin,
        )
        self.solution_mode_stack.addWidget(mass_percent_page)

        molarity_page = QWidget()
        molarity_form = QFormLayout(molarity_page)
        self.molarity_spin = self._new_float_spin(
            maximum=100_000.0,
            step=0.01,
            decimals=6,
            value=0.0,
        )
        self.molarity_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )
        molarity_form.addRow("Molarity (mol/L)", self.molarity_spin)
        self.molarity_element_edit = QLineEdit()
        self.molarity_element_edit.setPlaceholderText("e.g. Pb")
        self.molarity_element_edit.textChanged.connect(
            self._on_solution_settings_changed
        )
        molarity_form.addRow(
            "Molarity element",
            self.molarity_element_edit,
        )
        self.solution_mode_stack.addWidget(molarity_page)

        layout.addWidget(self.solution_mode_stack)

        button_row = QHBoxLayout()
        self.calculate_button = QPushButton("Calculate Volume Fraction")
        self.calculate_button.clicked.connect(self._calculate_estimate)
        button_row.addWidget(self.calculate_button)
        button_row.addStretch(1)
        layout.addLayout(button_row)

        output_header = QHBoxLayout()
        self.output_toggle_button = QToolButton()
        self.output_toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.output_toggle_button.setAutoRaise(True)
        self.output_toggle_button.clicked.connect(
            self._toggle_output_collapsed
        )
        output_header.addWidget(self.output_toggle_button)
        output_header.addStretch(1)
        layout.addLayout(output_header)

        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(180)
        layout.addWidget(self.output_box)
        self.set_output_collapsed(True)
        self._update_solution_mode_widgets()

    @staticmethod
    def _new_float_spin(
        *,
        maximum: float,
        step: float,
        decimals: int,
        value: float,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(0.0, maximum)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def set_target_parameter(
        self,
        parameter_name: str | None,
        fraction_kind: str | None,
    ) -> None:
        if parameter_name and fraction_kind:
            label = "solute" if fraction_kind == "solute" else "solvent"
            self.target_label.setText(
                "Active Prefit target: "
                f"{parameter_name} ({label} volume fraction)."
            )
        else:
            self.target_label.setText(
                "This calculator is not currently linked to a Prefit "
                "solute/solvent fraction parameter."
            )

    def append_application_note(self, message: str) -> None:
        self.set_output_collapsed(False)
        text = self.output_box.toPlainText().strip()
        if not text:
            self.output_box.setPlainText(message.strip())
            return
        self.output_box.setPlainText(text + "\n\n" + message.strip())

    def current_estimate(self) -> SoluteVolumeFractionEstimate | None:
        return self._current_estimate

    def _reload_presets(self, *, selected_name: str | None = None) -> None:
        previous_name = selected_name or self._selected_solution_preset_name()
        self._solution_presets = load_solution_property_presets()
        self.solution_preset_combo.blockSignals(True)
        self.solution_preset_combo.clear()
        self.solution_preset_combo.addItem("Current values", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_solution_property_preset_names(self._solution_presets),
            start=1,
        ):
            preset = self._solution_presets[name]
            label = f"{name} (Built-in)" if preset.builtin else name
            self.solution_preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = index
        self.solution_preset_combo.setCurrentIndex(selected_index)
        self.solution_preset_combo.blockSignals(False)

    def _selected_solution_preset_name(self) -> str | None:
        payload = self.solution_preset_combo.currentData()
        if payload is None:
            return None
        return str(payload)

    def _load_selected_solution_preset(self) -> None:
        preset_name = self._selected_solution_preset_name()
        if preset_name is None:
            self.append_application_note("Select a solution preset to load.")
            return
        preset = self._solution_presets.get(preset_name)
        if preset is None:
            self.append_application_note(
                f"Unknown solution preset: {preset_name}"
            )
            return
        self._apply_solution_preset(preset)
        self._select_solution_preset_name(preset.name)

    def _apply_solution_preset(
        self,
        preset: SolutionPropertiesPreset,
    ) -> None:
        self._apply_solution_settings(preset.settings)
        if preset.solute_density_g_per_ml is not None:
            self.solute_density_spin.setValue(preset.solute_density_g_per_ml)
        if preset.solvent_density_g_per_ml is not None:
            self.solvent_density_spin.setValue(preset.solvent_density_g_per_ml)

    def _apply_solution_settings(
        self,
        settings: SolutionPropertiesSettings,
    ) -> None:
        previous_updating = self._updating_solution_preset_selection
        self._updating_solution_preset_selection = True
        try:
            self._set_combo_value(self.solution_mode_combo, settings.mode)
            self.solution_density_spin.setValue(settings.solution_density)
            self.solute_stoich_edit.setText(settings.solute_stoich)
            self.solvent_stoich_edit.setText(settings.solvent_stoich)
            self.molar_mass_solute_spin.setValue(settings.molar_mass_solute)
            self.molar_mass_solvent_spin.setValue(settings.molar_mass_solvent)
            self.mass_solute_spin.setValue(settings.mass_solute)
            self.mass_solvent_spin.setValue(settings.mass_solvent)
            self.mass_percent_solute_spin.setValue(
                settings.mass_percent_solute
            )
            self.total_mass_solution_spin.setValue(
                settings.total_mass_solution
            )
            self.molarity_spin.setValue(settings.molarity)
            self.molarity_element_edit.setText(settings.molarity_element)
            self._update_solution_mode_widgets()
        finally:
            self._updating_solution_preset_selection = previous_updating

    @staticmethod
    def _set_combo_value(combo: QComboBox, value: str) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _selected_solution_mode(self) -> str:
        return str(self.solution_mode_combo.currentData() or "mass")

    def _current_solution_settings(self) -> SolutionPropertiesSettings:
        return SolutionPropertiesSettings(
            mode=self._selected_solution_mode(),
            solution_density=float(self.solution_density_spin.value()),
            solute_stoich=self.solute_stoich_edit.text().strip(),
            solvent_stoich=self.solvent_stoich_edit.text().strip(),
            molar_mass_solute=float(self.molar_mass_solute_spin.value()),
            molar_mass_solvent=float(self.molar_mass_solvent_spin.value()),
            mass_solute=float(self.mass_solute_spin.value()),
            mass_solvent=float(self.mass_solvent_spin.value()),
            mass_percent_solute=float(self.mass_percent_solute_spin.value()),
            total_mass_solution=float(self.total_mass_solution_spin.value()),
            molarity=float(self.molarity_spin.value()),
            molarity_element=self.molarity_element_edit.text().strip(),
        )

    def current_estimator_settings(self) -> SoluteVolumeFractionSettings:
        mode = self._selected_solution_mode()
        return SoluteVolumeFractionSettings(
            solution=self._current_solution_settings(),
            solute_density_g_per_ml=(
                None
                if mode == "molarity_per_liter"
                else float(self.solute_density_spin.value())
            ),
            solvent_density_g_per_ml=float(self.solvent_density_spin.value()),
        )

    def _on_solution_mode_changed(self) -> None:
        self._update_solution_mode_widgets()

    def _update_solution_mode_widgets(self) -> None:
        selected_mode = self._selected_solution_mode()
        mode_to_index = {
            "mass": 0,
            "mass_percent": 1,
            "molarity_per_liter": 2,
        }
        self.solution_mode_stack.setCurrentIndex(
            mode_to_index.get(selected_mode, 0)
        )
        show_solute_density = selected_mode != "molarity_per_liter"
        show_solvent_density = True
        self.solute_density_label.setVisible(show_solute_density)
        self.solute_density_spin.setVisible(show_solute_density)
        self.solvent_density_label.setVisible(show_solvent_density)
        self.solvent_density_spin.setVisible(show_solvent_density)
        self.solution_mode_hint_label.setText(
            self._estimator_mode_hint_text(selected_mode)
        )

    @staticmethod
    def _estimator_mode_hint_text(mode: str) -> str:
        base = solution_properties_mode_hint_text(mode)
        if mode == "molarity_per_liter":
            return (
                f"{base} For this volume-fraction estimate, molarity mode "
                "uses solvent-density closure: solvent density stays visible "
                "so SAXSShell can estimate V_solvent = m_solvent / "
                "rho_solvent and then V_solute ~= V_solution - V_solvent. "
                "Solute density is hidden in molarity mode."
            )
        return (
            f"{base} In these modes, both pure-component densities remain "
            "visible so the additive solute and solvent volumes can be "
            "reported alongside the fitted fraction estimate."
        )

    def _on_solution_settings_changed(self, *_args: object) -> None:
        if self._updating_solution_preset_selection:
            return
        self._select_solution_preset_name(
            self._matching_solution_preset_name(
                self._current_solution_settings()
            )
        )

    def _select_solution_preset_name(self, preset_name: str | None) -> None:
        target_index = 0
        if preset_name is not None:
            for index in range(self.solution_preset_combo.count()):
                if self.solution_preset_combo.itemData(index) == preset_name:
                    target_index = index
                    break
        previous_updating = self._updating_solution_preset_selection
        self._updating_solution_preset_selection = True
        try:
            self.solution_preset_combo.setCurrentIndex(target_index)
        finally:
            self._updating_solution_preset_selection = previous_updating

    def _matching_solution_preset_name(
        self,
        settings: SolutionPropertiesSettings,
    ) -> str | None:
        for name in ordered_solution_property_preset_names(
            self._solution_presets
        ):
            preset = self._solution_presets.get(name)
            if preset is None:
                continue
            if self._solution_settings_match(settings, preset.settings):
                return name
        return None

    @staticmethod
    def _solution_settings_match(
        left: SolutionPropertiesSettings,
        right: SolutionPropertiesSettings,
    ) -> bool:
        float_fields = (
            "solution_density",
            "molar_mass_solute",
            "molar_mass_solvent",
            "mass_solute",
            "mass_solvent",
            "mass_percent_solute",
            "total_mass_solution",
            "molarity",
        )
        text_fields = (
            "mode",
            "solute_stoich",
            "solvent_stoich",
            "molarity_element",
        )
        for field_name in float_fields:
            if (
                abs(
                    float(getattr(left, field_name))
                    - float(getattr(right, field_name))
                )
                > 1e-9
            ):
                return False
        for field_name in text_fields:
            if str(getattr(left, field_name)) != str(
                getattr(right, field_name)
            ):
                return False
        return True

    def _calculate_estimate(self) -> None:
        try:
            estimate = calculate_solute_volume_fraction_estimate(
                self.current_estimator_settings()
            )
        except Exception as exc:
            message = f"Unable to estimate the volume fraction: {exc}"
            self.set_output_collapsed(False)
            self.output_box.setPlainText(message)
            self._current_estimate = None
            self.estimate_failed.emit(str(exc))
            return
        self._current_estimate = estimate
        self.set_output_collapsed(False)
        self.output_box.setPlainText(estimate.summary_text())
        self.estimate_calculated.emit(estimate)

    def output_is_collapsed(self) -> bool:
        return self.output_box.isHidden()

    def set_output_collapsed(self, collapsed: bool) -> None:
        is_collapsed = bool(collapsed)
        self.output_box.setVisible(not is_collapsed)
        self.output_toggle_button.setArrowType(
            Qt.ArrowType.RightArrow if is_collapsed else Qt.ArrowType.DownArrow
        )
        self.output_toggle_button.setText(
            "Show Output" if is_collapsed else "Hide Output"
        )

    def _toggle_output_collapsed(self) -> None:
        self.set_output_collapsed(not self.output_box.isHidden())


class SoluteVolumeFractionToolWindow(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Volume Fraction Estimate")
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        citation_label = QLabel(
            "Estimate the physical solute or solvent volume fraction from "
            "solution composition plus pure-component densities using a "
            "SAXS-style concentration x specific-volume estimate. "
            f'<a href="{SOLUTE_VOLUME_FRACTION_CITATION_URL}">'
            "Citation: Hajizadeh et al. (2018)</a>"
        )
        citation_label.setWordWrap(True)
        citation_label.setOpenExternalLinks(True)
        layout.addWidget(citation_label)
        self.estimator_widget = SoluteVolumeFractionWidget(self)
        layout.addWidget(self.estimator_widget)
        self.resize(720, 760)


__all__ = [
    "SOLUTE_VOLUME_FRACTION_CITATION_URL",
    "SOLUTE_VOLUME_FRACTION_HELP_TEXT",
    "SoluteVolumeFractionToolWindow",
    "SoluteVolumeFractionWidget",
]
