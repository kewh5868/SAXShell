from __future__ import annotations

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
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
)
from saxshell.saxs.beam_geometry_presets import (
    DEFAULT_BEAM_GEOMETRY_PRESET_NAME,
    BeamGeometryPreset,
    delete_custom_beam_geometry_preset,
    load_beam_geometry_presets,
    ordered_beam_geometry_preset_names,
    save_custom_beam_geometry_preset,
)
from saxshell.saxs.solution_scattering_estimator import (
    BEAM_PROFILE_ITEMS,
    CAPILLARY_GEOMETRY_ITEMS,
    DEFAULT_BEAM_FOOTPRINT_HEIGHT_MM,
    DEFAULT_BEAM_FOOTPRINT_WIDTH_MM,
    DEFAULT_BEAM_PROFILE,
    DEFAULT_CAPILLARY_GEOMETRY,
    DEFAULT_CAPILLARY_SIZE_MM,
    DEFAULT_INCIDENT_ENERGY_KEV,
    BeamGeometrySettings,
    SolutionScatteringEstimate,
    SolutionScatteringEstimatorSettings,
    calculate_solution_scattering_estimate,
    wavelength_angstrom_from_energy_kev,
)

SOLUTION_MODE_ITEMS = (
    ("Masses", "mass"),
    ("Mass Percent", "mass_percent"),
    ("Molarity (per liter)", "molarity_per_liter"),
)

SOLUTE_VOLUME_FRACTION_CITATION_URL = (
    "https://www.nature.com/articles/s41598-018-25355-2"
)
XRAYDB_REFERENCE_URL = "https://scikit-beam.github.io/XrayDB/python.html"
NIST_ATTENUATION_REFERENCE_URL = "https://doi.org/10.6028/NBS.NSRDS.29"
XRF_FORWARD_MODEL_REFERENCE_URL = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC12871215/"
)
SELF_ABSORPTION_REFERENCE_URL = (
    "https://pmc.ncbi.nlm.nih.gov/articles/PMC6608621/"
)

SOLUTION_SCATTERING_HELP_TEXT = (
    "Solution scattering estimators\n\n"
    "This widget combines five related calculations that all start from the "
    "same solution-composition inputs.\n\n"
    "1. Number density\n"
    "   n = N_atoms / V_solution\n"
    "   SAXSShell reports the result in atoms/A^3.\n\n"
    "2. Physical solute-associated volume fraction\n"
    "   phi_phys ~= c_solute * vbar_solute\n"
    "            = (m_solute / V_solution) * (1 / rho_solute)\n"
    "   The physical solvent-associated fraction is reported as\n"
    "   1 - phi_phys. This bulk-density estimate stays in the output\n"
    "   console for reference.\n\n"
    "3. SAXS-effective interaction contrast ratio at energy E\n"
    "   rho_eff(E) = rho_mass * N_A / M * sum_i n_i [Z_i + f'_i(E)]\n"
    "   C(E) = ((rho_eff,solute(E) - rho_eff,solvent(E))\n"
    "          / rho_eff,solvent(E))^2\n"
    "   V_eff,solute(E) = C(E) * V_solute,phys\n"
    "   R_saxs(E) = V_eff,solute(E)\n"
    "             / (V_eff,solute(E) + V_solvent,phys)\n"
    "   This contrast-weighted ratio is the default model-facing\n"
    "   solute fraction for phi_solute / phi_solvent.\n\n"
    "4. Attenuation and solvent contribution scaling\n"
    "   mu(E) ~= c_solute * (mu/rho)_solute(E)\n"
    "          + c_solvent * (mu/rho)_solvent(E)\n"
    "   T(E, L) = exp(-mu(E) * L)\n"
    "   For SAXS transmission geometry, SAXSShell estimates the solvent "
    "   scattering scale factor from the ratio of beam-profile-averaged "
    "   L * exp(-mu * L) terms for the solvent in the sample versus the "
    "   neat solvent reference. If a template only exposes a single\n"
    "   solvent-weight parameter, SAXSShell recommends\n"
    "   w_model = (1 - R_saxs(E)) * w_att.\n\n"
    "5. Fluorescence background proxy\n"
    "   SAXSShell estimates primary fluorescence from element-resolved "
    "   photoelectric attenuation, edge jump-ratio partitioning, "
    "   fluorescence yields, and line branching. A first-order secondary "
    "   fluorescence pass is then added from re-absorption of the primary "
    "   fluorescent lines inside the sample. This is a screening estimate, "
    "   not a full Monte Carlo transport calculation.\n\n"
    "Key assumptions in the current implementation:\n"
    "- the beam profile is uniform\n"
    "- the beam footprint is centered on the capillary\n"
    "- cylindrical capillaries are treated as transmission through a round "
    "cross-section, so the footprint width controls the path-length average\n"
    "- fluorescence escape is modeled with a first-order self-absorption / "
    "secondary-emission approximation\n\n"
    "References:\n"
    f"- Hajizadeh et al. (2018): {SOLUTE_VOLUME_FRACTION_CITATION_URL}\n"
    f"- Hubbell / NIST attenuation reference: {NIST_ATTENUATION_REFERENCE_URL}\n"
    f"- XrayDB Python reference and Elam-based atomic data: {XRAYDB_REFERENCE_URL}\n"
    f"- Roter et al. XRF forward-model discussion and jump-ratio considerations: {XRF_FORWARD_MODEL_REFERENCE_URL}\n"
    f"- Trevorah et al. self-absorption discussion: {SELF_ABSORPTION_REFERENCE_URL}"
)

SOLUTE_VOLUME_FRACTION_HELP_TEXT = SOLUTION_SCATTERING_HELP_TEXT


class BeamEnergyWavelengthDialog(QDialog):
    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        energy_kev: float,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Beam Energy and Wavelength")
        layout = QVBoxLayout(self)
        explanation = QLabel(
            "The X-ray wavelength is computed from the incident energy using "
            "lambda (Å) = 12.3984198433 / E(keV)."
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)

        form_layout = QFormLayout()
        self.energy_value_label = QLabel()
        self.wavelength_value_label = QLabel()
        form_layout.addRow("Energy (keV)", self.energy_value_label)
        form_layout.addRow("Wavelength (Å)", self.wavelength_value_label)
        layout.addLayout(form_layout)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        buttons.accepted.connect(self.close)
        layout.addWidget(buttons)

        self.set_energy_kev(energy_kev)
        self.resize(340, 140)

    def set_energy_kev(self, energy_kev: float) -> None:
        wavelength = wavelength_angstrom_from_energy_kev(energy_kev)
        self.energy_value_label.setText(f"{float(energy_kev):.12g}")
        self.wavelength_value_label.setText(f"{float(wavelength):.12g}")


class SolutionScatteringEstimatorWidget(QWidget):
    estimate_calculated = Signal(object)
    estimate_failed = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        default_number_density: bool = True,
        default_volume_fraction: bool = True,
        default_attenuation: bool = True,
        default_fluorescence: bool = False,
    ) -> None:
        super().__init__(parent)
        self._solution_presets: dict[str, SolutionPropertiesPreset] = {}
        self._beam_presets: dict[str, BeamGeometryPreset] = {}
        self._updating_solution_preset_selection = False
        self._updating_beam_preset_selection = False
        self._current_estimate: SolutionScatteringEstimate | None = None
        self._wavelength_dialog: BeamEnergyWavelengthDialog | None = None
        self._default_number_density = bool(default_number_density)
        self._default_volume_fraction = bool(default_volume_fraction)
        self._default_attenuation = bool(default_attenuation)
        self._default_fluorescence = bool(default_fluorescence)
        self._build_ui()
        self._reload_solution_presets()
        self._reload_beam_presets(
            selected_name=DEFAULT_BEAM_GEOMETRY_PRESET_NAME
        )
        self._select_beam_preset_name(DEFAULT_BEAM_GEOMETRY_PRESET_NAME)
        self._load_selected_beam_preset()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(8)

        controls_widget = QWidget(self)
        layout = QVBoxLayout(controls_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.target_label = QLabel(
            "This calculator is not currently linked to an automatic Prefit "
            "parameter update."
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
            "These presets populate the composition inputs. Density and "
            "beam/capillary settings remain editable below."
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
            minimum=0.0,
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
            minimum=0.0,
            maximum=1_000_000.0,
            step=1.0,
            decimals=6,
            value=0.0,
        )
        self.molar_mass_solute_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        self.molar_mass_solvent_spin = self._new_float_spin(
            minimum=0.0,
            maximum=1_000_000.0,
            step=1.0,
            decimals=6,
            value=0.0,
        )
        self.molar_mass_solvent_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        self.solute_density_spin = self._new_float_spin(
            minimum=0.0,
            maximum=100.0,
            step=0.01,
            decimals=6,
            value=1.0,
        )
        self.solute_density_spin.valueChanged.connect(
            self._on_solution_settings_changed
        )

        self.solvent_density_spin = self._new_float_spin(
            minimum=0.0,
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
            minimum=0.0,
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
            minimum=0.0,
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
            minimum=0.0,
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
            minimum=0.0,
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
            minimum=0.0,
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

        calculations_group = QGroupBox("Calculations")
        calculations_layout = QHBoxLayout(calculations_group)
        self.calculate_number_density_checkbox = QCheckBox("Number Density")
        self.calculate_number_density_checkbox.setChecked(
            self._default_number_density
        )
        self.calculate_volume_fraction_checkbox = QCheckBox(
            "Solute Volume Fraction"
        )
        self.calculate_volume_fraction_checkbox.setChecked(
            self._default_volume_fraction
        )
        self.calculate_attenuation_checkbox = QCheckBox(
            "Solvent Scattering Contribution"
        )
        self.calculate_attenuation_checkbox.setChecked(
            self._default_attenuation
        )
        self.calculate_fluorescence_checkbox = QCheckBox(
            "Sample Fluorescence Yield"
        )
        self.calculate_fluorescence_checkbox.setChecked(
            self._default_fluorescence
        )
        for checkbox in (
            self.calculate_number_density_checkbox,
            self.calculate_volume_fraction_checkbox,
            self.calculate_attenuation_checkbox,
            self.calculate_fluorescence_checkbox,
        ):
            checkbox.toggled.connect(self._on_solution_settings_changed)
            calculations_layout.addWidget(checkbox)
        calculations_layout.addStretch(1)
        layout.addWidget(calculations_group)

        beam_preset_group = QGroupBox("Beam and Capillary Presets")
        beam_preset_layout = QVBoxLayout(beam_preset_group)
        beam_preset_row = QHBoxLayout()
        self.beam_preset_combo = QComboBox()
        beam_preset_row.addWidget(self.beam_preset_combo, stretch=1)
        self.load_beam_preset_button = QPushButton("Load")
        self.load_beam_preset_button.clicked.connect(
            self._load_selected_beam_preset
        )
        beam_preset_row.addWidget(self.load_beam_preset_button)
        self.save_beam_preset_button = QPushButton("Save Current")
        self.save_beam_preset_button.clicked.connect(
            self._save_current_beam_preset
        )
        beam_preset_row.addWidget(self.save_beam_preset_button)
        self.delete_beam_preset_button = QPushButton("Delete")
        self.delete_beam_preset_button.clicked.connect(
            self._delete_selected_beam_preset
        )
        beam_preset_row.addWidget(self.delete_beam_preset_button)
        beam_preset_layout.addLayout(beam_preset_row)
        beam_preset_hint = QLabel(
            "Presets include beam energy, capillary size, geometry, beam "
            "profile, and beam footprint. Custom presets are saved to a "
            "JSON file for reuse."
        )
        beam_preset_hint.setWordWrap(True)
        beam_preset_layout.addWidget(beam_preset_hint)
        layout.addWidget(beam_preset_group)

        beam_group = QGroupBox("Beam and Capillary")
        beam_layout = QGridLayout(beam_group)
        beam_layout.setColumnStretch(1, 1)
        beam_layout.setColumnStretch(3, 1)
        self.incident_energy_spin = self._new_float_spin(
            minimum=0.0,
            maximum=200.0,
            step=0.1,
            decimals=6,
            value=DEFAULT_INCIDENT_ENERGY_KEV,
        )
        self.incident_energy_spin.valueChanged.connect(
            self._on_beam_settings_changed
        )
        self.incident_energy_spin.valueChanged.connect(
            self._update_wavelength_dialog_energy
        )
        beam_layout.addWidget(QLabel("X-ray energy (keV)"), 0, 0)
        incident_energy_row = QHBoxLayout()
        incident_energy_row.addWidget(self.incident_energy_spin, stretch=1)
        self.show_wavelength_button = QPushButton("Wavelength...")
        self.show_wavelength_button.clicked.connect(
            self._show_wavelength_dialog
        )
        incident_energy_row.addWidget(self.show_wavelength_button)
        beam_layout.addLayout(incident_energy_row, 0, 1)
        self.capillary_size_spin = self._new_float_spin(
            minimum=0.0,
            maximum=100.0,
            step=0.1,
            decimals=6,
            value=DEFAULT_CAPILLARY_SIZE_MM,
        )
        self.capillary_size_spin.valueChanged.connect(
            self._on_beam_settings_changed
        )
        beam_layout.addWidget(QLabel("Capillary size (mm)"), 0, 2)
        beam_layout.addWidget(self.capillary_size_spin, 0, 3)
        self.capillary_geometry_combo = QComboBox()
        for label, value in CAPILLARY_GEOMETRY_ITEMS:
            self.capillary_geometry_combo.addItem(label, userData=value)
        self.capillary_geometry_combo.currentIndexChanged.connect(
            self._on_beam_settings_changed
        )
        beam_layout.addWidget(QLabel("Capillary geometry"), 1, 0)
        beam_layout.addWidget(self.capillary_geometry_combo, 1, 1)
        self.beam_profile_combo = QComboBox()
        for label, value in BEAM_PROFILE_ITEMS:
            self.beam_profile_combo.addItem(label, userData=value)
        self.beam_profile_combo.currentIndexChanged.connect(
            self._on_beam_settings_changed
        )
        beam_layout.addWidget(QLabel("Beam profile"), 1, 2)
        beam_layout.addWidget(self.beam_profile_combo, 1, 3)
        self.beam_footprint_width_spin = self._new_float_spin(
            minimum=0.0,
            maximum=100.0,
            step=0.1,
            decimals=6,
            value=DEFAULT_BEAM_FOOTPRINT_WIDTH_MM,
        )
        self.beam_footprint_width_spin.valueChanged.connect(
            self._on_beam_settings_changed
        )
        beam_layout.addWidget(QLabel("Beam footprint width (mm)"), 2, 0)
        beam_layout.addWidget(self.beam_footprint_width_spin, 2, 1)
        self.beam_footprint_height_spin = self._new_float_spin(
            minimum=0.0,
            maximum=100.0,
            step=0.1,
            decimals=6,
            value=DEFAULT_BEAM_FOOTPRINT_HEIGHT_MM,
        )
        self.beam_footprint_height_spin.valueChanged.connect(
            self._on_beam_settings_changed
        )
        beam_layout.addWidget(QLabel("Beam footprint height (mm)"), 2, 2)
        beam_layout.addWidget(self.beam_footprint_height_spin, 2, 3)
        layout.addWidget(beam_group)

        button_row = QHBoxLayout()
        self.calculate_button = QPushButton("Run Selected Calculations")
        self.calculate_button.clicked.connect(self._calculate_estimate)
        button_row.addWidget(self.calculate_button)
        button_row.addStretch(1)
        self.output_toggle_button = QToolButton()
        self.output_toggle_button.setToolButtonStyle(
            Qt.ToolButtonStyle.ToolButtonTextBesideIcon
        )
        self.output_toggle_button.setAutoRaise(True)
        self.output_toggle_button.clicked.connect(
            self._toggle_output_collapsed
        )
        button_row.addWidget(self.output_toggle_button)
        layout.addLayout(button_row)

        self.output_panel = QWidget(controls_widget)
        output_layout = QVBoxLayout(self.output_panel)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.setSpacing(8)
        self.output_title_label = QLabel("Calculation Output")
        output_layout.addWidget(self.output_title_label)
        self.output_box = QTextEdit()
        self.output_box.setReadOnly(True)
        self.output_box.setMinimumHeight(220)
        self.output_box.setPlaceholderText(
            "Run the selected calculations to populate this pane."
        )
        output_layout.addWidget(self.output_box, stretch=1)
        layout.addWidget(self.output_panel)
        layout.addStretch(1)

        controls_scroll = QScrollArea(self)
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        controls_scroll.setWidget(controls_widget)
        root_layout.addWidget(controls_scroll)

        self.set_output_collapsed(True)
        self._update_solution_mode_widgets()

    @staticmethod
    def _new_float_spin(
        *,
        minimum: float,
        maximum: float,
        step: float,
        decimals: int,
        value: float,
    ) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(float(minimum), float(maximum))
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    def set_target_parameter(
        self,
        parameter_name: str | None,
        fraction_kind: str | None,
        solvent_weight_parameter: str | None = None,
    ) -> None:
        messages: list[str] = []
        if parameter_name and fraction_kind:
            label = "solute" if fraction_kind == "solute" else "solvent"
            messages.append(
                f"{parameter_name} ({label} SAXS-effective interaction fraction)"
            )
        if solvent_weight_parameter:
            if parameter_name and fraction_kind:
                messages.append(
                    f"{solvent_weight_parameter} (attenuation solvent scale)"
                )
            else:
                messages.append(
                    f"{solvent_weight_parameter} (combined solvent background multiplier)"
                )
        if messages:
            self.target_label.setText(
                "Active Prefit targets: " + "; ".join(messages) + "."
            )
        else:
            self.target_label.setText(
                "This calculator is not currently linked to an automatic "
                "Prefit parameter update."
            )

    def set_calculation_selection(
        self,
        *,
        number_density: bool | None = None,
        volume_fraction: bool | None = None,
        attenuation: bool | None = None,
        fluorescence: bool | None = None,
    ) -> None:
        if number_density is not None:
            self.calculate_number_density_checkbox.setChecked(
                bool(number_density)
            )
        if volume_fraction is not None:
            self.calculate_volume_fraction_checkbox.setChecked(
                bool(volume_fraction)
            )
        if attenuation is not None:
            self.calculate_attenuation_checkbox.setChecked(bool(attenuation))
        if fluorescence is not None:
            self.calculate_fluorescence_checkbox.setChecked(bool(fluorescence))

    def append_application_note(self, message: str) -> None:
        self.set_output_collapsed(False)
        text = self.output_box.toPlainText().strip()
        if not text:
            self.output_box.setPlainText(message.strip())
            return
        self.output_box.setPlainText(text + "\n\n" + message.strip())

    def current_estimate(self) -> SolutionScatteringEstimate | None:
        return self._current_estimate

    def _reload_solution_presets(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
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

    def _reload_beam_presets(
        self,
        *,
        selected_name: str | None = None,
    ) -> None:
        previous_name = selected_name or self._selected_beam_preset_name()
        self._beam_presets = load_beam_geometry_presets()
        self.beam_preset_combo.blockSignals(True)
        self.beam_preset_combo.clear()
        self.beam_preset_combo.addItem("Current values", None)
        selected_index = 0
        for index, name in enumerate(
            ordered_beam_geometry_preset_names(self._beam_presets),
            start=1,
        ):
            preset = self._beam_presets[name]
            label = f"{name} (Built-in)" if preset.builtin else name
            self.beam_preset_combo.addItem(label, name)
            if name == previous_name:
                selected_index = index
        self.beam_preset_combo.setCurrentIndex(selected_index)
        self.beam_preset_combo.blockSignals(False)

    def _selected_beam_preset_name(self) -> str | None:
        payload = self.beam_preset_combo.currentData()
        if payload is None:
            return None
        return str(payload)

    def _load_selected_beam_preset(self) -> None:
        preset_name = self._selected_beam_preset_name()
        if preset_name is None:
            self.append_application_note("Select a beam preset to load.")
            return
        preset = self._beam_presets.get(preset_name)
        if preset is None:
            self.append_application_note(f"Unknown beam preset: {preset_name}")
            return
        self._apply_beam_preset(preset)
        self._select_beam_preset_name(preset.name)

    def _save_current_beam_preset(self) -> None:
        suggested_name = self._selected_beam_preset_name() or ""
        preset_name, accepted = QInputDialog.getText(
            self,
            "Save Beam Preset",
            (
                "Enter a name for this beam/capillary preset.\n"
                "The preset stores energy, capillary size, geometry, beam "
                "profile, and footprint."
            ),
            text=suggested_name,
        )
        if not accepted:
            return
        normalized_name = preset_name.strip()
        if not normalized_name:
            self.append_application_note("Beam preset name cannot be empty.")
            return
        preset = BeamGeometryPreset(
            name=normalized_name,
            beam=self._current_beam_settings(),
        )
        save_custom_beam_geometry_preset(preset)
        self._reload_beam_presets(selected_name=normalized_name)
        self._select_beam_preset_name(normalized_name)
        self.append_application_note(f"Saved beam preset {normalized_name!r}.")

    def _delete_selected_beam_preset(self) -> None:
        preset_name = self._selected_beam_preset_name()
        if preset_name is None:
            self.append_application_note(
                "Select a custom beam preset to delete."
            )
            return
        preset = self._beam_presets.get(preset_name)
        if preset is None:
            self.append_application_note(f"Unknown beam preset: {preset_name}")
            return
        if preset.builtin:
            QMessageBox.information(
                self,
                "Built-in beam preset",
                (
                    "Built-in beam presets cannot be deleted. Save a custom "
                    "preset with the same name if you want to override it."
                ),
            )
            return
        if not delete_custom_beam_geometry_preset(preset_name):
            self.append_application_note(
                f"No custom beam preset named {preset_name!r} was found."
            )
            return
        self._reload_beam_presets(
            selected_name=DEFAULT_BEAM_GEOMETRY_PRESET_NAME
        )
        self.append_application_note(f"Deleted beam preset {preset_name!r}.")

    def _apply_beam_preset(self, preset: BeamGeometryPreset) -> None:
        self._apply_beam_settings(preset.beam)

    def _apply_beam_settings(self, settings: BeamGeometrySettings) -> None:
        previous_updating = self._updating_beam_preset_selection
        self._updating_beam_preset_selection = True
        try:
            self.incident_energy_spin.setValue(settings.incident_energy_kev)
            self.capillary_size_spin.setValue(settings.capillary_size_mm)
            self._set_combo_value(
                self.capillary_geometry_combo,
                settings.capillary_geometry,
            )
            self._set_combo_value(
                self.beam_profile_combo,
                settings.beam_profile,
            )
            self.beam_footprint_width_spin.setValue(
                settings.beam_footprint_width_mm
            )
            self.beam_footprint_height_spin.setValue(
                settings.beam_footprint_height_mm
            )
        finally:
            self._updating_beam_preset_selection = previous_updating
        self._update_wavelength_dialog_energy()

    def _current_beam_settings(self) -> BeamGeometrySettings:
        return BeamGeometrySettings(
            incident_energy_kev=float(self.incident_energy_spin.value()),
            capillary_size_mm=float(self.capillary_size_spin.value()),
            capillary_geometry=str(
                self.capillary_geometry_combo.currentData()
                or DEFAULT_CAPILLARY_GEOMETRY
            ),
            beam_profile=str(
                self.beam_profile_combo.currentData() or DEFAULT_BEAM_PROFILE
            ),
            beam_footprint_width_mm=float(
                self.beam_footprint_width_spin.value()
            ),
            beam_footprint_height_mm=float(
                self.beam_footprint_height_spin.value()
            ),
        )

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

    def current_estimator_settings(
        self,
    ) -> SolutionScatteringEstimatorSettings:
        mode = self._selected_solution_mode()
        return SolutionScatteringEstimatorSettings(
            solution=self._current_solution_settings(),
            solute_density_g_per_ml=(
                None
                if mode == "molarity_per_liter"
                else float(self.solute_density_spin.value())
            ),
            solvent_density_g_per_ml=float(self.solvent_density_spin.value()),
            calculate_number_density=(
                self.calculate_number_density_checkbox.isChecked()
            ),
            calculate_solute_volume_fraction=(
                self.calculate_volume_fraction_checkbox.isChecked()
            ),
            calculate_solvent_scattering_contribution=(
                self.calculate_attenuation_checkbox.isChecked()
            ),
            calculate_sample_fluorescence_yield=(
                self.calculate_fluorescence_checkbox.isChecked()
            ),
            beam=self._current_beam_settings(),
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
        self.solute_density_label.setVisible(show_solute_density)
        self.solute_density_spin.setVisible(show_solute_density)
        self.solvent_density_label.setVisible(True)
        self.solvent_density_spin.setVisible(True)
        self.solution_mode_hint_label.setText(
            self._estimator_mode_hint_text(selected_mode)
        )

    @staticmethod
    def _estimator_mode_hint_text(mode: str) -> str:
        base = solution_properties_mode_hint_text(mode)
        if mode == "molarity_per_liter":
            return (
                f"{base} In molarity mode the solute density is hidden, but "
                "the solvent density remains active for attenuation and "
                "volume-closure calculations."
            )
        return (
            f"{base} In these modes, both component densities remain "
            "available for the volume-fraction and attenuation estimates."
        )

    def _on_solution_settings_changed(self, *_args: object) -> None:
        if self._updating_solution_preset_selection:
            return
        self._select_solution_preset_name(
            self._matching_solution_preset_name(
                self._current_solution_settings()
            )
        )

    def _on_beam_settings_changed(self, *_args: object) -> None:
        if self._updating_beam_preset_selection:
            return
        self._select_beam_preset_name(
            self._matching_beam_preset_name(self._current_beam_settings())
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

    def _select_beam_preset_name(self, preset_name: str | None) -> None:
        target_index = 0
        if preset_name is not None:
            for index in range(self.beam_preset_combo.count()):
                if self.beam_preset_combo.itemData(index) == preset_name:
                    target_index = index
                    break
        previous_updating = self._updating_beam_preset_selection
        self._updating_beam_preset_selection = True
        try:
            self.beam_preset_combo.setCurrentIndex(target_index)
        finally:
            self._updating_beam_preset_selection = previous_updating

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

    def _matching_beam_preset_name(
        self,
        settings: BeamGeometrySettings,
    ) -> str | None:
        for name in ordered_beam_geometry_preset_names(self._beam_presets):
            preset = self._beam_presets.get(name)
            if preset is None:
                continue
            if self._beam_settings_match(settings, preset.beam):
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

    @staticmethod
    def _beam_settings_match(
        left: BeamGeometrySettings,
        right: BeamGeometrySettings,
    ) -> bool:
        float_fields = (
            "incident_energy_kev",
            "capillary_size_mm",
            "beam_footprint_width_mm",
            "beam_footprint_height_mm",
        )
        text_fields = (
            "capillary_geometry",
            "beam_profile",
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

    def _show_wavelength_dialog(self) -> None:
        if self._wavelength_dialog is None:
            dialog = BeamEnergyWavelengthDialog(
                self,
                energy_kev=float(self.incident_energy_spin.value()),
            )
            dialog.destroyed.connect(self._clear_wavelength_dialog)
            self._wavelength_dialog = dialog
        self._update_wavelength_dialog_energy()
        self._wavelength_dialog.show()
        self._wavelength_dialog.raise_()
        self._wavelength_dialog.activateWindow()

    def _update_wavelength_dialog_energy(self, *_args: object) -> None:
        if self._wavelength_dialog is None:
            return
        self._wavelength_dialog.set_energy_kev(
            float(self.incident_energy_spin.value())
        )

    def _clear_wavelength_dialog(self, *_args: object) -> None:
        self._wavelength_dialog = None

    def _calculate_estimate(self) -> None:
        settings = self.current_estimator_settings()
        if not (
            settings.calculate_number_density
            or settings.calculate_solute_volume_fraction
            or settings.calculate_solvent_scattering_contribution
            or settings.calculate_sample_fluorescence_yield
        ):
            message = "Select at least one calculation before running."
            self.set_output_collapsed(False)
            self.output_box.setPlainText(message)
            self._current_estimate = None
            self.estimate_failed.emit(message)
            return
        try:
            estimate = calculate_solution_scattering_estimate(settings)
        except Exception as exc:
            message = f"Unable to run the selected calculations: {exc}"
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
        return self.output_panel.isHidden()

    def set_output_collapsed(self, collapsed: bool) -> None:
        is_collapsed = bool(collapsed)
        if is_collapsed:
            self.output_panel.hide()
        else:
            self.output_panel.show()
        self.output_toggle_button.setArrowType(
            Qt.ArrowType.DownArrow if is_collapsed else Qt.ArrowType.UpArrow
        )
        self.output_toggle_button.setText(
            "Show Output" if is_collapsed else "Hide Output"
        )

    def _toggle_output_collapsed(self) -> None:
        self.set_output_collapsed(not self.output_is_collapsed())


class SolutionScatteringToolWindow(QWidget):
    def __init__(
        self,
        title: str,
        subtitle_html: str,
        *,
        default_number_density: bool,
        default_volume_fraction: bool,
        default_attenuation: bool,
        default_fluorescence: bool,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        citation_label = QLabel(subtitle_html)
        citation_label.setWordWrap(True)
        citation_label.setOpenExternalLinks(True)
        layout.addWidget(citation_label)
        self.estimator_widget = SolutionScatteringEstimatorWidget(
            self,
            default_number_density=default_number_density,
            default_volume_fraction=default_volume_fraction,
            default_attenuation=default_attenuation,
            default_fluorescence=default_fluorescence,
        )
        layout.addWidget(self.estimator_widget)
        self.resize(self._default_window_size())

    @staticmethod
    def _default_window_size() -> QSize:
        app = QApplication.instance()
        screen = app.primaryScreen() if app is not None else None
        if screen is None:
            return QSize(1100, 720)

        available = screen.availableGeometry()
        target_width = min(1120, max(960, available.width() - 180))
        target_height = min(760, max(640, available.height() - 180))
        target_width = min(target_width, max(760, available.width() - 48))
        target_height = min(target_height, max(560, available.height() - 72))
        return QSize(target_width, target_height)


class SoluteVolumeFractionToolWindow(SolutionScatteringToolWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "Volume Fraction Estimate",
            (
                "Estimate solution-scattering quantities from composition, "
                "density, and beam/capillary settings. "
                f'<a href="{SOLUTE_VOLUME_FRACTION_CITATION_URL}">'
                "Hajizadeh et al. (2018)</a>"
            ),
            default_number_density=True,
            default_volume_fraction=True,
            default_attenuation=False,
            default_fluorescence=False,
            parent=parent,
        )


class AttenuationEstimateToolWindow(SolutionScatteringToolWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "Attenuation Estimate",
            (
                "Estimate attenuation and the solvent scattering scale factor "
                "used to map a neat-solvent reference onto the solvent "
                "contribution inside the sample. "
                f'<a href="{NIST_ATTENUATION_REFERENCE_URL}">NIST</a>, '
                f'<a href="{XRAYDB_REFERENCE_URL}">XrayDB</a>'
            ),
            default_number_density=False,
            default_volume_fraction=True,
            default_attenuation=True,
            default_fluorescence=False,
            parent=parent,
        )


class FluorescenceEstimateToolWindow(SolutionScatteringToolWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "Fluorescence Estimate",
            (
                "Estimate fluorescence background tendencies from the sample "
                "composition, attenuation, and beam energy using a "
                "first-order primary plus secondary fluorescence model. "
                f'<a href="{XRF_FORWARD_MODEL_REFERENCE_URL}">XRF forward model</a>, '
                f'<a href="{SELF_ABSORPTION_REFERENCE_URL}">self-absorption</a>'
            ),
            default_number_density=False,
            default_volume_fraction=False,
            default_attenuation=False,
            default_fluorescence=True,
            parent=parent,
        )


class NumberDensityEstimateToolWindow(SolutionScatteringToolWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(
            "Number Density Estimate",
            (
                "Estimate the total atomic number density of the solution "
                "from the composition inputs and report it in atoms/A^3."
            ),
            default_number_density=True,
            default_volume_fraction=False,
            default_attenuation=False,
            default_fluorescence=False,
            parent=parent,
        )


SoluteVolumeFractionWidget = SolutionScatteringEstimatorWidget

__all__ = [
    "AttenuationEstimateToolWindow",
    "FluorescenceEstimateToolWindow",
    "NumberDensityEstimateToolWindow",
    "SOLUTE_VOLUME_FRACTION_CITATION_URL",
    "SOLUTE_VOLUME_FRACTION_HELP_TEXT",
    "SOLUTION_SCATTERING_HELP_TEXT",
    "SolutionScatteringEstimatorWidget",
    "SolutionScatteringToolWindow",
    "SoluteVolumeFractionToolWindow",
    "SoluteVolumeFractionWidget",
]
