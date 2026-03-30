from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import xraydb

from saxshell.fullrmc.solution_properties import (
    SolutionPropertiesResult,
    SolutionPropertiesSettings,
    calculate_solution_properties,
)
from saxshell.saxs.solute_volume_fraction import (
    SoluteVolumeFractionEstimate,
    SoluteVolumeFractionSettings,
    calculate_solute_volume_fraction_estimate,
)

DEFAULT_INCIDENT_ENERGY_KEV = 17.0
DEFAULT_CAPILLARY_SIZE_MM = 1.0
DEFAULT_BEAM_FOOTPRINT_WIDTH_MM = 0.4
DEFAULT_BEAM_FOOTPRINT_HEIGHT_MM = 0.4
DEFAULT_CAPILLARY_GEOMETRY = "cylindrical"
DEFAULT_BEAM_PROFILE = "uniform"
PATH_SAMPLE_COUNT = 801

CAPILLARY_GEOMETRY_ITEMS = (
    ("Cylindrical", "cylindrical"),
    ("Flat plate", "flat_plate"),
)
BEAM_PROFILE_ITEMS = (("Uniform", "uniform"),)
EDGE_LINE_FAMILIES = {
    "K": ("Ka", "Kb"),
    "L3": ("La", "Lb"),
    "L2": ("Lb", "Lg"),
    "L1": ("Lb", "Lg"),
    "M5": ("Ma",),
    "M4": ("Mb",),
}
MINIMUM_LINE_CONTRIBUTION = 1e-15
ENERGY_WAVELENGTH_KEV_ANGSTROM = 12.398419843320026
AVOGADRO_NUMBER = 6.02214076e23


def _format_number(value: float, digits: int = 6) -> str:
    return f"{float(value):.{digits}g}"


def _format_fraction(value: float) -> str:
    return f"{float(value):.6f}"


def _format_scattering_density(value: float) -> str:
    return f"{float(value):.6e}"


def wavelength_angstrom_from_energy_kev(energy_kev: float) -> float:
    validated_energy = _validate_positive(energy_kev, "Incident energy")
    return float(ENERGY_WAVELENGTH_KEV_ANGSTROM / validated_energy)


@dataclass(slots=True)
class BeamGeometrySettings:
    incident_energy_kev: float = DEFAULT_INCIDENT_ENERGY_KEV
    capillary_size_mm: float = DEFAULT_CAPILLARY_SIZE_MM
    capillary_geometry: str = DEFAULT_CAPILLARY_GEOMETRY
    beam_profile: str = DEFAULT_BEAM_PROFILE
    beam_footprint_width_mm: float = DEFAULT_BEAM_FOOTPRINT_WIDTH_MM
    beam_footprint_height_mm: float = DEFAULT_BEAM_FOOTPRINT_HEIGHT_MM


@dataclass(slots=True)
class SolutionScatteringEstimatorSettings:
    solution: SolutionPropertiesSettings
    solute_density_g_per_ml: float | None = 1.0
    solvent_density_g_per_ml: float | None = 1.0
    calculate_number_density: bool = True
    calculate_solute_volume_fraction: bool = True
    calculate_solvent_scattering_contribution: bool = True
    calculate_sample_fluorescence_yield: bool = False
    beam: BeamGeometrySettings = field(default_factory=BeamGeometrySettings)


@dataclass(slots=True)
class NumberDensityEstimate:
    number_density_cm3: float
    number_density_a3: float
    total_atoms: float
    element_ratio_string: str

    def summary_text(self) -> str:
        lines = [
            "Number density estimate",
            (
                "Atomic number density: "
                f"{_format_number(self.number_density_a3)} atoms/A^3"
            ),
            (
                "Atomic number density: "
                f"{_format_number(self.number_density_cm3)} atoms/cm^3"
            ),
            (
                "Total atoms in source solution: "
                f"{_format_number(self.total_atoms)}"
            ),
        ]
        if self.element_ratio_string:
            lines.append(
                "Integer ratio of elements: " f"{self.element_ratio_string}"
            )
        return "\n".join(lines)


@dataclass(slots=True)
class SAXSInteractionEstimate:
    incident_energy_kev: float
    wavelength_angstrom: float
    solute_formula: str
    solvent_formula: str
    solute_density_g_per_cm3: float
    solvent_density_g_per_cm3: float
    solute_effective_scattering_density_electrons_per_cm3: float
    solvent_effective_scattering_density_electrons_per_cm3: float
    contrast_scattering_density_electrons_per_cm3: float
    physical_solute_associated_volume_cm3: float
    physical_solvent_associated_volume_cm3: float
    physical_solute_associated_volume_fraction: float
    physical_solvent_associated_volume_fraction: float
    contrast_weight_factor: float
    effective_solute_interaction_volume_cm3: float
    effective_solvent_background_volume_cm3: float
    saxs_effective_solute_interaction_ratio: float
    saxs_effective_solvent_background_ratio: float

    def summary_text(self) -> str:
        lines = [
            "SAXS-effective interaction contrast estimate",
            f"Incident energy: {self.incident_energy_kev:.6g} keV",
            f"Wavelength: {self.wavelength_angstrom:.6g} A",
            (
                "Solute effective scattering density: "
                f"{_format_scattering_density(self.solute_effective_scattering_density_electrons_per_cm3)} "
                "electrons/cm^3"
            ),
            (
                "Solvent effective scattering density: "
                f"{_format_scattering_density(self.solvent_effective_scattering_density_electrons_per_cm3)} "
                "electrons/cm^3"
            ),
            (
                "Energy-dependent solute-solvent contrast: "
                f"{_format_scattering_density(self.contrast_scattering_density_electrons_per_cm3)} "
                "electrons/cm^3"
            ),
            (
                "Physical solute-associated volume fraction: "
                f"{_format_fraction(self.physical_solute_associated_volume_fraction)}"
            ),
            (
                "Physical solvent-associated volume fraction: "
                f"{_format_fraction(self.physical_solvent_associated_volume_fraction)}"
            ),
            (
                "Contrast weight factor: "
                f"{_format_number(self.contrast_weight_factor)}"
            ),
            (
                "Effective solute interaction volume: "
                f"{self.effective_solute_interaction_volume_cm3:.6f} cm^3"
            ),
            (
                "SAXS-effective solute interaction ratio: "
                f"{_format_fraction(self.saxs_effective_solute_interaction_ratio)}"
            ),
            (
                "SAXS-effective solvent background ratio: "
                f"{_format_fraction(self.saxs_effective_solvent_background_ratio)}"
            ),
            "",
            "Interpretation:",
            "The physical fraction above is still reported for transparency, "
            "but the model-facing phi_solute / phi_solvent default is based "
            "on the contrast-weighted SAXS interaction ratio at the selected "
            "energy.",
            "SAXS-effective ratio model:",
            ("rho_eff(E) = rho_mass * N_A / M * sum_i n_i [Z_i + f'_i(E)]"),
            (
                "C(E) = ((rho_eff,solute(E) - rho_eff,solvent(E)) / "
                "rho_eff,solvent(E))^2"
            ),
            "V_eff,solute(E) = C(E) * V_solute,phys",
            (
                "R_saxs(E) = V_eff,solute(E) / "
                "(V_eff,solute(E) + V_solvent,phys)"
            ),
            "Prefit uses R_saxs(E) as the default model solute fraction when "
            "the active template exposes phi_solute / phi_solvent.",
        ]
        return "\n".join(lines)


@dataclass(slots=True)
class AttenuationEstimate:
    incident_energy_kev: float
    capillary_geometry: str
    sample_average_path_length_mm: float
    sample_linear_attenuation_inv_cm: float
    solute_linear_attenuation_inv_cm: float
    sample_solvent_linear_attenuation_inv_cm: float
    neat_solvent_linear_attenuation_inv_cm: float
    sample_transmission: float
    solute_only_transmission: float
    sample_solvent_only_transmission: float
    neat_solvent_transmission: float
    sample_scattering_weight: float
    neat_solvent_scattering_weight: float
    solvent_mass_concentration_g_per_cm3: float
    neat_solvent_density_g_per_cm3: float
    solvent_scattering_scale_factor: float
    neat_solvent_to_sample_ratio: float | None

    def summary_text(self) -> str:
        lines = [
            "Attenuation and solvent contribution estimate",
            f"Incident energy: {self.incident_energy_kev:.6g} keV",
            f"Capillary geometry: {self.capillary_geometry}",
            (
                "Average illuminated path length: "
                f"{self.sample_average_path_length_mm:.4f} mm"
            ),
            (
                "Sample total linear attenuation: "
                f"{_format_number(self.sample_linear_attenuation_inv_cm)} 1/cm"
            ),
            (
                "Solute-only linear attenuation in sample: "
                f"{_format_number(self.solute_linear_attenuation_inv_cm)} 1/cm"
            ),
            (
                "Solvent-only linear attenuation in sample: "
                f"{_format_number(self.sample_solvent_linear_attenuation_inv_cm)} 1/cm"
            ),
            (
                "Neat-solvent linear attenuation: "
                f"{_format_number(self.neat_solvent_linear_attenuation_inv_cm)} 1/cm"
            ),
            (
                "Sample total transmission: "
                f"{_format_fraction(self.sample_transmission)}"
            ),
            (
                "Solute-only transmission in sample: "
                f"{_format_fraction(self.solute_only_transmission)}"
            ),
            (
                "Solvent-only transmission in sample: "
                f"{_format_fraction(self.sample_solvent_only_transmission)}"
            ),
            (
                "Neat-solvent transmission: "
                f"{_format_fraction(self.neat_solvent_transmission)}"
            ),
            (
                "Sample solvent mass concentration: "
                f"{_format_number(self.solvent_mass_concentration_g_per_cm3)} g/cm^3"
            ),
            (
                "Neat-solvent density: "
                f"{_format_number(self.neat_solvent_density_g_per_cm3)} g/cm^3"
            ),
            (
                "Recommended solvent scattering scale factor: "
                f"{_format_fraction(self.solvent_scattering_scale_factor)}"
            ),
        ]
        if self.neat_solvent_to_sample_ratio is not None:
            lines.append(
                "Neat-solvent / sample-solvent intensity ratio: "
                f"{_format_number(self.neat_solvent_to_sample_ratio)}"
            )
        lines.extend(
            [
                "",
                "Interpretation:",
                "The solvent scale factor is the estimated proportionality "
                "needed to reduce the neat-solvent trace to the solvent "
                "contribution present inside the measured sample.",
            ]
        )
        return "\n".join(lines)


@dataclass(slots=True)
class FluorescenceLineEstimate:
    element: str
    edge: str
    family: str
    line_energy_ev: float
    primary_detected_yield: float
    secondary_detected_yield: float

    @property
    def total_detected_yield(self) -> float:
        return float(
            self.primary_detected_yield + self.secondary_detected_yield
        )


@dataclass(slots=True)
class FluorescenceEstimate:
    incident_energy_kev: float
    capillary_geometry: str
    total_primary_detected_yield: float
    total_secondary_detected_yield: float
    line_estimates: list[FluorescenceLineEstimate]

    def summary_text(self) -> str:
        lines = [
            "Fluorescence yield estimate",
            f"Incident energy: {self.incident_energy_kev:.6g} keV",
            f"Capillary geometry: {self.capillary_geometry}",
            (
                "Detected primary fluorescence proxy: "
                f"{_format_number(self.total_primary_detected_yield)}"
            ),
            (
                "Detected secondary fluorescence proxy: "
                f"{_format_number(self.total_secondary_detected_yield)}"
            ),
            (
                "Detected total fluorescence proxy: "
                f"{_format_number(self.total_primary_detected_yield + self.total_secondary_detected_yield)}"
            ),
            "",
            "Strongest line families:",
        ]
        for estimate in sorted(
            self.line_estimates,
            key=lambda item: item.total_detected_yield,
            reverse=True,
        )[:12]:
            lines.append(
                f"  {estimate.element} {estimate.family} "
                f"({estimate.edge}, {estimate.line_energy_ev:.1f} eV): "
                f"primary={_format_number(estimate.primary_detected_yield)}, "
                f"secondary={_format_number(estimate.secondary_detected_yield)}, "
                f"total={_format_number(estimate.total_detected_yield)}"
            )
        lines.extend(
            [
                "",
                "Interpretation:",
                "This is a first-order fluorescence-background proxy. The "
                "primary term uses edge jump-ratio partitioning together "
                "with Elam-style fluorescence yields and line branching. The "
                "secondary term is a single re-absorption and re-emission "
                "pass, not a full Monte Carlo transport model.",
            ]
        )
        return "\n".join(lines)


@dataclass(slots=True)
class SolutionScatteringEstimate:
    settings: SolutionScatteringEstimatorSettings
    solution_result: SolutionPropertiesResult
    number_density_estimate: NumberDensityEstimate | None = None
    volume_fraction_estimate: SoluteVolumeFractionEstimate | None = None
    interaction_contrast_estimate: SAXSInteractionEstimate | None = None
    attenuation_estimate: AttenuationEstimate | None = None
    fluorescence_estimate: FluorescenceEstimate | None = None

    def summary_text(self) -> str:
        sections = ["Solution scattering estimator"]
        if self.number_density_estimate is not None:
            sections.append(self.number_density_estimate.summary_text())
        if self.volume_fraction_estimate is not None:
            sections.append(self.volume_fraction_estimate.summary_text())
        if self.interaction_contrast_estimate is not None:
            sections.append(self.interaction_contrast_estimate.summary_text())
        if self.attenuation_estimate is not None:
            sections.append(self.attenuation_estimate.summary_text())
            if self.interaction_contrast_estimate is not None:
                sections.append(
                    "\n".join(
                        [
                            "Model-facing solvent defaults",
                            (
                                "Split-fraction templates "
                                "(phi_solute / phi_solvent + solvent_scale): "
                                f"phi ratio = {_format_fraction(self.interaction_contrast_estimate.saxs_effective_solute_interaction_ratio)}, "
                                f"solvent_scale = {_format_fraction(self.attenuation_estimate.solvent_scattering_scale_factor)}"
                            ),
                            (
                                "Single-solvent-weight templates "
                                "(solv_w only): "
                                f"solvent multiplier = {_format_fraction(self.attenuation_estimate.solvent_scattering_scale_factor * self.interaction_contrast_estimate.saxs_effective_solvent_background_ratio)}"
                            ),
                        ]
                    )
                )
        if self.fluorescence_estimate is not None:
            sections.append(self.fluorescence_estimate.summary_text())
        return "\n\n".join(section.strip() for section in sections if section)


def _build_number_density_estimate(
    solution_result: SolutionPropertiesResult,
) -> NumberDensityEstimate:
    return NumberDensityEstimate(
        number_density_cm3=float(solution_result.number_density_cm3),
        number_density_a3=float(solution_result.number_density_a3),
        total_atoms=float(solution_result.total_atoms),
        element_ratio_string=str(solution_result.element_ratio_string).strip(),
    )


def _validate_positive(value: float, label: str) -> float:
    number = float(value)
    if number <= 0.0:
        raise ValueError(f"{label} must be greater than zero.")
    return number


def _validate_supported_formula(formula: str, label: str) -> str:
    text = str(formula or "").strip()
    if not text:
        raise ValueError(f"{label} formula is required.")
    if not xraydb.validate_formula(text):
        raise ValueError(
            f"{label} formula {text!r} is not recognized by the X-ray "
            "attenuation database. Enter an empirical formula such as H2O "
            "or C3H7NO."
        )
    return text


def _effective_scattering_density(
    formula: str,
    energy_ev: float,
    density_g_per_cm3: float,
) -> float:
    if density_g_per_cm3 <= 0.0:
        raise ValueError("Density must be greater than zero.")
    try:
        composition = xraydb.chemparse(formula)
    except Exception as exc:
        raise ValueError(
            f"Unable to parse formula {formula!r} for scattering-density "
            "estimation."
        ) from exc
    if not composition:
        raise ValueError(
            f"Formula {formula!r} could not be parsed for scattering-density "
            "estimation."
        )
    formula_mass = 0.0
    effective_electrons = 0.0
    for element, abundance in composition.items():
        amount = float(abundance)
        formula_mass += amount * float(xraydb.atomic_mass(element))
        effective_electrons += amount * float(
            xraydb.atomic_number(element)
            + xraydb.f1_chantler(element, energy_ev)
        )
    if formula_mass <= 0.0:
        raise ValueError(
            f"Formula {formula!r} produced a non-positive molar mass."
        )
    return float(
        float(density_g_per_cm3)
        * AVOGADRO_NUMBER
        * effective_electrons
        / formula_mass
    )


def _normalize_geometry_name(value: str) -> str:
    normalized = str(value or DEFAULT_CAPILLARY_GEOMETRY).strip().lower()
    if normalized not in {"cylindrical", "flat_plate"}:
        raise ValueError(
            "Capillary geometry must be 'cylindrical' or 'flat_plate'."
        )
    return normalized


def _normalize_profile_name(value: str) -> str:
    normalized = str(value or DEFAULT_BEAM_PROFILE).strip().lower()
    if normalized != "uniform":
        raise ValueError(
            "Only a uniform beam profile is currently implemented."
        )
    return normalized


def _path_lengths_cm(settings: BeamGeometrySettings) -> np.ndarray:
    profile = _normalize_profile_name(settings.beam_profile)
    del profile
    geometry = _normalize_geometry_name(settings.capillary_geometry)
    capillary_size_cm = (
        _validate_positive(
            settings.capillary_size_mm,
            "Capillary size",
        )
        * 0.1
    )
    beam_width_cm = (
        _validate_positive(
            settings.beam_footprint_width_mm,
            "Beam footprint width",
        )
        * 0.1
    )
    _validate_positive(
        settings.beam_footprint_height_mm,
        "Beam footprint height",
    )
    if geometry == "flat_plate":
        return np.full(PATH_SAMPLE_COUNT, capillary_size_cm, dtype=float)

    radius_cm = 0.5 * capillary_size_cm
    x_values = np.linspace(
        -0.5 * beam_width_cm,
        0.5 * beam_width_cm,
        PATH_SAMPLE_COUNT,
        dtype=float,
    )
    path_lengths = np.zeros_like(x_values)
    inside = np.abs(x_values) <= radius_cm
    path_lengths[inside] = 2.0 * np.sqrt(
        np.clip(radius_cm**2 - x_values[inside] ** 2, 0.0, None)
    )
    return path_lengths


def _average_transmission(
    path_lengths_cm: np.ndarray, mu_inv_cm: float
) -> float:
    return float(np.mean(np.exp(-float(mu_inv_cm) * path_lengths_cm)))


def _average_weighted_path(
    path_lengths_cm: np.ndarray,
    mu_inv_cm: float,
) -> float:
    mu_inv_cm = float(mu_inv_cm)
    return float(
        np.mean(path_lengths_cm * np.exp(-mu_inv_cm * path_lengths_cm))
    )


def _average_source_integral(
    path_lengths_cm: np.ndarray,
    mu_inv_cm: float,
) -> float:
    mu_inv_cm = float(mu_inv_cm)
    if abs(mu_inv_cm) <= 1e-15:
        return float(np.mean(path_lengths_cm))
    return float(
        np.mean((1.0 - np.exp(-mu_inv_cm * path_lengths_cm)) / mu_inv_cm)
    )


def _average_detected_integral(
    path_lengths_cm: np.ndarray,
    mu_in_inv_cm: float,
    mu_out_inv_cm: float,
) -> float:
    mu_in_inv_cm = float(mu_in_inv_cm)
    mu_out_inv_cm = float(mu_out_inv_cm)
    if abs(mu_in_inv_cm - mu_out_inv_cm) <= 1e-15:
        return float(
            np.mean(path_lengths_cm * np.exp(-mu_in_inv_cm * path_lengths_cm))
        )
    return float(
        np.mean(
            (
                np.exp(-mu_out_inv_cm * path_lengths_cm)
                - np.exp(-mu_in_inv_cm * path_lengths_cm)
            )
            / (mu_in_inv_cm - mu_out_inv_cm)
        )
    )


def _average_half_path_escape_factor(
    path_lengths_cm: np.ndarray,
    mu_inv_cm: float,
) -> float:
    return float(np.mean(np.exp(-0.5 * float(mu_inv_cm) * path_lengths_cm)))


def _average_half_path_absorption_fraction(
    path_lengths_cm: np.ndarray,
    mu_inv_cm: float,
) -> float:
    return float(
        np.mean(1.0 - np.exp(-0.5 * float(mu_inv_cm) * path_lengths_cm))
    )


def _material_mu(
    formula: str,
    energy_ev: float,
    density_g_per_cm3: float,
    *,
    kind: str = "total",
) -> float:
    if density_g_per_cm3 <= 0.0:
        return 0.0
    try:
        return float(
            xraydb.material_mu(
                formula,
                float(energy_ev),
                density=float(density_g_per_cm3),
                kind=kind,
            )
        )
    except Warning as exc:
        raise ValueError(str(exc)) from exc


def _material_element_coefficients(
    formula: str,
    energy_ev: float,
    density_g_per_cm3: float,
    *,
    kind: str = "total",
) -> dict[str, float]:
    if density_g_per_cm3 <= 0.0:
        return {}
    try:
        payload = xraydb.material_mu_components(
            formula,
            float(energy_ev),
            density=float(density_g_per_cm3),
            kind=kind,
        )
    except Warning as exc:
        raise ValueError(str(exc)) from exc
    coefficients: dict[str, float] = {}
    for element in payload.get("elements", []):
        value = payload.get(str(element))
        if not isinstance(value, tuple) or len(value) < 3:
            continue
        coefficients[str(element)] = float(value[2])
    return coefficients


def _merged_element_coefficients(
    contributions: list[dict[str, float]],
) -> dict[str, float]:
    merged: dict[str, float] = {}
    for contribution in contributions:
        for element, value in contribution.items():
            merged[element] = merged.get(element, 0.0) + float(value)
    return merged


def _accessible_edge_shares(
    element: str,
    incident_energy_ev: float,
) -> dict[str, float]:
    raw_shares: dict[str, float] = {}
    for edge_name, edge_data in xraydb.xray_edges(element).items():
        if edge_name not in EDGE_LINE_FAMILIES:
            continue
        if float(edge_data.energy) >= float(incident_energy_ev):
            continue
        jump_ratio = float(edge_data.jump_ratio)
        if jump_ratio <= 1.0:
            continue
        raw_shares[str(edge_name)] = (jump_ratio - 1.0) / jump_ratio
    if not raw_shares:
        return {}
    total = sum(raw_shares.values())
    return {
        edge_name: value / total
        for edge_name, value in raw_shares.items()
        if value > 0.0
    }


def _solution_component_concentrations(
    solution_result: SolutionPropertiesResult,
) -> tuple[float, float]:
    volume_solution_cm3 = float(solution_result.volume_solution_cm3)
    if volume_solution_cm3 <= 0.0:
        raise ValueError("Solution volume must be greater than zero.")
    return (
        float(solution_result.mass_solute) / volume_solution_cm3,
        float(solution_result.mass_solvent) / volume_solution_cm3,
    )


def _calculate_saxs_interaction_estimate(
    settings: SolutionScatteringEstimatorSettings,
    volume_fraction_estimate: SoluteVolumeFractionEstimate,
) -> SAXSInteractionEstimate:
    incident_energy_kev = _validate_positive(
        settings.beam.incident_energy_kev,
        "Incident energy",
    )
    incident_energy_ev = incident_energy_kev * 1000.0
    solute_formula = _validate_supported_formula(
        settings.solution.solute_stoich,
        "Solute",
    )
    solvent_formula = _validate_supported_formula(
        settings.solution.solvent_stoich,
        "Solvent",
    )
    if settings.solute_density_g_per_ml is None:
        specific_volume = float(
            volume_fraction_estimate.approximate_solute_specific_volume_cm3_per_g
        )
        if specific_volume <= 0.0:
            raise ValueError(
                "The SAXS-effective interaction estimate needs a positive "
                "solute specific volume or solute density."
            )
        solute_density = 1.0 / specific_volume
    else:
        solute_density = _validate_positive(
            settings.solute_density_g_per_ml,
            "Solute density",
        )
    if settings.solvent_density_g_per_ml is not None:
        solvent_density = _validate_positive(
            settings.solvent_density_g_per_ml,
            "Solvent density",
        )
    else:
        solvent_volume = float(
            volume_fraction_estimate.solution_result.volume_solution_cm3
            - volume_fraction_estimate.solute_volume_cm3
        )
        if solvent_volume <= 0.0:
            raise ValueError(
                "The SAXS-effective interaction estimate needs a positive "
                "solvent-associated volume."
            )
        solvent_density = (
            float(volume_fraction_estimate.solution_result.mass_solvent)
            / solvent_volume
        )
    solute_scattering_density = _effective_scattering_density(
        solute_formula,
        incident_energy_ev,
        solute_density,
    )
    solvent_scattering_density = _effective_scattering_density(
        solvent_formula,
        incident_energy_ev,
        solvent_density,
    )
    if abs(solvent_scattering_density) <= 1e-30:
        raise ValueError(
            "The solvent effective scattering density is too small to form a "
            "contrast-weighted SAXS interaction ratio."
        )
    contrast_scattering_density = (
        solute_scattering_density - solvent_scattering_density
    )
    contrast_weight_factor = (
        contrast_scattering_density / solvent_scattering_density
    ) ** 2
    physical_solute_volume_cm3 = float(
        volume_fraction_estimate.solute_volume_cm3
    )
    physical_solvent_volume_cm3 = float(
        volume_fraction_estimate.solution_result.volume_solution_cm3
        - volume_fraction_estimate.solute_volume_cm3
    )
    if physical_solvent_volume_cm3 < 0.0:
        physical_solvent_volume_cm3 = 0.0
    effective_solute_interaction_volume_cm3 = (
        physical_solute_volume_cm3 * contrast_weight_factor
    )
    effective_solvent_background_volume_cm3 = physical_solvent_volume_cm3
    total_effective_volume = (
        effective_solute_interaction_volume_cm3
        + effective_solvent_background_volume_cm3
    )
    if total_effective_volume <= 0.0:
        raise ValueError(
            "The contrast-weighted interaction volume is non-positive."
        )
    return SAXSInteractionEstimate(
        incident_energy_kev=incident_energy_kev,
        wavelength_angstrom=wavelength_angstrom_from_energy_kev(
            incident_energy_kev
        ),
        solute_formula=solute_formula,
        solvent_formula=solvent_formula,
        solute_density_g_per_cm3=float(solute_density),
        solvent_density_g_per_cm3=float(solvent_density),
        solute_effective_scattering_density_electrons_per_cm3=(
            solute_scattering_density
        ),
        solvent_effective_scattering_density_electrons_per_cm3=(
            solvent_scattering_density
        ),
        contrast_scattering_density_electrons_per_cm3=(
            contrast_scattering_density
        ),
        physical_solute_associated_volume_cm3=physical_solute_volume_cm3,
        physical_solvent_associated_volume_cm3=physical_solvent_volume_cm3,
        physical_solute_associated_volume_fraction=float(
            volume_fraction_estimate.solute_volume_fraction
        ),
        physical_solvent_associated_volume_fraction=float(
            volume_fraction_estimate.solvent_volume_fraction
        ),
        contrast_weight_factor=float(contrast_weight_factor),
        effective_solute_interaction_volume_cm3=float(
            effective_solute_interaction_volume_cm3
        ),
        effective_solvent_background_volume_cm3=float(
            effective_solvent_background_volume_cm3
        ),
        saxs_effective_solute_interaction_ratio=float(
            effective_solute_interaction_volume_cm3 / total_effective_volume
        ),
        saxs_effective_solvent_background_ratio=float(
            effective_solvent_background_volume_cm3 / total_effective_volume
        ),
    )


def _calculate_attenuation_estimate(
    settings: SolutionScatteringEstimatorSettings,
    solution_result: SolutionPropertiesResult,
    path_lengths_cm: np.ndarray,
) -> AttenuationEstimate:
    solvent_density = settings.solvent_density_g_per_ml
    if solvent_density is None:
        raise ValueError(
            "Solvent density is required for the attenuation estimate."
        )
    solvent_density = _validate_positive(solvent_density, "Solvent density")
    incident_energy_ev = (
        _validate_positive(
            settings.beam.incident_energy_kev,
            "Incident energy",
        )
        * 1000.0
    )
    solute_formula = _validate_supported_formula(
        settings.solution.solute_stoich,
        "Solute",
    )
    solvent_formula = _validate_supported_formula(
        settings.solution.solvent_stoich,
        "Solvent",
    )
    solute_concentration, solvent_concentration = (
        _solution_component_concentrations(solution_result)
    )

    mu_solute = _material_mu(
        solute_formula,
        incident_energy_ev,
        solute_concentration,
        kind="total",
    )
    mu_sample_solvent = _material_mu(
        solvent_formula,
        incident_energy_ev,
        solvent_concentration,
        kind="total",
    )
    mu_neat_solvent = _material_mu(
        solvent_formula,
        incident_energy_ev,
        solvent_density,
        kind="total",
    )
    mu_sample = mu_solute + mu_sample_solvent
    sample_scattering_weight = _average_weighted_path(
        path_lengths_cm,
        mu_sample,
    )
    neat_solvent_scattering_weight = _average_weighted_path(
        path_lengths_cm,
        mu_neat_solvent,
    )
    if neat_solvent_scattering_weight <= 0.0:
        solvent_scale_factor = 0.0
        neat_to_sample_ratio = None
    else:
        solvent_scale_factor = (
            solvent_concentration * sample_scattering_weight
        ) / (solvent_density * neat_solvent_scattering_weight)
        neat_to_sample_ratio = (
            None if solvent_scale_factor <= 0.0 else 1.0 / solvent_scale_factor
        )
    return AttenuationEstimate(
        incident_energy_kev=float(settings.beam.incident_energy_kev),
        capillary_geometry=_normalize_geometry_name(
            settings.beam.capillary_geometry
        ),
        sample_average_path_length_mm=(float(np.mean(path_lengths_cm)) * 10.0),
        sample_linear_attenuation_inv_cm=mu_sample,
        solute_linear_attenuation_inv_cm=mu_solute,
        sample_solvent_linear_attenuation_inv_cm=mu_sample_solvent,
        neat_solvent_linear_attenuation_inv_cm=mu_neat_solvent,
        sample_transmission=_average_transmission(path_lengths_cm, mu_sample),
        solute_only_transmission=_average_transmission(
            path_lengths_cm,
            mu_solute,
        ),
        sample_solvent_only_transmission=_average_transmission(
            path_lengths_cm,
            mu_sample_solvent,
        ),
        neat_solvent_transmission=_average_transmission(
            path_lengths_cm,
            mu_neat_solvent,
        ),
        sample_scattering_weight=sample_scattering_weight,
        neat_solvent_scattering_weight=neat_solvent_scattering_weight,
        solvent_mass_concentration_g_per_cm3=solvent_concentration,
        neat_solvent_density_g_per_cm3=solvent_density,
        solvent_scattering_scale_factor=solvent_scale_factor,
        neat_solvent_to_sample_ratio=neat_to_sample_ratio,
    )


def _calculate_fluorescence_estimate(
    settings: SolutionScatteringEstimatorSettings,
    solution_result: SolutionPropertiesResult,
    path_lengths_cm: np.ndarray,
) -> FluorescenceEstimate:
    incident_energy_ev = (
        _validate_positive(
            settings.beam.incident_energy_kev,
            "Incident energy",
        )
        * 1000.0
    )
    solute_formula = _validate_supported_formula(
        settings.solution.solute_stoich,
        "Solute",
    )
    solvent_formula = _validate_supported_formula(
        settings.solution.solvent_stoich,
        "Solvent",
    )
    solute_concentration, solvent_concentration = (
        _solution_component_concentrations(solution_result)
    )
    sample_total_mu_cache: dict[float, float] = {}
    sample_photo_mu_cache: dict[float, dict[str, float]] = {}

    def sample_total_mu(energy_ev: float) -> float:
        rounded = round(float(energy_ev), 6)
        if rounded not in sample_total_mu_cache:
            sample_total_mu_cache[rounded] = _material_mu(
                solute_formula,
                rounded,
                solute_concentration,
                kind="total",
            ) + _material_mu(
                solvent_formula,
                rounded,
                solvent_concentration,
                kind="total",
            )
        return sample_total_mu_cache[rounded]

    def sample_photo_element_mus(energy_ev: float) -> dict[str, float]:
        rounded = round(float(energy_ev), 6)
        if rounded not in sample_photo_mu_cache:
            sample_photo_mu_cache[rounded] = _merged_element_coefficients(
                [
                    _material_element_coefficients(
                        solute_formula,
                        rounded,
                        solute_concentration,
                        kind="photo",
                    ),
                    _material_element_coefficients(
                        solvent_formula,
                        rounded,
                        solvent_concentration,
                        kind="photo",
                    ),
                ]
            )
        return sample_photo_mu_cache[rounded]

    mu_incident_total = sample_total_mu(incident_energy_ev)
    source_integral = _average_source_integral(
        path_lengths_cm,
        mu_incident_total,
    )
    line_lookup: dict[tuple[str, str, str], FluorescenceLineEstimate] = {}
    primary_generated_records: list[tuple[str, float, float]] = []

    for element, photo_mu in sample_photo_element_mus(
        incident_energy_ev
    ).items():
        if photo_mu <= 0.0:
            continue
        edge_shares = _accessible_edge_shares(element, incident_energy_ev)
        if not edge_shares:
            continue
        for edge_name, edge_share in edge_shares.items():
            edge_photo_mu = float(photo_mu) * float(edge_share)
            for family in EDGE_LINE_FAMILIES.get(edge_name, ()):
                try:
                    fyield, line_energy_ev, line_probability = (
                        xraydb.fluor_yield(
                            element,
                            edge_name,
                            family,
                            incident_energy_ev,
                        )
                    )
                except Exception:
                    continue
                line_branch = float(fyield) * float(line_probability)
                if (
                    line_energy_ev <= 0.0
                    or line_branch <= MINIMUM_LINE_CONTRIBUTION
                ):
                    continue
                mu_line_total = sample_total_mu(line_energy_ev)
                primary_generated = (
                    edge_photo_mu * line_branch * source_integral
                )
                primary_detected = (
                    edge_photo_mu
                    * line_branch
                    * _average_detected_integral(
                        path_lengths_cm,
                        mu_incident_total,
                        mu_line_total,
                    )
                )
                if (
                    primary_generated <= MINIMUM_LINE_CONTRIBUTION
                    and primary_detected <= MINIMUM_LINE_CONTRIBUTION
                ):
                    continue
                key = (str(element), str(edge_name), str(family))
                line_lookup[key] = FluorescenceLineEstimate(
                    element=str(element),
                    edge=str(edge_name),
                    family=str(family),
                    line_energy_ev=float(line_energy_ev),
                    primary_detected_yield=float(primary_detected),
                    secondary_detected_yield=0.0,
                )
                primary_generated_records.append(
                    (
                        str(element),
                        float(line_energy_ev),
                        float(primary_generated),
                    )
                )

    for (
        source_element,
        line_energy_ev,
        primary_generated,
    ) in primary_generated_records:
        if primary_generated <= MINIMUM_LINE_CONTRIBUTION:
            continue
        mu_line_total = sample_total_mu(line_energy_ev)
        reabsorbed_fraction = _average_half_path_absorption_fraction(
            path_lengths_cm,
            mu_line_total,
        )
        if reabsorbed_fraction <= MINIMUM_LINE_CONTRIBUTION:
            continue
        photo_element_mus = sample_photo_element_mus(line_energy_ev)
        total_photo_mu = sum(photo_element_mus.values())
        if total_photo_mu <= 0.0:
            continue
        absorbed_primary = primary_generated * reabsorbed_fraction
        for target_element, target_photo_mu in photo_element_mus.items():
            if target_element == source_element or target_photo_mu <= 0.0:
                continue
            element_absorption_share = float(target_photo_mu) / float(
                total_photo_mu
            )
            edge_shares = _accessible_edge_shares(
                target_element, line_energy_ev
            )
            if not edge_shares:
                continue
            for edge_name, edge_share in edge_shares.items():
                for family in EDGE_LINE_FAMILIES.get(edge_name, ()):
                    try:
                        fyield, emitted_energy_ev, line_probability = (
                            xraydb.fluor_yield(
                                target_element,
                                edge_name,
                                family,
                                line_energy_ev,
                            )
                        )
                    except Exception:
                        continue
                    line_branch = float(fyield) * float(line_probability)
                    if (
                        emitted_energy_ev <= 0.0
                        or line_branch <= MINIMUM_LINE_CONTRIBUTION
                    ):
                        continue
                    secondary_generated = (
                        absorbed_primary
                        * element_absorption_share
                        * float(edge_share)
                        * line_branch
                    )
                    if secondary_generated <= MINIMUM_LINE_CONTRIBUTION:
                        continue
                    mu_secondary_total = sample_total_mu(emitted_energy_ev)
                    secondary_detected = secondary_generated * (
                        _average_half_path_escape_factor(
                            path_lengths_cm,
                            mu_secondary_total,
                        )
                    )
                    if secondary_detected <= MINIMUM_LINE_CONTRIBUTION:
                        continue
                    key = (str(target_element), str(edge_name), str(family))
                    existing = line_lookup.get(key)
                    if existing is None:
                        line_lookup[key] = FluorescenceLineEstimate(
                            element=str(target_element),
                            edge=str(edge_name),
                            family=str(family),
                            line_energy_ev=float(emitted_energy_ev),
                            primary_detected_yield=0.0,
                            secondary_detected_yield=float(secondary_detected),
                        )
                    else:
                        existing.secondary_detected_yield = float(
                            existing.secondary_detected_yield
                            + secondary_detected
                        )

    line_estimates = list(line_lookup.values())
    return FluorescenceEstimate(
        incident_energy_kev=float(settings.beam.incident_energy_kev),
        capillary_geometry=_normalize_geometry_name(
            settings.beam.capillary_geometry
        ),
        total_primary_detected_yield=float(
            sum(item.primary_detected_yield for item in line_estimates)
        ),
        total_secondary_detected_yield=float(
            sum(item.secondary_detected_yield for item in line_estimates)
        ),
        line_estimates=line_estimates,
    )


def calculate_solution_scattering_estimate(
    settings: SolutionScatteringEstimatorSettings,
) -> SolutionScatteringEstimate:
    path_lengths_cm = _path_lengths_cm(settings.beam)
    solution_result = calculate_solution_properties(settings.solution)

    number_density_estimate = None
    if settings.calculate_number_density:
        number_density_estimate = _build_number_density_estimate(
            solution_result
        )

    volume_fraction_estimate = None
    if settings.calculate_solute_volume_fraction:
        volume_fraction_estimate = calculate_solute_volume_fraction_estimate(
            SoluteVolumeFractionSettings(
                solution=SolutionPropertiesSettings.from_dict(
                    settings.solution.to_dict()
                ),
                solute_density_g_per_ml=settings.solute_density_g_per_ml,
                solvent_density_g_per_ml=settings.solvent_density_g_per_ml,
            )
        )

    interaction_contrast_estimate = None
    if volume_fraction_estimate is not None:
        interaction_contrast_estimate = _calculate_saxs_interaction_estimate(
            settings,
            volume_fraction_estimate,
        )

    attenuation_estimate = None
    if settings.calculate_solvent_scattering_contribution:
        attenuation_estimate = _calculate_attenuation_estimate(
            settings,
            solution_result,
            path_lengths_cm,
        )

    fluorescence_estimate = None
    if settings.calculate_sample_fluorescence_yield:
        fluorescence_estimate = _calculate_fluorescence_estimate(
            settings,
            solution_result,
            path_lengths_cm,
        )

    return SolutionScatteringEstimate(
        settings=SolutionScatteringEstimatorSettings(
            solution=SolutionPropertiesSettings.from_dict(
                settings.solution.to_dict()
            ),
            solute_density_g_per_ml=settings.solute_density_g_per_ml,
            solvent_density_g_per_ml=settings.solvent_density_g_per_ml,
            calculate_number_density=bool(settings.calculate_number_density),
            calculate_solute_volume_fraction=bool(
                settings.calculate_solute_volume_fraction
            ),
            calculate_solvent_scattering_contribution=bool(
                settings.calculate_solvent_scattering_contribution
            ),
            calculate_sample_fluorescence_yield=bool(
                settings.calculate_sample_fluorescence_yield
            ),
            beam=BeamGeometrySettings(
                incident_energy_kev=float(settings.beam.incident_energy_kev),
                capillary_size_mm=float(settings.beam.capillary_size_mm),
                capillary_geometry=str(settings.beam.capillary_geometry),
                beam_profile=str(settings.beam.beam_profile),
                beam_footprint_width_mm=float(
                    settings.beam.beam_footprint_width_mm
                ),
                beam_footprint_height_mm=float(
                    settings.beam.beam_footprint_height_mm
                ),
            ),
        ),
        solution_result=solution_result,
        number_density_estimate=number_density_estimate,
        volume_fraction_estimate=volume_fraction_estimate,
        interaction_contrast_estimate=interaction_contrast_estimate,
        attenuation_estimate=attenuation_estimate,
        fluorescence_estimate=fluorescence_estimate,
    )


__all__ = [
    "BEAM_PROFILE_ITEMS",
    "CAPILLARY_GEOMETRY_ITEMS",
    "DEFAULT_BEAM_FOOTPRINT_HEIGHT_MM",
    "DEFAULT_BEAM_FOOTPRINT_WIDTH_MM",
    "DEFAULT_BEAM_PROFILE",
    "DEFAULT_CAPILLARY_GEOMETRY",
    "DEFAULT_CAPILLARY_SIZE_MM",
    "DEFAULT_INCIDENT_ENERGY_KEV",
    "ENERGY_WAVELENGTH_KEV_ANGSTROM",
    "AttenuationEstimate",
    "BeamGeometrySettings",
    "FluorescenceEstimate",
    "FluorescenceLineEstimate",
    "NumberDensityEstimate",
    "SAXSInteractionEstimate",
    "SolutionScatteringEstimate",
    "SolutionScatteringEstimatorSettings",
    "calculate_solution_scattering_estimate",
    "wavelength_angstrom_from_energy_kev",
]
