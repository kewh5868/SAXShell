from __future__ import annotations

from dataclasses import dataclass

from saxshell.fullrmc.solution_properties import (
    SolutionPropertiesResult,
    SolutionPropertiesSettings,
    calculate_solution_properties,
)

DISPLAY_MASS_DECIMALS_G = 6
DISPLAY_VOLUME_DECIMALS_CM3 = 3
DISPLAY_FRACTION_DECIMALS = 6
DISPLAY_DENSITY_DECIMALS = 4


def _format_mass_g(value: float) -> str:
    return f"{float(value):.{DISPLAY_MASS_DECIMALS_G}f} g"


def _format_volume_cm3(value: float) -> str:
    return f"{float(value):.{DISPLAY_VOLUME_DECIMALS_CM3}f} cm^3"


def _format_fraction(value: float) -> str:
    return f"{float(value):.{DISPLAY_FRACTION_DECIMALS}f}"


def _format_density_g_per_ml(value: float) -> str:
    return f"{float(value):.{DISPLAY_DENSITY_DECIMALS}f} g/mL"


@dataclass(slots=True)
class SoluteVolumeFractionSettings:
    solution: SolutionPropertiesSettings
    solute_density_g_per_ml: float | None = 1.0
    solvent_density_g_per_ml: float | None = 1.0


@dataclass(slots=True)
class SoluteVolumeFractionEstimate:
    calculation_method: str
    settings: SoluteVolumeFractionSettings
    solution_result: SolutionPropertiesResult
    solute_mass_concentration_g_per_cm3: float
    approximate_solute_specific_volume_cm3_per_g: float
    solute_volume_cm3: float
    solvent_volume_cm3: float | None
    additive_volume_cm3: float | None
    solute_volume_fraction: float
    solvent_volume_fraction: float
    additive_to_solution_volume_ratio: float | None

    def summary_text(self) -> str:
        lines = [
            "Volume fraction estimate",
            f"Mode: {self.solution_result.mode}",
            (
                "Solution volume from measured density: "
                f"{_format_volume_cm3(self.solution_result.volume_solution_cm3)}"
            ),
            (
                "Solute mass concentration: "
                f"{self.solute_mass_concentration_g_per_cm3:.6f} g/cm^3"
            ),
            (
                "Approx. solute specific volume: "
                f"{self.approximate_solute_specific_volume_cm3_per_g:.6f} "
                "cm^3/g"
            ),
            (
                "Estimated solute volume: "
                f"{_format_volume_cm3(self.solute_volume_cm3)}"
            ),
        ]
        if self.calculation_method == "solute_density":
            lines.insert(
                5,
                (
                    "Solute mass / density: "
                    f"{_format_mass_g(self.solution_result.mass_solute)} / "
                    f"{_format_density_g_per_ml(self.settings.solute_density_g_per_ml)}"
                ),
            )
        else:
            lines.insert(
                5,
                (
                    "Solute volume from solvent-density closure: "
                    "V_solute ~= V_solution - V_solvent"
                ),
            )

        if self.settings.solvent_density_g_per_ml is None:
            lines.extend(
                [
                    "Solvent-density additive-volume diagnostics: hidden for "
                    "this input mode.",
                    (
                        "Estimated solvent volume fraction: "
                        f"{_format_fraction(self.solvent_volume_fraction)}"
                    ),
                ]
            )
        elif self.calculation_method == "solvent_density_closure":
            lines.extend(
                [
                    (
                        "Solvent mass / density: "
                        f"{_format_mass_g(self.solution_result.mass_solvent)} / "
                        f"{_format_density_g_per_ml(self.settings.solvent_density_g_per_ml)}"
                    ),
                    (
                        "Estimated solvent volume: "
                        f"{_format_volume_cm3(self.solvent_volume_cm3)}"
                    ),
                    (
                        "Estimated solvent volume fraction: "
                        f"{_format_fraction(self.solvent_volume_fraction)}"
                    ),
                ]
            )
        else:
            lines.extend(
                [
                    (
                        "Solvent mass / density: "
                        f"{_format_mass_g(self.solution_result.mass_solvent)} / "
                        f"{_format_density_g_per_ml(self.settings.solvent_density_g_per_ml)}"
                    ),
                    (
                        "Estimated solvent volume: "
                        f"{_format_volume_cm3(self.solvent_volume_cm3)}"
                    ),
                    (
                        "Additive component volume: "
                        f"{_format_volume_cm3(self.additive_volume_cm3)}"
                    ),
                    (
                        "Additive / measured solution volume: "
                        f"{_format_fraction(self.additive_to_solution_volume_ratio)}"
                    ),
                    (
                        "Estimated solvent volume fraction: "
                        f"{_format_fraction(self.solvent_volume_fraction)}"
                    ),
                ]
            )
        lines.extend(
            [
                (
                    "Estimated solute volume fraction: "
                    f"{_format_fraction(self.solute_volume_fraction)}"
                ),
                "",
                "Interpretation:",
                "This estimate follows the SAXS-style concentration x "
                "specific-volume picture for solute occupancy in the measured "
                "solution volume:",
                (
                    "phi_solute ~= c_solute * vbar_solute "
                    "= (m_solute / V_solution) * (1 / rho_solute)."
                    if self.calculation_method == "solute_density"
                    else "phi_solute ~= V_solute / V_solution, with "
                    "V_solute ~= V_solution - (m_solvent / rho_solvent)."
                ),
            ]
        )
        if self.settings.solvent_density_g_per_ml is None:
            lines.append(
                "The fitted estimate above only needs the measured solution "
                "volume plus the solute density, so solvent-density-based "
                "additive diagnostics were skipped."
            )
        elif self.calculation_method == "solvent_density_closure":
            lines.append(
                "In this mode the solute density was not required. SAXSShell "
                "estimated the solute volume by subtracting the solvent "
                "volume from the measured solution volume, using the entered "
                "solvent density to convert solvent mass into solvent volume."
            )
        else:
            lines.append(
                "The additive solute and solvent volumes are still reported "
                "as a consistency check against ideal-volume assumptions."
            )
        return "\n".join(lines)


def calculate_solute_volume_fraction_estimate(
    settings: SoluteVolumeFractionSettings,
) -> SoluteVolumeFractionEstimate:
    if (
        settings.solute_density_g_per_ml is not None
        and settings.solute_density_g_per_ml <= 0.0
    ):
        raise ValueError("Solute density must be greater than zero.")
    if (
        settings.solvent_density_g_per_ml is not None
        and settings.solvent_density_g_per_ml <= 0.0
    ):
        raise ValueError("Solvent density must be greater than zero.")
    if (
        settings.solute_density_g_per_ml is None
        and settings.solvent_density_g_per_ml is None
    ):
        raise ValueError(
            "Provide either a solute density or a solvent density to "
            "estimate the solute volume fraction."
        )

    solution_result = calculate_solution_properties(settings.solution)
    measured_solution_volume = float(solution_result.volume_solution_cm3)
    if measured_solution_volume <= 0.0:
        raise ValueError("Measured solution volume must be greater than zero.")

    solute_mass_concentration_g_per_cm3 = (
        float(solution_result.mass_solute) / measured_solution_volume
    )
    calculation_method = "solute_density"
    solvent_volume_cm3: float | None = None
    additive_volume_cm3: float | None = None
    ratio: float | None = None
    if settings.solute_density_g_per_ml is not None:
        approximate_solute_specific_volume_cm3_per_g = 1.0 / float(
            settings.solute_density_g_per_ml
        )
        solute_volume_cm3 = float(solution_result.mass_solute) / float(
            settings.solute_density_g_per_ml
        )
    else:
        calculation_method = "solvent_density_closure"
        approximate_solute_specific_volume_cm3_per_g = 0.0
        solute_volume_cm3 = 0.0

    if settings.solvent_density_g_per_ml is not None:
        solvent_volume_cm3 = float(solution_result.mass_solvent) / float(
            settings.solvent_density_g_per_ml
        )
        if calculation_method == "solvent_density_closure":
            solute_volume_cm3 = measured_solution_volume - solvent_volume_cm3
            if solute_volume_cm3 <= 0.0:
                raise ValueError(
                    "The current solution and solvent densities imply a "
                    "non-positive solute volume. Review the solution density, "
                    "solvent density, and molarity inputs."
                )
            if float(solution_result.mass_solute) <= 0.0:
                raise ValueError("Solute mass must be greater than zero.")
            approximate_solute_specific_volume_cm3_per_g = (
                solute_volume_cm3 / float(solution_result.mass_solute)
            )
            additive_volume_cm3 = None
            ratio = None
        else:
            additive_volume_cm3 = solute_volume_cm3 + solvent_volume_cm3
            if additive_volume_cm3 <= 0.0:
                raise ValueError(
                    "The additive solute + solvent volume must be greater than zero."
                )
            ratio = additive_volume_cm3 / measured_solution_volume
    solute_volume_fraction = solute_volume_cm3 / measured_solution_volume
    solvent_volume_fraction = 1.0 - solute_volume_fraction
    if solute_volume_fraction < 0.0 or solvent_volume_fraction < 0.0:
        raise ValueError(
            "The current masses and densities imply a non-physical volume "
            "fraction. Review the solution density and component densities."
        )
    return SoluteVolumeFractionEstimate(
        calculation_method=calculation_method,
        settings=SoluteVolumeFractionSettings(
            solution=SolutionPropertiesSettings.from_dict(
                settings.solution.to_dict()
            ),
            solute_density_g_per_ml=(
                None
                if settings.solute_density_g_per_ml is None
                else float(settings.solute_density_g_per_ml)
            ),
            solvent_density_g_per_ml=(
                None
                if settings.solvent_density_g_per_ml is None
                else float(settings.solvent_density_g_per_ml)
            ),
        ),
        solution_result=solution_result,
        solute_mass_concentration_g_per_cm3=(
            solute_mass_concentration_g_per_cm3
        ),
        approximate_solute_specific_volume_cm3_per_g=(
            approximate_solute_specific_volume_cm3_per_g
        ),
        solute_volume_cm3=solute_volume_cm3,
        solvent_volume_cm3=solvent_volume_cm3,
        additive_volume_cm3=additive_volume_cm3,
        solute_volume_fraction=solute_volume_fraction,
        solvent_volume_fraction=solvent_volume_fraction,
        additive_to_solution_volume_ratio=ratio,
    )


__all__ = [
    "DISPLAY_FRACTION_DECIMALS",
    "DISPLAY_MASS_DECIMALS_G",
    "DISPLAY_VOLUME_DECIMALS_CM3",
    "SoluteVolumeFractionEstimate",
    "SoluteVolumeFractionSettings",
    "calculate_solute_volume_fraction_estimate",
]
