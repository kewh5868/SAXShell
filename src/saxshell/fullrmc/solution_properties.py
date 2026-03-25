from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_MODE_DEFAULT = "mass"


@dataclass(slots=True)
class SolutionPropertiesSettings:
    mode: str = _MODE_DEFAULT
    solution_density: float = 1.0
    solute_stoich: str = ""
    solvent_stoich: str = ""
    molar_mass_solute: float = 0.0
    molar_mass_solvent: float = 0.0
    mass_solute: float = 0.0
    mass_solvent: float = 0.0
    mass_percent_solute: float = 0.0
    total_mass_solution: float = 0.0
    molarity: float = 0.0
    molarity_element: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "solution_density": self.solution_density,
            "solute_stoich": self.solute_stoich,
            "solvent_stoich": self.solvent_stoich,
            "molar_mass_solute": self.molar_mass_solute,
            "molar_mass_solvent": self.molar_mass_solvent,
            "mass_solute": self.mass_solute,
            "mass_solvent": self.mass_solvent,
            "mass_percent_solute": self.mass_percent_solute,
            "total_mass_solution": self.total_mass_solution,
            "molarity": self.molarity,
            "molarity_element": self.molarity_element,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, object] | None
    ) -> "SolutionPropertiesSettings":
        source = dict(payload or {})
        return cls(
            mode=_normalized_mode(source.get("mode")),
            solution_density=_float_value(source.get("solution_density"), 1.0),
            solute_stoich=str(source.get("solute_stoich", "") or "").strip(),
            solvent_stoich=str(source.get("solvent_stoich", "") or "").strip(),
            molar_mass_solute=_float_value(
                source.get("molar_mass_solute"), 0.0
            ),
            molar_mass_solvent=_float_value(
                source.get("molar_mass_solvent"), 0.0
            ),
            mass_solute=_float_value(source.get("mass_solute"), 0.0),
            mass_solvent=_float_value(source.get("mass_solvent"), 0.0),
            mass_percent_solute=_float_value(
                source.get("mass_percent_solute"),
                0.0,
            ),
            total_mass_solution=_float_value(
                source.get("total_mass_solution"),
                0.0,
            ),
            molarity=_float_value(source.get("molarity"), 0.0),
            molarity_element=str(
                source.get("molarity_element", "") or ""
            ).strip(),
        )


def solution_properties_mode_hint_text(mode: str) -> str:
    normalized = _normalized_mode(mode)
    common = (
        "Common inputs: solution density, solute/solvent stoichiometry, and "
        "both molar masses are always required."
    )
    if normalized == "mass":
        return (
            f"{common} Mass mode uses the directly entered solute and solvent "
            "masses."
        )
    if normalized == "mass_percent":
        return (
            f"{common} Mass-percent mode uses the entered solute mass percent "
            "and total measured solution mass."
        )
    return (
        f"{common} Molarity mode assumes 1 L of solution. The solution "
        "density is still required so that liter can be converted into total "
        "solution mass, and the solvent mass is then inferred by subtraction."
    )


@dataclass(slots=True)
class SolutionPropertiesResult:
    mode: str
    solution_density_g_per_ml: float
    solution_density_g_per_cm3: float
    solute_stoich: str
    solvent_stoich: str
    solute_dict: dict[str, int]
    solvent_dict: dict[str, int]
    element_mole_dict: dict[str, float]
    volume_solution_cm3: float
    mass_solute: float
    mass_solvent: float
    total_mass_solution: float
    moles_solute: float
    moles_solvent: float
    total_atoms: float
    number_density_cm3: float
    number_density_a3: float
    element_ratio_string: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "solution_density_g_per_ml": self.solution_density_g_per_ml,
            "solution_density_g_per_cm3": self.solution_density_g_per_cm3,
            "solute_stoich": self.solute_stoich,
            "solvent_stoich": self.solvent_stoich,
            "solute_dict": self.solute_dict,
            "solvent_dict": self.solvent_dict,
            "element_mole_dict": self.element_mole_dict,
            "volume_solution_cm3": self.volume_solution_cm3,
            "mass_solute": self.mass_solute,
            "mass_solvent": self.mass_solvent,
            "total_mass_solution": self.total_mass_solution,
            "moles_solute": self.moles_solute,
            "moles_solvent": self.moles_solvent,
            "total_atoms": self.total_atoms,
            "number_density_cm3": self.number_density_cm3,
            "number_density_a3": self.number_density_a3,
            "element_ratio_string": self.element_ratio_string,
        }

    @classmethod
    def from_dict(
        cls, payload: dict[str, object] | None
    ) -> "SolutionPropertiesResult | None":
        if not payload:
            return None
        source = dict(payload)
        return cls(
            mode=_normalized_mode(source.get("mode")),
            solution_density_g_per_ml=_float_value(
                source.get("solution_density_g_per_ml"),
                0.0,
            ),
            solution_density_g_per_cm3=_float_value(
                source.get("solution_density_g_per_cm3"),
                0.0,
            ),
            solute_stoich=str(source.get("solute_stoich", "") or "").strip(),
            solvent_stoich=str(source.get("solvent_stoich", "") or "").strip(),
            solute_dict={
                str(key): int(value)
                for key, value in dict(source.get("solute_dict", {})).items()
            },
            solvent_dict={
                str(key): int(value)
                for key, value in dict(source.get("solvent_dict", {})).items()
            },
            element_mole_dict={
                str(key): float(value)
                for key, value in dict(
                    source.get("element_mole_dict", {})
                ).items()
            },
            volume_solution_cm3=_float_value(
                source.get("volume_solution_cm3"),
                0.0,
            ),
            mass_solute=_float_value(source.get("mass_solute"), 0.0),
            mass_solvent=_float_value(source.get("mass_solvent"), 0.0),
            total_mass_solution=_float_value(
                source.get("total_mass_solution"),
                0.0,
            ),
            moles_solute=_float_value(source.get("moles_solute"), 0.0),
            moles_solvent=_float_value(source.get("moles_solvent"), 0.0),
            total_atoms=_float_value(source.get("total_atoms"), 0.0),
            number_density_cm3=_float_value(
                source.get("number_density_cm3"),
                0.0,
            ),
            number_density_a3=_float_value(
                source.get("number_density_a3"),
                0.0,
            ),
            element_ratio_string=str(
                source.get("element_ratio_string", "") or ""
            ).strip(),
        )

    def summary_text(self, *, updated_at: str | None = None) -> str:
        lines = []
        if updated_at:
            lines.append(f"Saved calculation: {updated_at}")
        lines.extend(
            [
                f"Mode: {self.mode}",
                f"Solution density: {self.solution_density_g_per_ml:.6g} g/mL",
                f"Solute stoichiometry: {self.solute_stoich or 'n/a'}",
                f"Solvent stoichiometry: {self.solvent_stoich or 'n/a'}",
                f"Solution volume: {self.volume_solution_cm3:.6g} cm^3",
                f"Mass solute: {self.mass_solute:.6g} g",
                f"Mass solvent: {self.mass_solvent:.6g} g",
                f"Total solution mass: {self.total_mass_solution:.6g} g",
                f"Moles solute: {self.moles_solute:.6g}",
                f"Moles solvent: {self.moles_solvent:.6g}",
                (
                    "Number density: "
                    f"{self.number_density_cm3:.6g} atoms/cm^3"
                ),
                f"                {self.number_density_a3:.6g} atoms/A^3",
                f"Total atoms in source solution: {self.total_atoms:.6g}",
                (
                    "Integer ratio of elements: "
                    f"{self.element_ratio_string or 'n/a'}"
                ),
            ]
        )
        if self.element_mole_dict:
            lines.append("")
            lines.append("Element mole totals:")
            for element in sorted(self.element_mole_dict):
                lines.append(
                    f"  {element}: {self.element_mole_dict[element]:.6g}"
                )
        return "\n".join(lines)


@dataclass(slots=True)
class SolutionPropertiesMetadata:
    settings: SolutionPropertiesSettings
    result: SolutionPropertiesResult | None
    updated_at: str | None


class SolutionProperties:
    AVOGADRO = 6.022_140_76e23
    CM3_TO_ANGSTROM3 = 1.0e24

    def __init__(
        self,
        *,
        mode: str,
        solution_density: float,
        solute_stoich: str,
        solvent_stoich: str,
        molar_mass_solute: float,
        molar_mass_solvent: float,
        mass_solute: float | None = None,
        mass_solvent: float | None = None,
        mass_percent_solute: float | None = None,
        total_mass_solution: float | None = None,
        molarity: float | None = None,
        molarity_element: str | None = None,
    ) -> None:
        self.mode = _normalized_mode(mode)
        self.molarity_element = molarity_element
        self.molarity = molarity
        self.solution_density_g_per_mL = solution_density
        self.solution_density = solution_density
        self.solute_stoich = solute_stoich
        self.solvent_stoich = solvent_stoich
        self.molar_mass_solute = molar_mass_solute
        self.molar_mass_solvent = molar_mass_solvent
        self.solute_dict = self._parse_stoichiometry(solute_stoich)
        self.solvent_dict = self._parse_stoichiometry(solvent_stoich)
        self.moles_solute = 0.0
        self.moles_solvent = 0.0
        self.mass_solute = 0.0
        self.mass_solvent = 0.0
        self.total_mass_solution = 0.0
        self.volume_solution_cm3 = 0.0

        if self.mode == "mass":
            if mass_solute is None or mass_solvent is None:
                raise ValueError(
                    "For mode='mass', provide mass_solute and mass_solvent."
                )
            self.mass_solute = mass_solute
            self.mass_solvent = mass_solvent
            self.total_mass_solution = self.mass_solute + self.mass_solvent
            self.volume_solution_cm3 = (
                self.total_mass_solution / self.solution_density
            )
        elif self.mode == "mass_percent":
            if mass_percent_solute is None or total_mass_solution is None:
                raise ValueError(
                    "For mode='mass_percent', provide mass_percent_solute "
                    "and total_mass_solution."
                )
            self.mass_solute = (
                mass_percent_solute / 100.0
            ) * total_mass_solution
            self.mass_solvent = total_mass_solution - self.mass_solute
            self.total_mass_solution = total_mass_solution
            self.volume_solution_cm3 = (
                self.total_mass_solution / self.solution_density
            )
        elif self.mode == "molarity_per_liter":
            if molarity is None or molarity_element is None:
                raise ValueError(
                    "For mode='molarity_per_liter', provide molarity and "
                    "molarity_element."
                )
            if molarity_element not in self.solute_dict:
                raise ValueError(
                    f"Element '{molarity_element}' is not in solute "
                    f"stoichiometry '{solute_stoich}'."
                )
            element_count = self.solute_dict[molarity_element]
            self.moles_solute = molarity / element_count
            self.mass_solute = self.moles_solute * self.molar_mass_solute
            self.total_mass_solution = 1000.0 * self.solution_density_g_per_mL
            self.mass_solvent = self.total_mass_solution - self.mass_solute
            if self.mass_solvent < 0:
                raise ValueError(
                    "Calculated negative solvent mass for the requested "
                    "density and molarity."
                )
            self.volume_solution_cm3 = 1000.0
        else:
            raise ValueError(
                "mode must be 'mass', 'mass_percent', or "
                "'molarity_per_liter'."
            )

        if self.mode in {"mass", "mass_percent"}:
            self.moles_solute = self.mass_solute / self.molar_mass_solute
        if self.mass_solvent > 0.0:
            self.moles_solvent = self.mass_solvent / self.molar_mass_solvent
        else:
            self.moles_solvent = 0.0

        self.element_mole_dict = self._compute_element_moles()
        self.total_atoms = sum(
            moles * self.AVOGADRO for moles in self.element_mole_dict.values()
        )
        if self.volume_solution_cm3 <= 0:
            raise ValueError("Calculated non-positive solution volume.")
        self.number_density_cm3 = self.total_atoms / self.volume_solution_cm3
        self.number_density_A3 = (
            self.number_density_cm3 / self.CM3_TO_ANGSTROM3
        )
        self.element_ratio_string = self._compute_element_ratio_string()

    @classmethod
    def from_settings(
        cls,
        settings: SolutionPropertiesSettings,
    ) -> "SolutionProperties":
        _validate_solution_settings(settings)
        return cls(
            mode=settings.mode,
            solution_density=settings.solution_density,
            solute_stoich=settings.solute_stoich,
            solvent_stoich=settings.solvent_stoich,
            molar_mass_solute=settings.molar_mass_solute,
            molar_mass_solvent=settings.molar_mass_solvent,
            mass_solute=settings.mass_solute,
            mass_solvent=settings.mass_solvent,
            mass_percent_solute=settings.mass_percent_solute,
            total_mass_solution=settings.total_mass_solution,
            molarity=settings.molarity,
            molarity_element=settings.molarity_element or None,
        )

    def to_result(self) -> SolutionPropertiesResult:
        return SolutionPropertiesResult(
            mode=self.mode,
            solution_density_g_per_ml=self.solution_density_g_per_mL,
            solution_density_g_per_cm3=self.solution_density,
            solute_stoich=self.solute_stoich,
            solvent_stoich=self.solvent_stoich,
            solute_dict=dict(self.solute_dict),
            solvent_dict=dict(self.solvent_dict),
            element_mole_dict=dict(self.element_mole_dict),
            volume_solution_cm3=self.volume_solution_cm3,
            mass_solute=self.mass_solute,
            mass_solvent=self.mass_solvent,
            total_mass_solution=self.total_mass_solution,
            moles_solute=self.moles_solute,
            moles_solvent=self.moles_solvent,
            total_atoms=self.total_atoms,
            number_density_cm3=self.number_density_cm3,
            number_density_a3=self.number_density_A3,
            element_ratio_string=self.element_ratio_string,
        )

    def _parse_stoichiometry(self, formula: str) -> dict[str, int]:
        pattern = r"([A-Z][a-z]?)(\d*)"
        parts = re.findall(pattern, formula)
        result: dict[str, int] = {}
        for element, count_text in parts:
            count = int(count_text) if count_text.strip() else 1
            result[element] = result.get(element, 0) + count
        return result

    def _compute_element_moles(self) -> dict[str, float]:
        element_moles: dict[str, float] = {}
        for element, count in self.solute_dict.items():
            element_moles[element] = (
                element_moles.get(element, 0.0) + count * self.moles_solute
            )
        for element, count in self.solvent_dict.items():
            element_moles[element] = (
                element_moles.get(element, 0.0) + count * self.moles_solvent
            )
        return element_moles

    def _compute_element_ratio_string(self) -> str:
        nonzero_moles = [
            value for value in self.element_mole_dict.values() if value > 0
        ]
        if not nonzero_moles:
            return ""
        min_moles = min(nonzero_moles)
        ratio_dict: dict[str, int] = {}
        for element, moles in self.element_mole_dict.items():
            if moles > 0:
                ratio_dict[element] = round(moles / min_moles)
        return "".join(
            f"{element}{ratio_dict[element]}" for element in sorted(ratio_dict)
        )

    def get_box_composition(
        self,
        side_length_A: float,
        *,
        round_values: bool = True,
    ) -> dict[str, object]:
        volume_A3 = side_length_A**3
        total_atoms_in_box = self.number_density_A3 * volume_A3
        if self.moles_solvent > 0:
            ratio_molecules = self.moles_solute / self.moles_solvent
        else:
            ratio_molecules = float("inf") if self.moles_solute > 0 else 0.0

        atoms_per_solute = (
            sum(self.solute_dict.values()) if self.solute_dict else 0
        )
        atoms_per_solvent = (
            sum(self.solvent_dict.values()) if self.solvent_dict else 0
        )

        if ratio_molecules == float("inf"):
            n_solvent = 0.0
            n_solute = (
                total_atoms_in_box / atoms_per_solute
                if atoms_per_solute > 0
                else 0.0
            )
        elif ratio_molecules == 0:
            n_solute = 0.0
            n_solvent = (
                total_atoms_in_box / atoms_per_solvent
                if atoms_per_solvent > 0
                else 0.0
            )
        else:
            denominator = (
                atoms_per_solute * ratio_molecules + atoms_per_solvent
            )
            if denominator > 0:
                n_solvent = total_atoms_in_box / denominator
                n_solute = ratio_molecules * n_solvent
            else:
                n_solvent = 0.0
                n_solute = 0.0

        if round_values:
            n_solute_out = round(n_solute)
            n_solvent_out = round(n_solvent)
        else:
            n_solute_out = n_solute
            n_solvent_out = n_solvent

        actual_total_atoms = (
            atoms_per_solute * n_solute_out + atoms_per_solvent * n_solvent_out
        )
        return {
            "box_side_length_A": side_length_A,
            "box_volume_A3": volume_A3,
            "total_atoms_in_box": total_atoms_in_box,
            "solute_molecules": n_solute_out,
            "solvent_molecules": n_solvent_out,
            "solute_to_solvent_molecule_ratio": (
                f"{n_solute_out}:{n_solvent_out}"
                if round_values
                else f"{n_solute:.4f}:{n_solvent:.4f}"
            ),
            "actual_total_atoms": actual_total_atoms,
        }

    def get_box_for_solute_molecules(
        self,
        solute_molecule_count: float,
        *,
        round_values: bool = True,
    ) -> dict[str, object]:
        if self.moles_solvent > 0:
            ratio_molecules = self.moles_solute / self.moles_solvent
        else:
            if self.moles_solute <= 0:
                raise ValueError(
                    "Invalid solution with zero solute and solvent."
                )
            ratio_molecules = float("inf")

        atoms_per_solute = sum(self.solute_dict.values())
        atoms_per_solvent = sum(self.solvent_dict.values())
        if ratio_molecules == float("inf"):
            solvent_molecule_count = 0.0
        else:
            solvent_molecule_count = solute_molecule_count / ratio_molecules

        if round_values:
            solvent_molecule_count = round(solvent_molecule_count)

        total_atoms_box = (
            atoms_per_solute * solute_molecule_count
            + atoms_per_solvent * solvent_molecule_count
        )
        if self.number_density_A3 <= 0:
            raise ValueError("Non-positive number density.")
        box_volume_A3 = total_atoms_box / self.number_density_A3
        if box_volume_A3 < 0:
            raise ValueError("Calculated negative box volume.")
        return {
            "solute_molecules": solute_molecule_count,
            "solvent_molecules": solvent_molecule_count,
            "box_volume_A3": box_volume_A3,
            "box_side_length_A": box_volume_A3 ** (1.0 / 3.0),
            "total_atoms": total_atoms_box,
        }

    def adjust_box_for_target_atom_count(
        self,
        target_atoms: float,
        solute_molecule_count: float,
        *,
        round_values: bool = True,
    ) -> dict[str, object]:
        atoms_per_solute = sum(self.solute_dict.values())
        atoms_per_solvent = sum(self.solvent_dict.values())
        min_atoms = atoms_per_solute * solute_molecule_count
        if target_atoms < min_atoms:
            raise ValueError(
                "Target atoms is less than the atoms contributed by the "
                "solute molecules alone."
            )

        if self.moles_solvent > 0 and self.moles_solute > 0:
            ratio = self.moles_solute / self.moles_solvent
            original_solvent = solute_molecule_count / ratio
        else:
            original_solvent = 0.0

        adjusted_solvent = (
            target_atoms - atoms_per_solute * solute_molecule_count
        ) / atoms_per_solvent
        if adjusted_solvent < 0:
            raise ValueError("Computed negative solvent molecule count.")

        if round_values:
            solute_out = round(solute_molecule_count)
            adjusted_solvent_out = round(adjusted_solvent)
            original_solvent_out = round(original_solvent)
        else:
            solute_out = solute_molecule_count
            adjusted_solvent_out = adjusted_solvent
            original_solvent_out = original_solvent

        total_atoms_adjusted = (
            atoms_per_solute * solute_out
            + atoms_per_solvent * adjusted_solvent_out
        )
        new_box_volume = target_atoms / self.number_density_A3
        return {
            "target_atoms": target_atoms,
            "solute_molecules": solute_out,
            "original_solvent_molecules": original_solvent_out,
            "adjusted_solvent_molecules": adjusted_solvent_out,
            "solvent_molecules_removed": (
                original_solvent_out - adjusted_solvent_out
            ),
            "new_box_volume_A3": new_box_volume,
            "new_box_side_length_A": new_box_volume ** (1.0 / 3.0),
            "total_atoms": total_atoms_adjusted,
        }

    def get_concentration_and_box_for_target_atoms(
        self,
        target_atoms: float,
        solute_molecule_count: float,
        conc_coef_a: float,
        conc_coef_b: float,
        *,
        round_values: bool = True,
    ) -> dict[str, object]:
        atoms_per_solute = sum(self.solute_dict.values())
        atoms_per_solvent = sum(self.solvent_dict.values())
        if target_atoms < solute_molecule_count * atoms_per_solute:
            raise ValueError(
                "Target atoms is less than the atoms contributed by the "
                "fixed solute molecules."
            )

        n_solvent = (
            target_atoms - solute_molecule_count * atoms_per_solute
        ) / atoms_per_solvent
        if round_values:
            n_solvent = round(n_solvent)

        actual_total_atoms = (
            solute_molecule_count * atoms_per_solute
            + n_solvent * atoms_per_solvent
        )
        box_volume_A3 = actual_total_atoms / self.number_density_A3
        box_volume_L = box_volume_A3 * 1e-27
        solute_moles = solute_molecule_count / self.AVOGADRO
        concentration = solute_moles / box_volume_L
        return {
            "target_atoms": target_atoms,
            "solute_molecule_count": solute_molecule_count,
            "computed_solvent_molecule_count": n_solvent,
            "actual_total_atoms": actual_total_atoms,
            "computed_concentration_molarity": concentration,
            "extrapolated_density": conc_coef_a * concentration + conc_coef_b,
            "box_volume_A3": box_volume_A3,
            "box_side_length_A": box_volume_A3 ** (1.0 / 3.0),
            "box_number_density": actual_total_atoms / box_volume_A3,
        }


def calculate_solution_properties(
    settings: SolutionPropertiesSettings,
) -> SolutionPropertiesResult:
    return SolutionProperties.from_settings(settings).to_result()


def load_solution_properties_metadata(
    metadata_path: str | Path,
) -> SolutionPropertiesMetadata:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return SolutionPropertiesMetadata(
            settings=SolutionPropertiesSettings(),
            result=None,
            updated_at=None,
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}
    settings_payload: dict[str, object] | None
    if "settings" in payload or "result" in payload:
        settings_payload = payload.get("settings")  # type: ignore[assignment]
        result_payload = payload.get("result")
        updated_at = _optional_text(payload.get("updated_at"))
    else:
        settings_payload = payload
        result_payload = None
        updated_at = None
    return SolutionPropertiesMetadata(
        settings=SolutionPropertiesSettings.from_dict(settings_payload),
        result=SolutionPropertiesResult.from_dict(result_payload),
        updated_at=updated_at,
    )


def save_solution_properties_metadata(
    metadata_path: str | Path,
    *,
    settings: SolutionPropertiesSettings,
    result: SolutionPropertiesResult | None,
    updated_at: str | None = None,
) -> SolutionPropertiesMetadata:
    timestamp = updated_at or datetime.now().isoformat(timespec="seconds")
    path = Path(metadata_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "updated_at": timestamp,
        "settings": settings.to_dict(),
        "result": result.to_dict() if result is not None else None,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return SolutionPropertiesMetadata(
        settings=settings,
        result=result,
        updated_at=timestamp,
    )


def _validate_solution_settings(
    settings: SolutionPropertiesSettings,
) -> None:
    if settings.solution_density <= 0:
        raise ValueError("Solution density must be greater than zero.")
    if not settings.solute_stoich.strip():
        raise ValueError("Enter a solute stoichiometry.")
    if not settings.solvent_stoich.strip():
        raise ValueError("Enter a solvent stoichiometry.")
    if settings.molar_mass_solute <= 0:
        raise ValueError("Solute molar mass must be greater than zero.")
    if settings.molar_mass_solvent <= 0:
        raise ValueError("Solvent molar mass must be greater than zero.")

    if settings.mode == "mass":
        if settings.mass_solute < 0 or settings.mass_solvent < 0:
            raise ValueError("Mass inputs cannot be negative.")
        if settings.mass_solute == 0 and settings.mass_solvent == 0:
            raise ValueError(
                "Enter a non-zero mass for the solute or solvent."
            )
        return

    if settings.mode == "mass_percent":
        if not 0 <= settings.mass_percent_solute <= 100:
            raise ValueError("Mass percent must be between 0 and 100.")
        if settings.total_mass_solution <= 0:
            raise ValueError("Total solution mass must be greater than zero.")
        return

    if settings.mode == "molarity_per_liter":
        if settings.molarity <= 0:
            raise ValueError("Molarity must be greater than zero.")
        if not settings.molarity_element.strip():
            raise ValueError(
                "Enter the element whose molarity defines the solution."
            )
        return

    raise ValueError(f"Unsupported solution-properties mode: {settings.mode}")


def _normalized_mode(value: object) -> str:
    text = str(value or _MODE_DEFAULT).strip().lower()
    if text not in {"mass", "mass_percent", "molarity_per_liter"}:
        return _MODE_DEFAULT
    return text


def _float_value(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "SolutionProperties",
    "SolutionPropertiesMetadata",
    "SolutionPropertiesResult",
    "SolutionPropertiesSettings",
    "calculate_solution_properties",
    "load_solution_properties_metadata",
    "save_solution_properties_metadata",
    "solution_properties_mode_hint_text",
]
