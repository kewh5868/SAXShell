from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from saxshell.fullrmc.representatives import RepresentativeSelectionMetadata
from saxshell.fullrmc.solution_properties import (
    SolutionProperties,
    SolutionPropertiesMetadata,
)
from saxshell.fullrmc.solvent_handling import SolventHandlingMetadata
from saxshell.structure import PDBStructure

if False:  # pragma: no cover
    from .project_loader import RMCDreamProjectSource


@dataclass(slots=True)
class PackmolPlanningSettings:
    planning_mode: str = "per_element"
    box_side_length_a: float = 100.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "PackmolPlanningSettings":
        source = dict(payload or {})
        mode = str(source.get("planning_mode", "per_element")).strip()
        if mode not in {"per_element", "total"}:
            mode = "per_element"
        try:
            box_side_length_a = float(source.get("box_side_length_a", 100.0))
        except (TypeError, ValueError):
            box_side_length_a = 100.0
        return cls(
            planning_mode=mode,
            box_side_length_a=max(box_side_length_a, 1.0),
        )


@dataclass(slots=True)
class PackmolPlanningEntry:
    structure: str
    motif: str
    param: str
    selected_weight: float
    planned_count: int
    atom_count: int
    element_counts: dict[str, int]
    planned_count_weight: float
    planned_atom_weight: float
    composition_source: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object],
    ) -> "PackmolPlanningEntry":
        return cls(
            structure=str(payload.get("structure", "")).strip(),
            motif=str(payload.get("motif", "no_motif")).strip() or "no_motif",
            param=str(payload.get("param", "")).strip(),
            selected_weight=float(payload.get("selected_weight", 0.0)),
            planned_count=int(payload.get("planned_count", 0)),
            atom_count=int(payload.get("atom_count", 0)),
            element_counts={
                str(key): int(value)
                for key, value in dict(
                    payload.get("element_counts", {})
                ).items()
            },
            planned_count_weight=float(
                payload.get("planned_count_weight", 0.0)
            ),
            planned_atom_weight=float(payload.get("planned_atom_weight", 0.0)),
            composition_source=str(
                payload.get("composition_source", "")
            ).strip(),
        )


@dataclass(slots=True)
class PackmolPlanningMetadata:
    settings: PackmolPlanningSettings
    updated_at: str
    representative_selection_mode: str
    target_box_composition: dict[str, object]
    target_total_number_density_a3: float
    target_element_number_density_a3: dict[str, float]
    achieved_total_number_density_a3: float
    achieved_element_number_density_a3: dict[str, float]
    entries: list[PackmolPlanningEntry]
    report_text: str

    def to_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "settings": self.settings.to_dict(),
            "updated_at": self.updated_at,
            "representative_selection_mode": (
                self.representative_selection_mode
            ),
            "target_box_composition": dict(self.target_box_composition),
            "target_total_number_density_a3": (
                self.target_total_number_density_a3
            ),
            "target_element_number_density_a3": dict(
                self.target_element_number_density_a3
            ),
            "achieved_total_number_density_a3": (
                self.achieved_total_number_density_a3
            ),
            "achieved_element_number_density_a3": dict(
                self.achieved_element_number_density_a3
            ),
            "entries": [entry.to_dict() for entry in self.entries],
            "report_text": self.report_text,
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "PackmolPlanningMetadata | None":
        if not payload:
            return None
        return cls(
            settings=PackmolPlanningSettings.from_dict(
                payload.get("settings")
                if isinstance(payload.get("settings"), dict)
                else None
            ),
            updated_at=str(payload.get("updated_at", "")).strip(),
            representative_selection_mode=str(
                payload.get("representative_selection_mode", "")
            ).strip(),
            target_box_composition=dict(
                payload.get("target_box_composition", {})
            ),
            target_total_number_density_a3=float(
                payload.get("target_total_number_density_a3", 0.0)
            ),
            target_element_number_density_a3={
                str(key): float(value)
                for key, value in dict(
                    payload.get("target_element_number_density_a3", {})
                ).items()
            },
            achieved_total_number_density_a3=float(
                payload.get("achieved_total_number_density_a3", 0.0)
            ),
            achieved_element_number_density_a3={
                str(key): float(value)
                for key, value in dict(
                    payload.get("achieved_element_number_density_a3", {})
                ).items()
            },
            entries=[
                PackmolPlanningEntry.from_dict(dict(entry))
                for entry in payload.get("entries", [])
                if isinstance(entry, dict)
            ],
            report_text=str(payload.get("report_text", "")).strip(),
        )

    def summary_text(self) -> str:
        lines = [
            f"Planning mode: {self.settings.planning_mode}",
            f"Box side: {self.settings.box_side_length_a:.3f} A",
            f"Saved at: {self.updated_at}",
            f"Planned clusters: {sum(entry.planned_count for entry in self.entries)}",
            (
                "Target total number density: "
                f"{self.target_total_number_density_a3:.6g} atoms/A^3"
            ),
            (
                "Achieved total number density: "
                f"{self.achieved_total_number_density_a3:.6g} atoms/A^3"
            ),
        ]
        if self.entries:
            first = self.entries[0]
            lines.extend(
                [
                    "",
                    "Highest-priority cluster:",
                    f"  {first.structure}/{first.motif}",
                    f"  planned count: {first.planned_count}",
                    f"  selected weight: {first.selected_weight:.6g}",
                    f"  count weight: {first.planned_count_weight:.6g}",
                ]
            )
        return "\n".join(lines)


def build_packmol_plan(
    project_source: "RMCDreamProjectSource",
    settings: PackmolPlanningSettings,
    *,
    representative_metadata: RepresentativeSelectionMetadata | None = None,
    solution_metadata: SolutionPropertiesMetadata | None = None,
    solvent_metadata: SolventHandlingMetadata | None = None,
) -> PackmolPlanningMetadata:
    active_representatives = (
        representative_metadata or project_source.representative_selection
    )
    if (
        active_representatives is None
        or not active_representatives.representative_entries
    ):
        raise ValueError(
            "Compute representative clusters before planning the Packmol box."
        )
    active_solution = solution_metadata or project_source.solution_properties
    if active_solution.result is None:
        raise ValueError(
            "Calculate solution properties before planning the Packmol box."
        )

    solution = SolutionProperties.from_settings(active_solution.settings)
    box_targets = solution.get_box_composition(
        settings.box_side_length_a,
        round_values=True,
    )
    target_total_nd = float(solution.number_density_A3)
    target_element_nd = _element_number_density(solution)

    composition_lookup = _build_composition_lookup(
        active_representatives,
        solvent_metadata or project_source.solvent_handling,
    )
    keys: list[tuple[str, str, str]] = []
    weights: list[float] = []
    atom_counts: list[int] = []
    element_keys: set[str] = set()
    composition_sources: list[str] = []
    for entry in active_representatives.representative_entries:
        lookup_key = (entry.structure, entry.motif, entry.param)
        composition, atom_count, composition_source = composition_lookup[
            lookup_key
        ]
        keys.append(lookup_key)
        weights.append(max(entry.selected_weight, 0.0))
        atom_counts.append(atom_count)
        composition_sources.append(composition_source)
        element_keys.update(composition.keys())

    if not keys:
        raise ValueError(
            "No representative entries are available for planning."
        )

    ordered_elements = sorted(element_keys)
    element_matrix = np.zeros((len(ordered_elements), len(keys)), dtype=float)
    atom_vector = np.zeros(len(keys), dtype=float)
    for index, key in enumerate(keys):
        composition, atom_count, _composition_source = composition_lookup[key]
        atom_vector[index] = float(atom_count)
        for element_index, element in enumerate(ordered_elements):
            element_matrix[element_index, index] = float(
                composition.get(element, 0)
            )

    counts = _plan_counts(
        weights=np.asarray(weights, dtype=float),
        atom_vector=atom_vector,
        element_matrix=element_matrix,
        ordered_elements=ordered_elements,
        box_side_length_a=settings.box_side_length_a,
        target_total_nd=target_total_nd,
        target_element_nd=target_element_nd,
        planning_mode=settings.planning_mode,
    )
    planned_count_weights = _normalized_weights(counts.astype(float))
    planned_atom_weights = _normalized_weights(counts * atom_vector)

    volume_a3 = settings.box_side_length_a**3
    achieved_element_nd = {
        element: float(
            np.dot(counts, element_matrix[element_index]) / volume_a3
        )
        for element_index, element in enumerate(ordered_elements)
    }
    achieved_total_nd = float(np.dot(counts, atom_vector) / volume_a3)

    representative_entries = active_representatives.representative_entries
    entries: list[PackmolPlanningEntry] = []
    for index, representative_entry in enumerate(representative_entries):
        composition, atom_count, composition_source = composition_lookup[
            (
                representative_entry.structure,
                representative_entry.motif,
                representative_entry.param,
            )
        ]
        entries.append(
            PackmolPlanningEntry(
                structure=representative_entry.structure,
                motif=representative_entry.motif,
                param=representative_entry.param,
                selected_weight=float(weights[index]),
                planned_count=int(counts[index]),
                atom_count=int(atom_count),
                element_counts=dict(composition),
                planned_count_weight=float(planned_count_weights[index]),
                planned_atom_weight=float(planned_atom_weights[index]),
                composition_source=composition_source,
            )
        )

    report_text = _build_plan_report(
        settings=settings,
        box_targets=box_targets,
        entries=entries,
        target_total_nd=target_total_nd,
        target_element_nd=target_element_nd,
        achieved_total_nd=achieved_total_nd,
        achieved_element_nd=achieved_element_nd,
    )
    metadata = PackmolPlanningMetadata(
        settings=settings,
        updated_at=datetime.now().isoformat(timespec="seconds"),
        representative_selection_mode=active_representatives.selection_mode,
        target_box_composition=box_targets,
        target_total_number_density_a3=target_total_nd,
        target_element_number_density_a3=target_element_nd,
        achieved_total_number_density_a3=achieved_total_nd,
        achieved_element_number_density_a3=achieved_element_nd,
        entries=entries,
        report_text=report_text,
    )
    save_packmol_planning_metadata(
        project_source.rmcsetup_paths.packmol_plan_path,
        metadata,
    )
    _write_plan_reports(project_source, metadata)
    return metadata


def save_packmol_planning_metadata(
    output_path: str | Path,
    metadata: PackmolPlanningMetadata,
) -> Path:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(metadata.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def load_packmol_planning_metadata(
    metadata_path: str | Path,
) -> PackmolPlanningMetadata | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return PackmolPlanningMetadata.from_dict(payload)


def _element_number_density(solution: SolutionProperties) -> dict[str, float]:
    result: dict[str, float] = {}
    if solution.volume_solution_cm3 <= 0:
        return result
    for element, moles in solution.element_mole_dict.items():
        atoms = float(moles) * solution.AVOGADRO
        nd_cm3 = atoms / solution.volume_solution_cm3
        result[element] = nd_cm3 / solution.CM3_TO_ANGSTROM3
    return result


def _build_composition_lookup(
    metadata: RepresentativeSelectionMetadata,
    solvent_metadata: SolventHandlingMetadata | None,
) -> dict[tuple[str, str, str], tuple[dict[str, int], int, str]]:
    lookup: dict[tuple[str, str, str], tuple[dict[str, int], int, str]] = {}
    solvent_lookup: dict[tuple[str, str, str], Path] = {}
    if solvent_metadata is not None:
        for entry in solvent_metadata.entries:
            solvent_lookup[(entry.structure, entry.motif, entry.param)] = Path(
                entry.no_solvent_pdb
            )

    for entry in metadata.representative_entries:
        key = (entry.structure, entry.motif, entry.param)
        no_solvent_pdb = solvent_lookup.get(key)
        if no_solvent_pdb is not None and no_solvent_pdb.is_file():
            structure = PDBStructure.from_file(no_solvent_pdb)
            counts: dict[str, int] = {}
            for atom in structure.atoms:
                counts[atom.element] = counts.get(atom.element, 0) + 1
            lookup[key] = (
                counts,
                len(structure.atoms),
                "pdb_no_solvent",
            )
            continue
        lookup[key] = (
            dict(entry.element_counts),
            int(entry.atom_count),
            "representative_selection",
        )
    return lookup


def _plan_counts(
    *,
    weights: np.ndarray,
    atom_vector: np.ndarray,
    element_matrix: np.ndarray,
    ordered_elements: list[str],
    box_side_length_a: float,
    target_total_nd: float,
    target_element_nd: dict[str, float],
    planning_mode: str,
) -> np.ndarray:
    volume_a3 = float(box_side_length_a) ** 3
    if planning_mode == "per_element":
        cluster_targets = np.array(
            [
                float(target_element_nd.get(element, 0.0))
                for element in ordered_elements
            ],
            dtype=float,
        )
        if np.any(cluster_targets > 0.0):
            target_atoms = cluster_targets * volume_a3
            baseline = element_matrix @ weights
            denom = float(np.dot(baseline, baseline))
            denom = denom if denom > 0 else 1.0
            scale = float(np.dot(target_atoms, baseline) / denom)
            candidate = _largest_remainders(weights * max(scale, 0.0))
            best_error = _vector_error(
                target_atoms,
                element_matrix @ candidate,
            )
            for _ in range(10_000):
                residual = target_atoms - (element_matrix @ candidate)
                gains = element_matrix.T @ residual
                index = int(np.argmax(gains))
                if gains[index] <= 0:
                    break
                candidate[index] += 1
                new_error = _vector_error(
                    target_atoms,
                    element_matrix @ candidate,
                )
                if new_error >= best_error:
                    candidate[index] -= 1
                    break
                best_error = new_error
            return candidate.astype(int)

    total_target_atoms = float(target_total_nd) * volume_a3
    denom = float(np.dot(weights, atom_vector))
    denom = denom if denom > 0 else 1.0
    scale = total_target_atoms / denom
    candidate = _largest_remainders(weights * max(scale, 0.0))
    total_atoms = float(np.dot(candidate, atom_vector))
    normalized_weights = _normalized_weights(weights)
    while total_atoms + float(np.min(atom_vector)) <= total_target_atoms:
        next_counts = candidate + 1
        deviations = np.abs(
            _normalized_weights(next_counts.astype(float)) - normalized_weights
        )
        index = int(np.lexsort((-atom_vector, deviations))[0])
        candidate[index] += 1
        total_atoms = float(np.dot(candidate, atom_vector))
    return candidate.astype(int)


def _largest_remainders(values: np.ndarray) -> np.ndarray:
    total_count = int(max(1, round(float(np.sum(values)))))
    if total_count <= 0 or float(np.sum(values)) <= 0:
        return np.zeros_like(values, dtype=int)
    raw = values / float(np.sum(values)) * total_count
    base = np.floor(raw).astype(int)
    remainders = raw - base
    shortfall = total_count - int(np.sum(base))
    if shortfall > 0:
        indices = np.argsort(-remainders)[:shortfall]
        base[indices] += 1
    return base


def _normalized_weights(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= 0:
        return np.zeros_like(values, dtype=float)
    return np.asarray(values, dtype=float) / total


def _vector_error(target: np.ndarray, achieved: np.ndarray) -> float:
    return float(np.linalg.norm(target - achieved, ord=2))


def _build_plan_report(
    *,
    settings: PackmolPlanningSettings,
    box_targets: dict[str, object],
    entries: list[PackmolPlanningEntry],
    target_total_nd: float,
    target_element_nd: dict[str, float],
    achieved_total_nd: float,
    achieved_element_nd: dict[str, float],
) -> str:
    lines = [
        "== Packmol Planning ==",
        f"Planning mode: {settings.planning_mode}",
        f"Box side: {settings.box_side_length_a:.3f} A",
        (
            "Target bulk composition: "
            f"{int(box_targets.get('solute_molecules', 0))} solute molecules, "
            f"{int(box_targets.get('solvent_molecules', 0))} solvent molecules"
        ),
        ("Target total number density: " f"{target_total_nd:.6f} atoms/A^3"),
        (
            "Achieved total number density (cluster plan): "
            f"{achieved_total_nd:.6f} atoms/A^3"
        ),
        "",
        "Counts per cluster bin:",
    ]
    for entry in entries:
        lines.append(
            f"  - {entry.structure}/{entry.motif}: {entry.planned_count} "
            f"(selected weight {entry.selected_weight:.6g})"
        )
    if target_element_nd:
        lines.extend(
            [
                "",
                "Per-element number densities:",
            ]
        )
        for element in sorted(target_element_nd):
            lines.append(
                f"  - {element}: target={target_element_nd[element]:.6f}, "
                f"achieved={achieved_element_nd.get(element, 0.0):.6f}"
            )
    return "\n".join(lines)


def _write_plan_reports(
    project_source: "RMCDreamProjectSource",
    metadata: PackmolPlanningMetadata,
) -> None:
    paths = project_source.rmcsetup_paths
    paths.packmol_plan_report_path.write_text(
        metadata.report_text + "\n",
        encoding="utf-8",
    )

    with paths.cluster_counts_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["structure", "motif", "param", "planned_count", "atom_count"]
        )
        for entry in metadata.entries:
            writer.writerow(
                [
                    entry.structure,
                    entry.motif,
                    entry.param,
                    entry.planned_count,
                    entry.atom_count,
                ]
            )

    with paths.planned_count_weights_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure", "motif", "param", "weight_planned"])
        for entry in metadata.entries:
            writer.writerow(
                [
                    entry.structure,
                    entry.motif,
                    entry.param,
                    f"{entry.planned_count_weight:.12g}",
                ]
            )

    with paths.planned_atom_weights_csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(["structure", "motif", "param", "weight_atom_planned"])
        for entry in metadata.entries:
            writer.writerow(
                [
                    entry.structure,
                    entry.motif,
                    entry.param,
                    f"{entry.planned_atom_weight:.12g}",
                ]
            )


__all__ = [
    "PackmolPlanningEntry",
    "PackmolPlanningMetadata",
    "PackmolPlanningSettings",
    "build_packmol_plan",
    "load_packmol_planning_metadata",
    "save_packmol_planning_metadata",
]
