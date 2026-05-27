from __future__ import annotations

import numpy as np

from saxshell.saxs.stoichiometry import parse_stoich_label

STOICHIOMETRY_COMPENSATOR_TEMPLATE_NAME = (
    "template_pydream_monosq_normalized_scaled_solvent_stoich_compensator"
)
STOICH_TARGET_RATIO_INPUT = "stoich_target_ratio"
STOICH_COMPONENT_COUNTS_INPUT = "stoich_component_counts"
STOICH_COMPENSATOR_MASK_INPUT = "stoich_compensator_mask"
STOICH_COMPENSATOR_BASE_WEIGHTS_INPUT = "stoich_compensator_base_weights"
STOICHIOMETRY_COMPENSATOR_RUNTIME_INPUTS = (
    STOICH_TARGET_RATIO_INPUT,
    STOICH_COMPONENT_COUNTS_INPUT,
    STOICH_COMPENSATOR_MASK_INPUT,
    STOICH_COMPENSATOR_BASE_WEIGHTS_INPUT,
)


def template_uses_stoichiometry_compensator(template_name: object) -> bool:
    return str(template_name or "").strip() == (
        STOICHIOMETRY_COMPENSATOR_TEMPLATE_NAME
    )


def component_count_matrix(
    structures: list[str] | tuple[str, ...],
    elements: list[str] | tuple[str, ...],
) -> np.ndarray:
    rows: list[list[float]] = []
    for structure in structures:
        counts = parse_stoich_label(str(structure))
        rows.append([float(counts.get(element, 0)) for element in elements])
    return np.asarray(rows, dtype=float)


def guess_single_atom_compensator_names(
    components: list[tuple[str, str]] | tuple[tuple[str, str], ...],
    elements: list[str] | tuple[str, ...],
) -> tuple[str, ...]:
    target_elements = {str(element).strip() for element in elements}
    guessed: list[str] = []
    for parameter_name, structure in components:
        counts = parse_stoich_label(str(structure))
        target_atom_count = sum(
            int(counts.get(element, 0)) for element in target_elements
        )
        total_atom_count = sum(int(value) for value in counts.values())
        if total_atom_count == 1 and target_atom_count == 1:
            guessed.append(str(parameter_name))
    return tuple(dict.fromkeys(guessed))


def compute_compensated_weights(
    weights: np.ndarray,
    *,
    target_ratio: np.ndarray,
    component_counts: np.ndarray,
    compensator_mask: np.ndarray,
    compensator_base_weights: np.ndarray,
) -> np.ndarray:
    weights = np.asarray(weights, dtype=float).reshape(-1).copy()
    target_ratio = np.asarray(target_ratio, dtype=float).reshape(-1)
    component_counts = np.asarray(component_counts, dtype=float)
    compensator_mask = (
        np.asarray(compensator_mask, dtype=float).reshape(-1) > 0.5
    )
    base_weights = np.asarray(
        compensator_base_weights,
        dtype=float,
    ).reshape(-1)

    if (
        weights.size == 0
        or target_ratio.size < 2
        or component_counts.shape != (weights.size, target_ratio.size)
        or compensator_mask.size != weights.size
        or base_weights.size != weights.size
        or not np.any(compensator_mask)
    ):
        return weights

    ratio = np.asarray(target_ratio, dtype=float)
    if not np.all(np.isfinite(ratio)) or np.any(ratio <= 0.0):
        return weights

    noncompensator_weights = np.maximum(weights, 0.0)
    noncompensator_weights[compensator_mask] = 0.0
    noncompensator_totals = noncompensator_weights @ component_counts

    compensator_indices = np.flatnonzero(compensator_mask)
    compensator_design = component_counts[compensator_indices, :].T
    if compensator_design.size == 0 or not np.any(compensator_design > 0.0):
        return weights

    design = np.column_stack([compensator_design, -ratio])
    try:
        solution, *_ = np.linalg.lstsq(
            design,
            -noncompensator_totals,
            rcond=None,
        )
    except np.linalg.LinAlgError:
        return weights
    compensator_weights = np.asarray(
        solution[: compensator_indices.size],
        dtype=float,
    )
    if not np.all(np.isfinite(compensator_weights)):
        return weights

    weights[compensator_indices] = np.maximum(compensator_weights, 0.0)
    return weights


__all__ = [
    "STOICH_COMPENSATOR_BASE_WEIGHTS_INPUT",
    "STOICH_COMPENSATOR_MASK_INPUT",
    "STOICH_COMPONENT_COUNTS_INPUT",
    "STOICH_TARGET_RATIO_INPUT",
    "STOICHIOMETRY_COMPENSATOR_RUNTIME_INPUTS",
    "STOICHIOMETRY_COMPENSATOR_TEMPLATE_NAME",
    "component_count_matrix",
    "compute_compensated_weights",
    "guess_single_atom_compensator_names",
    "template_uses_stoichiometry_compensator",
]
