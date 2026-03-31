from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass


def parse_stoich_label(label: str) -> dict[str, int]:
    """Parse a stoichiometry label like ``Pb2I4O`` into element
    counts."""

    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", label)
    counts: dict[str, int] = {}
    for element, number in tokens:
        count = int(number) if number else 1
        counts[element] = counts.get(element, 0) + count
    return counts


@dataclass(slots=True, frozen=True)
class StoichiometryTarget:
    elements: tuple[str, ...]
    ratio: tuple[float, ...]
    normalized_ratio: tuple[float, ...]


@dataclass(slots=True, frozen=True)
class StoichiometryEvaluation:
    target: StoichiometryTarget
    element_totals: dict[str, float]
    observed_ratio: tuple[float, ...] | None
    deviation_percent_by_element: dict[str, float]
    max_deviation_percent: float | None
    is_valid: bool


def parse_stoichiometry_elements_input(
    value: str | Iterable[str],
) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = re.split(r"[\s,:;]+", value.strip())
    else:
        tokens = [str(token).strip() for token in value]
    elements = tuple(token for token in tokens if token)
    return elements


def parse_stoichiometry_ratio_input(
    value: str | Iterable[float | int | str],
) -> tuple[float, ...]:
    if isinstance(value, str):
        tokens = re.split(r"[\s,:;]+", value.strip())
    else:
        tokens = [str(token).strip() for token in value]
    ratios: list[float] = []
    for token in tokens:
        if not token:
            continue
        ratios.append(float(token))
    return tuple(ratios)


def build_stoichiometry_target(
    elements: str | Iterable[str],
    ratio: str | Iterable[float | int | str],
) -> StoichiometryTarget | None:
    parsed_elements = parse_stoichiometry_elements_input(elements)
    parsed_ratio = parse_stoichiometry_ratio_input(ratio)
    if not parsed_elements and not parsed_ratio:
        return None
    if not parsed_elements or not parsed_ratio:
        raise ValueError(
            "Enter both stoichiometry target elements and a target ratio."
        )
    if len(parsed_elements) != len(parsed_ratio):
        raise ValueError(
            "The stoichiometry target elements and ratio must have the same length."
        )
    if len(parsed_elements) < 2:
        raise ValueError(
            "Enter at least two target elements to define a stoichiometric ratio."
        )
    if any(component <= 0.0 for component in parsed_ratio):
        raise ValueError("Stoichiometry target ratios must all be positive.")
    reference = float(parsed_ratio[0])
    normalized = tuple(
        float(component) / reference for component in parsed_ratio
    )
    return StoichiometryTarget(
        elements=parsed_elements,
        ratio=tuple(float(component) for component in parsed_ratio),
        normalized_ratio=normalized,
    )


def evaluate_weighted_stoichiometry(
    structure_weights: Iterable[tuple[str, float]],
    target: StoichiometryTarget,
) -> StoichiometryEvaluation:
    element_totals = {element: 0.0 for element in target.elements}
    for label, raw_weight in structure_weights:
        counts = parse_stoich_label(str(label))
        weight = max(float(raw_weight), 0.0)
        if weight <= 0.0:
            continue
        for element in target.elements:
            element_totals[element] += weight * float(counts.get(element, 0))

    reference_total = float(element_totals.get(target.elements[0], 0.0))
    if reference_total <= 0.0:
        return StoichiometryEvaluation(
            target=target,
            element_totals=element_totals,
            observed_ratio=None,
            deviation_percent_by_element={},
            max_deviation_percent=None,
            is_valid=False,
        )

    observed_ratio = tuple(
        float(element_totals[element]) / reference_total
        for element in target.elements
    )
    deviation_percent_by_element: dict[str, float] = {}
    max_deviation = 0.0
    for index, element in enumerate(target.elements):
        target_value = float(target.normalized_ratio[index])
        observed_value = float(observed_ratio[index])
        if target_value <= 0.0:
            deviation = 0.0 if abs(observed_value) <= 1e-12 else float("inf")
        else:
            deviation = (
                abs(observed_value - target_value) / target_value * 100.0
            )
        deviation_percent_by_element[element] = deviation
        if index > 0:
            max_deviation = max(max_deviation, float(deviation))
    return StoichiometryEvaluation(
        target=target,
        element_totals=element_totals,
        observed_ratio=observed_ratio,
        deviation_percent_by_element=deviation_percent_by_element,
        max_deviation_percent=float(max_deviation),
        is_valid=True,
    )


def stoichiometry_target_text(target: StoichiometryTarget) -> str:
    return (
        f"{':'.join(target.elements)} = "
        f"{':'.join(f'{value:g}' for value in target.ratio)}"
    )


def stoichiometry_ratio_text(
    target: StoichiometryTarget,
    observed_ratio: tuple[float, ...] | None,
) -> str:
    if observed_ratio is None:
        return f"{':'.join(target.elements)} = unavailable"
    return (
        f"{':'.join(target.elements)} = "
        f"{':'.join(f'{value:.4g}' for value in observed_ratio)}"
    )


def stoichiometry_deviation_text(
    evaluation: StoichiometryEvaluation | None,
) -> str:
    if evaluation is None or not evaluation.is_valid:
        return "Deviation from target: unavailable"
    return (
        "Deviation from target: "
        f"{float(evaluation.max_deviation_percent or 0.0):.3g}%"
    )


def format_stoich_for_axis(label: str) -> str:
    """Format a stoichiometry label with mathtext subscripts.

    Pb/I-containing structures follow the legacy order: Pb first, I
    second, then any other elements alphabetically.
    """

    counts = parse_stoich_label(label)
    if not counts:
        return label

    def _format_element(element: str, count: int) -> str:
        if count <= 0:
            return ""
        if count == 1:
            return element
        return f"{element}$_{{{count}}}$"

    elements: list[str]
    if "Pb" in counts or "I" in counts:
        elements = ["Pb", "I"] + sorted(
            element for element in counts if element not in {"Pb", "I"}
        )
    else:
        elements = sorted(counts)
    return (
        "".join(
            _format_element(element, counts.get(element, 0))
            for element in elements
        )
        or label
    )


def stoich_sort_key(label: str) -> tuple[object, ...]:
    """Return a stable sort key for stoichiometry labels.

    Pb/I-containing structures follow the legacy ordering from the deprecated
    cluster histogram workflow:
    1. I-only species
    2. Pb-only species
    3. Mixed Pb/I species

    Within the mixed group, the order is Pb count, then I count, then any
    additional elements such as O.
    """

    counts = parse_stoich_label(label)
    if not counts:
        return (2, _natural_sort_key(label))

    if "Pb" in counts or "I" in counts:
        n_pb = counts.get("Pb", 0)
        n_i = counts.get("I", 0)
        if n_pb == 0 and n_i > 0:
            group = 0
        elif n_pb > 0 and n_i == 0:
            group = 1
        else:
            group = 2
        extra_counts = tuple(
            (element, counts[element])
            for element in sorted(counts)
            if element not in {"Pb", "I"}
        )
        return (0, group, n_pb, n_i, extra_counts, label)

    alphabetical_counts = tuple(
        (element, counts[element]) for element in sorted(counts)
    )
    return (1, alphabetical_counts, label)


def sort_stoich_labels(labels: Iterable[str]) -> list[str]:
    return sorted(labels, key=stoich_sort_key)


def _natural_sort_key(value: str) -> list[object]:
    return [
        int(token) if token.isdigit() else token.lower()
        for token in re.split(r"(\d+)", value)
        if token
    ]
