from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass

_ELEMENT_SYMBOL_PATTERN = re.compile(r"^[A-Za-z]{1,2}$")
_ELEMENT_SYMBOLS = frozenset(
    {
        "H",
        "He",
        "Li",
        "Be",
        "B",
        "C",
        "N",
        "O",
        "F",
        "Ne",
        "Na",
        "Mg",
        "Al",
        "Si",
        "P",
        "S",
        "Cl",
        "Ar",
        "K",
        "Ca",
        "Sc",
        "Ti",
        "V",
        "Cr",
        "Mn",
        "Fe",
        "Co",
        "Ni",
        "Cu",
        "Zn",
        "Ga",
        "Ge",
        "As",
        "Se",
        "Br",
        "Kr",
        "Rb",
        "Sr",
        "Y",
        "Zr",
        "Nb",
        "Mo",
        "Tc",
        "Ru",
        "Rh",
        "Pd",
        "Ag",
        "Cd",
        "In",
        "Sn",
        "Sb",
        "Te",
        "I",
        "Xe",
        "Cs",
        "Ba",
        "La",
        "Ce",
        "Pr",
        "Nd",
        "Pm",
        "Sm",
        "Eu",
        "Gd",
        "Tb",
        "Dy",
        "Ho",
        "Er",
        "Tm",
        "Yb",
        "Lu",
        "Hf",
        "Ta",
        "W",
        "Re",
        "Os",
        "Ir",
        "Pt",
        "Au",
        "Hg",
        "Tl",
        "Pb",
        "Bi",
        "Po",
        "At",
        "Rn",
        "Fr",
        "Ra",
        "Ac",
        "Th",
        "Pa",
        "U",
        "Np",
        "Pu",
        "Am",
        "Cm",
        "Bk",
        "Cf",
        "Es",
        "Fm",
        "Md",
        "No",
        "Lr",
        "Rf",
        "Db",
        "Sg",
        "Bh",
        "Hs",
        "Mt",
        "Ds",
        "Rg",
        "Cn",
        "Nh",
        "Fl",
        "Mc",
        "Lv",
        "Ts",
        "Og",
    }
)

DEFAULT_FORMAL_CHARGE_BY_SPECIES = {
    "Pb": 2.0,
    "Sn": 2.0,
    "I": -1.0,
    "Br": -1.0,
    "Cl": -1.0,
    "F": -1.0,
    "Cs": 1.0,
    "Rb": 1.0,
    "K": 1.0,
    "Na": 1.0,
    "Li": 1.0,
    "MA": 1.0,
    "FA": 1.0,
    "NH4": 1.0,
}


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


@dataclass(slots=True, frozen=True)
class FormalChargeComponent:
    label: str
    weight: float
    signed_charge_e: float
    counts: dict[str, int]
    unassigned_species: tuple[str, ...]


@dataclass(slots=True, frozen=True)
class FormalChargeEstimate:
    components: tuple[FormalChargeComponent, ...]
    weighted_mean_signed_charge_e: float | None
    weighted_mean_absolute_charge_e: float | None
    absolute_weighted_mean_charge_e: float | None
    total_weight: float
    unassigned_species: tuple[str, ...]

    @property
    def is_valid(self) -> bool:
        return self.absolute_weighted_mean_charge_e is not None


def parse_stoichiometry_elements_input(
    value: str | Iterable[str],
) -> tuple[str, ...]:
    if isinstance(value, str):
        tokens = re.split(r"[\s,:;]+", value.strip())
    else:
        tokens = [str(token).strip() for token in value]
    elements = tuple(
        _normalize_stoichiometry_element_token(token)
        for token in tokens
        if str(token).strip()
    )
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
        try:
            ratios.append(float(token))
        except ValueError as exc:
            raise ValueError(
                "Stoichiometry target ratios must be numeric values."
            ) from exc
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
    if len(set(parsed_elements)) != len(parsed_elements):
        raise ValueError("Stoichiometry target elements must be unique.")
    if len(parsed_elements) < 2:
        raise ValueError(
            "Enter at least two target elements to define a stoichiometric ratio."
        )
    if any(
        (not math.isfinite(component)) or component <= 0.0
        for component in parsed_ratio
    ):
        raise ValueError(
            "Stoichiometry target ratios must all be finite positive values."
        )
    reference = float(parsed_ratio[0])
    normalized = tuple(
        float(component) / reference for component in parsed_ratio
    )
    return StoichiometryTarget(
        elements=parsed_elements,
        ratio=tuple(float(component) for component in parsed_ratio),
        normalized_ratio=normalized,
    )


def _normalize_stoichiometry_element_token(token: object) -> str:
    value = str(token).strip()
    if not _ELEMENT_SYMBOL_PATTERN.fullmatch(value):
        raise ValueError(
            "Stoichiometry target elements must be element symbols, "
            "for example Pb, I."
        )
    return value[0].upper() + value[1:].lower()


def parse_formal_charge_species_counts(label: str) -> dict[str, int]:
    species_tokens = (
        frozenset(DEFAULT_FORMAL_CHARGE_BY_SPECIES) | _ELEMENT_SYMBOLS
    )
    ordered_tokens = sorted(species_tokens, key=len, reverse=True)
    counts: dict[str, int] = {}
    index = 0
    value = str(label)
    while index < len(value):
        if not value[index].isupper():
            index += 1
            continue
        matched_token = None
        for token in ordered_tokens:
            if value.startswith(token, index):
                matched_token = token
                break
        if matched_token is None:
            index += 1
            continue
        number_start = index + len(matched_token)
        number_end = number_start
        while number_end < len(value) and value[number_end].isdigit():
            number_end += 1
        count = int(value[number_start:number_end] or "1")
        counts[matched_token] = counts.get(matched_token, 0) + count
        index = number_end
    return counts


def formal_charge_for_stoich_label(
    label: str,
    *,
    charge_states: dict[str, float] | None = None,
) -> tuple[float, dict[str, int], tuple[str, ...]]:
    charges = (
        DEFAULT_FORMAL_CHARGE_BY_SPECIES
        if charge_states is None
        else charge_states
    )
    counts = parse_formal_charge_species_counts(label)
    total_charge = 0.0
    unassigned_species: list[str] = []
    for species, count in counts.items():
        if species in charges:
            total_charge += float(charges[species]) * float(count)
        else:
            unassigned_species.append(species)
    return (
        float(total_charge),
        counts,
        tuple(sorted(unassigned_species)),
    )


def estimate_weighted_formal_charge(
    structure_weights: Iterable[tuple[str, float]],
    *,
    charge_states: dict[str, float] | None = None,
) -> FormalChargeEstimate:
    components: list[FormalChargeComponent] = []
    weighted_signed_sum = 0.0
    weighted_absolute_sum = 0.0
    total_weight = 0.0
    unassigned_species: set[str] = set()
    for label, raw_weight in structure_weights:
        weight = max(float(raw_weight), 0.0)
        if weight <= 0.0:
            continue
        charge, counts, unresolved = formal_charge_for_stoich_label(
            str(label),
            charge_states=charge_states,
        )
        unassigned_species.update(unresolved)
        components.append(
            FormalChargeComponent(
                label=str(label),
                weight=weight,
                signed_charge_e=charge,
                counts=counts,
                unassigned_species=unresolved,
            )
        )
        weighted_signed_sum += weight * charge
        weighted_absolute_sum += weight * abs(charge)
        total_weight += weight
    if total_weight <= 0.0:
        return FormalChargeEstimate(
            components=tuple(components),
            weighted_mean_signed_charge_e=None,
            weighted_mean_absolute_charge_e=None,
            absolute_weighted_mean_charge_e=None,
            total_weight=0.0,
            unassigned_species=tuple(sorted(unassigned_species)),
        )
    weighted_mean_signed = weighted_signed_sum / total_weight
    weighted_mean_absolute = weighted_absolute_sum / total_weight
    return FormalChargeEstimate(
        components=tuple(components),
        weighted_mean_signed_charge_e=float(weighted_mean_signed),
        weighted_mean_absolute_charge_e=float(weighted_mean_absolute),
        absolute_weighted_mean_charge_e=float(abs(weighted_mean_signed)),
        total_weight=float(total_weight),
        unassigned_species=tuple(sorted(unassigned_species)),
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


def weighted_stoichiometry_text(
    structure_weights: Iterable[tuple[str, float]],
    *,
    label: str = "Stoich",
) -> str | None:
    element_totals: dict[str, float] = {}
    for structure, raw_weight in structure_weights:
        try:
            weight = max(float(raw_weight), 0.0)
        except (TypeError, ValueError):
            continue
        if weight <= 0.0:
            continue
        counts = parse_stoich_label(str(structure))
        for element, count in counts.items():
            if count <= 0:
                continue
            element_totals[element] = element_totals.get(element, 0.0) + (
                weight * float(count)
            )
    positive_elements = [
        element
        for element, total in element_totals.items()
        if math.isfinite(total) and total > 0.0
    ]
    if not positive_elements:
        return None
    elements = _weighted_stoichiometry_element_order(positive_elements)
    reference_element = next(
        (
            element
            for element in elements
            if float(element_totals.get(element, 0.0)) > 0.0
        ),
        None,
    )
    if reference_element is None:
        return None
    reference_total = float(element_totals[reference_element])
    if reference_total <= 0.0 or not math.isfinite(reference_total):
        return None
    ratio_values = [
        float(element_totals.get(element, 0.0)) / reference_total
        for element in elements
    ]
    return (
        f"{label}: {':'.join(elements)} = "
        f"{':'.join(f'{value:.4g}' for value in ratio_values)}"
    )


def _weighted_stoichiometry_element_order(
    elements: Iterable[str],
) -> list[str]:
    unique = set(elements)
    if "Pb" in unique or "I" in unique:
        ordered = [element for element in ("Pb", "I") if element in unique]
        ordered.extend(
            sorted(element for element in unique if element not in {"Pb", "I"})
        )
        return ordered
    return sorted(unique)


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
