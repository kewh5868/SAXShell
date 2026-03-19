from __future__ import annotations

import re
from collections.abc import Iterable


def parse_stoich_label(label: str) -> dict[str, int]:
    """Parse a stoichiometry label like ``Pb2I4O`` into element
    counts."""

    tokens = re.findall(r"([A-Z][a-z]*)(\d*)", label)
    counts: dict[str, int] = {}
    for element, number in tokens:
        count = int(number) if number else 1
        counts[element] = counts.get(element, 0) + count
    return counts


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
