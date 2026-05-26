from __future__ import annotations

import pytest

from saxshell.saxs.stoichiometry import build_stoichiometry_target


def test_build_stoichiometry_target_normalizes_element_symbols() -> None:
    target = build_stoichiometry_target("pb, i", "1:2")

    assert target is not None
    assert target.elements == ("Pb", "I")
    assert target.ratio == (1.0, 2.0)
    assert target.normalized_ratio == (1.0, 2.0)


def test_build_stoichiometry_target_rejects_malformed_input() -> None:
    with pytest.raises(ValueError, match="numeric values"):
        build_stoichiometry_target("Pb, I", "1:two")

    with pytest.raises(ValueError, match="element symbols"):
        build_stoichiometry_target("Pb2, I", "1:2")

    with pytest.raises(ValueError, match="unique"):
        build_stoichiometry_target("Pb, pb", "1:2")

    with pytest.raises(ValueError, match="finite positive"):
        build_stoichiometry_target("Pb, I", "1:nan")
