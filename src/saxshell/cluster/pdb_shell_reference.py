from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class PDBShellReferenceDefinition:
    """Reference-molecule metadata for one PDB shell atom rule.

    The optional backbone fields are retained only for backwards
    compatibility with earlier saved settings and deprecated placement
    experiments. New UI flows only collect the shell rule and reference
    molecule.
    """

    shell_element: str
    shell_residue: str | None
    reference_name: str
    backbone_atom1_name: str | None = None
    backbone_atom2_name: str | None = None


__all__ = ["PDBShellReferenceDefinition"]
