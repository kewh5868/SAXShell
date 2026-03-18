"""Shared structure parsers and atom containers."""

from .pdb_structure import (
    AtomTypeDefinitions,
    PDBAtom,
    PDBStructure,
    normalize_atom_type_definitions,
)

__all__ = [
    "AtomTypeDefinitions",
    "PDBAtom",
    "PDBStructure",
    "normalize_atom_type_definitions",
]
