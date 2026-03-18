"""Compatibility shim for the legacy pdbhandlerplus module name."""

from .pdb_structure import PDBAtom, PDBStructure

AtomPlus = PDBAtom
PDBHandlerPlus = PDBStructure

__all__ = [
    "AtomPlus",
    "PDBAtom",
    "PDBHandlerPlus",
    "PDBStructure",
]
