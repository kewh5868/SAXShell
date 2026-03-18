from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from string import ascii_letters, digits

import numpy as np
from scipy.optimize import linear_sum_assignment

from saxshell.structure import PDBAtom, PDBStructure

_SAFE_FILENAME_CHARS = frozenset(ascii_letters + digits + "-_")


def rotation_matrix_from_to(
    source_vector: Sequence[float],
    target_vector: Sequence[float],
    *,
    tolerance: float = 1.0e-8,
) -> np.ndarray:
    """Return the rotation matrix that aligns one vector to another."""
    source = np.asarray(source_vector, dtype=float)
    target = np.asarray(target_vector, dtype=float)
    source_norm = float(np.linalg.norm(source))
    target_norm = float(np.linalg.norm(target))
    if source_norm < tolerance or target_norm < tolerance:
        raise ValueError("Rotation vectors must have non-zero length.")

    source = source / source_norm
    target = target / target_norm
    cross = np.cross(source, target)
    dot = float(np.dot(source, target))
    cross_norm = float(np.linalg.norm(cross))
    if cross_norm < tolerance:
        if dot > 0.0:
            return np.eye(3)
        perpendicular = np.zeros(3, dtype=float)
        perpendicular[int(np.argmin(np.abs(source)))] = 1.0
        axis = np.cross(source, perpendicular)
        axis = axis / float(np.linalg.norm(axis))
        skew = np.array(
            [
                [0.0, -axis[2], axis[1]],
                [axis[2], 0.0, -axis[0]],
                [-axis[1], axis[0], 0.0],
            ],
            dtype=float,
        )
        return np.eye(3) + 2.0 * (skew @ skew)

    axis = cross / cross_norm
    angle = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=float,
    )
    return np.eye(3) + np.sin(angle) * skew + (
        1.0 - np.cos(angle)
    ) * (skew @ skew)


def rigid_alignment_from_points(
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the rigid-body transform that best aligns two point sets."""
    if source_points.shape != target_points.shape:
        raise ValueError("Point sets must have matching shapes for alignment.")
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Point sets must have shape (n, 3).")
    if len(source_points) < 2:
        raise ValueError("At least two points are required for alignment.")

    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    covariance = source_centered.T @ target_centered
    left_vectors, _singular_values, right_vectors_t = np.linalg.svd(
        covariance
    )
    rotation = left_vectors @ right_vectors_t
    if float(np.linalg.det(rotation)) < 0.0:
        left_vectors[:, -1] *= -1.0
        rotation = left_vectors @ right_vectors_t
    return rotation, source_centroid, target_centroid


def default_reference_library_dir() -> Path:
    """Return the bundled reference-library folder."""
    return Path(__file__).resolve().parent / "reference_library"


def next_available_output_dir(parent_dir: Path, folder_name: str) -> Path:
    """Return the next available output directory path."""
    candidate = parent_dir / folder_name
    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = parent_dir / f"{folder_name}{index:04d}"
        if not candidate.exists():
            return candidate
        index += 1


def suggest_output_dir(input_path: str | Path) -> Path:
    """Suggest a new output directory beside the XYZ input."""
    source_path = Path(input_path)
    if source_path.is_dir():
        folder_name = source_path.name
        parent_dir = source_path.parent
    else:
        folder_name = source_path.stem
        parent_dir = source_path.parent
    folder_label = re.sub(r"[^0-9A-Za-z]+", "_", folder_name).strip("_")
    if not folder_label:
        folder_label = "xyz"
    return next_available_output_dir(parent_dir, f"xyz2pdb_{folder_label}")


def _normalized_element_symbol(value: str) -> str:
    text = re.sub(r"[^A-Za-z]", "", str(value or "")).strip()
    if not text:
        raise ValueError("Element symbols must contain at least one letter.")
    if len(text) == 1:
        return text.upper()
    return text[0].upper() + text[1:].lower()


def _normalized_atom_name(value: str, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).strip().upper()
    if not text:
        text = fallback.upper()
    return text[:4]


def _normalized_residue_name(value: str | None, *, fallback: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]", "", str(value or "")).strip().upper()
    if not text:
        text = re.sub(r"[^A-Za-z0-9]", "", fallback).strip().upper()
    if not text:
        text = "UNK"
    return text[:3]


def _safe_reference_filename(name: str) -> str:
    text = "".join(
        character if character in _SAFE_FILENAME_CHARS else "_"
        for character in str(name or "").strip()
    )
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "reference"


def _build_cryst1_line(pbc_params: dict[str, float | str]) -> str | None:
    a = pbc_params.get("a")
    b = pbc_params.get("b")
    c = pbc_params.get("c")
    if a is None and b is None and c is None:
        return None

    if a is None:
        raise ValueError("PBC parameter 'a' is required when writing CRYST1.")
    a_value = float(a)
    b_value = float(b if b is not None else a_value)
    c_value = float(c if c is not None else a_value)
    alpha = float(pbc_params.get("alpha", 90.0))
    beta = float(pbc_params.get("beta", 90.0))
    gamma = float(pbc_params.get("gamma", 90.0))
    space_group = str(pbc_params.get("space_group", "P 1"))[:11]
    return (
        f"CRYST1{a_value:9.3f}{b_value:9.3f}{c_value:9.3f}"
        f"{alpha:7.2f}{beta:7.2f}{gamma:7.2f} {space_group:<11}1"
    )


@dataclass(slots=True)
class XYZAtomRecord:
    """One atom parsed from a single XYZ frame."""

    atom_id: int
    element: str
    coordinates: np.ndarray


@dataclass(slots=True)
class XYZFrame:
    """Single XYZ frame used by the converter."""

    filepath: Path
    comment: str
    atoms: list[XYZAtomRecord]


@dataclass(slots=True)
class AnchorPairDefinition:
    """One anchor pair used to place a reference molecule."""

    atom1_name: str
    atom2_name: str
    tolerance: float


@dataclass(slots=True)
class FreeAtomDefinition:
    """Residue assignment for one ungrouped element."""

    element: str
    residue_name: str
    atom_name: str | None = None


@dataclass(slots=True)
class ReferenceLibraryEntry:
    """Summary of one available reference PDB."""

    name: str
    path: Path
    residue_name: str
    atom_count: int
    atom_names: tuple[str, ...]


@dataclass(slots=True)
class ReferenceCreationResult:
    """Information about a newly written reference PDB."""

    name: str
    path: Path
    residue_name: str
    atom_count: int


@dataclass(slots=True)
class MoleculeDefinition:
    """Resolved molecule-matching definition built from JSON config."""

    name: str
    reference_name: str
    reference_path: Path
    residue_name: str
    reference_atoms: tuple[PDBAtom, ...]
    anchors: tuple[AnchorPairDefinition, ...]
    resolved_anchor_indices: tuple[tuple[int, int, float], ...]
    max_assignment_distance: float | None = None


@dataclass(slots=True)
class XYZToPDBConfiguration:
    """Fully parsed xyz2pdb configuration."""

    molecules: tuple[MoleculeDefinition, ...]
    free_atoms: dict[str, FreeAtomDefinition]
    exclude_hydrogen: bool
    pbc_params: dict[str, float | str]

    @property
    def configured_reference_names(self) -> tuple[str, ...]:
        return tuple(molecule.reference_name for molecule in self.molecules)


@dataclass(slots=True)
class ConvertedResidue:
    """One residue assignment prepared for PDB writing."""

    residue_number: int
    residue_name: str
    molecule_name: str
    atoms: tuple[PDBAtom, ...]
    source_atom_indices: tuple[int, ...]


@dataclass(slots=True)
class XYZToPDBInspectionResult:
    """Summary of the selected XYZ input and reference library."""

    input_path: Path
    input_mode: str
    xyz_files: tuple[Path, ...]
    config_file: Path | None
    reference_library_dir: Path
    available_references: tuple[ReferenceLibraryEntry, ...]
    configured_molecules: tuple[str, ...] = ()
    configured_reference_names: tuple[str, ...] = ()
    free_atom_elements: tuple[str, ...] = ()

    @property
    def total_files(self) -> int:
        return len(self.xyz_files)

    @property
    def first_file(self) -> Path | None:
        return self.xyz_files[0] if self.xyz_files else None

    def to_dict(self) -> dict[str, object]:
        return {
            "input_path": str(self.input_path),
            "input_mode": self.input_mode,
            "xyz_files": [str(path) for path in self.xyz_files],
            "config_file": (
                None if self.config_file is None else str(self.config_file)
            ),
            "reference_library_dir": str(self.reference_library_dir),
            "available_references": [entry.name for entry in self.available_references],
            "configured_molecules": list(self.configured_molecules),
            "configured_reference_names": list(
                self.configured_reference_names
            ),
            "free_atom_elements": list(self.free_atom_elements),
        }


@dataclass(slots=True)
class XYZToPDBPreviewResult:
    """Preview of the first-frame conversion and output selection."""

    inspection: XYZToPDBInspectionResult
    output_dir: Path
    residues: tuple[ConvertedResidue, ...]
    molecule_counts: dict[str, int]
    residue_counts: dict[str, int]
    total_atoms: int

    @property
    def first_output_file(self) -> Path | None:
        if self.inspection.first_file is None:
            return None
        return self.output_dir / f"{self.inspection.first_file.stem}.pdb"

    def to_dict(self) -> dict[str, object]:
        return {
            "inspection": self.inspection.to_dict(),
            "output_dir": str(self.output_dir),
            "first_output_file": (
                None
                if self.first_output_file is None
                else str(self.first_output_file)
            ),
            "molecule_counts": dict(self.molecule_counts),
            "residue_counts": dict(self.residue_counts),
            "total_atoms": self.total_atoms,
            "total_residues": len(self.residues),
        }


@dataclass(slots=True)
class XYZToPDBExportResult:
    """Result of writing converted PDB files to disk."""

    output_dir: Path
    written_files: tuple[Path, ...]
    preview: XYZToPDBPreviewResult

    def to_dict(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "written_files": [str(path) for path in self.written_files],
            "written_count": len(self.written_files),
            "preview": self.preview.to_dict(),
        }


def list_reference_library(
    library_dir: str | Path | None = None,
) -> list[ReferenceLibraryEntry]:
    """List the reference PDB files available in one library folder."""
    resolved_dir = (
        default_reference_library_dir()
        if library_dir is None
        else Path(library_dir)
    )
    if not resolved_dir.exists():
        return []

    entries: list[ReferenceLibraryEntry] = []
    for path in sorted(resolved_dir.glob("*.pdb"), key=lambda item: item.name.lower()):
        structure = PDBStructure.from_file(path)
        residue_name = (
            structure.atoms[0].residue_name if structure.atoms else "UNK"
        )
        atom_names = tuple(atom.atom_name for atom in structure.atoms)
        entries.append(
            ReferenceLibraryEntry(
                name=path.stem,
                path=path,
                residue_name=residue_name,
                atom_count=len(structure.atoms),
                atom_names=atom_names,
            )
        )
    return entries


def resolve_reference_path(
    reference: str | Path,
    *,
    library_dir: str | Path | None = None,
) -> Path:
    """Resolve a reference identifier or path to a PDB file."""
    candidate = Path(reference)
    if candidate.exists():
        return candidate

    resolved_dir = (
        default_reference_library_dir()
        if library_dir is None
        else Path(library_dir)
    )
    bare_name = str(reference).strip()
    names_to_try = [bare_name]
    if not bare_name.lower().endswith(".pdb"):
        names_to_try.append(f"{bare_name}.pdb")

    for name in names_to_try:
        library_candidate = resolved_dir / name
        if library_candidate.exists():
            return library_candidate
    raise FileNotFoundError(
        f"Reference molecule {reference!r} was not found in {resolved_dir}."
    )


def create_reference_molecule(
    source_file: str | Path,
    *,
    reference_name: str,
    residue_name: str | None = None,
    library_dir: str | Path | None = None,
) -> ReferenceCreationResult:
    """Convert one source structure into a stored reference PDB."""
    source_path = Path(source_file)
    if not source_path.exists():
        raise FileNotFoundError(
            f"Reference source file does not exist: {source_path}"
        )

    output_dir = (
        default_reference_library_dir()
        if library_dir is None
        else Path(library_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_name = _safe_reference_filename(reference_name)
    output_path = output_dir / f"{safe_name}.pdb"

    if source_path.suffix.lower() == ".pdb":
        source_structure = PDBStructure.from_file(source_path)
        atoms = [atom.copy() for atom in source_structure.atoms]
        if not atoms:
            raise ValueError("The selected PDB reference source has no atoms.")
        resolved_residue_name = _normalized_residue_name(
            residue_name,
            fallback=reference_name,
        )
        unique_names = {atom.atom_name for atom in atoms if atom.atom_name}
        should_rename = len(unique_names) != len(atoms)
        if should_rename:
            counters: Counter[str] = Counter()
            for atom in atoms:
                counters[atom.element] += 1
                atom.atom_name = _normalized_atom_name(
                    atom.atom_name,
                    fallback=f"{atom.element}{counters[atom.element]}",
                )
        for index, atom in enumerate(atoms, start=1):
            atom.atom_id = index
            atom.residue_number = 1
            atom.residue_name = resolved_residue_name
            atom.atom_name = _normalized_atom_name(
                atom.atom_name,
                fallback=f"{atom.element}{index}",
            )
        structure = PDBStructure(atoms=atoms, source_name=safe_name)
    elif source_path.suffix.lower() == ".xyz":
        frame = XYZToPDBWorkflow.read_xyz_frame(source_path)
        resolved_residue_name = _normalized_residue_name(
            residue_name,
            fallback=reference_name,
        )
        counters: Counter[str] = Counter()
        atoms = []
        for index, xyz_atom in enumerate(frame.atoms, start=1):
            counters[xyz_atom.element] += 1
            atoms.append(
                PDBAtom(
                    atom_id=index,
                    atom_name=_normalized_atom_name(
                        "",
                        fallback=(
                            f"{xyz_atom.element}{counters[xyz_atom.element]}"
                        ),
                    ),
                    residue_name=resolved_residue_name,
                    residue_number=1,
                    coordinates=xyz_atom.coordinates.copy(),
                    element=xyz_atom.element,
                )
            )
        structure = PDBStructure(atoms=atoms, source_name=safe_name)
    else:
        raise ValueError(
            "Reference creation supports only .pdb and .xyz source files."
        )

    structure.write_pdb_file(output_path)
    return ReferenceCreationResult(
        name=safe_name,
        path=output_path,
        residue_name=resolved_residue_name,
        atom_count=len(structure.atoms),
    )


class XYZToPDBWorkflow:
    """Headless xyz-to-pdb conversion workflow with reference matching."""

    def __init__(
        self,
        input_path: str | Path,
        *,
        config_file: str | Path | None = None,
        reference_library_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
    ) -> None:
        self.input_path = Path(input_path)
        self.config_file = None if config_file is None else Path(config_file)
        self.reference_library_dir = (
            default_reference_library_dir()
            if reference_library_dir is None
            else Path(reference_library_dir)
        )
        self.output_dir = None if output_dir is None else Path(output_dir)
        self._configuration: XYZToPDBConfiguration | None = None

    @staticmethod
    def read_xyz_frame(filepath: str | Path) -> XYZFrame:
        """Read a single XYZ frame from a file."""
        path = Path(filepath)
        lines = path.read_text(encoding="utf-8").splitlines()
        if len(lines) < 2:
            raise ValueError(f"The XYZ file is incomplete: {path}")
        try:
            atom_count = int(lines[0].strip())
        except ValueError as exc:
            raise ValueError(
                f"The XYZ file does not start with an atom count: {path}"
            ) from exc

        atoms: list[XYZAtomRecord] = []
        atom_lines = [line for line in lines[2:] if line.strip()]
        if len(atom_lines) < atom_count:
            raise ValueError(
                f"The XYZ file does not contain {atom_count} atom lines: {path}"
            )

        for index, line in enumerate(atom_lines[:atom_count], start=1):
            parts = line.split()
            if len(parts) < 4:
                raise ValueError(f"Malformed XYZ atom line: {line!r}")
            atoms.append(
                XYZAtomRecord(
                    atom_id=index,
                    element=_normalized_element_symbol(parts[0]),
                    coordinates=np.array(
                        [float(parts[1]), float(parts[2]), float(parts[3])],
                        dtype=float,
                    ),
                )
            )
        return XYZFrame(filepath=path, comment=lines[1], atoms=atoms)

    def inspect(self) -> XYZToPDBInspectionResult:
        """Inspect the selected input path and optional configuration."""
        input_mode, xyz_files = self._resolve_xyz_inputs()
        available_references = tuple(
            list_reference_library(self.reference_library_dir)
        )
        configured_molecules: tuple[str, ...] = ()
        configured_reference_names: tuple[str, ...] = ()
        free_atom_elements: tuple[str, ...] = ()
        if self.config_file is not None:
            configuration = self.load_configuration()
            configured_molecules = tuple(
                molecule.name for molecule in configuration.molecules
            )
            configured_reference_names = (
                configuration.configured_reference_names
            )
            free_atom_elements = tuple(sorted(configuration.free_atoms))
        return XYZToPDBInspectionResult(
            input_path=self.input_path,
            input_mode=input_mode,
            xyz_files=tuple(xyz_files),
            config_file=self.config_file,
            reference_library_dir=self.reference_library_dir,
            available_references=available_references,
            configured_molecules=configured_molecules,
            configured_reference_names=configured_reference_names,
            free_atom_elements=free_atom_elements,
        )

    def load_configuration(self) -> XYZToPDBConfiguration:
        """Load and cache the JSON residue-assignment configuration."""
        if self._configuration is not None:
            return self._configuration
        if self.config_file is None:
            raise ValueError("A residue-assignment JSON config file is required.")

        payload = json.loads(self.config_file.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("The xyz2pdb config must be a JSON object.")

        exclude_hydrogen = bool(payload.get("exclude_hydrogen", False))
        pbc_params = self._parse_pbc_params(payload.get("pbc"))
        molecules = self._parse_molecules(
            payload.get("molecules"),
            exclude_hydrogen=exclude_hydrogen,
        )
        free_atoms = self._parse_free_atoms(payload.get("free_atoms"))
        self._configuration = XYZToPDBConfiguration(
            molecules=tuple(molecules),
            free_atoms=free_atoms,
            exclude_hydrogen=exclude_hydrogen,
            pbc_params=pbc_params,
        )
        return self._configuration

    def preview_conversion(
        self,
        *,
        output_dir: str | Path | None = None,
    ) -> XYZToPDBPreviewResult:
        """Preview the first-frame residue assignments and output path."""
        inspection = self.inspect()
        if not inspection.xyz_files:
            raise ValueError("No XYZ files were found for conversion.")

        frame = self.read_xyz_frame(inspection.xyz_files[0])
        configuration = self.load_configuration()
        residues = tuple(self._convert_first_frame(frame, configuration))
        resolved_output_dir = (
            Path(output_dir)
            if output_dir is not None
            else self.output_dir
            if self.output_dir is not None
            else suggest_output_dir(self.input_path)
        )
        molecule_counts: Counter[str] = Counter()
        residue_counts: Counter[str] = Counter()
        total_atoms = 0
        for residue in residues:
            molecule_counts[residue.molecule_name] += 1
            residue_counts[residue.residue_name] += 1
            total_atoms += len(residue.atoms)
        return XYZToPDBPreviewResult(
            inspection=inspection,
            output_dir=resolved_output_dir,
            residues=residues,
            molecule_counts=dict(molecule_counts),
            residue_counts=dict(residue_counts),
            total_atoms=total_atoms,
        )

    def export_pdbs(
        self,
        *,
        output_dir: str | Path | None = None,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> XYZToPDBExportResult:
        """Convert the selected XYZ input into PDB files on disk."""
        preview = self.preview_conversion(output_dir=output_dir)
        configuration = self.load_configuration()
        output_path = preview.output_dir
        output_path.mkdir(parents=True, exist_ok=True)

        inspection = preview.inspection
        written_files: list[Path] = []
        headers = self._header_lines(configuration)
        total_files = len(inspection.xyz_files)
        template_residues = preview.residues

        for index, xyz_file in enumerate(inspection.xyz_files, start=1):
            if index == 1:
                residues = template_residues
            else:
                frame = self.read_xyz_frame(xyz_file)
                residues = tuple(self._apply_template(frame, template_residues))
            atoms = [atom.copy() for residue in residues for atom in residue.atoms]
            structure = PDBStructure(atoms=atoms, source_name=xyz_file.stem)
            written_path = structure.write_pdb_file(
                output_path / f"{xyz_file.stem}.pdb",
                header_lines=headers,
            )
            written_files.append(written_path)
            if progress_callback is not None:
                progress_callback(index, total_files, xyz_file.name)

        return XYZToPDBExportResult(
            output_dir=output_path,
            written_files=tuple(written_files),
            preview=preview,
        )

    def _resolve_xyz_inputs(self) -> tuple[str, list[Path]]:
        if self.input_path.is_file():
            if self.input_path.suffix.lower() != ".xyz":
                raise ValueError(
                    f"The input file must be an .xyz file: {self.input_path}"
                )
            return "single_xyz", [self.input_path]

        if not self.input_path.is_dir():
            raise ValueError(
                f"The selected XYZ input does not exist: {self.input_path}"
            )

        xyz_files = sorted(
            (
                path
                for path in self.input_path.iterdir()
                if path.is_file() and path.suffix.lower() == ".xyz"
            ),
            key=lambda path: path.name.lower(),
        )
        if not xyz_files:
            raise ValueError(
                f"No .xyz files were found in {self.input_path}"
            )
        return "xyz_folder", xyz_files

    def _parse_pbc_params(
        self,
        value: object | None,
    ) -> dict[str, float | str]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("The optional 'pbc' config section must be an object.")

        parsed: dict[str, float | str] = {}
        for key in ("a", "b", "c", "alpha", "beta", "gamma"):
            if key in value and value[key] is not None:
                parsed[key] = float(value[key])
        if "space_group" in value and value["space_group"] is not None:
            parsed["space_group"] = str(value["space_group"])
        return parsed

    def _parse_molecules(
        self,
        value: object,
        *,
        exclude_hydrogen: bool,
    ) -> list[MoleculeDefinition]:
        if not isinstance(value, list) or not value:
            raise ValueError(
                "The xyz2pdb config must contain a non-empty 'molecules' list."
            )

        molecules: list[MoleculeDefinition] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("Each molecule config entry must be an object.")
            reference_value = item.get("reference")
            if reference_value is None:
                raise ValueError(
                    "Each molecule config entry needs a 'reference' value."
                )
            reference_path = resolve_reference_path(
                reference_value,
                library_dir=self.reference_library_dir,
            )
            reference_structure = PDBStructure.from_file(reference_path)
            reference_atoms = tuple(
                atom.copy()
                for atom in reference_structure.atoms
                if not (
                    exclude_hydrogen
                    and atom.element.upper() == "H"
                )
            )
            if not reference_atoms:
                raise ValueError(
                    f"Reference {reference_path} has no atoms after filtering."
                )
            reference_name = Path(reference_path).stem
            molecule_name = str(item.get("name") or reference_name).strip()
            residue_name = _normalized_residue_name(
                item.get("residue_name"),
                fallback=reference_atoms[0].residue_name or molecule_name,
            )
            anchors_value = item.get("anchors")
            if not isinstance(anchors_value, list) or not anchors_value:
                raise ValueError(
                    f"Molecule {molecule_name!r} needs at least one anchor pair."
                )

            anchors: list[AnchorPairDefinition] = []
            resolved_anchor_indices: list[tuple[int, int, float]] = []
            for anchor_item in anchors_value:
                if not isinstance(anchor_item, dict):
                    raise ValueError(
                        f"Molecule {molecule_name!r} has an invalid anchor entry."
                    )
                pair_value = anchor_item.get("pair")
                if not isinstance(pair_value, list | tuple) or len(pair_value) != 2:
                    raise ValueError(
                        f"Molecule {molecule_name!r} anchor pairs must contain two atom names."
                    )
                atom1_name = _normalized_atom_name(
                    str(pair_value[0]),
                    fallback="A1",
                )
                atom2_name = _normalized_atom_name(
                    str(pair_value[1]),
                    fallback="A2",
                )
                tolerance = float(anchor_item.get("tol", 0.15))
                anchors.append(
                    AnchorPairDefinition(
                        atom1_name=atom1_name,
                        atom2_name=atom2_name,
                        tolerance=tolerance,
                    )
                )
                resolved_anchor_indices.append(
                    self._resolve_anchor_pair_indices(
                        reference_atoms,
                        atom1_name,
                        atom2_name,
                        molecule_name=molecule_name,
                    )
                    + (tolerance,)
                )

            max_assignment_distance = item.get("max_assignment_distance")
            molecules.append(
                MoleculeDefinition(
                    name=molecule_name,
                    reference_name=reference_name,
                    reference_path=reference_path,
                    residue_name=residue_name,
                    reference_atoms=reference_atoms,
                    anchors=tuple(anchors),
                    resolved_anchor_indices=tuple(resolved_anchor_indices),
                    max_assignment_distance=(
                        None
                        if max_assignment_distance is None
                        else float(max_assignment_distance)
                    ),
                )
            )
        return molecules

    def _parse_free_atoms(
        self,
        value: object | None,
    ) -> dict[str, FreeAtomDefinition]:
        if value is None:
            return {}

        parsed: dict[str, FreeAtomDefinition] = {}
        if isinstance(value, dict):
            items = []
            for element, definition in value.items():
                if isinstance(definition, str):
                    items.append(
                        {
                            "element": element,
                            "residue_name": definition,
                        }
                    )
                elif isinstance(definition, dict):
                    items.append({"element": element, **definition})
                else:
                    raise ValueError(
                        "Free-atom definitions must be strings or objects."
                    )
        elif isinstance(value, list):
            items = value
        else:
            raise ValueError(
                "The optional 'free_atoms' config section must be an object or list."
            )

        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Each free-atom entry must be an object.")
            element = _normalized_element_symbol(str(item.get("element") or ""))
            residue_name = _normalized_residue_name(
                item.get("residue_name"),
                fallback=element,
            )
            atom_name_value = item.get("atom_name")
            parsed[element] = FreeAtomDefinition(
                element=element,
                residue_name=residue_name,
                atom_name=(
                    None
                    if atom_name_value is None
                    else _normalized_atom_name(
                        str(atom_name_value),
                        fallback=element,
                    )
                ),
            )
        return parsed

    def _resolve_anchor_pair_indices(
        self,
        reference_atoms: Sequence[PDBAtom],
        atom1_name: str,
        atom2_name: str,
        *,
        molecule_name: str,
    ) -> tuple[int, int]:
        atom_names = [atom.atom_name.upper()[:4] for atom in reference_atoms]
        try:
            atom1_index = atom_names.index(atom1_name)
            atom2_index = atom_names.index(atom2_name)
        except ValueError as exc:
            raise ValueError(
                f"Anchor pair ({atom1_name}, {atom2_name}) was not found in "
                f"reference molecule {molecule_name!r}."
            ) from exc
        if atom1_index == atom2_index:
            raise ValueError(
                f"Molecule {molecule_name!r} uses the same atom twice in one anchor pair."
            )
        return atom1_index, atom2_index

    def _convert_first_frame(
        self,
        frame: XYZFrame,
        configuration: XYZToPDBConfiguration,
    ) -> list[ConvertedResidue]:
        residues: list[ConvertedResidue] = []
        used = [False] * len(frame.atoms)
        residue_number = 1
        atom_serial = 1

        for molecule in configuration.molecules:
            while True:
                matched_indices = self._find_best_match(
                    frame,
                    used,
                    molecule,
                )
                if matched_indices is None:
                    break
                for source_index in matched_indices:
                    used[source_index] = True

                residue_atoms: list[PDBAtom] = []
                for reference_atom, source_index in zip(
                    molecule.reference_atoms,
                    matched_indices,
                ):
                    xyz_atom = frame.atoms[source_index]
                    residue_atoms.append(
                        PDBAtom(
                            atom_id=atom_serial,
                            atom_name=_normalized_atom_name(
                                reference_atom.atom_name,
                                fallback=f"{reference_atom.element}{atom_serial}",
                            ),
                            residue_name=molecule.residue_name,
                            residue_number=residue_number,
                            coordinates=xyz_atom.coordinates.copy(),
                            element=reference_atom.element,
                        )
                    )
                    atom_serial += 1
                residues.append(
                    ConvertedResidue(
                        residue_number=residue_number,
                        residue_name=molecule.residue_name,
                        molecule_name=molecule.name,
                        atoms=tuple(residue_atoms),
                        source_atom_indices=tuple(matched_indices),
                    )
                )
                residue_number += 1

        free_atom_counters: Counter[str] = Counter()
        for atom_index, xyz_atom in enumerate(frame.atoms):
            if used[atom_index]:
                continue
            assignment = configuration.free_atoms.get(xyz_atom.element)
            residue_name = (
                assignment.residue_name if assignment is not None else "UNK"
            )
            free_atom_counters[xyz_atom.element] += 1
            generated_atom_name = (
                assignment.atom_name
                if assignment is not None and assignment.atom_name is not None
                else f"{xyz_atom.element}{free_atom_counters[xyz_atom.element]}"
            )
            residues.append(
                ConvertedResidue(
                    residue_number=residue_number,
                    residue_name=residue_name,
                    molecule_name=(
                        residue_name
                        if assignment is not None
                        else f"UNMATCHED_{xyz_atom.element}"
                    ),
                    atoms=(
                        PDBAtom(
                            atom_id=atom_serial,
                            atom_name=_normalized_atom_name(
                                generated_atom_name,
                                fallback=xyz_atom.element,
                            ),
                            residue_name=residue_name,
                            residue_number=residue_number,
                            coordinates=xyz_atom.coordinates.copy(),
                            element=xyz_atom.element,
                        ),
                    ),
                    source_atom_indices=(atom_index,),
                )
            )
            atom_serial += 1
            residue_number += 1

        return residues

    def _find_best_match(
        self,
        frame: XYZFrame,
        used: Sequence[bool],
        molecule: MoleculeDefinition,
    ) -> tuple[int, ...] | None:
        reference_atoms = molecule.reference_atoms
        reference_coordinates = np.array(
            [atom.coordinates for atom in reference_atoms],
            dtype=float,
        )
        best_assignment: tuple[int, ...] | None = None
        best_score: float | None = None
        anchor_constraints = self._anchor_constraints(
            reference_coordinates,
            molecule.resolved_anchor_indices,
        )
        anchor_reference_indices = tuple(sorted(anchor_constraints))

        for anchor_index1, anchor_index2, tolerance in molecule.resolved_anchor_indices:
            anchor_atom1 = reference_atoms[anchor_index1]
            anchor_atom2 = reference_atoms[anchor_index2]
            reference_vector = (
                reference_coordinates[anchor_index2]
                - reference_coordinates[anchor_index1]
            )
            reference_length = float(np.linalg.norm(reference_vector))
            if reference_length <= 0.0:
                continue

            candidate_indices1 = [
                index
                for index, atom in enumerate(frame.atoms)
                if not used[index] and atom.element == anchor_atom1.element
            ]
            candidate_indices2 = [
                index
                for index, atom in enumerate(frame.atoms)
                if not used[index] and atom.element == anchor_atom2.element
            ]
            for source_index1 in candidate_indices1:
                position1 = frame.atoms[source_index1].coordinates
                for source_index2 in candidate_indices2:
                    if source_index1 == source_index2:
                        continue
                    position2 = frame.atoms[source_index2].coordinates
                    candidate_distance = float(
                        np.linalg.norm(position2 - position1)
                    )
                    if abs(candidate_distance - reference_length) > tolerance:
                        continue
                    try:
                        rotation = rotation_matrix_from_to(
                            reference_vector,
                            position2 - position1,
                        )
                    except ValueError:
                        continue

                    transformed = (
                        (reference_coordinates - reference_coordinates[anchor_index1])
                        @ rotation.T
                    ) + position1
                    initial_anchor_assignment = {
                        anchor_index1: source_index1,
                        anchor_index2: source_index2,
                    }
                    expanded_anchor_assignments = self._expand_anchor_assignments(
                        frame,
                        used,
                        reference_atoms=reference_atoms,
                        anchor_reference_indices=anchor_reference_indices,
                        anchor_constraints=anchor_constraints,
                        initial_assignment=initial_anchor_assignment,
                    )
                    for anchor_assignment in expanded_anchor_assignments:
                        target_coordinates = transformed
                        if len(anchor_assignment) >= 3:
                            target_coordinates = self._transform_reference_with_anchors(
                                reference_coordinates,
                                frame,
                                anchor_assignment,
                            )
                        assignment, score = self._assign_reference_atoms(
                            frame,
                            used,
                            reference_atoms=reference_atoms,
                            target_coordinates=target_coordinates,
                            fixed_assignment=anchor_assignment,
                            max_assignment_distance=molecule.max_assignment_distance,
                        )
                        if assignment is None:
                            continue
                        if best_score is None or score < best_score:
                            best_assignment = assignment
                            best_score = score
        return best_assignment

    def _anchor_constraints(
        self,
        reference_coordinates: np.ndarray,
        resolved_anchor_indices: Sequence[tuple[int, int, float]],
    ) -> dict[int, list[tuple[int, float, float]]]:
        constraints: dict[int, list[tuple[int, float, float]]] = defaultdict(list)
        for anchor_index1, anchor_index2, tolerance in resolved_anchor_indices:
            reference_distance = float(
                np.linalg.norm(
                    reference_coordinates[anchor_index2]
                    - reference_coordinates[anchor_index1]
                )
            )
            constraints[anchor_index1].append(
                (anchor_index2, reference_distance, tolerance)
            )
            constraints[anchor_index2].append(
                (anchor_index1, reference_distance, tolerance)
            )
        return constraints

    def _expand_anchor_assignments(
        self,
        frame: XYZFrame,
        used: Sequence[bool],
        *,
        reference_atoms: Sequence[PDBAtom],
        anchor_reference_indices: Sequence[int],
        anchor_constraints: dict[int, list[tuple[int, float, float]]],
        initial_assignment: dict[int, int],
    ) -> list[dict[int, int]]:
        if not anchor_reference_indices:
            return [dict(initial_assignment)]

        assignments: list[dict[int, int]] = []

        def recurse(current_assignment: dict[int, int]) -> None:
            if len(current_assignment) == len(anchor_reference_indices):
                assignments.append(dict(current_assignment))
                return

            remaining_indices = [
                reference_index
                for reference_index in anchor_reference_indices
                if reference_index not in current_assignment
            ]
            candidate_sets: list[tuple[int, list[int]]] = []
            for reference_index in remaining_indices:
                candidates = self._candidate_anchor_source_indices(
                    frame,
                    used,
                    reference_atoms=reference_atoms,
                    reference_index=reference_index,
                    current_assignment=current_assignment,
                    anchor_constraints=anchor_constraints,
                )
                if not candidates:
                    return
                candidate_sets.append((reference_index, candidates))

            next_reference_index, next_candidates = min(
                candidate_sets,
                key=lambda item: len(item[1]),
            )
            for source_index in next_candidates:
                current_assignment[next_reference_index] = source_index
                recurse(current_assignment)
                del current_assignment[next_reference_index]

        recurse(dict(initial_assignment))
        if assignments:
            return assignments
        return [dict(initial_assignment)]

    def _candidate_anchor_source_indices(
        self,
        frame: XYZFrame,
        used: Sequence[bool],
        *,
        reference_atoms: Sequence[PDBAtom],
        reference_index: int,
        current_assignment: dict[int, int],
        anchor_constraints: dict[int, list[tuple[int, float, float]]],
    ) -> list[int]:
        reference_atom = reference_atoms[reference_index]
        claimed_source_indices = set(current_assignment.values())
        constrained_candidates: list[int] = []
        for source_index, source_atom in enumerate(frame.atoms):
            if used[source_index] or source_index in claimed_source_indices:
                continue
            if source_atom.element != reference_atom.element:
                continue

            satisfied = False
            valid = True
            for other_reference_index, reference_distance, tolerance in (
                anchor_constraints.get(reference_index, [])
            ):
                if other_reference_index not in current_assignment:
                    continue
                satisfied = True
                other_source_index = current_assignment[other_reference_index]
                other_coordinates = frame.atoms[other_source_index].coordinates
                candidate_distance = float(
                    np.linalg.norm(source_atom.coordinates - other_coordinates)
                )
                if abs(candidate_distance - reference_distance) > tolerance:
                    valid = False
                    break
            if satisfied and valid:
                constrained_candidates.append(source_index)

        if constrained_candidates:
            return constrained_candidates

        return [
            source_index
            for source_index, source_atom in enumerate(frame.atoms)
            if not used[source_index]
            and source_index not in claimed_source_indices
            and source_atom.element == reference_atom.element
        ]

    def _transform_reference_with_anchors(
        self,
        reference_coordinates: np.ndarray,
        frame: XYZFrame,
        anchor_assignment: dict[int, int],
    ) -> np.ndarray:
        ordered_anchor_indices = tuple(sorted(anchor_assignment))
        source_points = np.array(
            [reference_coordinates[index] for index in ordered_anchor_indices],
            dtype=float,
        )
        target_points = np.array(
            [
                frame.atoms[anchor_assignment[index]].coordinates
                for index in ordered_anchor_indices
            ],
            dtype=float,
        )
        rotation, source_centroid, target_centroid = rigid_alignment_from_points(
            source_points,
            target_points,
        )
        return ((reference_coordinates - source_centroid) @ rotation) + target_centroid

    def _assign_reference_atoms(
        self,
        frame: XYZFrame,
        used: Sequence[bool],
        *,
        reference_atoms: Sequence[PDBAtom],
        target_coordinates: np.ndarray,
        fixed_assignment: dict[int, int],
        max_assignment_distance: float | None,
    ) -> tuple[tuple[int, ...] | None, float]:
        assignment_by_reference_index = dict(fixed_assignment)
        distances: list[float] = []

        for reference_index, source_index in fixed_assignment.items():
            distance = float(
                np.linalg.norm(
                    target_coordinates[reference_index]
                    - frame.atoms[source_index].coordinates
                )
            )
            if (
                max_assignment_distance is not None
                and distance > max_assignment_distance
            ):
                return None, 0.0
            distances.append(distance)

        reference_indices_by_element: dict[str, list[int]] = defaultdict(list)
        for reference_index, reference_atom in enumerate(reference_atoms):
            if reference_index in fixed_assignment:
                continue
            reference_indices_by_element[reference_atom.element].append(
                reference_index
            )

        claimed_source_indices = set(fixed_assignment.values())
        for element, reference_indices in reference_indices_by_element.items():
            source_indices = [
                index
                for index, atom in enumerate(frame.atoms)
                if not used[index]
                and index not in claimed_source_indices
                and atom.element == element
            ]
            if len(source_indices) < len(reference_indices):
                return None, 0.0

            cost_matrix = np.zeros(
                (len(reference_indices), len(source_indices)),
                dtype=float,
            )
            for row_index, reference_index in enumerate(reference_indices):
                target = target_coordinates[reference_index]
                for column_index, source_index in enumerate(source_indices):
                    cost_matrix[row_index, column_index] = float(
                        np.linalg.norm(
                            target - frame.atoms[source_index].coordinates
                        )
                    )

            row_indices, column_indices = linear_sum_assignment(cost_matrix)
            for row_index, column_index in zip(row_indices, column_indices):
                reference_index = reference_indices[row_index]
                source_index = source_indices[column_index]
                distance = float(cost_matrix[row_index, column_index])
                if (
                    max_assignment_distance is not None
                    and distance > max_assignment_distance
                ):
                    return None, 0.0
                assignment_by_reference_index[reference_index] = source_index
                claimed_source_indices.add(source_index)
                distances.append(distance)

        if len(assignment_by_reference_index) != len(reference_atoms):
            return None, 0.0
        ordered_assignment = tuple(
            assignment_by_reference_index[index]
            for index in range(len(reference_atoms))
        )
        score = float(np.mean(distances)) if distances else 0.0
        return ordered_assignment, score

    def _apply_template(
        self,
        frame: XYZFrame,
        template_residues: Sequence[ConvertedResidue],
    ) -> list[ConvertedResidue]:
        residues: list[ConvertedResidue] = []
        for residue in template_residues:
            copied_atoms: list[PDBAtom] = []
            for template_atom, source_index in zip(
                residue.atoms,
                residue.source_atom_indices,
            ):
                if source_index >= len(frame.atoms):
                    raise ValueError(
                        f"Frame {frame.filepath.name} does not contain atom index {source_index}."
                    )
                xyz_atom = frame.atoms[source_index]
                if xyz_atom.element != template_atom.element:
                    raise ValueError(
                        f"Frame {frame.filepath.name} changed element order at atom "
                        f"index {source_index + 1}: expected {template_atom.element}, "
                        f"found {xyz_atom.element}."
                    )
                copied_atom = template_atom.copy()
                copied_atom.coordinates = xyz_atom.coordinates.copy()
                copied_atoms.append(copied_atom)
            residues.append(
                ConvertedResidue(
                    residue_number=residue.residue_number,
                    residue_name=residue.residue_name,
                    molecule_name=residue.molecule_name,
                    atoms=tuple(copied_atoms),
                    source_atom_indices=tuple(residue.source_atom_indices),
                )
            )
        return residues

    def _header_lines(
        self,
        configuration: XYZToPDBConfiguration,
    ) -> list[str] | None:
        cryst1_line = _build_cryst1_line(configuration.pbc_params)
        if cryst1_line is None:
            return None
        return [cryst1_line]


__all__ = [
    "AnchorPairDefinition",
    "ConvertedResidue",
    "FreeAtomDefinition",
    "MoleculeDefinition",
    "ReferenceCreationResult",
    "ReferenceLibraryEntry",
    "XYZAtomRecord",
    "XYZFrame",
    "XYZToPDBConfiguration",
    "XYZToPDBExportResult",
    "XYZToPDBInspectionResult",
    "XYZToPDBPreviewResult",
    "XYZToPDBWorkflow",
    "create_reference_molecule",
    "default_reference_library_dir",
    "list_reference_library",
    "next_available_output_dir",
    "resolve_reference_path",
    "suggest_output_dir",
]
