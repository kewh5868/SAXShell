from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

if False:  # pragma: no cover
    from .packmol_setup import PackmolSetupMetadata

DEFAULT_PACKMOL_CONTAINER_ROOT = "/packmol_input_files"
_DOCKER_DAEMON_UNAVAILABLE_HINT = (
    "Docker Desktop or the Docker daemon does not appear to be running. "
    "Start Docker Desktop (or another Docker runtime such as OrbStack or "
    "Colima), wait for `docker info` to succeed, and retry."
)
_DOCKER_DAEMON_UNAVAILABLE_PATTERNS = (
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "error during connect",
)


@dataclass(slots=True)
class PackmolDockerLink:
    display_name: str
    container_name: str
    container_project_root: str = DEFAULT_PACKMOL_CONTAINER_ROOT
    packmol_command: str = "packmol"
    shell_command: str = "sh"
    packmol_version: str | None = None
    linked_at: str | None = None
    last_verified_at: str | None = None
    container_id: str | None = None
    image_name: str | None = None
    packmol_command_path: str | None = None
    last_sync_at: str | None = None
    last_sync_status: str | None = None
    last_sync_message: str | None = None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["display_name"] = self.resolved_display_name
        payload["container_name"] = self.container_name.strip()
        payload["container_project_root"] = normalize_container_directory(
            self.container_project_root
        )
        payload["packmol_command"] = self.packmol_command.strip() or "packmol"
        payload["shell_command"] = self.shell_command.strip() or "sh"
        return payload

    def to_preset_dict(self) -> dict[str, object]:
        return {
            "display_name": self.resolved_display_name,
            "container_name": self.container_name.strip(),
            "container_project_root": normalize_container_directory(
                self.container_project_root
            ),
            "packmol_command": self.packmol_command.strip() or "packmol",
            "shell_command": self.shell_command.strip() or "sh",
        }

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, object] | None,
    ) -> "PackmolDockerLink | None":
        if not payload:
            return None
        container_name = str(payload.get("container_name", "")).strip()
        display_name = str(payload.get("display_name", container_name)).strip()
        if not container_name:
            return None
        return cls(
            display_name=display_name or container_name,
            container_name=container_name,
            container_project_root=normalize_container_directory(
                payload.get("container_project_root")
            ),
            packmol_command=str(
                payload.get("packmol_command", "packmol")
            ).strip()
            or "packmol",
            shell_command=str(payload.get("shell_command", "sh")).strip()
            or "sh",
            packmol_version=_optional_text(payload.get("packmol_version")),
            linked_at=_optional_text(payload.get("linked_at")),
            last_verified_at=_optional_text(payload.get("last_verified_at")),
            container_id=_optional_text(payload.get("container_id")),
            image_name=_optional_text(payload.get("image_name")),
            packmol_command_path=_optional_text(
                payload.get("packmol_command_path")
            ),
            last_sync_at=_optional_text(payload.get("last_sync_at")),
            last_sync_status=_optional_text(payload.get("last_sync_status")),
            last_sync_message=_optional_text(payload.get("last_sync_message")),
        )

    @property
    def resolved_display_name(self) -> str:
        return self.display_name.strip() or self.container_name.strip()

    def remote_rmcsetup_dir(self) -> PurePosixPath:
        return (
            PurePosixPath(
                normalize_container_directory(self.container_project_root)
            )
            / "rmcsetup"
        )

    def remote_packmol_inputs_dir(self) -> PurePosixPath:
        return self.remote_rmcsetup_dir() / "packmol_inputs"

    def remote_packmol_input_path(
        self,
        packmol_setup_metadata: "PackmolSetupMetadata | None" = None,
    ) -> str:
        input_name = "packmol_combined.inp"
        if (
            packmol_setup_metadata is not None
            and packmol_setup_metadata.packmol_input_path
        ):
            input_name = Path(packmol_setup_metadata.packmol_input_path).name
        return str(self.remote_packmol_inputs_dir() / input_name)

    def remote_packmol_output_path(
        self,
        packmol_setup_metadata: "PackmolSetupMetadata | None" = None,
    ) -> str:
        output_name = "packed_combined.pdb"
        if (
            packmol_setup_metadata is not None
            and packmol_setup_metadata.packed_output_filename
        ):
            output_name = packmol_setup_metadata.packed_output_filename
        return str(self.remote_packmol_inputs_dir() / output_name)

    def summary_text(
        self,
        *,
        packmol_setup_metadata: "PackmolSetupMetadata | None" = None,
    ) -> str:
        lines = [
            f"Preset: {self.resolved_display_name}",
            f"Container: {self.container_name}",
            (
                "Container project root: "
                f"{normalize_container_directory(self.container_project_root)}"
            ),
            f"Packmol command: {self.packmol_command}",
            (
                "Resolved Packmol binary: "
                f"{self.packmol_command_path or '(not verified yet)'}"
            ),
            f"Packmol version: {self.packmol_version or '(not verified yet)'}",
            f"Shell command: {self.shell_command}",
            f"Image: {self.image_name or '(not verified yet)'}",
            f"Container ID: {self.container_id or '(not verified yet)'}",
            f"Linked at: {self.linked_at or '(not linked yet)'}",
            f"Last verified: {self.last_verified_at or '(not verified yet)'}",
            (
                "Remote Packmol inputs folder: "
                f"{self.remote_packmol_inputs_dir()}"
            ),
        ]
        if packmol_setup_metadata is not None:
            lines.extend(
                [
                    (
                        "Remote Packmol input file: "
                        f"{self.remote_packmol_input_path(packmol_setup_metadata)}"
                    ),
                    (
                        "Remote packed output path: "
                        f"{self.remote_packmol_output_path(packmol_setup_metadata)}"
                    ),
                ]
            )
        if self.last_sync_status:
            lines.append(
                f"Last sync status: {self.last_sync_status} at "
                f"{self.last_sync_at or '(unknown time)'}"
            )
        if self.last_sync_message:
            lines.append(f"Last sync details: {self.last_sync_message}")
        return "\n".join(lines)


@dataclass(slots=True, frozen=True)
class PackmolDockerDirectoryEntry:
    name: str
    path: str


@dataclass(slots=True, frozen=True)
class PackmolDockerContainerRecord:
    name: str
    image_name: str
    status: str

    @property
    def summary_label(self) -> str:
        details = [part for part in (self.image_name, self.status) if part]
        if not details:
            return self.name
        return f"{self.name} ({' | '.join(details)})"


@dataclass(slots=True, frozen=True)
class PackmolDockerValidationResult:
    verified_at: str
    container_id: str
    image_name: str
    packmol_command_path: str
    packmol_version: str
    container_project_root: str

    def summary_text(self, link: PackmolDockerLink) -> str:
        lines = [
            "Docker validation succeeded.",
            "",
            f"Preset: {link.resolved_display_name}",
            f"Container: {link.container_name}",
            f"Container ID: {self.container_id}",
            f"Image: {self.image_name}",
            f"Packmol command: {link.packmol_command}",
            f"Resolved Packmol binary: {self.packmol_command_path}",
            f"Packmol version: {self.packmol_version}",
            f"Verified workspace root: {self.container_project_root}",
            (
                "Remote Packmol sync folder: "
                f"{link.remote_packmol_inputs_dir()}"
            ),
            f"Verified at: {self.verified_at}",
        ]
        return "\n".join(lines)


@dataclass(slots=True, frozen=True)
class PackmolDockerSyncResult:
    synced_at: str
    remote_packmol_inputs_dir: str
    remote_packmol_input_path: str
    remote_packed_output_path: str
    synced_file_count: int

    def summary_text(self) -> str:
        return (
            f"Synced {self.synced_file_count} file(s) to "
            f"{self.remote_packmol_inputs_dir} at {self.synced_at}."
        )


class PackmolDockerClient:
    def __init__(self, *, docker_executable: str = "docker") -> None:
        self.docker_executable = docker_executable

    def list_containers(self) -> list[PackmolDockerContainerRecord]:
        stdout = self._run_docker(
            "ps",
            "-a",
            "--format",
            "{{.Names}}\t{{.Image}}\t{{.Status}}",
        ).stdout
        records: list[PackmolDockerContainerRecord] = []
        seen_names: set[str] = set()
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            name, separator, remainder = line.partition("\t")
            if not separator:
                continue
            image_name, _, status = remainder.partition("\t")
            normalized_name = name.strip()
            if not normalized_name or normalized_name in seen_names:
                continue
            seen_names.add(normalized_name)
            records.append(
                PackmolDockerContainerRecord(
                    name=normalized_name,
                    image_name=image_name.strip(),
                    status=status.strip(),
                )
            )
        records.sort(key=lambda record: record.name.lower())
        return records

    def verify_link(
        self,
        link: PackmolDockerLink,
    ) -> PackmolDockerValidationResult:
        self._run_docker("info")
        if not self._container_is_running(link.container_name):
            self._run_docker("start", link.container_name)
        if not self._container_is_running(link.container_name):
            raise RuntimeError(
                "Docker could not keep the selected container running. If "
                "your container requires an attached shell, start it "
                "manually with `docker start -i <container_name>` and try "
                "linking it again."
            )
        container_id = self._run_docker(
            "inspect",
            "--format",
            "{{.Id}}",
            link.container_name,
        ).stdout.strip()
        image_name = self._run_docker(
            "inspect",
            "--format",
            "{{.Config.Image}}",
            link.container_name,
        ).stdout.strip()
        project_root = normalize_container_directory(
            link.container_project_root
        )
        _validate_container_project_root(project_root)
        self._run_in_container(
            link,
            f"target={shlex.quote(project_root)}; "
            '[ -d "$target" ] || exit 3',
        )
        packmol_command_path = self._resolve_packmol_command_path(link)
        packmol_version = self._resolve_packmol_version(
            link,
            packmol_command_path,
        )
        return PackmolDockerValidationResult(
            verified_at=datetime.now().isoformat(timespec="seconds"),
            container_id=container_id,
            image_name=image_name,
            packmol_command_path=packmol_command_path,
            packmol_version=packmol_version,
            container_project_root=project_root,
        )

    def list_directories(
        self,
        link: PackmolDockerLink,
        directory: str,
    ) -> list[PackmolDockerDirectoryEntry]:
        normalized = normalize_container_directory(directory)
        stdout = self._run_in_container(
            link,
            f"target={shlex.quote(normalized)}; "
            '[ -d "$target" ] || exit 3; '
            'for entry in "$target"/*; do '
            '  [ -d "$entry" ] || continue; '
            '  name=$(basename "$entry"); '
            '  printf "%s\\t%s\\n" "$name" "$entry"; '
            "done",
        )
        entries: list[PackmolDockerDirectoryEntry] = []
        for raw_line in stdout.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            name, separator, path = line.partition("\t")
            if not separator:
                continue
            entries.append(
                PackmolDockerDirectoryEntry(
                    name=name.strip(),
                    path=normalize_container_directory(path),
                )
            )
        entries.sort(key=lambda entry: entry.name.lower())
        return entries

    def sync_packmol_inputs(
        self,
        link: PackmolDockerLink,
        local_packmol_inputs_dir: str | Path,
        *,
        packmol_setup_metadata: "PackmolSetupMetadata | None" = None,
    ) -> PackmolDockerSyncResult:
        local_dir = Path(local_packmol_inputs_dir).expanduser().resolve()
        if not local_dir.is_dir():
            raise ValueError(
                f"Local Packmol inputs directory does not exist: {local_dir}"
            )
        verified = self.verify_link(link)
        remote_inputs_dir = str(link.remote_packmol_inputs_dir())
        self._run_in_container(
            link,
            f'target={shlex.quote(remote_inputs_dir)}; mkdir -p "$target"',
        )
        self._run_docker(
            "cp",
            str(local_dir) + "/.",
            f"{link.container_name}:{remote_inputs_dir}",
        )
        synced_file_count = sum(
            1 for path in local_dir.iterdir() if path.is_file()
        )
        return PackmolDockerSyncResult(
            synced_at=verified.verified_at,
            remote_packmol_inputs_dir=remote_inputs_dir,
            remote_packmol_input_path=link.remote_packmol_input_path(
                packmol_setup_metadata
            ),
            remote_packed_output_path=link.remote_packmol_output_path(
                packmol_setup_metadata
            ),
            synced_file_count=synced_file_count,
        )

    def _container_is_running(self, container_name: str) -> bool:
        stdout = self._run_docker(
            "inspect",
            "--format",
            "{{.State.Running}}",
            container_name,
        ).stdout.strip()
        return stdout.lower() == "true"

    def _resolve_packmol_command_path(self, link: PackmolDockerLink) -> str:
        candidate = link.packmol_command.strip() or "packmol"
        stdout = self._run_in_container(
            link,
            f"candidate={shlex.quote(candidate)}; "
            'if command -v "$candidate" >/dev/null 2>&1; then '
            '  command -v "$candidate"; '
            'elif [ -x "$candidate" ]; then '
            '  printf "%s\\n" "$candidate"; '
            "else "
            "  exit 127; "
            "fi",
        )
        resolved = stdout.strip()
        if not resolved:
            raise RuntimeError(
                f"Unable to resolve Packmol command inside container: {candidate}"
            )
        return resolved

    def _resolve_packmol_version(
        self,
        link: PackmolDockerLink,
        command_path: str,
    ) -> str:
        stdout = self._run_in_container(
            link,
            f"candidate={shlex.quote(command_path)}; "
            'version_output=$("$candidate" --version 2>&1); '
            'if [ -z "$version_output" ]; then '
            '  version_output=$("$candidate" -version 2>&1); '
            "fi; "
            'if [ -z "$version_output" ]; then '
            '  version_output=$("$candidate" -v 2>&1); '
            "fi; "
            'if [ -z "$version_output" ]; then '
            '  version_output=$("$candidate" -h 2>&1); '
            "fi; "
            '[ -n "$version_output" ] || exit 125; '
            'printf "%s\\n" "$version_output" | sed -n \'/./{p;q;}\'',
        )
        version_line = stdout.strip()
        if not version_line:
            raise RuntimeError(
                "Packmol executable was found inside the container, but its "
                "version output could not be read."
            )
        return version_line

    def _run_in_container(
        self,
        link: PackmolDockerLink,
        script: str,
    ) -> str:
        completed = self._run_docker(
            "exec",
            link.container_name,
            link.shell_command.strip() or "sh",
            "-lc",
            script,
        )
        return completed.stdout

    def _run_docker(self, *args: str) -> subprocess.CompletedProcess[str]:
        try:
            completed = subprocess.run(
                [self.docker_executable, *args],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Docker CLI was not found on PATH. Install Docker and make "
                "sure the `docker` command is available before linking a "
                "Packmol container."
            ) from exc
        if completed.returncode != 0:
            message = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or "Docker command failed."
            )
            raise RuntimeError(message)
        return completed


def normalize_container_directory(value: object) -> str:
    text = _optional_text(value) or DEFAULT_PACKMOL_CONTAINER_ROOT
    if not text.startswith("/"):
        text = "/" + text
    normalized_parts: list[str] = []
    for part in text.split("/"):
        if not part or part == ".":
            continue
        if part == "..":
            if normalized_parts:
                normalized_parts.pop()
            continue
        normalized_parts.append(part)
    if not normalized_parts:
        return "/"
    return "/" + "/".join(normalized_parts)


def container_project_root_is_valid(value: object) -> bool:
    try:
        normalized = normalize_container_directory(value)
    except Exception:
        return False
    return _is_within_container_root(
        normalized,
        DEFAULT_PACKMOL_CONTAINER_ROOT,
    )


def docker_daemon_unavailable_hint(value: object) -> str | None:
    text = _optional_text(value)
    if text is None:
        return None
    normalized = " ".join(text.lower().split())
    if any(
        pattern in normalized
        for pattern in _DOCKER_DAEMON_UNAVAILABLE_PATTERNS
    ):
        return _DOCKER_DAEMON_UNAVAILABLE_HINT
    return None


def _validate_container_project_root(value: object) -> str:
    normalized = normalize_container_directory(value)
    if not _is_within_container_root(
        normalized,
        DEFAULT_PACKMOL_CONTAINER_ROOT,
    ):
        raise RuntimeError(
            "Container project root must be inside "
            f"{DEFAULT_PACKMOL_CONTAINER_ROOT} so Packmol input files stay "
            "inside the expected bind-mounted folder."
        )
    return normalized


def _is_within_container_root(path_text: str, root_text: str) -> bool:
    path_parts = PurePosixPath(path_text).parts
    root_parts = PurePosixPath(root_text).parts
    if len(path_parts) < len(root_parts):
        return False
    return path_parts[: len(root_parts)] == root_parts


def load_packmol_docker_link_metadata(
    metadata_path: str | Path,
) -> PackmolDockerLink | None:
    path = Path(metadata_path).expanduser().resolve()
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict) and "link" in payload:
        payload = payload.get("link")
    return PackmolDockerLink.from_dict(
        payload if isinstance(payload, dict) else None
    )


def save_packmol_docker_link_metadata(
    metadata_path: str | Path,
    link: PackmolDockerLink | None,
) -> Path:
    path = Path(metadata_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object]
    if link is None:
        payload = {}
    else:
        payload = {
            "version": 1,
            "link": link.to_dict(),
        }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


__all__ = [
    "DEFAULT_PACKMOL_CONTAINER_ROOT",
    "container_project_root_is_valid",
    "docker_daemon_unavailable_hint",
    "PackmolDockerClient",
    "PackmolDockerDirectoryEntry",
    "PackmolDockerLink",
    "PackmolDockerSyncResult",
    "PackmolDockerValidationResult",
    "load_packmol_docker_link_metadata",
    "normalize_container_directory",
    "save_packmol_docker_link_metadata",
]
