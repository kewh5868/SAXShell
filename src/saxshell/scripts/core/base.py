from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class SAXShellConfig:
    """Base configuration for a SAXShell workflow."""

    project_dir: Path = Path(".")
    verbose: bool = True


@dataclass(slots=True)
class SAXShellResult:
    """Base result object for a SAXShell workflow."""

    workflow_name: str
    message: str

    def summary(self) -> str:
        return f"[{self.workflow_name}] {self.message}"


class SAXShellWorkflow:
    """Base workflow interface for SAXShell applications."""

    workflow_name: str = "workflow"

    def __init__(self, config: SAXShellConfig) -> None:
        self.config = config

    def run(self) -> SAXShellResult:
        raise NotImplementedError("Subclasses must implement run().")
