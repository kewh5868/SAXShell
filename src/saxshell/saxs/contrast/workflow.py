from __future__ import annotations

from dataclasses import dataclass

from .settings import ContrastModeLaunchContext


@dataclass(slots=True, frozen=True)
class ContrastWorkflowPreview:
    launch_context: ContrastModeLaunchContext
    stage_labels: tuple[str, ...]
    warnings: tuple[str, ...]
    notes: tuple[str, ...]

    def summary_text(self) -> str:
        lines = [
            "Contrast-mode SAXS workspace",
            (
                "This supporting application is reserved for the contrast-enabled "
                "Debye workflow and does not replace the existing no-contrast "
                "component builder."
            ),
            "",
            "Launch context",
            (
                "Project folder: "
                + (
                    str(self.launch_context.project_dir)
                    if self.launch_context.project_dir is not None
                    else "Not provided"
                )
            ),
            (
                "Clusters folder: "
                + (
                    str(self.launch_context.clusters_dir)
                    if self.launch_context.clusters_dir is not None
                    else "Not provided"
                )
            ),
            (
                "Experimental data: "
                + (
                    str(self.launch_context.experimental_data_file)
                    if self.launch_context.experimental_data_file is not None
                    else "Not provided"
                )
            ),
            f"q-range: {self.launch_context.q_range_text}",
            (
                "Active template: "
                + (self.launch_context.active_template_name or "Not provided")
            ),
            "",
            "Planned stages",
        ]
        lines.extend(
            f"  {index}. {label}"
            for index, label in enumerate(self.stage_labels, start=1)
        )
        if self.warnings:
            lines.extend(["", "Warnings"])
            lines.extend(f"  - {warning}" for warning in self.warnings)
        if self.notes:
            lines.extend(["", "Scaffold notes"])
            lines.extend(f"  - {note}" for note in self.notes)
        return "\n".join(lines)


def build_contrast_workflow_preview(
    launch_context: ContrastModeLaunchContext,
) -> ContrastWorkflowPreview:
    warnings: list[str] = []
    if launch_context.project_dir is None:
        warnings.append(
            "No project folder was supplied, so the scaffold is running without "
            "saved-project context."
        )
    if launch_context.clusters_dir is None:
        warnings.append(
            "No clusters folder is linked yet. Representative-structure analysis "
            "will need this in a later implementation step."
        )
    if launch_context.experimental_data_file is None:
        warnings.append(
            "No experimental data file is linked yet. Plot previews are "
            "currently placeholder panels."
        )
    if launch_context.q_min is None or launch_context.q_max is None:
        warnings.append(
            "The launch q-range is incomplete, so the scaffold will inherit the "
            "main UI range later when build-mode routing is added."
        )

    notes = (
        "The current no-contrast Debye builder remains unchanged.",
        "This window now provides the contrast workflow shell, including "
        "inherited inputs, preview plots, cluster inspection tables, and a "
        "placeholder representative-structure viewer.",
        "The representative-structure analysis backend now screens each "
        "recognized stoichiometry bin and writes human-readable outputs into "
        "the project contrast workflow folder.",
        "Later queue items will connect the retained mesh construction, "
        "electron-density estimation, and contrast-aware Debye calculation "
        "backends to this UI.",
    )
    stages = (
        "Resolve inherited project inputs and active computed-distribution context.",
        "Analyze cluster bins and select representative structures.",
        "Build a retained volume mesh and compute electron-density terms.",
        "Run the contrast-aware Debye forward simulation.",
        "Push the completed component outputs back into the main SAXS project.",
    )
    return ContrastWorkflowPreview(
        launch_context=launch_context,
        stage_labels=stages,
        warnings=tuple(warnings),
        notes=notes,
    )
