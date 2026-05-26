from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from saxshell.saxs.dream.distributions import (
    DreamParameterEntry,
    build_default_parameter_map_from_prefit_entries,
    load_parameter_map,
    recentered_parameter_entry,
    save_parameter_map,
)
from saxshell.saxs.dream.settings import (
    DreamRunSettings,
    dream_run_settings_to_dict,
    load_dream_settings,
    save_dream_settings,
)
from saxshell.saxs.prefit import SAXSPrefitWorkflow
from saxshell.saxs.prefit.workflow import PrefitParameterEntry
from saxshell.saxs.project_manager import (
    ProjectSettings,
    SAXSProjectManager,
    build_project_paths,
    project_artifact_paths,
)
from saxshell.saxs.stoichiometry import build_stoichiometry_target
from saxshell.saxs.stoichiometry_compensator import (
    guess_single_atom_compensator_names,
    template_uses_stoichiometry_compensator,
)


@dataclass(slots=True)
class DreamRunBundle:
    run_dir: Path
    runtime_script_path: Path
    metadata_path: Path
    settings_path: Path
    parameter_map_path: Path


class SAXSDreamWorkflow:
    """Create and manage SAXS DREAM runtime bundles."""

    def __init__(
        self,
        project_dir: str | Path,
        *,
        template_dir: str | Path | None = None,
    ) -> None:
        self.project_manager = SAXSProjectManager()
        self.settings = self.project_manager.load_project(project_dir)
        self.paths = build_project_paths(self.settings.project_dir)
        self.artifact_paths = project_artifact_paths(self.settings)
        self.template_dir = (
            Path(template_dir).expanduser().resolve()
            if template_dir is not None
            else None
        )
        self.prefit_workflow = SAXSPrefitWorkflow(
            self.paths.project_dir,
            template_dir=self.template_dir,
        )
        self.dream_dir = self.artifact_paths.dream_dir
        self.dream_runtime_dir = self.artifact_paths.dream_runtime_dir
        self.prefit_dir = self.artifact_paths.prefit_dir
        self.project_manager.ensure_artifact_dirs(self.artifact_paths)
        self.dream_settings_path = self.dream_dir / "pd_settings.json"
        self.parameter_map_path = self.dream_dir / "pd_param_map.json"
        self.settings_presets_dir = self.dream_dir / "settings_presets"
        self.settings_presets_dir.mkdir(parents=True, exist_ok=True)

    def apply_project_settings(
        self,
        settings: ProjectSettings,
    ) -> None:
        incoming_settings = ProjectSettings.from_dict(settings.to_dict())
        if incoming_settings.resolved_project_dir != self.paths.project_dir:
            raise ValueError(
                "Cannot apply project settings from a different SAXS project."
            )
        self.settings = incoming_settings
        self.paths = build_project_paths(self.settings.project_dir)
        self.artifact_paths = project_artifact_paths(self.settings)
        self.dream_dir = self.artifact_paths.dream_dir
        self.dream_runtime_dir = self.artifact_paths.dream_runtime_dir
        self.prefit_dir = self.artifact_paths.prefit_dir
        self.project_manager.ensure_artifact_dirs(self.artifact_paths)
        self.dream_settings_path = self.dream_dir / "pd_settings.json"
        self.parameter_map_path = self.dream_dir / "pd_param_map.json"
        self.settings_presets_dir = self.dream_dir / "settings_presets"
        self.settings_presets_dir.mkdir(parents=True, exist_ok=True)
        self.prefit_workflow.apply_project_settings(incoming_settings)

    def load_settings(self) -> DreamRunSettings:
        settings = load_dream_settings(self.dream_settings_path)
        if settings.model_name is None:
            settings.model_name = self.prefit_workflow.template_spec.name
        return settings

    def save_settings(self, settings: DreamRunSettings) -> Path:
        if settings.model_name is None:
            settings.model_name = self.prefit_workflow.template_spec.name
        return save_dream_settings(self.dream_settings_path, settings)

    def list_settings_presets(self) -> list[str]:
        return sorted(
            [
                path.stem
                for path in self.settings_presets_dir.glob("*.json")
                if path.is_file()
            ]
        )

    def load_settings_preset(
        self, preset_name: str | None
    ) -> DreamRunSettings:
        if preset_name is None:
            return self.load_settings()
        preset_path = self.settings_presets_dir / (
            f"{self._sanitize_preset_name(preset_name)}.json"
        )
        settings = load_dream_settings(preset_path)
        if settings.model_name is None:
            settings.model_name = self.prefit_workflow.template_spec.name
        return settings

    def save_settings_preset(
        self,
        settings: DreamRunSettings,
        preset_name: str | None,
    ) -> Path:
        active_path = self.save_settings(settings)
        if preset_name is None:
            return active_path
        preset_path = self.settings_presets_dir / (
            f"{self._sanitize_preset_name(preset_name)}.json"
        )
        return save_dream_settings(preset_path, settings)

    def load_parameter_map(
        self,
        *,
        persist_if_missing: bool = True,
    ) -> list[DreamParameterEntry]:
        if not self.parameter_map_path.is_file():
            return self.create_default_parameter_map(
                persist=persist_if_missing
            )
        payload = json.loads(
            self.parameter_map_path.read_text(encoding="utf-8")
        )
        entries = load_parameter_map(self.parameter_map_path)
        normalized_entries = self._normalize_parameter_map_entries(entries)
        normalized_payload = {
            "entries": [entry.to_dict() for entry in normalized_entries]
        }
        if payload != normalized_payload:
            save_parameter_map(self.parameter_map_path, normalized_entries)
        return normalized_entries

    def save_parameter_map(
        self,
        entries: list[DreamParameterEntry],
    ) -> Path:
        return save_parameter_map(
            self.parameter_map_path,
            self._normalize_parameter_map_entries(entries),
        )

    def create_default_parameter_map(
        self,
        *,
        persist: bool = True,
    ) -> list[DreamParameterEntry]:
        self._reload_prefit_workflow()
        entries = build_default_parameter_map_from_prefit_entries(
            self.prefit_workflow.parameter_entries
        )
        entries = self._apply_stoichiometry_compensator_map_defaults(entries)
        if persist:
            self.save_parameter_map(entries)
        return entries

    def sync_parameter_map_to_prefit(
        self,
        *,
        persist: bool = True,
    ) -> list[DreamParameterEntry]:
        self._reload_prefit_workflow()
        current_entries = (
            load_parameter_map(self.parameter_map_path)
            if self.parameter_map_path.is_file()
            else []
        )
        synced_entries = self._sync_parameter_map_entries_to_prefit(
            current_entries
        )
        if persist:
            self.save_parameter_map(synced_entries)
        return synced_entries

    def create_runtime_bundle(
        self,
        *,
        settings: DreamRunSettings | None = None,
        entries: list[DreamParameterEntry] | None = None,
        prefit_parameter_entries: list[PrefitParameterEntry] | None = None,
        include_posterior_filter_settings: bool = True,
    ) -> DreamRunBundle:
        self._reload_prefit_workflow()
        active_prefit_entries = (
            [
                PrefitParameterEntry.from_dict(entry.to_dict())
                for entry in prefit_parameter_entries
            ]
            if prefit_parameter_entries is not None
            else self.prefit_workflow.parameter_entries
        )
        parameter_entries = self._normalize_parameter_map_entries(
            entries or self.load_parameter_map(),
            prefit_parameter_entries=active_prefit_entries,
        )
        active_settings = self._prepare_runtime_settings(
            settings or self.load_settings(),
            parameter_entries,
        )
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        run_name = (
            f"dream_{self._sanitize_runtime_name(self.paths.project_dir.name)}"
            f"_{timestamp}"
        )
        run_dir = self.dream_runtime_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=False)

        runtime_script_path = run_dir / f"{run_name}.py"
        metadata_path = run_dir / "dream_runtime_metadata.json"
        settings_path = run_dir / "pd_settings.json"
        parameter_map_path = run_dir / "pd_param_map.json"

        save_dream_settings(
            settings_path,
            active_settings,
            include_posterior_filter_settings=(
                include_posterior_filter_settings
            ),
        )
        save_parameter_map(parameter_map_path, parameter_entries)
        self._copy_prefit_snapshot(run_dir)

        metadata = self._build_runtime_metadata(
            active_settings,
            parameter_entries,
            prefit_parameter_entries=active_prefit_entries,
            include_posterior_filter_settings=(
                include_posterior_filter_settings
            ),
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        runtime_script_path.write_text(
            self._build_runtime_script(metadata),
            encoding="utf-8",
        )
        return DreamRunBundle(
            run_dir=run_dir,
            runtime_script_path=runtime_script_path,
            metadata_path=metadata_path,
            settings_path=settings_path,
            parameter_map_path=parameter_map_path,
        )

    def run_bundle(
        self,
        bundle: DreamRunBundle,
        *,
        output_callback: Callable[[str], None] | None = None,
        output_interval_seconds: float | None = None,
    ) -> dict[str, object]:
        captured_lines: list[str] = []
        buffered_lines: list[str] = []
        emit_interval = (
            None
            if output_interval_seconds is None
            else max(float(output_interval_seconds), 0.0)
        )
        last_emit_time = time.monotonic()

        def emit_buffered_output(*, force: bool = False) -> None:
            nonlocal last_emit_time
            if output_callback is None or not buffered_lines:
                return
            now = time.monotonic()
            if not force and emit_interval is not None and emit_interval > 0:
                if (now - last_emit_time) < emit_interval:
                    return
            output_callback("\n".join(buffered_lines))
            buffered_lines.clear()
            last_emit_time = now

        with subprocess.Popen(
            [sys.executable, str(bundle.runtime_script_path)],
            cwd=str(bundle.run_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as process:
            if process.stdout is not None:
                for raw_line in process.stdout:
                    line = raw_line.rstrip()
                    if not line:
                        continue
                    captured_lines.append(line)
                    if output_callback is not None:
                        if emit_interval is None or emit_interval <= 0:
                            output_callback(line)
                        else:
                            buffered_lines.append(line)
                            emit_buffered_output()
            returncode = process.wait()
        emit_buffered_output(force=True)
        if returncode != 0:
            details = "\n".join(captured_lines)
            raise RuntimeError(
                details
                or (
                    "DREAM runtime bundle failed with exit code "
                    f"{returncode}."
                )
            )
        sampled_params_path = bundle.run_dir / "dream_sampled_params.npy"
        log_ps_path = bundle.run_dir / "dream_log_ps.npy"
        if not sampled_params_path.is_file() or not log_ps_path.is_file():
            raise FileNotFoundError(
                "DREAM runtime bundle finished without writing the expected "
                "sample output files."
            )
        return {
            "sampled_params_path": str(sampled_params_path),
            "log_ps_path": str(log_ps_path),
        }

    def _load_runtime_module(
        self,
        bundle: DreamRunBundle,
    ) -> tuple[str, object, bool]:
        module_name = bundle.runtime_script_path.stem
        run_dir = str(bundle.run_dir)
        added_sys_path = False
        if run_dir not in sys.path:
            sys.path.insert(0, run_dir)
            added_sys_path = True
        import_spec = importlib.util.spec_from_file_location(
            module_name,
            bundle.runtime_script_path,
        )
        if import_spec is None or import_spec.loader is None:
            raise ImportError(
                f"Unable to import generated runtime script from "
                f"{bundle.runtime_script_path}"
            )
        module = importlib.util.module_from_spec(import_spec)
        sys.modules[module_name] = module
        try:
            import_spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            if added_sys_path:
                try:
                    sys.path.remove(run_dir)
                except ValueError:
                    pass
            raise
        return module_name, module, added_sys_path

    @staticmethod
    def _unload_runtime_module(
        module_name: str,
        *,
        added_sys_path: bool,
    ) -> None:
        module = sys.modules.pop(module_name, None)
        if added_sys_path and module is not None:
            module_path = getattr(module, "__file__", None)
            if module_path:
                run_dir = str(Path(module_path).resolve().parent)
                try:
                    sys.path.remove(run_dir)
                except ValueError:
                    pass

    def _copy_prefit_snapshot(self, run_dir: Path) -> None:
        prefit_json = self.prefit_dir / "pd_prefit_params.json"
        prefit_state = self.prefit_dir / "prefit_state.json"
        if prefit_json.is_file():
            shutil.copy2(prefit_json, run_dir / prefit_json.name)
        if prefit_state.is_file():
            shutil.copy2(prefit_state, run_dir / prefit_state.name)

    def _prepare_runtime_settings(
        self,
        settings: DreamRunSettings,
        entries: list[DreamParameterEntry],
    ) -> DreamRunSettings:
        active_settings = DreamRunSettings.from_dict(settings.to_dict())
        if not active_settings.model_name:
            active_settings.model_name = (
                self.prefit_workflow.template_spec.name
            )
        active_count = sum(1 for entry in entries if entry.vary)
        if active_count <= 0:
            raise ValueError(
                "No DREAM parameters are currently set to vary. Edit the "
                "priors and enable vary for at least one refinable parameter "
                "before writing or running a DREAM bundle."
            )
        if active_settings.nchains < active_count:
            active_settings.nchains = active_count
        if active_settings.nseedchains <= 0:
            active_settings.nseedchains = max(10 * active_settings.nchains, 1)
        active_settings.nseedchains = max(
            int(active_settings.nseedchains),
            int(active_settings.nchains) * 2,
        )
        if active_settings.history_file is not None:
            history_file = str(active_settings.history_file).strip()
            active_settings.history_file = history_file or None
        return active_settings

    def _reload_prefit_workflow(self) -> None:
        self.prefit_workflow = SAXSPrefitWorkflow(
            self.paths.project_dir,
            template_dir=self.template_dir,
        )

    def _normalize_parameter_map_entries(
        self,
        entries: list[DreamParameterEntry],
        *,
        prefit_parameter_entries: list[PrefitParameterEntry] | None = None,
    ) -> list[DreamParameterEntry]:
        default_entries = build_default_parameter_map_from_prefit_entries(
            prefit_parameter_entries or self.prefit_workflow.parameter_entries
        )
        existing_by_param = {entry.param: entry for entry in entries}
        existing_by_component = {
            self._component_key(entry): entry
            for entry in entries
            if self._is_weight_entry(entry)
        }
        default_weight_count = sum(
            1 for entry in default_entries if self._is_weight_entry(entry)
        )
        default_signature = [
            self._entry_signature(entry) for entry in default_entries
        ]
        existing_signature = [
            self._entry_signature(existing_by_param[entry.param])
            for entry in default_entries
            if entry.param in existing_by_param
        ]
        if existing_signature == default_signature:
            return self._apply_stoichiometry_compensator_map_defaults(
                [existing_by_param[entry.param] for entry in default_entries]
            )

        normalized_entries: list[DreamParameterEntry] = []
        for default_entry in default_entries:
            existing_entry = self._matching_existing_entry(
                default_entry,
                existing_by_param=existing_by_param,
                existing_by_component=existing_by_component,
                default_weight_count=default_weight_count,
            )
            if existing_entry is None:
                normalized_entries.append(default_entry)
                continue
            existing_param = str(existing_entry.param)
            should_recenter = self._is_weight_entry(
                default_entry
            ) and existing_param != str(default_entry.param)
            source_entry = (
                recentered_parameter_entry(existing_entry, default_entry.value)
                if should_recenter
                else existing_entry
            )
            normalized_entries.append(
                DreamParameterEntry(
                    structure=default_entry.structure,
                    motif=default_entry.motif,
                    param_type=default_entry.param_type,
                    param=default_entry.param,
                    value=source_entry.value,
                    vary=source_entry.vary,
                    distribution=source_entry.distribution,
                    dist_params=source_entry.dist_params,
                    smart_preset_status=source_entry.smart_preset_status,
                )
            )
        return self._apply_stoichiometry_compensator_map_defaults(
            normalized_entries
        )

    def _sync_parameter_map_entries_to_prefit(
        self,
        entries: list[DreamParameterEntry],
    ) -> list[DreamParameterEntry]:
        default_entries = build_default_parameter_map_from_prefit_entries(
            self.prefit_workflow.parameter_entries
        )
        existing_by_param = {entry.param: entry for entry in entries}
        existing_by_component = {
            self._component_key(entry): entry
            for entry in entries
            if self._is_weight_entry(entry)
        }
        default_weight_count = sum(
            1 for entry in default_entries if self._is_weight_entry(entry)
        )
        synced_entries: list[DreamParameterEntry] = []
        for default_entry in default_entries:
            existing_entry = self._matching_existing_entry(
                default_entry,
                existing_by_param=existing_by_param,
                existing_by_component=existing_by_component,
                default_weight_count=default_weight_count,
            )
            if existing_entry is None:
                synced_entries.append(default_entry)
                continue
            recentered_entry = recentered_parameter_entry(
                existing_entry,
                default_entry.value,
            )
            synced_entries.append(
                DreamParameterEntry(
                    structure=default_entry.structure,
                    motif=default_entry.motif,
                    param_type=default_entry.param_type,
                    param=default_entry.param,
                    value=recentered_entry.value,
                    vary=existing_entry.vary,
                    distribution=existing_entry.distribution,
                    dist_params=recentered_entry.dist_params,
                    smart_preset_status=existing_entry.smart_preset_status,
                )
            )
        return self._apply_stoichiometry_compensator_map_defaults(
            synced_entries
        )

    def _apply_stoichiometry_compensator_map_defaults(
        self,
        entries: list[DreamParameterEntry],
    ) -> list[DreamParameterEntry]:
        compensator_names = self._stoichiometry_compensator_weight_names()
        if not compensator_names:
            return entries
        adjusted_entries: list[DreamParameterEntry] = []
        for entry in entries:
            if str(entry.param).strip() not in compensator_names:
                adjusted_entries.append(entry)
                continue
            adjusted_entries.append(
                DreamParameterEntry(
                    structure=entry.structure,
                    motif=entry.motif,
                    param_type=entry.param_type,
                    param=entry.param,
                    value=entry.value,
                    vary=False,
                    distribution=entry.distribution,
                    dist_params=dict(entry.dist_params),
                    smart_preset_status=entry.smart_preset_status,
                )
            )
        return adjusted_entries

    def _stoichiometry_compensator_weight_names(self) -> set[str]:
        if not template_uses_stoichiometry_compensator(
            self.prefit_workflow.template_spec.name
        ):
            return set()
        selected_names = {
            str(name).strip()
            for name in (
                self.prefit_workflow.settings.stoichiometry_compensator_weight_names
            )
            if str(name).strip()
        }
        if selected_names:
            return selected_names

        workflow_settings = self.prefit_workflow.settings
        try:
            target = build_stoichiometry_target(
                workflow_settings.stoichiometry_compensator_target_elements_text,
                workflow_settings.stoichiometry_compensator_target_ratio_text,
            )
        except ValueError:
            target = None
        elements = (
            target.elements
            if target is not None
            else tuple(
                self._stoichiometry_compensator_elements_from_text(
                    workflow_settings.stoichiometry_compensator_target_elements_text
                )
            )
        )
        if not elements:
            return set()
        return set(
            guess_single_atom_compensator_names(
                tuple(
                    (component.param_name, component.structure)
                    for component in self.prefit_workflow.components
                ),
                elements,
            )
        )

    @staticmethod
    def _stoichiometry_compensator_elements_from_text(text: str) -> list[str]:
        return [
            token.strip()
            for token in str(text or "")
            .replace(":", ",")
            .replace(";", ",")
            .split(",")
            if token.strip()
        ]

    @staticmethod
    def _is_weight_entry(entry: DreamParameterEntry) -> bool:
        name = str(entry.param).strip()
        return bool(
            re.fullmatch(r"w\d+", name)
            and (str(entry.structure).strip() or str(entry.motif).strip())
        )

    @staticmethod
    def _component_key(entry: DreamParameterEntry) -> tuple[str, str]:
        return (str(entry.structure).strip(), str(entry.motif).strip())

    @classmethod
    def _entry_signature(
        cls,
        entry: DreamParameterEntry,
    ) -> tuple[str, str, str]:
        if cls._is_weight_entry(entry):
            structure, motif = cls._component_key(entry)
            return (str(entry.param).strip(), structure, motif)
        return (str(entry.param).strip(), "", "")

    @classmethod
    def _matching_existing_entry(
        cls,
        default_entry: DreamParameterEntry,
        *,
        existing_by_param: dict[str, DreamParameterEntry],
        existing_by_component: dict[tuple[str, str], DreamParameterEntry],
        default_weight_count: int,
    ) -> DreamParameterEntry | None:
        if cls._is_weight_entry(default_entry):
            component_match = existing_by_component.get(
                cls._component_key(default_entry)
            )
            if component_match is not None:
                return component_match
            param_match = existing_by_param.get(default_entry.param)
            if param_match is not None and cls._component_key(
                param_match
            ) == cls._component_key(default_entry):
                return param_match
            if param_match is not None and default_weight_count <= 1:
                return param_match
            return None
        return existing_by_param.get(default_entry.param)

    @staticmethod
    def _sanitize_preset_name(preset_name: str) -> str:
        cleaned = re.sub(r'[<>:"/\\\\|?*]+', "_", preset_name.strip())
        cleaned = re.sub(r"\s+", " ", cleaned)
        return cleaned.strip(" ._") or "dream_settings"

    @staticmethod
    def _sanitize_runtime_name(name: str) -> str:
        cleaned = re.sub(r"\W+", "_", name.strip())
        cleaned = cleaned.strip("_")
        if not cleaned:
            return "project"
        if cleaned[0].isdigit():
            return f"project_{cleaned}"
        return cleaned

    def _resolve_runtime_fit_q_range(
        self,
        settings: DreamRunSettings,
        prefit_parameter_entries: list[PrefitParameterEntry],
    ) -> tuple[float, float] | None:
        if settings.fit_q_min is None and settings.fit_q_max is None:
            return None
        active_components = (
            self.prefit_workflow._active_components_for_entries(
                prefit_parameter_entries
            )
        )
        q_values = self.prefit_workflow._component_q_values(active_components)
        if q_values.size == 0:
            raise ValueError("The active Prefit model trace has no q-values.")
        requested_q_min = (
            float(np.min(q_values))
            if settings.fit_q_min is None
            else float(settings.fit_q_min)
        )
        requested_q_max = (
            float(np.max(q_values))
            if settings.fit_q_max is None
            else float(settings.fit_q_max)
        )
        return self.prefit_workflow._resolve_prefit_fit_q_bounds(
            q_values,
            requested_q_min=requested_q_min,
            requested_q_max=requested_q_max,
        )

    def _build_runtime_metadata(
        self,
        settings: DreamRunSettings,
        entries: list[DreamParameterEntry],
        *,
        prefit_parameter_entries: list[PrefitParameterEntry] | None = None,
        include_posterior_filter_settings: bool = True,
    ) -> dict[str, object]:
        source_prefit_entries = (
            prefit_parameter_entries or self.prefit_workflow.parameter_entries
        )
        runtime_fit_q_range = self._resolve_runtime_fit_q_range(
            settings,
            source_prefit_entries,
        )
        prefit_entries = (
            self.prefit_workflow._active_parameter_entries_for_model(
                source_prefit_entries
            )
        )
        full_parameter_names = [entry.name for entry in prefit_entries]
        fixed_parameter_values = [entry.value for entry in prefit_entries]
        parameter_components = [
            {
                "param": entry.name,
                "structure": entry.structure,
                "motif": entry.motif,
                "category": entry.category,
            }
            for entry in prefit_entries
        ]
        active_indices: list[int] = []
        active_entries: list[dict[str, object]] = []
        for dream_entry in entries:
            if not dream_entry.vary:
                continue
            if dream_entry.param not in full_parameter_names:
                continue
            active_indices.append(
                full_parameter_names.index(dream_entry.param)
            )
            active_entries.append(dream_entry.to_dict())

        evaluation_kwargs = {}
        if runtime_fit_q_range is not None:
            evaluation_kwargs = {
                "fit_q_min": runtime_fit_q_range[0],
                "fit_q_max": runtime_fit_q_range[1],
            }
        evaluation = self.prefit_workflow.evaluate(
            source_prefit_entries,
            **evaluation_kwargs,
        )
        evaluation_q_values = np.asarray(evaluation.q_values, dtype=float)
        fit_mask = self.prefit_workflow.evaluation_fit_mask(evaluation)
        if fit_mask.shape != evaluation_q_values.shape or not np.any(fit_mask):
            raise ValueError(
                "The active Prefit fit q-range does not provide any q-points "
                "for DREAM."
            )
        runtime_q_values = np.asarray(
            evaluation_q_values[fit_mask],
            dtype=float,
        )
        output_q_values = evaluation_q_values
        template_runtime_inputs = (
            self.prefit_workflow.template_runtime_inputs_payload(
                parameter_entries=source_prefit_entries
            )
        )
        active_components = (
            self.prefit_workflow._active_components_for_entries(
                source_prefit_entries
            )
        )
        metadata = {
            "project_dir": str(self.paths.project_dir),
            "template_name": self.prefit_workflow.template_spec.name,
            "settings": dream_run_settings_to_dict(
                settings,
                include_posterior_filter_settings=(
                    include_posterior_filter_settings
                ),
            ),
            "parameter_map": [entry.to_dict() for entry in entries],
            "active_parameter_entries": active_entries,
            "active_parameter_indices": active_indices,
            "full_parameter_names": full_parameter_names,
            "parameter_components": parameter_components,
            "fixed_parameter_values": fixed_parameter_values,
            "q_values": runtime_q_values.tolist(),
            "output_q_values": output_q_values.tolist(),
            "output_fit_mask": np.asarray(fit_mask, dtype=bool).tolist(),
            "prefit_fit_q_range": {
                "q_min": (
                    None
                    if evaluation.fit_q_min is None
                    else float(evaluation.fit_q_min)
                ),
                "q_max": (
                    None
                    if evaluation.fit_q_max is None
                    else float(evaluation.fit_q_max)
                ),
                "model_q_min": (
                    float(np.min(evaluation_q_values))
                    if evaluation_q_values.size
                    else None
                ),
                "model_q_max": (
                    float(np.max(evaluation_q_values))
                    if evaluation_q_values.size
                    else None
                ),
                "point_count": int(runtime_q_values.size),
            },
            "experimental_intensities": (
                np.asarray(
                    evaluation.experimental_intensities,
                    dtype=float,
                )[fit_mask].tolist()
            ),
            "output_experimental_intensities": (
                np.asarray(
                    evaluation.experimental_intensities,
                    dtype=float,
                ).tolist()
            ),
            "theoretical_intensities": [
                intensities.tolist()
                for intensities in self.prefit_workflow._model_data_for_q_values(
                    runtime_q_values,
                    components=active_components,
                )
            ],
            "output_theoretical_intensities": [
                intensities.tolist()
                for intensities in self.prefit_workflow._model_data_for_q_values(
                    output_q_values,
                    components=active_components,
                )
            ],
            "solvent_intensities": (
                np.asarray(
                    self.prefit_workflow._solvent_trace_for_q_values(
                        runtime_q_values
                    ),
                    dtype=float,
                ).tolist()
                if self.prefit_workflow.solvent_data is not None
                else [0.0] * len(runtime_q_values)
            ),
            "output_solvent_intensities": (
                np.asarray(
                    self.prefit_workflow._solvent_trace_for_q_values(
                        output_q_values
                    ),
                    dtype=float,
                ).tolist()
                if self.prefit_workflow.solvent_data is not None
                else [0.0] * len(output_q_values)
            ),
            "template_runtime_inputs": template_runtime_inputs,
            "lmfit_extra_inputs": list(
                self.prefit_workflow.template_spec.extra_lmfit_inputs
            ),
            "cluster_geometry_metadata": (
                None
                if self.prefit_workflow.cluster_geometry_table is None
                else self.prefit_workflow.cluster_geometry_table.to_dict()
            ),
        }
        return metadata

    def _build_runtime_script(self, metadata: dict[str, object]) -> str:
        template_path = self.prefit_workflow.template_spec.module_path
        template_source = template_path.read_text(encoding="utf-8")
        lmfit_model_name = self.prefit_workflow.template_spec.lmfit_model_name
        dream_model_name = self.prefit_workflow.template_spec.dream_model_name
        return f"""from __future__ import annotations

import json
import multiprocessing as mp
import os
import warnings
from pathlib import Path

import numpy as np
from scipy import stats

try:
    from pydream.core import run_dream
    from pydream.Dream import Dream as _PyDreamDream, Dream_shared_vars
    from pydream.parameters import SampledParam
except ImportError:
    run_dream = None
    _PyDreamDream = None
    Dream_shared_vars = None
    SampledParam = None


RUN_DIR = Path(__file__).resolve().parent
RUNTIME_METADATA = json.loads(
    (RUN_DIR / "dream_runtime_metadata.json").read_text(encoding="utf-8")
)
SETTINGS = RUNTIME_METADATA["settings"]
FULL_PARAMETER_NAMES = list(RUNTIME_METADATA["full_parameter_names"])
ACTIVE_PARAMETER_INDICES = list(RUNTIME_METADATA["active_parameter_indices"])
FIXED_PARAMETER_VALUES = np.asarray(
    RUNTIME_METADATA["fixed_parameter_values"],
    dtype=float,
)
ACTIVE_PARAMETER_ENTRIES = list(RUNTIME_METADATA["active_parameter_entries"])
q_values = np.asarray(RUNTIME_METADATA["q_values"], dtype=float)
experimental_intensities = np.asarray(
    RUNTIME_METADATA["experimental_intensities"],
    dtype=float,
)
theoretical_intensities = [
    np.asarray(values, dtype=float)
    for values in RUNTIME_METADATA["theoretical_intensities"]
]
solvent_intensities = np.asarray(
    RUNTIME_METADATA["solvent_intensities"],
    dtype=float,
)
TEMPLATE_RUNTIME_INPUTS = {{
    str(name): np.asarray(values, dtype=float)
    for name, values in dict(
        RUNTIME_METADATA.get("template_runtime_inputs", {{}})
    ).items()
}}
LMFIT_EXTRA_INPUTS = list(RUNTIME_METADATA.get("lmfit_extra_inputs", []))
globals().update(TEMPLATE_RUNTIME_INPUTS)
GUIDE_INTERVAL_LOWER_Q = float(stats.norm.cdf(-3.0))
GUIDE_INTERVAL_UPPER_Q = float(stats.norm.cdf(3.0))
PRACTICAL_BOUND_TOLERANCE = 1e-12

warnings.filterwarnings(
    "ignore",
    message=r".*'where' used without 'out'.*memory in output.*",
    category=UserWarning,
    module=r"pydream\\.Dream",
)


def _coerce_runtime_scalar(value):
    array_value = np.asarray(value)
    if array_value.ndim == 0:
        return array_value.item()
    flat_value = array_value.reshape(-1)
    if flat_value.size != 1:
        raise ValueError(
            "Expected a single DREAM selector value, got "
            f"shape {{array_value.shape}}."
        )
    return flat_value[0].item()


def _choose_runtime_index(probabilities):
    draws = np.random.multinomial(1, np.asarray(probabilities, dtype=float))
    matches = np.flatnonzero(draws)
    if matches.size != 1:
        raise RuntimeError(
            "Unable to choose a unique DREAM selector index from "
            f"probabilities={{probabilities!r}}."
        )
    return int(matches[0])


def _apply_pydream_numpy_compatibility():
    if _PyDreamDream is None or Dream_shared_vars is None:
        return
    if getattr(_PyDreamDream, "_saxshell_numpy_compatibility", False):
        return

    def _patched_set_snooker(self):
        if self.snooker == 0:
            return False
        return _choose_runtime_index([self.snooker, 1 - self.snooker]) == 0

    def _patched_set_CR(self, CR_probs, CR_vals):
        return _coerce_runtime_scalar(
            np.asarray(CR_vals)[_choose_runtime_index(CR_probs)]
        )

    def _patched_set_gamma_level(self, gamma_level_probs, gamma_level_vals):
        return int(
            _coerce_runtime_scalar(
                np.asarray(gamma_level_vals)[
                    _choose_runtime_index(gamma_level_probs)
                ]
            )
        )

    def _patched_set_gamma(
        self,
        DEpairs,
        snooker_choice,
        gamma_level_choice,
        d_prime,
    ):
        if snooker_choice:
            return np.random.uniform(1.2, 2.2)
        gamma_unity = _choose_runtime_index(
            [self.p_gamma_unity, 1 - self.p_gamma_unity]
        )
        if gamma_unity == 0:
            return 1.0
        gamma_level_index = int(_coerce_runtime_scalar(gamma_level_choice)) - 1
        depairs_index = int(_coerce_runtime_scalar(DEpairs)) - 1
        d_prime_index = int(_coerce_runtime_scalar(d_prime)) - 1
        return self.gamma_arr[gamma_level_index][depairs_index][d_prime_index]

    def _patched_estimate_crossover_probabilities(self, ndim, q0, q_new, CR):
        cross_probs = Dream_shared_vars.cross_probs[0:self.nCR]
        cr_value = float(_coerce_runtime_scalar(CR))
        matches = np.flatnonzero(
            np.asarray(self.CR_values, dtype=float) == cr_value
        )
        if matches.size != 1:
            raise ValueError(
                "Unable to match DREAM crossover value "
                f"{{cr_value}} in {{self.CR_values!r}}."
            )
        m_loc = int(matches[0])

        Dream_shared_vars.ncr_updates[m_loc] += 1

        current_positions = np.frombuffer(
            Dream_shared_vars.current_positions.get_obj()
        )
        current_positions = current_positions.reshape((self.nchains, ndim))

        sd_by_dim = np.std(current_positions, axis=0)
        sd_by_dim[sd_by_dim == 0] = 1e-12

        change = np.nan_to_num(np.sum(((q_new - q0) / sd_by_dim) ** 2))
        Dream_shared_vars.delta_m[m_loc] = (
            Dream_shared_vars.delta_m[m_loc] + change
        )

        delta_ms = np.array(Dream_shared_vars.delta_m[0:self.nCR])
        ncr_updates = np.array(Dream_shared_vars.ncr_updates[0:self.nCR])

        if np.all(delta_ms != 0):
            for m in range(self.nCR):
                cross_probs[m] = (
                    Dream_shared_vars.delta_m[m]
                    / ncr_updates[m]
                ) * self.nchains
            cross_probs = cross_probs / np.sum(cross_probs)

        Dream_shared_vars.cross_probs[0:self.nCR] = cross_probs
        self.CR_probabilities = cross_probs
        return cross_probs

    def _patched_estimate_gamma_level_probs(
        self,
        ndim,
        q0,
        q_new,
        gamma_level,
    ):
        current_positions = np.frombuffer(
            Dream_shared_vars.current_positions.get_obj()
        )
        current_positions = current_positions.reshape((self.nchains, ndim))

        sd_by_dim = np.std(current_positions, axis=0)
        gamma_level_probs = Dream_shared_vars.gamma_level_probs[0:self.ngamma]

        gamma_value = int(_coerce_runtime_scalar(gamma_level))
        matches = np.flatnonzero(
            np.asarray(self.gamma_level_values, dtype=int) == gamma_value
        )
        if matches.size != 1:
            raise ValueError(
                "Unable to match DREAM gamma level "
                f"{{gamma_value}} in {{self.gamma_level_values!r}}."
            )
        gamma_loc = int(matches[0])

        Dream_shared_vars.ngamma_updates[gamma_loc] += 1
        Dream_shared_vars.delta_m_gamma[gamma_loc] = (
            Dream_shared_vars.delta_m_gamma[gamma_loc]
            + np.nan_to_num(np.sum(((q_new - q0) / sd_by_dim) ** 2))
        )

        delta_ms_gamma = np.array(Dream_shared_vars.delta_m_gamma[0:self.ngamma])
        if np.all(delta_ms_gamma != 0):
            for m in range(self.ngamma):
                gamma_level_probs[m] = (
                    Dream_shared_vars.delta_m_gamma[m]
                    / Dream_shared_vars.ngamma_updates[m]
                ) * self.nchains
            gamma_level_probs = gamma_level_probs / np.sum(gamma_level_probs)

        Dream_shared_vars.gamma_level_probs[0:self.ngamma] = gamma_level_probs
        return gamma_level_probs

    _PyDreamDream.set_snooker = _patched_set_snooker
    _PyDreamDream.set_CR = _patched_set_CR
    _PyDreamDream.set_gamma_level = _patched_set_gamma_level
    _PyDreamDream.set_gamma = _patched_set_gamma
    _PyDreamDream.estimate_crossover_probabilities = (
        _patched_estimate_crossover_probabilities
    )
    _PyDreamDream.estimate_gamma_level_probs = (
        _patched_estimate_gamma_level_probs
    )
    _PyDreamDream._saxshell_numpy_compatibility = True


_apply_pydream_numpy_compatibility()


def _build_runtime_mp_context():
    # pyDREAM always uses multiprocessing for chain execution. On macOS the
    # default "spawn" context can miss the live Dream monkeypatches that make
    # crossover adaptation NumPy-safe, so prefer a fork-based context when the
    # platform supports it.
    if os.name == "nt":
        return None
    for method_name in ("fork", "forkserver"):
        try:
            return mp.get_context(method_name)
        except ValueError:
            continue
    return None


{template_source}


def expand_active_params(active_params):
    full = FIXED_PARAMETER_VALUES.astype(float).copy()
    full[np.asarray(ACTIVE_PARAMETER_INDICES, dtype=int)] = np.asarray(
        active_params,
        dtype=float,
    )
    return full


def full_params_to_kwargs(full_params):
    return {{
        name: float(full_params[index])
        for index, name in enumerate(FULL_PARAMETER_NAMES)
    }}


class _TruncatedScipyDistribution:
    def __init__(
        self,
        distribution_name,
        dist_params,
        lower_bound,
        upper_bound,
    ):
        self.base = getattr(stats, str(distribution_name))(
            **dict(dist_params or {{}})
        )
        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        if not (
            np.isfinite(self.lower_bound)
            and np.isfinite(self.upper_bound)
            and self.lower_bound < self.upper_bound
        ):
            raise ValueError(
                "Invalid bounded DREAM prior interval for %s."
                % distribution_name
            )
        self.cdf_low = float(self.base.cdf(self.lower_bound))
        self.cdf_high = float(self.base.cdf(self.upper_bound))
        if not (
            np.isfinite(self.cdf_low)
            and np.isfinite(self.cdf_high)
            and self.cdf_low < self.cdf_high
        ):
            raise ValueError(
                "Invalid bounded DREAM prior mass for %s."
                % distribution_name
            )
        self.normalization = self.cdf_high - self.cdf_low

    def rvs(self, *args, size=None, random_state=None, **kwargs):
        rng = random_state if random_state is not None else np.random
        uniform_values = rng.uniform(
            self.cdf_low,
            self.cdf_high,
            size=size,
        )
        return self.base.ppf(uniform_values)

    def ppf(self, q):
        q_values = np.asarray(q, dtype=float)
        bounded_q = self.cdf_low + q_values * self.normalization
        return self.base.ppf(bounded_q)

    def interval(self, alpha=1):
        alpha_value = min(max(float(alpha), 0.0), 1.0)
        lower_q = (1.0 - alpha_value) / 2.0
        upper_q = 1.0 - lower_q
        return self.ppf(lower_q), self.ppf(upper_q)

    def logpdf(self, value):
        values = np.asarray(value, dtype=float)
        log_prior = self.base.logpdf(values) - np.log(self.normalization)
        outside = (values < self.lower_bound) | (values > self.upper_bound)
        return np.where(outside, -np.inf, log_prior)


def _entry_practical_prior_bounds(entry):
    def _apply_parameter_domain_bounds(lower_bound, upper_bound):
        param_name = str(entry.get("param", "")).strip()
        if param_name.startswith("w") and param_name[1:].isdigit():
            if lower_bound is None or not np.isfinite(float(lower_bound)):
                lower_bound = 0.0
            else:
                lower_bound = max(float(lower_bound), 0.0)
        return lower_bound, upper_bound

    distribution = getattr(stats, entry["distribution"])
    params = dict(entry.get("dist_params", {{}}))
    try:
        support_low, support_high = distribution.support(**params)
        support_low = float(support_low)
        support_high = float(support_high)
    except Exception:
        support_low = float("nan")
        support_high = float("nan")

    if np.isfinite(support_low) and np.isfinite(support_high):
        if support_low <= support_high:
            return _apply_parameter_domain_bounds(
                support_low,
                support_high,
            )

    try:
        guide_low = float(distribution.ppf(GUIDE_INTERVAL_LOWER_Q, **params))
        guide_high = float(distribution.ppf(GUIDE_INTERVAL_UPPER_Q, **params))
    except Exception:
        return None, None
    if np.isfinite(support_low):
        guide_low = (
            max(guide_low, support_low)
            if np.isfinite(guide_low)
            else support_low
        )
    if np.isfinite(support_high):
        guide_high = (
            min(guide_high, support_high)
            if np.isfinite(guide_high)
            else support_high
        )
    param_name = str(entry.get("param", "")).strip()
    if (
        str(entry.get("distribution", "")).strip() == "lognorm"
        and param_name.startswith("w")
        and param_name[1:].isdigit()
        and np.isfinite(support_low)
        and support_low <= 0.0
    ):
        guide_low = 0.0
    if not (np.isfinite(guide_low) and np.isfinite(guide_high)):
        return None, None
    if guide_low > guide_high:
        return None, None
    return _apply_parameter_domain_bounds(guide_low, guide_high)


def _value_inside_practical_prior_bounds(value, entry):
    lower_bound, upper_bound = _entry_practical_prior_bounds(entry)
    value = float(value)
    tolerance = PRACTICAL_BOUND_TOLERANCE * max(abs(value), 1.0)
    if lower_bound is not None and value < float(lower_bound) - tolerance:
        return False
    if upper_bound is not None and value > float(upper_bound) + tolerance:
        return False
    return True


def active_params_inside_prior_support(active_params):
    values = np.asarray(active_params, dtype=float).reshape(-1)
    if values.size != len(ACTIVE_PARAMETER_ENTRIES):
        return False
    for value, entry in zip(values, ACTIVE_PARAMETER_ENTRIES):
        param_name = str(entry.get("param", "")).strip()
        if param_name.startswith("w") and param_name[1:].isdigit():
            if float(value) < 0.0:
                return False
        distribution = getattr(stats, entry["distribution"])
        try:
            log_prior = distribution.logpdf(
                float(value),
                **entry["dist_params"],
            )
        except Exception:
            return False
        if not np.isfinite(log_prior):
            return False
        if not _value_inside_practical_prior_bounds(value, entry):
            return False
    return True


def model_from_active_params(active_params):
    full_params = expand_active_params(active_params)
    params = full_params_to_kwargs(full_params)
    extra_inputs = [
        np.asarray(TEMPLATE_RUNTIME_INPUTS[name], dtype=float)
        for name in LMFIT_EXTRA_INPUTS
    ]
    return {lmfit_model_name}(
        q_values,
        solvent_intensities,
        theoretical_intensities,
        *extra_inputs,
        **params,
    )


def active_log_likelihood(active_params):
    if not active_params_inside_prior_support(active_params):
        return -np.inf
    full_params = expand_active_params(active_params)
    return {dream_model_name}(full_params)


def build_sampled_parameters():
    if SampledParam is None:
        raise RuntimeError(
            "pydream is not installed. Install pydream to execute the "
            "generated DREAM runtime script."
    )
    sampled_parameters = []
    for entry in ACTIVE_PARAMETER_ENTRIES:
        lower_bound, upper_bound = _entry_practical_prior_bounds(entry)
        if lower_bound is None or upper_bound is None:
            raise ValueError(
                "Unable to derive a bounded DREAM prior interval for %s."
                % entry.get("param", "<unknown>")
            )
        sampled_parameters.append(
            SampledParam(
                _TruncatedScipyDistribution,
                entry["distribution"],
                dict(entry.get("dist_params", {{}})),
                lower_bound,
                upper_bound,
            )
        )
    return sampled_parameters


def run_sampler():
    if run_dream is None:
        raise RuntimeError(
            "pydream is not installed. Install pydream to run this bundle."
        )
    sampled_parameters = build_sampled_parameters()
    mp_context = _build_runtime_mp_context()
    sampled_params, log_ps = run_dream(
        parameters=sampled_parameters,
        likelihood=active_log_likelihood,
        nchains=int(SETTINGS["nchains"]),
        niterations=int(SETTINGS["niterations"]),
        burnin=int(  # codespell:ignore burnin
            int(SETTINGS["niterations"])
            * float(SETTINGS["burnin_percent"])
            / 100
        ),
        mp_context=mp_context,
        restart=bool(SETTINGS["restart"]),
        verbose=bool(SETTINGS["verbose"]),
        parallel=bool(SETTINGS["parallel"]),
        nseedchains=int(SETTINGS["nseedchains"]),
        adapt_crossover=bool(SETTINGS["adapt_crossover"]),
        crossover_burnin=int(SETTINGS["crossover_burnin"]),
        lamb=float(SETTINGS["lamb"]),
        zeta=float(SETTINGS["zeta"]),
        history_thin=int(SETTINGS["history_thin"]),
        snooker=float(SETTINGS["snooker"]),
        p_gamma_unity=float(SETTINGS["p_gamma_unity"]),
        model_name=SETTINGS.get("model_name") or False,
        history_file=SETTINGS.get("history_file") or False,
    )
    sampled_params = np.asarray(sampled_params, dtype=float)
    log_ps = np.asarray(log_ps, dtype=float)
    if log_ps.ndim == 3 and log_ps.shape[-1] == 1:
        log_ps = log_ps[..., 0]
    np.save(RUN_DIR / "dream_sampled_params.npy", sampled_params)
    np.save(RUN_DIR / "dream_log_ps.npy", log_ps)
    return {{
        "sampled_params_path": str(RUN_DIR / "dream_sampled_params.npy"),
        "log_ps_path": str(RUN_DIR / "dream_log_ps.npy"),
    }}


if __name__ == "__main__":
    run_sampler()
"""


__all__ = [
    "DreamRunBundle",
    "SAXSDreamWorkflow",
]
