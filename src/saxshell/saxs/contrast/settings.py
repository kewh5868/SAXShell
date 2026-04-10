from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

COMPONENT_BUILD_MODE_NO_CONTRAST = "no_contrast"
COMPONENT_BUILD_MODE_CONTRAST = "contrast"
COMPONENT_BUILD_MODE_BORN_APPROXIMATION = "born_approximation"
_COMPONENT_BUILD_MODE_LABELS = {
    COMPONENT_BUILD_MODE_NO_CONTRAST: "No Contrast (Debye)",
    COMPONENT_BUILD_MODE_CONTRAST: "Contrast (Debye)",
    COMPONENT_BUILD_MODE_BORN_APPROXIMATION: "Born Approximation (Average)",
}


def normalize_component_build_mode(value: object) -> str:
    normalized = str(value or "").strip().lower().replace("-", "_")
    if normalized in {
        COMPONENT_BUILD_MODE_CONTRAST,
        "contrast_mode",
    }:
        return COMPONENT_BUILD_MODE_CONTRAST
    if normalized in {
        COMPONENT_BUILD_MODE_BORN_APPROXIMATION,
        "born_approx",
        "born_approximation_mode",
        "born_approximation_average",
        "average",
    }:
        return COMPONENT_BUILD_MODE_BORN_APPROXIMATION
    return COMPONENT_BUILD_MODE_NO_CONTRAST


def component_build_mode_label(value: object) -> str:
    return _COMPONENT_BUILD_MODE_LABELS[normalize_component_build_mode(value)]


def component_build_mode_choices() -> tuple[tuple[str, str], ...]:
    return tuple(_COMPONENT_BUILD_MODE_LABELS.items())


def _optional_resolved_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return Path(text).expanduser().resolve()


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: object) -> str | None:
    text = str(value or "").strip()
    return text or None


@dataclass(slots=True, frozen=True)
class ContrastRepresentativeSamplerSettings:
    enabled: bool = True
    full_scan_threshold: int = 24
    target_distribution_samples: int = 64
    minimum_candidate_samples: int = 8
    max_candidate_samples: int = 32
    candidate_batch_size: int = 4
    random_seed: int = 1337
    convergence_patience: int = 2
    improvement_tolerance: float = 0.0025
    stratify_sampling: bool = True

    @classmethod
    def from_values(
        cls,
        *,
        enabled: object = True,
        full_scan_threshold: object = 24,
        target_distribution_samples: object = 64,
        minimum_candidate_samples: object = 8,
        max_candidate_samples: object = 32,
        candidate_batch_size: object = 4,
        random_seed: object = 1337,
        convergence_patience: object = 2,
        improvement_tolerance: object = 0.0025,
        stratify_sampling: object = True,
        minimum_samples: object | None = None,
        max_samples: object | None = None,
        batch_size: object | None = None,
        shortlist_multiplier: object | None = None,
    ) -> "ContrastRepresentativeSamplerSettings":
        normalized_full_scan_threshold = max(
            _optional_int(full_scan_threshold) or 0,
            0,
        )
        normalized_target_distribution_samples = max(
            _optional_int(target_distribution_samples) or 64,
            1,
        )
        legacy_max_samples = _optional_int(max_samples)
        legacy_minimum_samples = _optional_int(minimum_samples)
        legacy_batch_size = _optional_int(batch_size)
        normalized_max_candidate_samples = max(
            _optional_int(max_candidate_samples) or legacy_max_samples or 32,
            1,
        )
        normalized_minimum_candidate_samples = min(
            max(
                _optional_int(minimum_candidate_samples)
                or legacy_minimum_samples
                or 8,
                1,
            ),
            normalized_max_candidate_samples,
        )
        normalized_candidate_batch_size = max(
            min(
                _optional_int(candidate_batch_size) or legacy_batch_size or 4,
                normalized_max_candidate_samples,
            ),
            1,
        )
        normalized_random_seed = _optional_int(random_seed)
        if normalized_random_seed is None:
            normalized_random_seed = 1337
        normalized_convergence_patience = max(
            _optional_int(convergence_patience) or 2,
            1,
        )
        normalized_improvement_tolerance = max(
            _optional_float(improvement_tolerance) or 0.0,
            0.0,
        )
        return cls(
            enabled=bool(enabled),
            full_scan_threshold=normalized_full_scan_threshold,
            target_distribution_samples=normalized_target_distribution_samples,
            minimum_candidate_samples=normalized_minimum_candidate_samples,
            max_candidate_samples=normalized_max_candidate_samples,
            candidate_batch_size=normalized_candidate_batch_size,
            random_seed=normalized_random_seed,
            convergence_patience=normalized_convergence_patience,
            improvement_tolerance=normalized_improvement_tolerance,
            stratify_sampling=bool(stratify_sampling),
        )

    def candidate_sample_limit(self, total_candidates: int) -> int:
        return min(max(int(total_candidates), 0), self.max_candidate_samples)

    def distribution_sample_limit(self, total_candidates: int) -> int:
        return min(
            max(int(total_candidates), 0),
            max(int(self.target_distribution_samples), 1),
        )

    def should_use_sampling(self, total_candidates: int) -> bool:
        total = max(int(total_candidates), 0)
        if not self.enabled or total <= 0:
            return False
        if total <= self.full_scan_threshold:
            return False
        return self.candidate_sample_limit(total) < total

    def to_dict(self) -> dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "full_scan_threshold": int(self.full_scan_threshold),
            "target_distribution_samples": int(
                self.target_distribution_samples
            ),
            "minimum_candidate_samples": int(self.minimum_candidate_samples),
            "max_candidate_samples": int(self.max_candidate_samples),
            "candidate_batch_size": int(self.candidate_batch_size),
            "random_seed": int(self.random_seed),
            "convergence_patience": int(self.convergence_patience),
            "improvement_tolerance": float(self.improvement_tolerance),
            "stratify_sampling": bool(self.stratify_sampling),
        }


@dataclass(slots=True, frozen=True)
class ContrastModeLaunchContext:
    project_dir: Path | None = None
    clusters_dir: Path | None = None
    experimental_data_file: Path | None = None
    q_min: float | None = None
    q_max: float | None = None
    active_template_name: str | None = None
    active_distribution_id: str | None = None
    distribution_root_dir: Path | None = None
    contrast_artifact_dir: Path | None = None

    @classmethod
    def from_values(
        cls,
        *,
        project_dir: str | Path | None = None,
        clusters_dir: str | Path | None = None,
        experimental_data_file: str | Path | None = None,
        q_min: object = None,
        q_max: object = None,
        active_template_name: object = None,
        active_distribution_id: object = None,
        distribution_root_dir: str | Path | None = None,
        contrast_artifact_dir: str | Path | None = None,
    ) -> "ContrastModeLaunchContext":
        return cls(
            project_dir=_optional_resolved_path(project_dir),
            clusters_dir=_optional_resolved_path(clusters_dir),
            experimental_data_file=_optional_resolved_path(
                experimental_data_file
            ),
            q_min=_optional_float(q_min),
            q_max=_optional_float(q_max),
            active_template_name=_optional_text(active_template_name),
            active_distribution_id=_optional_text(active_distribution_id),
            distribution_root_dir=_optional_resolved_path(
                distribution_root_dir
            ),
            contrast_artifact_dir=_optional_resolved_path(
                contrast_artifact_dir
            ),
        )

    @property
    def q_range_text(self) -> str:
        if self.q_min is None or self.q_max is None:
            return "Inherited from the main UI when available"
        return f"{self.q_min:.6g} to {self.q_max:.6g}"

    def to_dict(self) -> dict[str, object]:
        return {
            "project_dir": (
                None if self.project_dir is None else str(self.project_dir)
            ),
            "clusters_dir": (
                None if self.clusters_dir is None else str(self.clusters_dir)
            ),
            "experimental_data_file": (
                None
                if self.experimental_data_file is None
                else str(self.experimental_data_file)
            ),
            "q_min": self.q_min,
            "q_max": self.q_max,
            "active_template_name": self.active_template_name,
            "active_distribution_id": self.active_distribution_id,
            "distribution_root_dir": (
                None
                if self.distribution_root_dir is None
                else str(self.distribution_root_dir)
            ),
            "contrast_artifact_dir": (
                None
                if self.contrast_artifact_dir is None
                else str(self.contrast_artifact_dir)
            ),
        }
