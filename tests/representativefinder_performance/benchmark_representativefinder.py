from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

from saxshell.bondanalysis import AngleTripletDefinition, BondPairDefinition
from saxshell.representativefinder import (
    RepresentativeFinderSettings,
    analyze_representative_structure_folder,
)

REFERENCE_PB2I4_DIR = Path(
    "/Users/keithwhite/repos/cluster_extraction/"
    "041_cp2k_pbi2_dmf_0p7M_RT/"
    "clusters_xyz2pdb_splitxyz_f1002_t497p5fs0001/Pb2I4"
)
OUTPUT_ROOT = Path(__file__).resolve().parent / "output_results"


def _settings(worker_count: int) -> RepresentativeFinderSettings:
    return RepresentativeFinderSettings(
        bond_pairs=(BondPairDefinition("Pb", "I", 4.2),),
        angle_triplets=(AngleTripletDefinition("Pb", "I", "I", 4.2, 4.2),),
        bond_weight=1.0,
        angle_weight=0.5,
        solvent_weight=0.5,
        parallel_workers=worker_count,
    )


def _prepare_sample(
    reference_dir: Path,
    workspace_dir: Path,
    *,
    sample_count: int,
) -> tuple[Path, list[str]]:
    sample_dir = workspace_dir / reference_dir.name
    sample_dir.mkdir(parents=True)
    source_files = sorted(reference_dir.glob("*.pdb"))[:sample_count]
    for source_path in source_files:
        target_path = sample_dir / source_path.name
        try:
            target_path.symlink_to(source_path)
        except OSError:
            shutil.copy2(source_path, target_path)
    return sample_dir, [path.name for path in source_files]


def _run_once(
    sample_dir: Path,
    output_dir: Path,
    *,
    worker_count: int,
) -> dict[str, object]:
    started = time.perf_counter()
    result = analyze_representative_structure_folder(
        sample_dir,
        settings=_settings(worker_count),
        output_dir=output_dir,
    )
    elapsed = time.perf_counter() - started
    return {
        "worker_count": int(worker_count),
        "elapsed_seconds": round(float(elapsed), 6),
        "candidate_count": len(result.candidates),
        "selected_candidate": result.selected_candidate.file_name,
        "selected_score_total": result.selected_candidate.score_total,
        "skipped_files": len(result.skipped_files),
        "summary_json_path": str(result.summary_json_path),
        "score_table_path": str(result.score_table_path),
        "summary_text_path": str(result.summary_text_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Benchmark representative finder serial and parallel runs."
    )
    parser.add_argument(
        "--reference-dir",
        default=str(REFERENCE_PB2I4_DIR),
        help="Folder containing Pb2I4 .pdb representative candidates.",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=256,
        help="Number of reference files to include in the benchmark sample.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        nargs="+",
        default=[1, 0],
        help="Worker counts to benchmark. Use 0 for auto-sized parallelism.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_ROOT),
        help="Directory where benchmark outputs and reports are written.",
    )
    args = parser.parse_args()

    reference_dir = Path(args.reference_dir).expanduser().resolve()
    if not reference_dir.is_dir():
        raise SystemExit(f"Reference folder is missing: {reference_dir}")

    output_root = Path(args.output_root).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"pb2i4_benchmark_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(tempfile.mkdtemp(prefix="pb2i4_benchmark_"))
    try:
        sample_dir, sample_files = _prepare_sample(
            reference_dir,
            workspace_dir,
            sample_count=max(int(args.sample_size), 1),
        )
        runs = []
        for worker_count in args.workers:
            label = (
                "auto" if int(worker_count) == 0 else str(int(worker_count))
            )
            runs.append(
                _run_once(
                    sample_dir,
                    run_dir / f"workers_{label}",
                    worker_count=int(worker_count),
                )
            )
        report = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "reference_dir": str(reference_dir),
            "sample_size": len(sample_files),
            "sample_files": sample_files,
            "runs": runs,
        }
        report_path = run_dir / "benchmark_results.json"
        report_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        latest_path = output_root / "latest_benchmark_results.json"
        latest_path.write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
    finally:
        shutil.rmtree(workspace_dir, ignore_errors=True)

    print(report_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
