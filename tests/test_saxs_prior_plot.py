from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_hex

from saxshell.saxs.project_manager import (
    build_prior_histogram_export_payload,
    export_prior_histogram_npy,
    export_prior_histogram_table,
    list_secondary_filter_elements,
    plot_md_prior_histogram,
)


def test_prior_histogram_uses_pb_i_sorting_and_axis_formatting(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "md_prior_weights.json"
    json_path.write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 10,
                "structures": {
                    "Pb2I4": {
                        "no_motif": {
                            "count": 4,
                            "weight": 0.4,
                            "profile_file": "Pb2I4_no_motif.txt",
                        }
                    },
                    "I2": {
                        "no_motif": {
                            "count": 1,
                            "weight": 0.1,
                            "profile_file": "I2_no_motif.txt",
                        }
                    },
                    "Pb": {
                        "no_motif": {
                            "count": 2,
                            "weight": 0.2,
                            "profile_file": "Pb_no_motif.txt",
                        }
                    },
                    "PbI2O": {
                        "no_motif": {
                            "count": 3,
                            "weight": 0.3,
                            "profile_file": "PbI2O_no_motif.txt",
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axis = plot_md_prior_histogram(json_path, mode="structure_fraction")
    labels = [tick.get_text() for tick in axis.get_xticklabels()]

    assert labels == [
        "I$_{2}$",
        "Pb",
        "PbI$_{2}$O",
        "Pb$_{2}$I$_{4}$",
    ]
    plt.close(fig)


def test_prior_histogram_exports_csv_and_npy_for_selected_value_modes(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "md_prior_weights.json"
    json_path.write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 5,
                "structures": {
                    "PbI2": {
                        "motif_A": {
                            "count": 2,
                            "weight": 0.4,
                            "profile_file": "PbI2_motif_A.txt",
                        }
                    },
                    "Pb2I4O": {
                        "motif_B": {
                            "count": 3,
                            "weight": 0.6,
                            "profile_file": "Pb2I4O_motif_B.txt",
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = export_prior_histogram_table(
        json_path,
        tmp_path / "prior_histogram_structure_fraction_fraction.csv",
        mode="structure_fraction",
        value_mode="fraction",
    )
    npy_path = export_prior_histogram_npy(
        json_path,
        tmp_path / "prior_histogram_structure_fraction_count.npy",
        mode="structure_fraction",
        value_mode="count",
    )

    csv_text = csv_path.read_text(encoding="utf-8")
    npy_payload = np.load(npy_path, allow_pickle=True).item()

    assert "structure,axis_label,motif_motif_A,motif_motif_B,total" in csv_text
    assert "PbI$_{2}$" in csv_text
    assert npy_payload["value_mode"] == "count"
    assert list(npy_payload["labels"]) == ["PbI2", "Pb2I4O"]


def test_solvent_sort_payload_groups_secondary_atom_count_subbins(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "md_prior_weights.json"
    json_path.write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 5,
                "available_elements": ["Pb", "I", "O"],
                "structures": {
                    "PbI2": {
                        "motif_A": {
                            "count": 2,
                            "weight": 0.4,
                            "profile_file": "PbI2_motif_A.txt",
                            "secondary_atom_distributions": {
                                "O": {"0": 1, "1": 1}
                            },
                        }
                    },
                    "Pb2I4": {
                        "motif_B": {
                            "count": 3,
                            "weight": 0.6,
                            "profile_file": "Pb2I4_motif_B.txt",
                            "secondary_atom_distributions": {
                                "O": {"0": 1, "2": 2}
                            },
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    payload = build_prior_histogram_export_payload(
        json_path,
        mode="solvent_sort_structure_fraction",
        value_mode="count",
        secondary_element="O",
    )

    assert list_secondary_filter_elements(json_path) == ["O"]
    assert payload["segments"] == ["0", "1", "2"]
    assert payload["segment_labels"] == ["0 O", "1 O", "2 O"]
    assert list(payload["labels"]) == ["PbI2", "Pb2I4"]
    assert np.asarray(payload["matrix"]).tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 2.0],
    ]


def test_prior_histogram_can_use_structure_motif_trace_colors(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "md_prior_weights.json"
    json_path.write_text(
        json.dumps(
            {
                "origin": "clusters",
                "total_files": 5,
                "structures": {
                    "PbI2": {
                        "motif_A": {
                            "count": 2,
                            "weight": 0.4,
                            "profile_file": "PbI2_motif_A.txt",
                        }
                    },
                    "Pb2I4": {
                        "motif_B": {
                            "count": 3,
                            "weight": 0.6,
                            "profile_file": "Pb2I4_motif_B.txt",
                        }
                    },
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    fig, axis = plot_md_prior_histogram(
        json_path,
        mode="structure_fraction",
        structure_motif_colors={
            "PbI2_motif_A": "#112233",
            "Pb2I4_motif_B": "#445566",
        },
    )

    patch_colors = [to_hex(patch.get_facecolor()) for patch in axis.patches]

    assert patch_colors[0] == "#112233"
    assert patch_colors[-1] == "#445566"
    plt.close(fig)
