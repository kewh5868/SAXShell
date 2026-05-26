from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"


def _run_import_probe(code: str) -> list[str]:
    env = os.environ.copy()
    pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (
        str(SRC_ROOT)
        if not pythonpath
        else str(SRC_ROOT) + os.pathsep + pythonpath
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip().splitlines()


def test_saxs_package_import_does_not_load_workflow_dependencies():
    output = _run_import_probe(
        "import sys; import saxshell.saxs; "
        "print('lmfit', 'lmfit' in sys.modules); "
        "print('pyarrow', 'pyarrow' in sys.modules)"
    )

    assert output == ["lmfit False", "pyarrow False"]


def test_saxs_cli_import_does_not_load_workflow_dependencies():
    output = _run_import_probe(
        "import sys; import saxshell.saxs.cli; "
        "print('lmfit', 'lmfit' in sys.modules); "
        "print('pyarrow', 'pyarrow' in sys.modules)"
    )

    assert output == ["lmfit False", "pyarrow False"]
