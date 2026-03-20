from pathlib import Path
from types import SimpleNamespace

from saxshell.saxs.cli import main


def test_saxs_templates_command_lists_bundled_templates(capsys):
    assert main(["templates"]) == 0
    output = capsys.readouterr().out
    assert "template_pd_likelihood_monosq_decoupled" in output


def test_saxs_templates_validate_command_reports_success(capsys, monkeypatch):
    fake_result = SimpleNamespace(
        passed=True,
        template_name="candidate_template",
    )
    monkeypatch.setattr(
        "saxshell.saxs.cli.validate_template_candidate",
        lambda *args, **kwargs: fake_result,
    )
    monkeypatch.setattr(
        "saxshell.saxs.cli.format_validation_report",
        lambda result: f"validated {result.template_name}",
    )

    assert main(["templates", "validate", str(Path("candidate.py"))]) == 0
    output = capsys.readouterr().out
    assert "validated candidate_template" in output


def test_saxs_templates_install_command_reports_destination(
    capsys, monkeypatch
):
    fake_installed = SimpleNamespace(
        installed_template_path=Path("/tmp/template.py"),
        installed_metadata_path=Path("/tmp/template.json"),
        validation_result=SimpleNamespace(
            passed=True,
            template_name="candidate_template",
        ),
    )
    monkeypatch.setattr(
        "saxshell.saxs.cli.install_template_candidate",
        lambda *args, **kwargs: fake_installed,
    )
    monkeypatch.setattr(
        "saxshell.saxs.cli.format_validation_report",
        lambda result: f"validated {result.template_name}",
    )

    assert main(["templates", "install", str(Path("candidate.py"))]) == 0
    output = capsys.readouterr().out
    assert "validated candidate_template" in output
    assert "Installed template: /tmp/template.py" in output
