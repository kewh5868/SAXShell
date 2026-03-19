from saxshell.saxs.cli import main


def test_saxs_templates_command_lists_bundled_templates(capsys):
    assert main(["templates"]) == 0
    output = capsys.readouterr().out
    assert "template_pd_likelihood_monosq_decoupled" in output
