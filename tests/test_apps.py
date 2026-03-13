from _pytest.capture import CaptureFixture

from saxshell.apps.cluster_stats import main as cluster_stats_main
from saxshell.apps.extract_clusters import main as extract_main


def test_extract_app_runs(capsys: CaptureFixture[str]) -> None:
    extract_main()
    captured = capsys.readouterr()
    assert "extract-clusters" in captured.out


def test_cluster_stats_app_runs(capsys: CaptureFixture[str]) -> None:
    cluster_stats_main()
    captured = capsys.readouterr()
    assert "cluster-stats" in captured.out
