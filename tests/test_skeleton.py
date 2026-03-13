from saxshell.scripts.extract.config import ExtractionConfig
from saxshell.scripts.extract.extractor import ClusterExtractor
from saxshell.scripts.stats.analyzer import (
    ClusterStatsAnalyzer,
    ClusterStatsConfig,
)


def test_cluster_extractor_returns_result() -> None:
    config = ExtractionConfig()
    extractor = ClusterExtractor(config=config)
    result = extractor.run()

    assert result.workflow_name == "extract-clusters"
    assert "trajectory=" in result.message


def test_cluster_stats_returns_result() -> None:
    config = ClusterStatsConfig()
    analyzer = ClusterStatsAnalyzer(config=config)
    result = analyzer.run()

    assert result.workflow_name == "cluster-stats"
    assert "histogram_bins=" in result.message
