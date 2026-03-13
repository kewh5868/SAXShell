from ..scripts.stats.analyzer import ClusterStatsAnalyzer, ClusterStatsConfig


def main() -> None:
    config = ClusterStatsConfig()
    analyzer = ClusterStatsAnalyzer(config=config)
    result = analyzer.run()
    print(result.summary())
