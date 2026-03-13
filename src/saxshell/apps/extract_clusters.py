from ..scripts.extract.config import ExtractionConfig
from ..scripts.extract.extractor import ClusterExtractor


def main() -> None:
    config = ExtractionConfig()
    extractor = ClusterExtractor(config=config)
    result = extractor.run()
    print(result.summary())
