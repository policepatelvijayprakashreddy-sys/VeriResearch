import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_crossref_verification import run_benchmark


def test_dual_registry_source_quality_benchmark():
    run_benchmark()
