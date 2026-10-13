import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from test_consensus_verification import run_consensus_benchmark


def test_cross_paper_consensus_benchmark():
    run_consensus_benchmark()
