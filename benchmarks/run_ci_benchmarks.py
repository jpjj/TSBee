#!/usr/bin/env python3
"""
CI-friendly benchmark runner with pass/fail criteria.

Exit codes:
  0 - All benchmarks passed
  1 - Benchmarks failed (gap exceeded threshold)
  2 - Missing instance files
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from run_benchmarks import main as run_benchmarks

if __name__ == "__main__":
    # This will exit with appropriate code
    run_benchmarks()
