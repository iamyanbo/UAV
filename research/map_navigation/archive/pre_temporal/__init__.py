"""Photo-goal navigation with an immutable overhead prior.

The historical RGB modules are imported explicitly via this package bootstrap;
no collection/evaluator module is imported by the inference process.
"""
from pathlib import Path
import sys

LEGACY = Path(__file__).resolve().parents[1] / 'rgb_flight'
if str(LEGACY) not in sys.path:
    sys.path.insert(0, str(LEGACY))

