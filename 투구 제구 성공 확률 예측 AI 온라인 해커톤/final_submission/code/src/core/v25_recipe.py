"""Frozen v25 post-break anchor recipe constants.

Moved verbatim from ``src/train_v25_postbreak_anchor.py`` during the core
extraction so shared consumers do not import a training entry point.
"""

from __future__ import annotations


SOURCE_YEAR = 2024
MODEL_NAME = "logistic_c01"
ETA = 0.075
APPLY_DOMAIN = "R_ANCHOR"
