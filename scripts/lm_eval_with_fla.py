#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FLA = ROOT / "flash-linear-attention"
if str(FLA) not in sys.path:
    sys.path.insert(0, str(FLA))

import fla  # noqa: F401,E402
from lm_eval.__main__ import cli_evaluate  # noqa: E402


if __name__ == "__main__":
    cli_evaluate()
