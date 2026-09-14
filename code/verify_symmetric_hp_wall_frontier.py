"""Independent entry point for the symmetric hp frontier artifact checks."""
from __future__ import annotations

import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(_CODE))

from s6_symmetric_hp_wall_frontier import verify


if __name__ == "__main__":
    verify()
