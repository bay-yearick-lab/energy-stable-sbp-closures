from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CODEDIR = ROOT / 'code'
PAPERDIR = ROOT / 'paper'
FIGDIR = ROOT / 'figures'
RESULTSDIR = CODEDIR / 'results'


def ensure_generated_dirs() -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    RESULTSDIR.mkdir(parents=True, exist_ok=True)


def recommended_process_workers(job_count: int, *, cap: int | None = None, reserve_cores: int = 1) -> int:
    """Return a conservative process-pool size for local paper rebuilds."""
    override = os.getenv("BSPLINE_MAX_WORKERS")
    if override is not None:
        requested = max(1, int(override))
        return min(job_count, requested)

    usable = max(1, (os.cpu_count() or 4) - reserve_cores)
    if cap is not None:
        usable = min(usable, cap)
    return max(1, min(job_count, usable))


def save_figure(fig, stem: str, *, bbox_inches: str = 'tight', pad_inches: float = 0.03) -> None:
    """Save both raster and vector variants for manuscript figures."""
    import matplotlib

    # Type 3 (non-scalable) fonts crash pdfTeX's microtype font expansion.
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42
    fig.savefig(FIGDIR / f'{stem}.png', bbox_inches=bbox_inches, pad_inches=pad_inches)
    fig.savefig(FIGDIR / f'{stem}.pdf', bbox_inches=bbox_inches, pad_inches=pad_inches)