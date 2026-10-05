"""6.4 틱 성능 프로파일링.

입력: outputs/logs/tick_timing.csv
출력: outputs/timing/{tick_timing.csv, tick_duration.png,
                      timing_summary.json}
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._io import load_tick_timing


def analyze_timing(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    out_dir = output_dir / "timing"
    out_dir.mkdir(parents=True, exist_ok=True)

    records: List[Tuple[int, float, float, float, float, float]] = (
        load_tick_timing(logs_dir)
    )

    # ---- 1) tick_timing.csv 복사 ----
    with open(out_dir / "tick_timing.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tick", "step1_ms", "step2_ms", "step3_ms", "step4_ms",
            "total_ms",
        ])
        for rec in records:
            w.writerow(rec)

    ticks = np.asarray([r[0] for r in records], dtype=np.int64)
    totals = np.asarray([r[5] for r in records], dtype=np.float64)

    # ---- 2) tick_duration.png ----
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ticks, totals, lw=1.0)
    ax.set_xlabel("tick")
    ax.set_ylabel("total duration (ms)")
    ax.set_title("Tick duration")
    fig.tight_layout()
    fig.savefig(out_dir / "tick_duration.png", dpi=120)
    plt.close(fig)

    # ---- 3) timing_summary.json ----
    if len(totals) == 0:
        summary = {
            "total_ticks": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "ticks_per_second": 0.0,
        }
    else:
        sorted_totals = np.sort(totals)
        n = len(sorted_totals)
        mean_ms = float(sorted_totals.mean())
        summary = {
            "total_ticks": int(n),
            "mean_ms": mean_ms,
            "p50_ms": float(sorted_totals[n // 2]),
            "p99_ms": float(sorted_totals[min(n - 1, int(n * 0.99))]),
            "max_ms": float(sorted_totals[-1]),
            "ticks_per_second": (
                1000.0 / mean_ms if mean_ms > 0 else 0.0
            ),
        }

    with open(out_dir / "timing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)