"""6.5 In-Degree 분포 로깅.

입력: outputs/logs/edge_snapshots/*.csv
출력: outputs/in_degree/
        in_degree_hist_t{tick}.png
        in_degree_timeseries.csv
        in_degree_summary.csv
        in_degree_warnings.log
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._io import list_edge_snapshots, load_edge_snapshot, load_neuron_meta


def analyze_in_degree(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    snaps_dir = logs_dir / "edge_snapshots"
    out_dir = output_dir / "in_degree"
    out_dir.mkdir(parents=True, exist_ok=True)

    N = len(load_neuron_meta(logs_dir))
    warn_threshold = N / 2.0

    snaps = list_edge_snapshots(snaps_dir)
    timeseries_rows: List[tuple] = []
    summary_rows: List[tuple] = []
    warnings: List[str] = []

    for tick, path in snaps:
        edges = load_edge_snapshot(path)
        in_deg = np.zeros(N, dtype=np.int64)
        for _eid, _src, tgt, _w, _c in edges:
            in_deg[tgt] += 1

        # 히스토그램
        fig, ax = plt.subplots(figsize=(7, 4))
        max_d = int(in_deg.max()) if N else 0
        bins = np.arange(0, max_d + 2) - 0.5
        ax.hist(in_deg, bins=bins, edgecolor="black")
        ax.set_xlabel("in-degree")
        ax.set_ylabel("neuron count")
        ax.set_title(f"In-degree histogram (t={tick})")
        fig.tight_layout()
        fig.savefig(out_dir / f"in_degree_hist_t{tick:04d}.png", dpi=120)
        plt.close(fig)

        # 시계열용 rows
        for nid in range(N):
            timeseries_rows.append((tick, nid, int(in_deg[nid])))

        # 요약
        summary_rows.append((
            tick,
            int(in_deg.max()) if N else 0,
            float(np.percentile(in_deg, 99)) if N else 0.0,
            float(in_deg.mean()) if N else 0.0,
        ))

        # 경고
        for nid in range(N):
            if in_deg[nid] > warn_threshold:
                warnings.append(
                    f"tick={tick} neuron={nid} in_degree={in_deg[nid]} "
                    f"> threshold={warn_threshold}"
                )

    with open(out_dir / "in_degree_timeseries.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "neuron_id", "in_degree"])
        for row in timeseries_rows:
            w.writerow(row)

    with open(out_dir / "in_degree_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "max", "p99", "mean"])
        for row in summary_rows:
            w.writerow(row)

    with open(out_dir / "in_degree_warnings.log", "w", encoding="utf-8") as f:
        for line in warnings:
            f.write(line + "\n")