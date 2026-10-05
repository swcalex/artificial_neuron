"""6.6 가중치 분포 및 Co-fire 로깅.

입력: outputs/logs/edge_snapshots/*.csv, weight_stats.csv
출력: outputs/weight/
        weight_hist_t{tick}.png
        weight_boundary_ratio.csv
        top_cofire_trajectories.png
        top_cofire_data.csv
        weight_stats.csv
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._io import (
    list_edge_snapshots,
    load_edge_snapshot,
    load_weight_stats,
)

TOP_K = 10  # 상위 co-fire 엣지 수


def analyze_weight(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    snaps_dir = logs_dir / "edge_snapshots"
    out_dir = output_dir / "weight"
    out_dir.mkdir(parents=True, exist_ok=True)

    snaps = list_edge_snapshots(snaps_dir)

    # edge_id -> [(tick, src, tgt, W, cofire), ...]
    per_edge_track: dict = {}

    for tick, path in snaps:
        edges = load_edge_snapshot(path)
        weights = np.asarray([e[3] for e in edges], dtype=np.float64)

        if len(weights) > 0:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.hist(
                weights, bins=30, range=(0.0, 2.0), edgecolor="black"
            )
            ax.set_xlabel("weight")
            ax.set_ylabel("edge count")
            ax.set_title(f"Weight histogram (t={tick})")
            fig.tight_layout()
            fig.savefig(out_dir / f"weight_hist_t{tick:04d}.png", dpi=120)
            plt.close(fig)

        for eid, src, tgt, w, c in edges:
            per_edge_track.setdefault(eid, []).append(
                (tick, src, tgt, w, c)
            )

    # weight_boundary_ratio.csv (weight_stats.csv에서 복사)
    stats = load_weight_stats(logs_dir)
    with open(out_dir / "weight_boundary_ratio.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "ratio_W_le_0.01", "ratio_W_ge_1.99"])
        for s in stats:
            w.writerow([
                s["tick"], s["ratio_W_le_0.01"], s["ratio_W_ge_1.99"],
            ])

    # 최종 스냅샷 기준 상위 K개 edge_id
    if snaps:
        last_tick, last_path = snaps[-1]
        last_edges = load_edge_snapshot(last_path)
        ranked = sorted(last_edges, key=lambda e: e[4], reverse=True)
        top_edges = ranked[:TOP_K]
    else:
        top_edges = []

    with open(out_dir / "top_cofire_data.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tick", "edge_id", "source", "target", "W", "cofire_count",
        ])
        for eid, _src, _tgt, _w, _c in top_edges:
            for row in per_edge_track.get(eid, []):
                tick, src, tgt, ww, cc = row
                w.writerow([tick, eid, src, tgt, ww, cc])

    if top_edges:
        fig, ax = plt.subplots(figsize=(9, 4))
        for eid, src, tgt, _w, _c in top_edges:
            track = per_edge_track.get(eid, [])
            if not track:
                continue
            ticks = [r[0] for r in track]
            ws = [r[3] for r in track]
            ax.plot(
                ticks, ws, marker="o", ms=3, lw=1.0,
                label=f"e{eid} ({src}->{tgt})",
            )
        ax.set_xlabel("tick")
        ax.set_ylabel("weight")
        ax.set_ylim(0.0, 2.0)
        ax.set_title(f"Top-{len(top_edges)} co-fire edge trajectories")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / "top_cofire_trajectories.png", dpi=120)
        plt.close(fig)

    # weight_stats.csv 복사
    src_stats = logs_dir / "weight_stats.csv"
    if src_stats.exists():
        shutil.copyfile(src_stats, out_dir / "weight_stats.csv")