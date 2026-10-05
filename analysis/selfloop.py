"""6.7 Self-loop 관찰 지표.

입력: outputs/logs/edge_snapshots/*.csv, activation_history.csv
출력: outputs/selfloop/
        selfloop_weights.csv
        selfloop_firing.png
        selfloop_warnings.log
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Set

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._io import (
    list_edge_snapshots,
    load_activation_history,
    load_edge_snapshot,
    load_neuron_meta,
)


CONSECUTIVE_FIRING_THRESHOLD = 100


def analyze_selfloop(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    snaps_dir = logs_dir / "edge_snapshots"
    out_dir = output_dir / "selfloop"
    out_dir.mkdir(parents=True, exist_ok=True)

    load_neuron_meta(logs_dir)  # 스키마 검증용
    A = load_activation_history(logs_dir)  # (T, N)
    T = A.shape[0]

    snaps = list_edge_snapshots(snaps_dir)

    # selfloop_weights.csv (중복 self-loop은 합산)
    rows: List[tuple] = []
    selfloop_neurons: Set[int] = set()
    for tick, path in snaps:
        edges = load_edge_snapshot(path)
        per_neuron: Dict[int, float] = {}
        for _eid, src, tgt, w, _c in edges:
            if src == tgt:
                per_neuron[src] = per_neuron.get(src, 0.0) + w
                selfloop_neurons.add(src)
        for nid, wsum in sorted(per_neuron.items()):
            rows.append((tick, nid, wsum))

    with open(out_dir / "selfloop_weights.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "neuron_id", "W_selfloop"])
        for row in rows:
            w.writerow(row)

    # selfloop_firing.png
    if selfloop_neurons:
        fig, ax = plt.subplots(figsize=(10, 4))
        for nid in sorted(selfloop_neurons):
            ax.plot(np.arange(T), A[:, nid], lw=0.8, label=f"n{nid}")
        ax.set_xlabel("tick")
        ax.set_ylabel("A_i(t)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Self-loop neuron firing patterns")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / "selfloop_firing.png", dpi=120)
        plt.close(fig)

    # selfloop_warnings.log
    warnings: List[str] = []
    for nid in sorted(selfloop_neurons):
        series = A[:, nid]
        run = 0
        for t in range(T):
            if series[t] == 1:
                run += 1
                if run >= CONSECUTIVE_FIRING_THRESHOLD:
                    warnings.append(
                        f"neuron={nid} consecutive_firing>="
                        f"{CONSECUTIVE_FIRING_THRESHOLD} at tick={t}"
                    )
                    break
            else:
                run = 0

    with open(out_dir / "selfloop_warnings.log", "w", encoding="utf-8") as f:
        for line in warnings:
            f.write(line + "\n")