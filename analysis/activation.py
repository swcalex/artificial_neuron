"""6.1 뉴런 활성화 분포 시각화.

입력: outputs/logs/activation_history.csv, neuron_meta.csv
출력: outputs/activation/{activation_heatmap.png, firing_rate.png,
                         firing_rate_ei.png, activation_summary.csv}
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ._io import load_activation_history, load_neuron_meta


def analyze_activation(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    out_dir = output_dir / "activation"
    out_dir.mkdir(parents=True, exist_ok=True)

    A = load_activation_history(logs_dir)  # (T, N) int8
    neuron_meta = load_neuron_meta(logs_dir)

    T, N = A.shape
    S = np.asarray([m["S"] for m in neuron_meta], dtype=np.int8)
    exc_ids = np.where(S == +1)[0]
    inh_ids = np.where(S == -1)[0]

    # ---- 1) activation_heatmap.png ----
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(
        A.T,
        aspect="auto",
        cmap="binary",
        interpolation="nearest",
        origin="lower",
        extent=[0, T, 0, N],
    )
    ax.set_xlabel("tick")
    ax.set_ylabel("neuron id")
    ax.set_title("Activation heatmap (A_i(t))")
    fig.tight_layout()
    fig.savefig(out_dir / "activation_heatmap.png", dpi=120)
    plt.close(fig)

    # ---- 2) firing_rate.png ----
    total = A.mean(axis=1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(T), total, lw=1.0)
    ax.set_xlabel("tick")
    ax.set_ylabel("firing rate")
    ax.set_ylim(0, 1)
    ax.set_title("Firing rate (all neurons)")
    fig.tight_layout()
    fig.savefig(out_dir / "firing_rate.png", dpi=120)
    plt.close(fig)

    # ---- 3) firing_rate_ei.png ----
    exc_rate = A[:, exc_ids].mean(axis=1) if len(exc_ids) else np.zeros(T)
    inh_rate = A[:, inh_ids].mean(axis=1) if len(inh_ids) else np.zeros(T)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(T), exc_rate, lw=1.0, label="excitatory")
    ax.plot(np.arange(T), inh_rate, lw=1.0, label="inhibitory")
    ax.set_xlabel("tick")
    ax.set_ylabel("firing rate")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Firing rate by E/I")
    fig.tight_layout()
    fig.savefig(out_dir / "firing_rate_ei.png", dpi=120)
    plt.close(fig)

    # ---- 4) activation_summary.csv ----
    with open(out_dir / "activation_summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tick", "total_firing_rate",
            "exc_firing_rate", "inh_firing_rate",
        ])
        for t in range(T):
            w.writerow([
                t, float(total[t]), float(exc_rate[t]), float(inh_rate[t]),
            ])