"""6.3 동적 결정 경계 투영 (PCA / t-SNE).

입력: outputs/logs/voltage_history.csv, output_history.csv,
      edge_snapshots/*.csv (스냅샷 시점 추출용)
출력: outputs/projection/
        pca_projection_t{tick}.png
        tsne_projection_t{tick}.png
        separation_metrics.csv

설계 (7d):
    - PCA / t-SNE는 전체 시계열에 한 번 fit하여 축을 고정.
    - 각 스냅샷 시점 이미지는 그때까지의 점만 그린다. 축이 같으므로
      시간에 따른 상태 분포의 이동을 정성적으로 비교할 수 있다.
    - class_separation_score = silhouette score (출력 라벨 기준).
      클래스가 하나뿐이면 0.0.
    - cluster_count = PCA 2D 투영에 KMeans(k=2..min(5, n-1))를 적용해
      silhouette 최대인 k. 데이터가 너무 적으면 0.
    - t-SNE perplexity = min(30, max(5, n // 4)). n<6이면 t-SNE 스킵.
    - t=0 스냅샷은 점 1개이므로 스킵.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from ._io import (
    list_edge_snapshots,
    load_output_history,
    load_voltage_history,
)

RANDOM_STATE = 0
MAX_K = 5


def analyze_projection(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    snaps_dir = logs_dir / "edge_snapshots"
    out_dir = output_dir / "projection"
    out_dir.mkdir(parents=True, exist_ok=True)

    V = load_voltage_history(logs_dir)          # (T, N)
    outputs = load_output_history(logs_dir)     # (T, N_output)
    y = outputs[:, 0]                            # (T,) - N_output=1
    T = V.shape[0]

    snap_ticks = [t for t, _ in list_edge_snapshots(snaps_dir)]

    # ---- 전체에 PCA fit ----
    n_components = min(2, V.shape[1])
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    pca_2d = pca.fit_transform(V)
    # n_components < 2인 극단 케이스 대비: y축을 0으로
    if pca_2d.shape[1] < 2:
        pca_2d = np.hstack([pca_2d, np.zeros((T, 1))])

    # ---- 전체에 t-SNE fit ----
    tsne_2d = None
    if T >= 6:
        perplexity = float(min(30, max(5, T // 4)))
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=RANDOM_STATE,
            init="pca",
            learning_rate="auto",
        )
        tsne_2d = tsne.fit_transform(V)

    # ---- 스냅샷 시점별 이미지 ----
    for t in snap_ticks:
        if t < 1:
            continue  # 점 1개 이하는 투영 의미 없음

        # PCA
        _scatter(
            pca_2d[: t + 1],
            y[: t + 1],
            title=f"PCA projection (up to t={t})",
            path=out_dir / f"pca_projection_t{t:04d}.png",
        )

        # t-SNE
        if tsne_2d is not None:
            _scatter(
                tsne_2d[: t + 1],
                y[: t + 1],
                title=f"t-SNE projection (up to t={t})",
                path=out_dir / f"tsne_projection_t{t:04d}.png",
            )

    # ---- separation_metrics.csv ----
    rows: List[Tuple[int, float, int]] = []
    for t in snap_ticks:
        if t < 2:
            rows.append((t, 0.0, 0))
            continue
        pts = pca_2d[: t + 1]
        y_t = y[: t + 1]

        # class_separation_score (silhouette)
        if len(np.unique(y_t)) >= 2:
            sep = float(silhouette_score(pts, y_t))
        else:
            sep = 0.0

        # cluster_count (KMeans 최적 k)
        k_best = _best_k(pts)
        rows.append((t, sep, k_best))

    with open(out_dir / "separation_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tick", "class_separation_score", "cluster_count"])
        for row in rows:
            w.writerow(row)


def _scatter(points: np.ndarray, labels: np.ndarray, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    uniq = np.unique(labels)
    if len(uniq) == 1:
        color = "#ff6b6b" if uniq[0] == 1 else "#4c6ef5"
        ax.scatter(points[:, 0], points[:, 1], s=6, alpha=0.7, c=color)
    else:
        sc = ax.scatter(
            points[:, 0], points[:, 1], s=6, alpha=0.7,
            c=labels, cmap="coolwarm", vmin=0, vmax=1,
        )
        fig.colorbar(sc, ax=ax, label="output A")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _best_k(points: np.ndarray) -> int:
    n = points.shape[0]
    if n < 4:
        return 0
    k_max = min(MAX_K, n - 1)
    if k_max < 2:
        return 0
    best_k = 0
    best_score = -2.0
    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(points)
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(points, labels)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k