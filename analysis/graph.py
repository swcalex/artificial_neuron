"""6.2 시냅스 가중치 인접 행렬 및 그래프.

입력: outputs/logs/edge_snapshots/*.csv, neuron_meta.csv
출력: outputs/graph/
        adjacency_matrix_t{tick}.png
        graph_t{tick}.png
        graph_metrics.csv

참고: modularity/clustering은 directed 그래프를 undirected로 변환 후 계산.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ._io import (
    list_edge_snapshots,
    load_edge_snapshot,
    load_neuron_meta,
)


def analyze_graph(output_dir: Path) -> None:
    output_dir = Path(output_dir)
    logs_dir = output_dir / "logs"
    snaps_dir = logs_dir / "edge_snapshots"
    out_dir = output_dir / "graph"
    out_dir.mkdir(parents=True, exist_ok=True)

    neuron_meta = load_neuron_meta(logs_dir)
    N = len(neuron_meta)
    S = np.asarray([m["S"] for m in neuron_meta], dtype=np.int8)

    snaps = list_edge_snapshots(snaps_dir)
    metrics_rows: List[dict] = []

    for tick, path in snaps:
        edges = load_edge_snapshot(path)

        # ---- 1) adjacency matrix ----
        adj = np.zeros((N, N), dtype=np.float64)
        for _eid, src, tgt, w, _c in edges:
            adj[src, tgt] += w

        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(adj, cmap="viridis", aspect="equal")
        ax.set_xlabel("target")
        ax.set_ylabel("source")
        ax.set_title(f"Adjacency matrix (t={tick}, E={len(edges)})")
        fig.colorbar(im, ax=ax, label="weight sum")
        fig.tight_layout()
        fig.savefig(out_dir / f"adjacency_matrix_t{tick:04d}.png", dpi=120)
        plt.close(fig)

        # ---- 2) directed graph 시각화 ----
        G = nx.DiGraph()
        for nid in range(N):
            G.add_node(nid, S=int(S[nid]))
        for _eid, src, tgt, w, _c in edges:
            if G.has_edge(src, tgt):
                G[src][tgt]["weight"] += w
            else:
                G.add_edge(src, tgt, weight=w)

        pos = nx.spring_layout(G, seed=0)
        node_colors = [
            "#ff6b6b" if G.nodes[n]["S"] == +1 else "#4c6ef5"
            for n in G.nodes()
        ]
        edge_widths = [max(0.3, G[u][v]["weight"]) for u, v in G.edges()]
        fig, ax = plt.subplots(figsize=(7, 6))
        nx.draw_networkx_nodes(
            G, pos, ax=ax, node_color=node_colors, node_size=300,
        )
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=7)
        nx.draw_networkx_edges(
            G, pos, ax=ax, width=edge_widths, alpha=0.5,
            arrows=True, arrowsize=8,
        )
        ax.set_title(f"Graph (t={tick}, E={len(edges)})")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"graph_t{tick:04d}.png", dpi=120)
        plt.close(fig)

        # ---- 3) metrics ----
        avg_degree = len(edges) / N if N else 0.0
        G_und = G.to_undirected()
        clustering = (
            nx.average_clustering(G_und)
            if G_und.number_of_nodes() > 0 else 0.0
        )
        if G_und.number_of_edges() > 0:
            communities = nx.community.greedy_modularity_communities(G_und)
            modularity = nx.community.modularity(G_und, communities)
        else:
            modularity = 0.0

        metrics_rows.append({
            "tick": tick,
            "avg_degree": avg_degree,
            "clustering_coefficient": clustering,
            "modularity": modularity,
            "edge_count": len(edges),
        })

    # ---- 4) graph_metrics.csv ----
    with open(out_dir / "graph_metrics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "tick", "avg_degree", "clustering_coefficient",
            "modularity", "edge_count",
        ])
        for row in metrics_rows:
            w.writerow([
                row["tick"], row["avg_degree"],
                row["clustering_coefficient"], row["modularity"],
                row["edge_count"],
            ])