"""6.2/6.5/6.6/6.7 분석 모듈 스모크 테스트."""

from pathlib import Path

from artificial_neuron.analysis.graph import analyze_graph
from artificial_neuron.analysis.in_degree import analyze_in_degree
from artificial_neuron.analysis.selfloop import analyze_selfloop
from artificial_neuron.analysis.weight import analyze_weight

from .test_analysis_activation import _run_sim_and_save


# ----------------------------------------------------------------------
# 6.2
# ----------------------------------------------------------------------

def test_analyze_graph_creates_outputs(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_graph(out_dir)

    g = out_dir / "graph"
    assert (g / "adjacency_matrix_t0000.png").exists()
    assert (g / "adjacency_matrix_t0100.png").exists()
    assert (g / "adjacency_matrix_t0200.png").exists()
    assert (g / "graph_t0000.png").exists()
    assert (g / "graph_metrics.csv").exists()


def test_graph_metrics_schema(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_graph(out_dir)

    lines = (
        (out_dir / "graph" / "graph_metrics.csv")
        .read_text().strip().splitlines()
    )
    assert lines[0] == (
        "tick,avg_degree,clustering_coefficient,modularity,edge_count"
    )
    assert len(lines) == 1 + 3


# ----------------------------------------------------------------------
# 6.5
# ----------------------------------------------------------------------

def test_analyze_in_degree_creates_outputs(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_in_degree(out_dir)

    d = out_dir / "in_degree"
    assert (d / "in_degree_hist_t0000.png").exists()
    assert (d / "in_degree_timeseries.csv").exists()
    assert (d / "in_degree_summary.csv").exists()
    assert (d / "in_degree_warnings.log").exists()


def test_in_degree_summary_schema(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_in_degree(out_dir)

    lines = (
        (out_dir / "in_degree" / "in_degree_summary.csv")
        .read_text().strip().splitlines()
    )
    assert lines[0] == "tick,max,p99,mean"
    assert len(lines) == 1 + 3


# ----------------------------------------------------------------------
# 6.6
# ----------------------------------------------------------------------

def test_analyze_weight_creates_outputs(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_weight(out_dir)

    w = out_dir / "weight"
    assert (w / "weight_hist_t0000.png").exists()
    assert (w / "weight_boundary_ratio.csv").exists()
    assert (w / "weight_stats.csv").exists()
    assert (w / "top_cofire_data.csv").exists()
    assert (w / "top_cofire_trajectories.png").exists()


def test_weight_boundary_ratio_schema(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_weight(out_dir)

    lines = (
        (out_dir / "weight" / "weight_boundary_ratio.csv")
        .read_text().strip().splitlines()
    )
    assert lines[0] == "tick,ratio_W_le_0.01,ratio_W_ge_1.99"


# ----------------------------------------------------------------------
# 6.7
# ----------------------------------------------------------------------

def test_analyze_selfloop_creates_outputs(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 500)
    analyze_selfloop(out_dir)

    s = out_dir / "selfloop"
    assert (s / "selfloop_weights.csv").exists()
    assert (s / "selfloop_warnings.log").exists()

    lines = (
        (s / "selfloop_weights.csv").read_text().strip().splitlines()
    )
    assert lines[0] == "tick,neuron_id,W_selfloop"