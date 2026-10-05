"""main.py CLI 진입점 테스트."""

from pathlib import Path

import pytest

from artificial_neuron.main import main


def test_main_runs_minimal(tmp_path: Path):
    """--ticks 50 정도의 짧은 실행이 성공하고 로그를 생성한다."""
    out_dir = tmp_path / "run"
    rc = main([
        "--ticks", "50",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert rc == 0
    assert (out_dir / "run_meta.json").exists()
    assert (out_dir / "logs" / "activation_history.csv").exists()


def test_main_analyze_flag_creates_all_analysis_outputs(tmp_path: Path):
    out_dir = tmp_path / "run_analyze"
    rc = main([
        "--ticks", "150",
        "--output-dir", str(out_dir),
        "--analyze",
        "--quiet",
    ])
    assert rc == 0

    # 7종 분석의 대표 산출물
    assert (out_dir / "activation" / "activation_heatmap.png").exists()
    assert (out_dir / "graph" / "graph_metrics.csv").exists()
    assert (out_dir / "projection" / "separation_metrics.csv").exists()
    assert (out_dir / "timing" / "timing_summary.json").exists()
    assert (out_dir / "in_degree" / "in_degree_summary.csv").exists()
    assert (out_dir / "weight" / "weight_stats.csv").exists()
    assert (out_dir / "selfloop" / "selfloop_weights.csv").exists()


def test_main_no_plasticity(tmp_path: Path):
    out_dir = tmp_path / "run_no_plast"
    rc = main([
        "--ticks", "50",
        "--output-dir", str(out_dir),
        "--no-plasticity",
        "--quiet",
    ])
    assert rc == 0

    # edge_events.csv는 헤더만 있어야 함
    events = (
        (out_dir / "logs" / "edge_events.csv")
        .read_text().strip().splitlines()
    )
    assert len(events) == 1


def test_main_seed_override_reproducible(tmp_path: Path):
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    for out_dir in (out_a, out_b):
        rc = main([
            "--ticks", "100",
            "--output-dir", str(out_dir),
            "--init-seed", "777",
            "--gen-seed", "888",
            "--quiet",
        ])
        assert rc == 0

    act_a = (out_a / "logs" / "activation_history.csv").read_text()
    act_b = (out_b / "logs" / "activation_history.csv").read_text()
    assert act_a == act_b


def test_main_rejects_negative_ticks(tmp_path: Path):
    out_dir = tmp_path / "bad"
    rc = main([
        "--ticks", "-1",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert rc == 2


def test_main_rejects_non_positive_snapshot_interval(tmp_path: Path):
    out_dir = tmp_path / "bad"
    rc = main([
        "--ticks", "10",
        "--snapshot-interval", "0",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert rc == 2


def test_main_snapshot_interval_override(tmp_path: Path):
    out_dir = tmp_path / "custom_snap"
    rc = main([
        "--ticks", "60",
        "--snapshot-interval", "20",
        "--output-dir", str(out_dir),
        "--quiet",
    ])
    assert rc == 0
    snaps = sorted((out_dir / "logs" / "edge_snapshots").glob("edges_t*.csv"))
    names = [s.name for s in snaps]
    # t=0, 20, 40 세 스냅샷
    assert names == ["edges_t0000.csv", "edges_t0020.csv", "edges_t0040.csv"]