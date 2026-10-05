"""6.4 timing 분석 모듈 스모크 테스트."""

import json
from pathlib import Path

from artificial_neuron.analysis.timing import analyze_timing

from .test_analysis_activation import _run_sim_and_save


def test_analyze_timing_creates_outputs(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 200)
    analyze_timing(out_dir)

    timing_dir = out_dir / "timing"
    assert (timing_dir / "tick_timing.csv").exists()
    assert (timing_dir / "tick_duration.png").exists()
    assert (timing_dir / "timing_summary.json").exists()


def test_timing_summary_schema(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 50)
    analyze_timing(out_dir)

    summary = json.loads(
        (out_dir / "timing" / "timing_summary.json")
        .read_text(encoding="utf-8")
    )
    for key in (
        "total_ticks", "mean_ms", "p50_ms", "p99_ms", "max_ms",
        "ticks_per_second",
    ):
        assert key in summary

    assert summary["total_ticks"] == 50
    assert summary["mean_ms"] >= 0.0
    assert summary["p50_ms"] <= summary["max_ms"]


def test_tick_timing_csv_line_count(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 30)
    analyze_timing(out_dir)

    lines = (
        (out_dir / "timing" / "tick_timing.csv")
        .read_text().strip().splitlines()
    )
    assert len(lines) == 1 + 30
    assert lines[0].startswith("tick,")