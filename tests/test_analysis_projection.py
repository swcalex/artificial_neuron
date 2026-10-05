"""6.3 projection 분석 모듈 스모크 테스트."""

from pathlib import Path

from artificial_neuron.analysis.projection import analyze_projection

from .test_analysis_activation import _run_sim_and_save


def test_analyze_projection_creates_outputs(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_projection(out_dir)

    p = out_dir / "projection"
    assert p.is_dir()
    # t=100, 200 스냅샷 이미지 (t=0은 스킵)
    assert (p / "pca_projection_t0100.png").exists()
    assert (p / "pca_projection_t0200.png").exists()
    assert (p / "tsne_projection_t0100.png").exists()
    assert (p / "tsne_projection_t0200.png").exists()
    assert (p / "separation_metrics.csv").exists()


def test_separation_metrics_schema(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_projection(out_dir)

    lines = (
        (out_dir / "projection" / "separation_metrics.csv")
        .read_text().strip().splitlines()
    )
    assert lines[0] == "tick,class_separation_score,cluster_count"
    # 스냅샷 t=0, 100, 200 세 줄
    assert len(lines) == 1 + 3


def test_projection_skips_t0_snapshot(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 300)
    analyze_projection(out_dir)

    p = out_dir / "projection"
    assert not (p / "pca_projection_t0000.png").exists()


def test_analyze_projection_short_run(tmp_path: Path):
    """t-SNE가 스킵되는 짧은 실행에서도 예외 없이 동작."""
    out_dir = _run_sim_and_save(tmp_path, 150)
    analyze_projection(out_dir)

    p = out_dir / "projection"
    # t=100 스냅샷만 존재
    assert (p / "pca_projection_t0100.png").exists()
    assert (p / "separation_metrics.csv").exists()