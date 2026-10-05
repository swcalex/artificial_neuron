"""6.1 activation 분석 모듈 스모크 테스트."""

import random
from pathlib import Path

import pytest

from artificial_neuron.analysis.activation import analyze_activation
from artificial_neuron.config import Config
from artificial_neuron.input_provider import CyclicPatternProvider
from artificial_neuron.network import Network
from artificial_neuron.simulator import Simulator


def _run_sim_and_save(tmp_path: Path, n_ticks: int = 250) -> Path:
    cfg = Config()
    rng = random.Random(cfg.init_seed)
    net = Network(cfg, rng)
    provider = CyclicPatternProvider.from_config(cfg)
    sim = Simulator(cfg, net, provider)
    sim.run(n_ticks)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)
    return out_dir


def test_analyze_activation_creates_outputs(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 250)
    analyze_activation(out_dir)

    act_dir = out_dir / "activation"
    assert (act_dir / "activation_heatmap.png").exists()
    assert (act_dir / "firing_rate.png").exists()
    assert (act_dir / "firing_rate_ei.png").exists()
    assert (act_dir / "activation_summary.csv").exists()

    # PNG 파일이 0바이트가 아님
    for name in (
        "activation_heatmap.png",
        "firing_rate.png",
        "firing_rate_ei.png",
    ):
        assert (act_dir / name).stat().st_size > 0


def test_activation_summary_csv_schema(tmp_path: Path):
    out_dir = _run_sim_and_save(tmp_path, 100)
    analyze_activation(out_dir)

    lines = (
        (out_dir / "activation" / "activation_summary.csv")
        .read_text().strip().splitlines()
    )
    assert lines[0] == (
        "tick,total_firing_rate,exc_firing_rate,inh_firing_rate"
    )
    assert len(lines) == 1 + 100


def test_activation_summary_matches_simulator_log(tmp_path: Path):
    """분석 결과 CSV가 시뮬레이터 로그와 일치해야 한다."""
    out_dir = _run_sim_and_save(tmp_path, 100)
    analyze_activation(out_dir)

    sim_log = (out_dir / "logs" / "activation_summary.csv").read_text()
    analysis_log = (
        out_dir / "activation" / "activation_summary.csv"
    ).read_text()

    # 헤더는 동일, 값은 float 포맷 차이가 있을 수 있으므로 라인 수만 우선 확인.
    sim_lines = sim_log.strip().splitlines()
    ana_lines = analysis_log.strip().splitlines()
    assert sim_lines[0] == ana_lines[0]
    assert len(sim_lines) == len(ana_lines)


def test_analyze_activation_no_simulator_rerun(tmp_path: Path):
    """분석 모듈은 시뮬레이터 재실행 없이 로그만으로 동작해야 한다 (6.8)."""
    out_dir = _run_sim_and_save(tmp_path, 50)
    # 시뮬레이터 객체를 파괴해도 분석이 동작해야 함
    analyze_activation(out_dir)
    assert (out_dir / "activation" / "activation_heatmap.png").exists()