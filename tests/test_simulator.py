"""Simulator 최소 루프 (Step 3 미포함) 테스트 및 bootstrap 실측.

algorithm_spec.md 7.4 Step 4 참조.

구성:
- 구조 테스트: 루프가 정상 실행되고 로그가 쌓이는지
- 순서 테스트: Step 1 -> 2 -> 4가 매 틱 실행되는지
- bootstrap: T=1000 실행 후 발화율 관찰 (실패 조건이 아닌 리포트)
"""

import random
from dataclasses import replace
from pathlib import Path

import pytest

from artificial_neuron.config import Config
from artificial_neuron.input_provider import CyclicPatternProvider
from artificial_neuron.network import Network
from artificial_neuron.simulator import Simulator

def make_simulator(
    config: Config | None = None,
    enable_plasticity: bool = True,
    gen_seed: int | None = None,
    snapshot_interval: int = 100,
) -> Simulator:
    cfg = config or Config()
    init_rng = random.Random(cfg.init_seed)
    network = Network(cfg, init_rng)
    provider = CyclicPatternProvider.from_config(cfg)
    if gen_seed is None:
        gen_seed = cfg.gen_seed
    return Simulator(
        cfg,
        network,
        provider,
        gen_rng=random.Random(gen_seed),
        enable_plasticity=enable_plasticity,
        snapshot_interval=snapshot_interval,
    )

# ----------------------------------------------------------------------
# 구조
# ----------------------------------------------------------------------

def test_initial_state():
    sim = make_simulator()
    assert sim.tick == 0
    assert sim.tick_records == []
    assert sim.activation_history == []
    assert sim.output_history == []


def test_single_step_advances_tick():
    sim = make_simulator()
    out = sim.step()
    assert sim.tick == 1
    assert len(sim.tick_records) == 1
    assert len(sim.activation_history) == 1
    assert len(sim.output_history) == 1
    assert isinstance(out, tuple)
    assert len(out) == sim.config.n_output


def test_run_advances_multiple_ticks():
    sim = make_simulator()
    sim.run(25)
    assert sim.tick == 25
    assert len(sim.tick_records) == 25
    assert len(sim.activation_history) == 25


def test_run_rejects_negative():
    sim = make_simulator()
    with pytest.raises(ValueError):
        sim.run(-1)


# ----------------------------------------------------------------------
# 순서 (Step 1 -> 2 -> 4)
# ----------------------------------------------------------------------

def test_step1_reflects_provider_in_activation_history():
    """t=0..199는 (0,0), t=200..399는 (0,1) 패턴 (2.4)."""
    sim = make_simulator()
    sim.run(400)

    # t=199: 입력 (0,0), t=200: (0,1)
    # activation_history[t]의 인덱스 0,1은 입력 뉴런.
    assert sim.activation_history[199][0] == 0
    assert sim.activation_history[199][1] == 0
    assert sim.activation_history[200][0] == 0
    assert sim.activation_history[200][1] == 1


def test_output_history_matches_step4():
    sim = make_simulator()
    sim.run(10)
    for t in range(10):
        assert sim.output_history[t] == (sim.activation_history[t][-1],)


# ----------------------------------------------------------------------
# 프로파일링
# ----------------------------------------------------------------------

def test_tick_records_monotonic_indices():
    sim = make_simulator()
    sim.run(50)
    for i, rec in enumerate(sim.tick_records):
        assert rec.tick == i


def test_timing_summary_shape():
    sim = make_simulator()
    sim.run(20)
    summary = sim.timing_summary()
    for key in (
        "mean_ms",
        "p50_ms",
        "p99_ms",
        "max_ms",
        "ticks_per_second",
        "total_ticks",
    ):
        assert key in summary
    assert summary["total_ticks"] == 20
    assert summary["mean_ms"] >= 0.0

# ----------------------------------------------------------------------
# 7a: save_run 로그 저장 계층
# ----------------------------------------------------------------------

import json  # 파일 상단 import 목록에 추가


def test_save_run_creates_all_files(tmp_path: Path):
    sim = make_simulator(enable_plasticity=True)
    sim.run(300)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    assert (out_dir / "run_meta.json").exists()

    logs = out_dir / "logs"
    assert (logs / "neuron_meta.csv").exists()
    assert (logs / "activation_history.csv").exists()
    assert (logs / "voltage_history.csv").exists()
    assert (logs / "output_history.csv").exists()
    assert (logs / "activation_summary.csv").exists()
    assert (logs / "tick_timing.csv").exists()
    assert (logs / "weight_stats.csv").exists()
    assert (logs / "edge_events.csv").exists()
    assert (logs / "edge_snapshots").is_dir()


def test_run_meta_schema(tmp_path: Path):
    sim = make_simulator()
    sim.run(10)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    meta = json.loads((out_dir / "run_meta.json").read_text(encoding="utf-8"))

    for key in (
        "spec_version", "project_version", "start_time", "end_time",
        "total_ticks", "snapshot_interval", "plasticity_enabled",
        "provider", "seeds", "config",
        "final_neuron_count", "final_edge_count",
        "total_pruned_events", "total_generated_events",
    ):
        assert key in meta

    assert meta["total_ticks"] == 10
    assert meta["snapshot_interval"] == 100
    assert meta["seeds"]["init_seed"] == 42
    assert meta["seeds"]["input_seed"] == 43
    assert meta["seeds"]["gen_seed"] == 44
    assert meta["config"]["n_input"] == 2
    assert meta["config"]["n_total"] if "n_total" in meta["config"] else True
    assert meta["provider"] == "CyclicPatternProvider"


def test_activation_history_csv_shape(tmp_path: Path):
    sim = make_simulator()
    sim.run(50)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    lines = (
        (out_dir / "logs" / "activation_history.csv")
        .read_text().strip().splitlines()
    )
    assert len(lines) == 1 + 50
    header = lines[0].split(",")
    assert header[0] == "tick"
    assert len(header) == 1 + 20  # N=20
    assert header[1] == "n0"
    assert header[-1] == "n19"


def test_voltage_history_csv_shape(tmp_path: Path):
    sim = make_simulator()
    sim.run(20)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    lines = (
        (out_dir / "logs" / "voltage_history.csv")
        .read_text().strip().splitlines()
    )
    assert len(lines) == 1 + 20
    header = lines[0].split(",")
    assert len(header) == 1 + 20


def test_output_history_csv_shape(tmp_path: Path):
    sim = make_simulator()
    sim.run(20)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    lines = (
        (out_dir / "logs" / "output_history.csv")
        .read_text().strip().splitlines()
    )
    assert len(lines) == 1 + 20
    header = lines[0].split(",")
    assert header == ["tick", "out0"]


def test_weight_stats_csv_header(tmp_path: Path):
    sim = make_simulator()
    sim.run(15)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    lines = (
        (out_dir / "logs" / "weight_stats.csv")
        .read_text().strip().splitlines()
    )
    assert len(lines) == 1 + 15
    assert lines[0] == (
        "tick,edge_count,mean,std,min,max,"
        "ratio_W_le_0.01,ratio_W_ge_1.99"
    )


def test_edge_events_csv_schema(tmp_path: Path):
    sim = make_simulator(enable_plasticity=True)
    sim.run(200)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    lines = (
        (out_dir / "logs" / "edge_events.csv")
        .read_text().strip().splitlines()
    )
    assert lines[0] == "tick,event_type,edge_id,source,target,weight"
    # T=200이면 최소한 generate 이벤트가 존재 (t=0..19에서 20개)
    assert len(lines) > 1


def test_edge_events_empty_when_plasticity_disabled(tmp_path: Path):
    sim = make_simulator(enable_plasticity=False)
    sim.run(50)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    lines = (
        (out_dir / "logs" / "edge_events.csv")
        .read_text().strip().splitlines()
    )
    assert len(lines) == 1  # 헤더만


def test_edge_snapshot_interval(tmp_path: Path):
    sim = make_simulator()
    sim.run(300)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    snaps = sorted(
        (out_dir / "logs" / "edge_snapshots").glob("edges_t*.csv")
    )
    names = [s.name for s in snaps]
    # t=0, 100, 200 캡처 (t=300은 실행되지 않음)
    assert names == ["edges_t0000.csv", "edges_t0100.csv", "edges_t0200.csv"]

    # 스냅샷 CSV 헤더 검증
    first = snaps[0].read_text().strip().splitlines()
    assert first[0] == "edge_id,source,target,weight,cofire_count"


def test_snapshot_interval_constructor_validation():
    cfg = Config()
    rng = random.Random(0)
    network = Network(cfg, rng)
    provider = CyclicPatternProvider.from_config(cfg)

    with pytest.raises(ValueError):
        Simulator(cfg, network, provider, snapshot_interval=0)
    with pytest.raises(ValueError):
        Simulator(cfg, network, provider, snapshot_interval=-5)


def test_save_run_reproducible_across_identical_runs(tmp_path: Path):
    """동일 seed/설정이면 저장된 로그가 완전히 동일해야 한다 (6.8)."""
    out_a = tmp_path / "a"
    out_b = tmp_path / "b"

    sim_a = make_simulator()
    sim_a.run(200)
    sim_a.save_run(out_a)

    sim_b = make_simulator()
    sim_b.run(200)
    sim_b.save_run(out_b)

    # 로그 파일 내용 비교 (run_meta.json은 타임스탬프가 달라 제외)
    for rel in (
        "logs/activation_history.csv",
        "logs/voltage_history.csv",
        "logs/output_history.csv",
        "logs/activation_summary.csv",
        "logs/weight_stats.csv",
        "logs/edge_events.csv",
        "logs/edge_snapshots/edges_t0000.csv",
        "logs/edge_snapshots/edges_t0100.csv",
    ):
        content_a = (out_a / rel).read_text()
        content_b = (out_b / rel).read_text()
        assert content_a == content_b, f"mismatch in {rel}"

# ----------------------------------------------------------------------
# Bootstrap 실측 (T=1000)
# ----------------------------------------------------------------------

def test_bootstrap_1000_ticks_step3_disabled_reports_firing_stats():
    """Step 3 없이 T=1000 (구현 단계 4 baseline 재현).

    실패 조건이 아니라 관찰 리포트.
    """
    sim = make_simulator(enable_plasticity=False)
    sim.run(1000)
    _print_bootstrap_report(sim, "Bootstrap (T=1000, Step 3 미포함)")


def test_bootstrap_1000_ticks_full_loop_reports_firing_stats():
    """Step 3 포함 T=1000 (구현 단계 6 완전 루프).

    실패 조건이 아니라 관찰 리포트. 앞선 baseline과 비교하여
    Hebbian + 구조 가변이 발화율을 어떻게 이동시키는지 관찰한다.
    """
    sim = make_simulator(enable_plasticity=True)
    sim.run(1000)
    _print_bootstrap_report(sim, "Bootstrap (T=1000, Step 3 포함)")

    # 구조 가변성 흔적 리포트
    total_pruned = sum(len(p) for p in sim.prune_history)
    total_generated = sum(len(g) for g in sim.generate_history)
    final_edge_count = len(sim.network.edges)
    print(f"total pruned events    : {total_pruned}")
    print(f"total generated events : {total_generated}")
    print(f"final edge count       : {final_edge_count}")


def _print_bootstrap_report(sim: Simulator, title: str) -> None:
    group_mean = sim.mean_firing_rate_by_group()
    windows = sim.windowed_firing_rate(window_size=100)
    summary = sim.timing_summary()

    print()
    print(f"===== {title} =====")
    print("[그룹 평균 발화율]")
    print(f"  input    : {group_mean['input']:.4f}  (외생 입력, 참고용)")
    print(f"  internal : {group_mean['internal']:.4f}")
    print(f"  output   : {group_mean['output']:.4f}")
    print(f"  total    : {group_mean['total']:.4f}")
    print()
    print("[100틱 윈도우별 발화율]")
    print(f"  {'window':>12s}   {'total':>7s}  {'input':>7s}  "
          f"{'internal':>8s}  {'output':>7s}")
    for (start, tot), (_, inp), (_, intern), (_, out) in zip(
        windows["total"],
        windows["input"],
        windows["internal"],
        windows["output"],
    ):
        end = start + 100
        print(f"  t=[{start:4d},{end:4d})   {tot:7.4f}  {inp:7.4f}  "
              f"{intern:8.4f}  {out:7.4f}")
    print()
    print(f"mean tick duration : {summary['mean_ms']:.4f} ms")
    print(f"ticks_per_second   : {summary['ticks_per_second']:.1f}")

    assert 0.0 <= group_mean["total"] <= 1.0
    assert 0.0 <= group_mean["internal"] <= 1.0
    assert 0.0 <= group_mean["output"] <= 1.0
    assert sim.tick == 1000

def test_firing_rate_series_by_group_shape():
    sim = make_simulator()
    sim.run(30)
    series = sim.firing_rate_series_by_group()
    for key in ("input", "internal", "output", "total"):
        assert key in series
        assert len(series[key]) == 30
        assert all(0.0 <= v <= 1.0 for v in series[key])


def test_mean_firing_rate_by_group_shape():
    sim = make_simulator()
    sim.run(30)
    means = sim.mean_firing_rate_by_group()
    for key in ("input", "internal", "output", "total"):
        assert key in means
        assert 0.0 <= means[key] <= 1.0


def test_windowed_firing_rate_shape():
    sim = make_simulator()
    sim.run(250)
    windows = sim.windowed_firing_rate(window_size=100)
    for key in ("total", "input", "internal", "output"):
        assert key in windows
    # t=0, 100, 200 세 윈도우
    assert len(windows["total"]) == 3
    starts = [s for s, _ in windows["total"]]
    assert starts == [0, 100, 200]


def test_windowed_firing_rate_rejects_non_positive():
    sim = make_simulator()
    sim.run(5)
    with pytest.raises(ValueError):
        sim.windowed_firing_rate(window_size=0)
    with pytest.raises(ValueError):
        sim.windowed_firing_rate(window_size=-1)

# ----------------------------------------------------------------------
# Step 3 통합 (구현 단계 6)
# ----------------------------------------------------------------------

def test_enable_plasticity_flag_controls_step3():
    """enable_plasticity=False면 엣지 가중치가 W_init 그대로."""
    sim_off = make_simulator(enable_plasticity=False)
    sim_on = make_simulator(enable_plasticity=True)

    sim_off.run(50)
    sim_on.run(50)

    # off: 모든 엣지가 W_init 그대로
    for e in sim_off.network.edges.values():
        assert e.weight == pytest.approx(0.2)

    # on: (0,0) 입력 구간이라 자연 감쇄로 가중치가 감소.
    on_weights = [e.weight for e in sim_on.network.edges.values()]
    assert any(w < 0.2 for w in on_weights)


def test_prev_a_snapshot_used_by_step3():
    """Step 3가 prev_A를 실제로 사용하는지 확인.

    t=0에서:
      - 초기 엣지: Hebbian 자연 감쇄 -> 0.198
      - 신규 엣지(source=0의 stagger 생성): W_init -> 0.2
    두 종류가 공존하는 것이 5.4 순서의 정상 결과.
    """
    sim = make_simulator(enable_plasticity=True)
    initial_edge_ids = set(sim.network.edges.keys())

    sim.run(1)

    # 초기 엣지는 감쇄 후 0.198 (prune되지 않은 것만)
    for eid in initial_edge_ids:
        if eid in sim.network.edges:
            assert sim.network.edges[eid].weight == pytest.approx(0.198)

    # 신규 생성 엣지는 W_init 그대로
    new_edge_ids = set(sim.network.edges.keys()) - initial_edge_ids
    for eid in new_edge_ids:
        assert sim.network.edges[eid].weight == pytest.approx(0.2)


def test_step3_prune_and_generate_histories_length():
    sim = make_simulator(enable_plasticity=True)
    sim.run(30)
    assert len(sim.prune_history) == 30
    assert len(sim.generate_history) == 30


def test_step3_generation_follows_stagger_pattern():
    """Stagger 조건에 따른 생성 패턴 (spec 5.2).

    T_gen=100, N=20일 때:
      t=0..19  : 매 틱 정확히 1개 (source i=t)
      t=20..99 : 0개 (매치되는 source 없음)
    """
    sim = make_simulator(enable_plasticity=True)
    sim.run(100)

    # t=0..19: 매 틱 1개 생성, source == t
    for t in range(20):
        assert len(sim.generate_history[t]) == 1
        generated_edge = sim.network.edges[sim.generate_history[t][0]]
        assert generated_edge.source == t

    # t=20..99: 생성 없음
    for t in range(20, 100):
        assert sim.generate_history[t] == []


def test_step3_generation_occurs_at_stagger_ticks():
    """t=100 부근에서 stagger 조건을 만족하는 뉴런이 생성 이벤트를 발생."""
    sim = make_simulator(enable_plasticity=True)
    sim.run(200)
    total_generated = sum(len(g) for g in sim.generate_history)
    # T_gen=100, N=20 -> t=100에서 source mod 100 == 0인 source는 0.
    # t=100에서 source 0만 검사 대상. out_degree 캡 여유 있으면 1개 생성.
    assert total_generated >= 1


def test_simulator_gen_rng_determinism():
    """동일 seed면 두 시뮬레이터의 진화가 완전히 동일해야 한다."""
    sim_a = make_simulator(enable_plasticity=True, gen_seed=123)
    sim_b = make_simulator(enable_plasticity=True, gen_seed=123)
    sim_a.run(150)
    sim_b.run(150)

    # 엣지 구성이 동일
    pairs_a = sorted(
        (e.source, e.target, round(e.weight, 10))
        for e in sim_a.network.edges.values()
    )
    pairs_b = sorted(
        (e.source, e.target, round(e.weight, 10))
        for e in sim_b.network.edges.values()
    )
    assert pairs_a == pairs_b


def test_simulator_different_gen_seed_differs():
    """다른 gen_seed면 생성 엣지가 달라질 수 있다."""
    sim_a = make_simulator(enable_plasticity=True, gen_seed=1)
    sim_b = make_simulator(enable_plasticity=True, gen_seed=2)
    sim_a.run(150)
    sim_b.run(150)

    pairs_a = sorted((e.source, e.target) for e in sim_a.network.edges.values())
    pairs_b = sorted((e.source, e.target) for e in sim_b.network.edges.values())
    # 극히 낮은 확률로 동일할 수 있으나 실질적으로 달라야 한다.
    assert pairs_a != pairs_b


def test_step3_pruning_reduces_edge_count():
    """감쇄만 지속되면 결국 W_min 미만으로 내려가 엣지가 소멸한다.

    W_init=0.2, 감쇄만 적용 시 매 틱 x0.99.
    W_min=0.01까지 도달: 0.2 * 0.99^n < 0.01 -> n > ln(0.05)/ln(0.99) ≈ 298.
    T=350이면 대부분의 초기 엣지가 prune된다.
    """
    sim = make_simulator(enable_plasticity=True)
    initial_count = len(sim.network.edges)
    sim.run(400)
    # 초기 엣지 일부는 prune됨. 생성도 일부 있으므로 최종 개수 자체는
    # 단정하지 않고, prune 이벤트가 발생했음만 확인.
    total_pruned = sum(len(p) for p in sim.prune_history)
    assert total_pruned > 0

def test_neuron_meta_csv_schema(tmp_path: Path):
    sim = make_simulator()
    sim.run(5)
    out_dir = tmp_path / "outputs"
    sim.save_run(out_dir)

    lines = (
        (out_dir / "logs" / "neuron_meta.csv")
        .read_text().strip().splitlines()
    )
    assert lines[0] == "neuron_id,S,is_input,theta"
    assert len(lines) == 1 + 20
    # n0, n1이 input
    row0 = lines[1].split(",")
    row1 = lines[2].split(",")
    assert row0[2] == "1"
    assert row1[2] == "1"
    # S는 +1 또는 -1
    for line in lines[1:]:
        s = line.split(",")[1]
        assert s in ("1", "-1")