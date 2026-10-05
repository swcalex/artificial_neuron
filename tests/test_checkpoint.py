"""io/checkpoint.py 단위·통합 테스트 (7.4 단계 8, 7.7).

커버:
- 저장 후 로드 시 뉴런/엣지/카운터/상태 동일
- 이어하기: T1 저장 → 로드 → T2 실행 == T1+T2 한 번에 실행
- edge_id 재사용 금지 규칙 유지
- 잘못된 format_version 거부
- Config round-trip
"""

import random
from dataclasses import replace
from pathlib import Path

import pytest

from artificial_neuron.config import Config
from artificial_neuron.input_provider import CyclicPatternProvider
from artificial_neuron.io.checkpoint import (
    CHECKPOINT_FORMAT_VERSION,
    load_checkpoint,
    save_checkpoint,
)
from artificial_neuron.network import Network
from artificial_neuron.simulator import Simulator


def make_sim(config: Config | None = None) -> Simulator:
    cfg = config or Config()
    init_rng = random.Random(cfg.init_seed)
    net = Network(cfg, init_rng)
    provider = CyclicPatternProvider.from_config(cfg)
    return Simulator(
        cfg,
        net,
        provider,
        gen_rng=random.Random(cfg.gen_seed),
        enable_plasticity=True,
        snapshot_interval=100,
    )


# ----------------------------------------------------------------------
# Round-trip
# ----------------------------------------------------------------------

def test_checkpoint_round_trip_state_identical(tmp_path: Path):
    sim = make_sim()
    sim.run(50)
    ckpt = tmp_path / "ckpt.json"
    save_checkpoint(sim, ckpt)

    loaded = load_checkpoint(ckpt)

    assert loaded.tick == sim.tick
    assert loaded.network.edge_id_counter == sim.network.edge_id_counter
    assert set(loaded.network.neurons.keys()) == set(sim.network.neurons.keys())
    for nid in sim.network.neurons:
        a = sim.network.neurons[nid]
        b = loaded.network.neurons[nid]
        assert a.S == b.S
        assert a.V == pytest.approx(b.V)
        assert a.A == b.A
        assert a.theta == pytest.approx(b.theta)
        assert a.is_input == b.is_input

    assert set(loaded.network.edges.keys()) == set(sim.network.edges.keys())
    for eid in sim.network.edges:
        ea = sim.network.edges[eid]
        eb = loaded.network.edges[eid]
        assert (ea.source, ea.target) == (eb.source, eb.target)
        assert ea.weight == pytest.approx(eb.weight)


def test_checkpoint_preserves_in_out_indices(tmp_path: Path):
    sim = make_sim()
    sim.run(30)
    ckpt = tmp_path / "ckpt.json"
    save_checkpoint(sim, ckpt)
    loaded = load_checkpoint(ckpt)

    for nid in loaded.network.neurons:
        assert set(loaded.network.out_edges[nid]) == set(
            sim.network.out_edges[nid]
        )
        assert set(loaded.network.in_edges[nid]) == set(
            sim.network.in_edges[nid]
        )


def test_checkpoint_config_round_trip(tmp_path: Path):
    cfg = replace(
        Config(),
        init_seed=999,
        gen_seed=888,
        input_pattern_period=50,
    )
    sim = make_sim(cfg)
    sim.run(5)
    ckpt = tmp_path / "ckpt.json"
    save_checkpoint(sim, ckpt)

    loaded = load_checkpoint(ckpt)
    assert loaded.config.init_seed == 999
    assert loaded.config.gen_seed == 888
    assert loaded.config.input_pattern_period == 50
    # tuple 필드 복원 확인
    assert isinstance(loaded.config.input_patterns, tuple)
    assert loaded.config.input_patterns == cfg.input_patterns


# ----------------------------------------------------------------------
# 이어하기 재현
# ----------------------------------------------------------------------

def test_checkpoint_continue_reproduces_full_run(tmp_path: Path):
    """T1 저장 → 로드 → T2 더 실행 == T1+T2 한 번에 실행."""
    T1 = 40
    T2 = 60

    sim_full = make_sim()
    sim_full.run(T1 + T2)

    sim_part = make_sim()
    sim_part.run(T1)
    ckpt = tmp_path / "ckpt.json"
    save_checkpoint(sim_part, ckpt)

    loaded = load_checkpoint(ckpt)
    loaded.run(T2)

    # 마지막 틱 활성화 일치
    assert loaded.tick == T1 + T2
    assert loaded.activation_history[-1] == sim_full.activation_history[-1]

    # 최종 엣지 상태 및 카운터 일치
    assert set(loaded.network.edges.keys()) == set(sim_full.network.edges.keys())
    for eid in sim_full.network.edges:
        ea = sim_full.network.edges[eid]
        eb = loaded.network.edges[eid]
        assert (ea.source, ea.target) == (eb.source, eb.target)
        assert ea.weight == pytest.approx(eb.weight)
    assert (
        loaded.network.edge_id_counter
        == sim_full.network.edge_id_counter
    )


def test_checkpoint_log_buffer_starts_empty(tmp_path: Path):
    """재개 시 로그 버퍼는 비어 있어야 한다 (정책)."""
    sim = make_sim()
    sim.run(20)
    ckpt = tmp_path / "ckpt.json"
    save_checkpoint(sim, ckpt)

    loaded = load_checkpoint(ckpt)
    assert loaded.activation_history == []
    assert loaded.tick_records == []
    assert loaded.output_history == []
    assert loaded.tick == 20

    loaded.run(5)
    assert len(loaded.activation_history) == 5


# ----------------------------------------------------------------------
# 오류 케이스
# ----------------------------------------------------------------------

def test_checkpoint_rejects_unknown_format_version(tmp_path: Path):
    ckpt = tmp_path / "bad.json"
    ckpt.write_text('{"format_version": 9999}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_checkpoint(ckpt)


def test_checkpoint_format_version_constant():
    assert CHECKPOINT_FORMAT_VERSION == 1


# ----------------------------------------------------------------------
# edge_id_counter 유지
# ----------------------------------------------------------------------

def test_checkpoint_edge_id_counter_continues(tmp_path: Path):
    """체크포인트 후 신규 엣지가 기존 edge_id와 겹치지 않아야 한다."""
    sim = make_sim()
    sim.run(30)
    ckpt = tmp_path / "ckpt.json"
    save_checkpoint(sim, ckpt)
    loaded = load_checkpoint(ckpt)

    before_ids = set(loaded.network.edges.keys())
    before_counter = loaded.network.edge_id_counter

    # 신규 엣지 하나 추가
    new_edge = loaded.network.add_edge(source=5, target=6, weight=0.2)
    assert new_edge.edge_id == before_counter
    assert new_edge.edge_id not in before_ids