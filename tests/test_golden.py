"""Golden regression test (spec 7.5-3).

N=5, T=20, seed 고정. 스냅샷과 실행 결과를 비교하여 회귀를 감지한다.
스냅샷은 `python scripts/generate_golden.py` 로 생성/갱신한다.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from artificial_neuron.config import Config
from artificial_neuron.input_provider import CyclicPatternProvider
from artificial_neuron.network import Network
from artificial_neuron.simulator import Simulator


GOLDEN_PATH = Path(__file__).parent / "golden" / "snapshot_n5_t20.json"

# scripts/generate_golden.py의 GOLDEN_CONFIG와 반드시 동일해야 함
GOLDEN_CONFIG = Config(
    n_input=1,
    n_internal=3,
    n_output=1,
    input_patterns=((0,), (1,)),
    input_pattern_period=5,
    init_seed=12345,
    input_seed=23456,
    gen_seed=34567,
)


def _load_golden() -> dict:
    if not GOLDEN_PATH.exists():
        pytest.skip(
            f"Golden snapshot not found at {GOLDEN_PATH}. "
            f"Run `python scripts/generate_golden.py` first."
        )
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def _run_golden_config(golden: dict) -> Simulator:
    cfg = GOLDEN_CONFIG
    init_rng = random.Random(cfg.init_seed)
    network = Network(cfg, init_rng)
    provider = CyclicPatternProvider.from_config(cfg)
    sim = Simulator(
        cfg,
        network,
        provider,
        gen_rng=random.Random(cfg.gen_seed),
        enable_plasticity=golden["plasticity_enabled"],
        snapshot_interval=golden["snapshot_interval"],
    )
    sim.run(golden["ticks"])
    return sim


def test_golden_snapshot_exists():
    assert GOLDEN_PATH.exists(), (
        f"Golden snapshot missing: {GOLDEN_PATH}. "
        f"Run `python scripts/generate_golden.py`."
    )


def test_golden_activation_history():
    golden = _load_golden()
    sim = _run_golden_config(golden)
    actual = [list(row) for row in sim.activation_history]
    assert actual == golden["activation_history"]


def test_golden_voltage_history():
    golden = _load_golden()
    sim = _run_golden_config(golden)
    actual = [
        [round(float(x), 10) for x in row] for row in sim.voltage_history
    ]
    expected = golden["voltage_history"]
    assert len(actual) == len(expected)
    for row_a, row_e in zip(actual, expected):
        assert len(row_a) == len(row_e)
        for a, e in zip(row_a, row_e):
            assert a == pytest.approx(e, abs=1e-9)


def test_golden_output_history():
    golden = _load_golden()
    sim = _run_golden_config(golden)
    actual = [list(row) for row in sim.output_history]
    assert actual == golden["output_history"]


def test_golden_final_edges():
    golden = _load_golden()
    sim = _run_golden_config(golden)
    actual_edges = sorted(
        [
            [e.edge_id, e.source, e.target, round(e.weight, 10)]
            for e in sim.network.edges.values()
        ],
        key=lambda row: row[0],
    )
    expected_edges = golden["final_edges"]
    assert len(actual_edges) == len(expected_edges)
    for row_a, row_e in zip(actual_edges, expected_edges):
        assert row_a[0] == row_e[0]  # edge_id
        assert row_a[1] == row_e[1]  # source
        assert row_a[2] == row_e[2]  # target
        assert row_a[3] == pytest.approx(row_e[3], abs=1e-9)


def test_golden_edge_id_counter():
    golden = _load_golden()
    sim = _run_golden_config(golden)
    assert sim.network.edge_id_counter == golden["final_edge_id_counter"]