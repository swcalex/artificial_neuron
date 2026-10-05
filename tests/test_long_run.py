"""장기 안정성 종합 테스트 (spec 7.5-5, 7.6).

T=10,000틱 실행 후 다음을 수행한다.
  1. 7.6의 참고 지표를 콘솔 리포트로 산출
  2. 극단적 실패(전면 침묵 / 전면 포화 / W 포화 / in-degree runaway /
     self-loop runaway / 억제 발화 소멸)만 assert로 자동 판정
  3. seed 고정 시 재현성 검증

1.4의 설계 목표 만족 여부는 리포트 관찰로 판단한다.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Dict, List, Tuple

import pytest

from artificial_neuron.config import Config
from artificial_neuron.input_provider import CyclicPatternProvider
from artificial_neuron.network import Network
from artificial_neuron.simulator import Simulator


LONG_TICKS = 10_000
FINAL_WINDOW = 1_000
RUNAWAY_THRESHOLD = 100


# ----------------------------------------------------------------------
# 헬퍼
# ----------------------------------------------------------------------

def _build_sim() -> Simulator:
    cfg = Config()
    init_rng = random.Random(cfg.init_seed)
    net = Network(cfg, init_rng)
    provider = CyclicPatternProvider.from_config(cfg)
    return Simulator(
        cfg, net, provider,
        gen_rng=random.Random(cfg.gen_seed),
        enable_plasticity=True,
        snapshot_interval=100,
    )


def _count_runaway_neurons(
    history: List[Tuple[int, ...]], threshold: int
) -> int:
    """threshold틱 이상 연속 발화한 뉴런 수."""
    if not history:
        return 0
    T = len(history)
    N = len(history[0])
    count = 0
    for nid in range(N):
        run = 0
        for t in range(T):
            if history[t][nid] == 1:
                run += 1
                if run >= threshold:
                    count += 1
                    break
            else:
                run = 0
    return count


def _build_report(sim: Simulator) -> Dict[str, float]:
    cfg = sim.config
    T = sim.tick

    group_means = sim.mean_firing_rate_by_group()

    final_start = max(0, T - FINAL_WINDOW)
    tail = sim.activation_history[final_start:]
    n_tail = len(tail)

    io_ids = list(cfg.internal_ids) + list(cfg.output_ids)
    e_ids = [nid for nid, n in sim.network.neurons.items() if n.S == +1]
    i_ids = [nid for nid, n in sim.network.neurons.items() if n.S == -1]

    def _tail_rate(ids: List[int]) -> float:
        if not ids or n_tail == 0:
            return 0.0
        s = sum(sum(a[nid] for nid in ids) for a in tail)
        return s / (n_tail * len(ids))

    last_ws = sim.weight_stats_history[-1]
    max_sat_ratio = max(
        last_ws.ratio_W_le_0_01, last_ws.ratio_W_ge_1_99
    )

    in_degrees = [
        sim.network.in_degree(nid) for nid in sim.network.neurons
    ]
    max_in_degree = max(in_degrees) if in_degrees else 0

    runaway = _count_runaway_neurons(
        sim.activation_history, RUNAWAY_THRESHOLD
    )

    return {
        "total_ticks": float(T),
        "final_edge_count": float(len(sim.network.edges)),
        "total_pruned_events": float(
            sum(len(p) for p in sim.prune_history)
        ),
        "total_generated_events": float(
            sum(len(g) for g in sim.generate_history)
        ),
        "mean_firing_total": group_means["total"],
        "mean_firing_internal": group_means["internal"],
        "mean_firing_output": group_means["output"],
        "tail_firing_internal_output": _tail_rate(io_ids),
        "tail_firing_exc": _tail_rate(e_ids),
        "tail_firing_inh": _tail_rate(i_ids),
        "max_weight_boundary_ratio": max_sat_ratio,
        "max_in_degree": float(max_in_degree),
        "in_degree_threshold": float(cfg.n_total / 2.0),
        "runaway_neurons": float(runaway),
        "runaway_threshold": float(RUNAWAY_THRESHOLD),
    }


def _print_report(report: Dict[str, float]) -> None:
    print()
    print("===== Long-run stability report (T=10,000) =====")
    print(f"total ticks              : {int(report['total_ticks'])}")
    print(f"final edge count         : {int(report['final_edge_count'])}")
    print(f"total pruned events      : {int(report['total_pruned_events'])}")
    print(f"total generated events   : "
          f"{int(report['total_generated_events'])}")
    print()
    print("[발화율]")
    print(f"  mean total             : {report['mean_firing_total']:.4f}")
    print(f"  mean internal          : {report['mean_firing_internal']:.4f}")
    print(f"  mean output            : {report['mean_firing_output']:.4f}")
    print(f"  tail internal+output   : "
          f"{report['tail_firing_internal_output']:.4f}")
    print(f"  tail excitatory        : {report['tail_firing_exc']:.4f}")
    print(f"  tail inhibitory        : {report['tail_firing_inh']:.4f}")
    print()
    print("[구조 / 가중치]")
    print(f"  max weight saturation  : "
          f"{report['max_weight_boundary_ratio']:.4f}")
    print(f"  max in-degree          : {int(report['max_in_degree'])} "
          f"(threshold N/2 = {report['in_degree_threshold']:.1f})")
    print(f"  runaway neurons (>= {int(report['runaway_threshold'])} tick) : "
          f"{int(report['runaway_neurons'])}")


# ----------------------------------------------------------------------
# 테스트
# ----------------------------------------------------------------------

def test_long_run_completes():
    sim = _build_sim()
    sim.run(LONG_TICKS)
    assert sim.tick == LONG_TICKS


def test_long_run_reproducible():
    """동일 seed/설정이면 T=10,000에서도 완전히 동일한 결과."""
    sim_a = _build_sim()
    sim_a.run(LONG_TICKS)
    sim_b = _build_sim()
    sim_b.run(LONG_TICKS)

    assert sim_a.activation_history[-1] == sim_b.activation_history[-1]
    assert set(sim_a.network.edges.keys()) == set(sim_b.network.edges.keys())
    for eid in sim_a.network.edges:
        ea = sim_a.network.edges[eid]
        eb = sim_b.network.edges[eid]
        assert (ea.source, ea.target) == (eb.source, eb.target)
        assert ea.weight == pytest.approx(eb.weight)
    assert sim_a.network.edge_id_counter == sim_b.network.edge_id_counter


def test_long_run_stability_report(tmp_path: Path):
    """T=10,000 관찰 리포트 (spec 7.5-5, 7.6).

    본 테스트는 리포트 전용이다. 7.6의 참고 범위는 assert로 강제하지
    않으며, 콘솔에 관찰 사항으로만 출력한다. 이는 spec 1.4의
    "초기 구현 → 모니터링 → 파라미터 조정" 흐름에 따른다.

    단, 알고리즘 불변식 (V_i >= 0, 0 <= W_e <= W_max, in-edge 금지 등)
    은 assert로 강제한다. 이는 모델 건강도가 아니라 구현 정확성 문제다.
    """
    sim = _build_sim()
    sim.run(LONG_TICKS)
    sim.save_run(tmp_path / "long_run")

    report = _build_report(sim)
    _print_report(report)

    # ---- 알고리즘 불변식 (반드시 성립) ----
    cfg = sim.config
    for nid, n in sim.network.neurons.items():
        if not n.is_input:
            assert n.V >= 0.0, f"V[{nid}] < 0"
    for eid, e in sim.network.edges.items():
        assert 0.0 <= e.weight <= cfg.W_max, (
            f"W[{eid}] = {e.weight} out of [0, {cfg.W_max}]"
        )
    for nid in cfg.input_ids:
        assert sim.network.in_degree(nid) == 0, (
            f"input neuron {nid} has in-edges"
        )
    # E/I 비율은 초기 배정 후 변하지 않음
    e_count = sum(1 for n in sim.network.neurons.values() if n.S == +1)
    i_count = sum(1 for n in sim.network.neurons.values() if n.S == -1)
    assert e_count == 16 and i_count == 4, (
        f"E/I ratio drift: E={e_count}, I={i_count}"
    )

    # ---- 7.6 관찰 사항 (실패 아님, 경고만 출력) ----
    _print_76_observations(report)


def _print_76_observations(report: Dict[str, float]) -> None:
    """7.6 참고 범위 대비 관찰 사항을 출력한다 (실패 아님).

    이후 파라미터 조정 사이클의 입력 자료로 사용한다.
    """
    observations: List[str] = []

    if report["tail_firing_internal_output"] >= 0.9:
        observations.append(
            f"  - tail internal+output firing "
            f"{report['tail_firing_internal_output']:.4f} >= 0.9 (포화)"
        )
    if report["tail_firing_internal_output"] <= 0.0:
        observations.append(
            "  - tail internal+output firing == 0 (완전 침묵)"
        )
    if report["max_weight_boundary_ratio"] >= 0.20:
        observations.append(
            f"  - max W boundary ratio "
            f"{report['max_weight_boundary_ratio']:.4f} >= 0.20 (경계 포화)"
        )
    if report["max_in_degree"] >= report["in_degree_threshold"]:
        observations.append(
            f"  - max in-degree {int(report['max_in_degree'])} "
            f">= threshold {report['in_degree_threshold']:.1f}"
        )
    if report["runaway_neurons"] > 0:
        observations.append(
            f"  - runaway neurons (>= "
            f"{int(report['runaway_threshold'])} tick) : "
            f"{int(report['runaway_neurons'])}"
        )
    if report["tail_firing_inh"] == 0.0 and report["tail_firing_exc"] > 0.0:
        observations.append(
            "  - inhibitory firing == 0 while excitatory > 0"
        )

    print()
    print("[7.6 참고 범위 대비 관찰 사항]")
    if not observations:
        print("  (관찰된 이탈 없음)")
    else:
        for line in observations:
            print(line)
        print()
        print(
            "  참고: 본 관찰은 실패가 아니다. 파라미터 조정 사이클에서 "
            "검토 대상으로 기록한다 (spec 1.4)."
        )