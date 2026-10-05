"""Step 3: Hebbian 업데이트, pruning, generation.

algorithm_spec.md 5.1, 5.2, 5.4 참조.

Step 3 내부 순서 (5.4, 고정):
    1. Hebbian 업데이트 (기존 엣지 전량)
    2. Pruning (W < W_min 엣지 제거)
    3. Generation (stagger 조건 충족 뉴런에 최대 1개 신규)

설계 메모:
    - 5.1 수식은 co-fire 지표로 A_j(t-1) · A_i(t)를 사용한다.
      입력 뉴런 pre에 대해서도 동일하게 t-1 값을 사용한다
      (dynamics의 입력 same-tick 예외와 다름).
    - Step 2가 A를 t 상태로 덮어썼으므로, A_j(t-1)은 외부에서
      prev_A 딕셔너리로 전달받는다 (Simulator가 캡처).
    - 5.2: 신규 생성 엣지는 같은 틱의 Hebbian 계산에 소급되지 않는다.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .network import Network


# ----------------------------------------------------------------------
# Step 3-1: Hebbian
# ----------------------------------------------------------------------

def hebbian_update(
    network: Network,
    prev_A: Dict[int, int],
    reward: float,
) -> None:
    """Step 3-1: 모든 기존 엣지에 대해 5.1 수식을 적용해 W_e(t) 확정.

    수식 (5.1):
        ΔW_e = eta_u · R · A_j^(pre)(t) · A_i(t) - eta_d · W_e(t-1)
        W_e(t) = clip(W_e(t-1) + ΔW_e, 0, W_max)

    A_j^(pre)(t) 규칙 (dynamics 4.1과 일치):
        - source j가 Input Neuron이면 A_j(t) (Step 1에서 갱신된 당 틱 값)
        - 그 외는 A_j(t-1) (외부에서 prev_A로 전달)

    Args:
        network: 대상 네트워크. 엣지의 weight가 in-place로 갱신된다.
        prev_A: {neuron_id: A_j(t-1)}. Simulator가 Step 1 이전에 캡처.
        reward: 전역 보상 신호 R(t). 비지도 환경에서는 config.reward(=1.0).
    """
    cfg = network.config

    for edge in network.edges.values():
        src_neuron = network.neurons[edge.source]
        # input pre는 same-tick, 그 외는 t-1.
        if src_neuron.is_input:
            a_pre = src_neuron.A
        else:
            a_pre = prev_A[edge.source]
        a_post = network.neurons[edge.target].A  # A_i(t) (Step 2에서 확정)

        delta = (
            cfg.eta_u * reward * a_pre * a_post
            - cfg.eta_d * edge.weight
        )
        new_w = edge.weight + delta

        # 5.1 클립: [0, W_max]
        if new_w < 0.0:
            new_w = 0.0
        elif new_w > cfg.W_max:
            new_w = cfg.W_max

        edge.weight = new_w


# ----------------------------------------------------------------------
# Step 3-2: Pruning
# ----------------------------------------------------------------------

def prune_edges(network: Network) -> List[int]:
    """Step 3-2: W_e(t) < W_min 엣지를 Out/In 인덱스에서 제거.

    Returns:
        제거된 edge_id 목록 (로그·테스트용).
    """
    cfg = network.config
    to_remove = [
        eid for eid, e in network.edges.items() if e.weight < cfg.W_min
    ]
    for eid in to_remove:
        network.remove_edge(eid)
    return to_remove


# ----------------------------------------------------------------------
# Step 3-3: Generation
# ----------------------------------------------------------------------

def generate_edges(
    network: Network,
    t: int,
    gen_rng: random.Random,
) -> List[int]:
    """Step 3-3: stagger 조건을 충족하는 뉴런에 최대 1개의 신규 엣지 생성.

    조건 (5.2):
        - stagger: t mod T_gen == source mod T_gen
        - out_degree(source) < max_out_edges (self-loop, 중복 포함)
        - target in non_input_ids (Input Neuron은 In-Edge 금지)
        - 한 생성 이벤트당 최대 1개 (per source, per tick)

    타깃 선택 (5.2):
        후보 집합 `non_input_ids`에서 uniform random 1개 추출.
        후보 집합이 비어 있으면 해당 소스는 스킵.

    Returns:
        생성된 신규 edge_id 목록.
    """
    cfg = network.config
    created: List[int] = []

    candidates = list(cfg.non_input_ids)
    if not candidates:
        return created

    for src in range(cfg.n_total):
        # stagger 조건
        if t % cfg.T_gen != src % cfg.T_gen:
            continue
        # out-degree 캡
        if network.out_degree(src) >= cfg.max_out_edges:
            continue

        tgt = gen_rng.choice(candidates)
        edge = network.add_edge(source=src, target=tgt, weight=cfg.W_init)
        created.append(edge.edge_id)

    return created


# ----------------------------------------------------------------------
# Step 3 전체
# ----------------------------------------------------------------------

def step3_apply_plasticity(
    network: Network,
    t: int,
    prev_A: Dict[int, int],
    gen_rng: random.Random,
    reward: float | None = None,
) -> None:
    """Step 3 전체를 5.4의 순서로 적용.

    순서 (5.4, 고정):
        1) Hebbian 업데이트 (기존 엣지 전량)
        2) Pruning (W < W_min 제거)
        3) Generation (stagger 조건, 최대 1개/source)
    """
    cfg = network.config
    r = cfg.reward if reward is None else reward

    hebbian_update(network, prev_A, r)
    prune_edges(network)
    generate_edges(network, t, gen_rng)