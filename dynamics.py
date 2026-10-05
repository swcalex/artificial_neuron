"""Forward dynamics: Step 1, 2, 4.

algorithm_spec.md 3.1, 4.1, 4.2 참조.

Step 1: 입력 뉴런 활성화 갱신 (External Signal Provider)
Step 2: 비입력 뉴런의 V_i(t), A_i(t) 동시 업데이트
Step 4: 출력 뉴런의 활성화 벡터 반환

V_i의 0 클립 (4.1):
    V_i(t) = max(0, gamma * V_i(t-1) * (1 - A_i(t-1)) + sum(...))
    하방 발산 방지 목적. W_e의 [0, W_max] 클립과 동일한 성격.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .input_provider import InputProvider
from .network import Network


def step1_update_inputs(
    network: Network,
    t: int,
    provider: InputProvider,
) -> None:
    """Step 1: 입력 뉴런의 A_input(t)를 provider로부터 갱신.

    입력 뉴런은 V를 갖지 않으므로 A만 갱신한다 (2.1).

    Args:
        network: 대상 네트워크. 입력 뉴런의 A가 in-place 갱신된다.
        t: 현재 틱 인덱스 (t >= 0).
        provider: External Signal Provider (2.4, 7.3-8).

    Raises:
        ValueError: provider가 반환한 벡터의 길이가 n_input과 다르거나,
                    이진값이 아닌 원소를 포함하는 경우.
    """
    cfg = network.config
    sample = provider.sample(t)
    if len(sample) != cfg.n_input:
        raise ValueError(
            f"provider returned vector of length {len(sample)}, "
            f"expected {cfg.n_input}"
        )
    for idx, nid in enumerate(cfg.input_ids):
        a = int(sample[idx])
        if a not in (0, 1):
            raise ValueError(f"input sample contains non-binary value: {a}")
        network.neurons[nid].A = a


def step2_update_dynamics(network: Network) -> None:
    """Step 2: 비입력 뉴런의 V_i(t), A_i(t) 동시 업데이트.

    동시 업데이트 원칙 (3.1):
        모든 비입력 뉴런의 새 V, A는 이전 틱 상태만으로 계산한다.
        새 값은 임시 dict에 모아두었다가 마지막에 일괄 반영하므로,
        계산 도중 확정된 다른 뉴런의 새 A가 같은 틱 계산에 섞이지 않는다.

    수식 (4.1):
        raw = gamma * V_i(t-1) * (1 - A_i(t-1))
              + sum_e A_j^(pre)(t) * S_j * W_e(t-1)
        V_i(t) = max(0, raw)             # 하방 발산 방지 클립

    활성화 (4.2):
        A_i(t) = 1 if V_i(t) >= theta_i else 0

    입력 뉴런 예외 (4.1):
        사전 뉴런 j가 입력 뉴런이면 Step 1에서 이미 t로 갱신된 A_j(t)를
        사용한다. 그 외의 j는 이전 틱 A_j(t-1)을 사용한다.
    """
    cfg = network.config

    new_V: Dict[int, float] = {}
    new_A: Dict[int, int] = {}

    for nid, neuron in network.neurons.items():
        if neuron.is_input:
            continue

        incoming = 0.0
        for edge in network.get_in_edges(nid):
            src_neuron = network.neurons[edge.source]
            # 입력 뉴런이면 현재 A(=A_j(t)), 그 외는 이전 틱 A(=A_j(t-1)).
            # 현재 A는 아직 갱신되지 않았으므로 비입력은 t-1 값을 담고 있다.
            a_pre = src_neuron.A
            incoming += a_pre * src_neuron.S * edge.weight

        raw = cfg.gamma * neuron.V * (1 - neuron.A) + incoming
        v = max(0.0, raw)  # 4.1 하방 발산 방지 클립
        a = 1 if v >= neuron.theta else 0

        new_V[nid] = v
        new_A[nid] = a

    # 동시 반영.
    for nid, v in new_V.items():
        network.neurons[nid].V = v
        network.neurons[nid].A = new_A[nid]


def step4_collect_outputs(network: Network) -> Tuple[int, ...]:
    """Step 4: 출력 뉴런의 활성화 상태를 ID 오름차순으로 반환.

    Step 2에서 이미 확정된 A를 그대로 반환한다 (3.1).

    Returns:
        길이 N_output의 튜플. 각 원소는 0 또는 1.
    """
    return tuple(network.neurons[nid].A for nid in network.config.output_ids)


def forward(
    network: Network,
    t: int,
    provider: InputProvider,
) -> Tuple[int, ...]:
    """Step 1 -> Step 2 -> Step 4를 순서대로 실행하는 편의 함수.

    Step 3 (plasticity)은 포함하지 않는다. 시뮬레이터가 Step별
    프로파일링을 수행할 때는 개별 step 함수를 직접 호출한다.
    """
    step1_update_inputs(network, t, provider)
    step2_update_dynamics(network)
    return step4_collect_outputs(network)