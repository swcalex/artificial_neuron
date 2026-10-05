"""dynamics.py (Step 1, 2, 4) 단위 테스트.

커버 (7.5-1):
- Step 1: provider -> A_input 반영 및 길이/이진값 검증
- Step 2: 손계산 3-뉴런 케이스
- Step 2: 활성화 임계값 경계 (θ-ε, θ, θ+ε)
- Step 2: 입력 뉴런 pre의 same-tick 사용 (4.1 예외)
- Step 2: 동시 업데이트 원칙
- Step 2: 발화 후 전하 리셋
- Step 2: 중복 엣지 합산, self-loop, 억제성 pre의 음수 V
- Step 4: 출력 벡터 반환
- forward: Step 1->2->4 순서
"""

import random
from dataclasses import replace

import pytest

from artificial_neuron.config import Config
from artificial_neuron.dynamics import (
    forward,
    step1_update_inputs,
    step2_update_dynamics,
    step4_collect_outputs,
)
from artificial_neuron.input_provider import CyclicPatternProvider
from artificial_neuron.network import Network


# ----------------------------------------------------------------------
# 헬퍼
# ----------------------------------------------------------------------

def make_3_neuron_network() -> tuple[Config, Network]:
    """3-뉴런 네트워크: 0=input, 1=internal, 2=output. 엣지는 전부 제거."""
    cfg = replace(
        Config(),
        n_input=1,
        n_internal=1,
        n_output=1,
        input_patterns=((0,), (1,)),
        input_pattern_period=1,
    )
    rng = random.Random(0)
    net = Network(cfg, rng)

    # 초기 시냅스 전부 제거 (테스트가 엣지를 직접 구성한다).
    for eid in list(net.edges.keys()):
        net.remove_edge(eid)

    return cfg, net


def set_all_excitatory(net: Network) -> None:
    for n in net.neurons.values():
        n.S = +1


def clear_state(net: Network) -> None:
    """모든 뉴런의 A를 0, V를 0으로 초기화."""
    for n in net.neurons.values():
        n.A = 0
        if not n.is_input:
            n.V = 0.0


# ----------------------------------------------------------------------
# Step 1
# ----------------------------------------------------------------------

def test_step1_sets_input_activation_from_provider():
    cfg, net = make_3_neuron_network()
    provider = CyclicPatternProvider.from_config(cfg)

    step1_update_inputs(net, t=0, provider=provider)
    assert net.neurons[0].A == 0

    step1_update_inputs(net, t=1, provider=provider)
    assert net.neurons[0].A == 1

    step1_update_inputs(net, t=2, provider=provider)
    assert net.neurons[0].A == 0


def test_step1_does_not_touch_non_input_neurons():
    cfg, net = make_3_neuron_network()
    provider = CyclicPatternProvider.from_config(cfg)

    net.neurons[1].A = 1
    net.neurons[2].A = 1

    step1_update_inputs(net, t=1, provider=provider)

    assert net.neurons[1].A == 1  # 변경 없음
    assert net.neurons[2].A == 1  # 변경 없음


def test_step1_rejects_wrong_length():
    cfg, net = make_3_neuron_network()

    class BadProvider:
        def sample(self, t):
            return (0, 1)  # 길이 2, n_input=1

    with pytest.raises(ValueError):
        step1_update_inputs(net, t=0, provider=BadProvider())  # type: ignore[arg-type]


def test_step1_rejects_non_binary():
    cfg, net = make_3_neuron_network()

    class BadProvider:
        def sample(self, t):
            return (2,)

    with pytest.raises(ValueError):
        step1_update_inputs(net, t=0, provider=BadProvider())  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Step 2 - 손계산
# ----------------------------------------------------------------------

def test_step2_manual_calculation_single_tick():
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=0.2)  # input -> internal
    net.add_edge(source=1, target=2, weight=0.3)  # internal -> output

    # 입력 뉴런 A는 Step 1에서 이미 t로 갱신되었다고 가정.
    net.neurons[0].A = 1
    net.neurons[1].V = 0.1
    net.neurons[1].A = 0
    net.neurons[2].V = 0.2
    net.neurons[2].A = 0

    step2_update_dynamics(net)

    # V_1(t) = 0.9 * 0.1 * (1 - 0) + 1 * (+1) * 0.2 = 0.09 + 0.2 = 0.29
    assert net.neurons[1].V == pytest.approx(0.29)
    assert net.neurons[1].A == 0  # 0.29 < 1.0

    # V_2(t) = 0.9 * 0.2 * (1 - 0) + A_1(t-1) * 1 * 0.3
    #        = 0.18 + 0 * 0.3 = 0.18
    assert net.neurons[2].V == pytest.approx(0.18)
    assert net.neurons[2].A == 0


def test_step2_decay_with_no_input():
    cfg, net = make_3_neuron_network()
    clear_state(net)
    net.neurons[1].V = 0.5
    net.neurons[2].V = 0.4

    step2_update_dynamics(net)

    assert net.neurons[1].V == pytest.approx(0.45)
    assert net.neurons[2].V == pytest.approx(0.36)


def test_step2_reset_after_firing():
    cfg, net = make_3_neuron_network()
    clear_state(net)
    net.neurons[1].V = 0.5
    net.neurons[1].A = 1  # 이전 틱에 발화했음
    net.neurons[2].V = 0.4

    step2_update_dynamics(net)

    # V_1 = 0.9 * 0.5 * (1 - 1) = 0
    assert net.neurons[1].V == pytest.approx(0.0)
    assert net.neurons[1].A == 0


def test_step2_duplicate_edges_are_summed_independently():
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=0.3)
    net.add_edge(source=0, target=1, weight=0.4)
    clear_state(net)
    net.neurons[0].A = 1

    step2_update_dynamics(net)

    # V_1 = 1*1*0.3 + 1*1*0.4 = 0.7
    assert net.neurons[1].V == pytest.approx(0.7)


def test_step2_self_loop_uses_previous_activation():
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=1, target=1, weight=0.5)
    clear_state(net)
    net.neurons[1].A = 1  # 이전 틱 발화
    net.neurons[1].V = 0.2

    step2_update_dynamics(net)

    # V_1 = 0.9 * 0.2 * (1 - 1) + A_1(t-1)*S_1*W = 0 + 1*1*0.5 = 0.5
    assert net.neurons[1].V == pytest.approx(0.5)


# ----------------------------------------------------------------------
# Step 2 - 입력 뉴런 same-tick (4.1 예외)
# ----------------------------------------------------------------------

def test_step2_input_pre_uses_same_tick_activation():
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=1.0)
    clear_state(net)
    net.neurons[0].A = 1  # Step 1에서 t로 갱신됨

    step2_update_dynamics(net)

    # V_1 = 0 + A_0(t)*S_0*W = 1*1*1.0 = 1.0
    assert net.neurons[1].V == pytest.approx(1.0)
    assert net.neurons[1].A == 1


def test_step2_non_input_pre_uses_previous_tick_activation():
    """internal/output 뉴런이 pre일 때는 이전 틱 A를 사용해야 한다."""
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=1, target=2, weight=1.0)
    clear_state(net)
    net.neurons[1].A = 1  # 이전 틱 A

    step2_update_dynamics(net)

    # V_2 = 0 + A_1(t-1)*S_1*W = 1*1*1.0 = 1.0
    assert net.neurons[2].V == pytest.approx(1.0)


# ----------------------------------------------------------------------
# Step 2 - 동시 업데이트
# ----------------------------------------------------------------------

def test_step2_simultaneous_update_chain():
    """뉴런 1의 새 발화가 같은 틱의 뉴런 2 계산에 사용되지 않아야 한다."""
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=1.0)
    net.add_edge(source=1, target=2, weight=1.0)
    clear_state(net)
    net.neurons[0].A = 1

    step2_update_dynamics(net)

    # 뉴런 1: V = 1.0 -> 발화
    assert net.neurons[1].A == 1
    # 뉴런 2: A_1(t-1)=0만 반영 -> V=0, 비발화
    assert net.neurons[2].V == pytest.approx(0.0)
    assert net.neurons[2].A == 0


# ----------------------------------------------------------------------
# Step 2 - 임계값 경계
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "weight,expected_A",
    [
        (0.999, 0),   # θ - ε
        (1.0, 1),     # θ
        (1.001, 1),   # θ + ε
    ],
)
def test_step2_activation_threshold_boundary(weight, expected_A):
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=weight)
    clear_state(net)
    net.neurons[0].A = 1

    step2_update_dynamics(net)

    assert net.neurons[1].A == expected_A


# ----------------------------------------------------------------------
# Step 2 - 억제성 및 0 클립 (4.1)
# ----------------------------------------------------------------------

def test_step2_inhibitory_pre_is_clipped_to_zero():
    """억제성 pre로 raw가 음수가 되면 V는 0으로 클립된다 (4.1)."""
    cfg, net = make_3_neuron_network()
    clear_state(net)
    net.neurons[0].S = -1  # 억제성 input
    net.neurons[1].S = +1
    net.neurons[2].S = +1
    net.add_edge(source=0, target=1, weight=0.5)
    net.neurons[0].A = 1

    step2_update_dynamics(net)

    # raw = 0 + 1 * (-1) * 0.5 = -0.5 -> max(0, -0.5) = 0.0
    assert net.neurons[1].V == pytest.approx(0.0)
    assert net.neurons[1].A == 0


def test_step2_clip_bounds_decay_then_inhibition():
    """감쇄 항이 양수여도 억제 입력이 더 크면 0으로 클립된다."""
    cfg, net = make_3_neuron_network()
    clear_state(net)
    net.neurons[0].S = -1  # 억제성 input
    net.add_edge(source=0, target=1, weight=1.0)
    net.neurons[1].V = 0.2   # 감쇄 항 = 0.9 * 0.2 = 0.18
    net.neurons[0].A = 1     # 억제 항 = -1.0

    step2_update_dynamics(net)

    # raw = 0.18 - 1.0 = -0.82 -> 0.0
    assert net.neurons[1].V == pytest.approx(0.0)


def test_step2_positive_result_is_not_clipped():
    """raw가 양수면 클립 없이 그대로 사용된다."""
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=0.3)
    clear_state(net)
    net.neurons[1].V = 0.5
    net.neurons[0].A = 1

    step2_update_dynamics(net)

    # raw = 0.9 * 0.5 + 0.3 = 0.45 + 0.3 = 0.75
    assert net.neurons[1].V == pytest.approx(0.75)


# ----------------------------------------------------------------------
# Step 4
# ----------------------------------------------------------------------

def test_step4_returns_output_vector():
    cfg, net = make_3_neuron_network()
    clear_state(net)

    net.neurons[2].A = 1
    assert step4_collect_outputs(net) == (1,)

    net.neurons[2].A = 0
    assert step4_collect_outputs(net) == (0,)


def test_step4_does_not_modify_state():
    cfg, net = make_3_neuron_network()
    clear_state(net)
    net.neurons[2].A = 1

    step4_collect_outputs(net)

    assert net.neurons[2].A == 1


# ----------------------------------------------------------------------
# forward
# ----------------------------------------------------------------------

def test_forward_runs_steps_in_order():
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=1.0)
    clear_state(net)

    provider = CyclicPatternProvider.from_config(cfg)
    out = forward(net, t=1, provider=provider)  # t=1 -> input A=(1,)

    # Step 1: A_0 = 1
    # Step 2: V_1 = 1.0 -> A_1 = 1, V_2 = 0 -> A_2 = 0
    # Step 4: (0,)
    assert net.neurons[0].A == 1
    assert net.neurons[1].A == 1
    assert out == (0,)


def test_forward_t0_uses_zero_previous_activation():
    """t=0에서 이전 틱 A는 0이어야 한다 (1.3)."""
    cfg, net = make_3_neuron_network()
    set_all_excitatory(net)
    net.add_edge(source=0, target=1, weight=1.0)
    clear_state(net)

    provider = CyclicPatternProvider.from_config(cfg)
    out = forward(net, t=0, provider=provider)  # t=0 -> input A=(0,)

    # Step 1: A_0 = 0
    # Step 2: V_1 = 0 (입력도 0, 이전 상태도 0)
    assert net.neurons[0].A == 0
    assert net.neurons[1].V == pytest.approx(0.0)
    assert net.neurons[1].A == 0
    assert out == (0,)