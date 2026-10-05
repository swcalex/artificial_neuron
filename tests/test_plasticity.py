"""plasticity.py (Step 3) 단위 테스트.

커버 (7.5-1, 7.5-2):
- Hebbian: (A_j, A_i) 4조합 전수, W_max 캡, 0 클립
- Hebbian: 중복 엣지 독립 학습
- Pruning: W_min 경계, in/out 인덱스 갱신
- Generation: stagger, out-degree 캡, Input 배제, 최대 1개/source
- Step 3 순서: Hebbian -> prune -> generate
- 신규 엣지가 같은 틱 Hebbian에 소급되지 않음
- gen_rng 결정성
"""

import random
from dataclasses import replace

import pytest

from artificial_neuron.config import Config
from artificial_neuron.network import Network
from artificial_neuron.plasticity import (
    generate_edges,
    hebbian_update,
    prune_edges,
    step3_apply_plasticity,
)


# ----------------------------------------------------------------------
# 헬퍼
# ----------------------------------------------------------------------

def make_bare_network(
    n_input: int = 1,
    n_internal: int = 2,
    n_output: int = 1,
    seed: int = 0,
    **cfg_overrides,
) -> Network:
    """엣지가 전혀 없는 네트워크."""
    cfg = replace(
        Config(),
        n_input=n_input,
        n_internal=n_internal,
        n_output=n_output,
        **cfg_overrides,
    )
    rng = random.Random(seed)
    net = Network(cfg, rng)
    for eid in list(net.edges.keys()):
        net.remove_edge(eid)
    return net


def zero_prev_A(net: Network) -> dict:
    return {nid: 0 for nid in net.neurons}


def clear_A(net: Network) -> None:
    for n in net.neurons.values():
        n.A = 0


# ----------------------------------------------------------------------
# Hebbian: 4조합 전수
# ----------------------------------------------------------------------

@pytest.mark.parametrize(
    "a_pre,a_post,expect_increase",
    [
        (1, 1, True),
        (1, 0, False),
        (0, 1, False),
        (0, 0, False),
    ],
)
def test_hebbian_four_combinations(a_pre, a_post, expect_increase):
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=0.2)  # internal -> internal
    prev_A = zero_prev_A(net)
    prev_A[1] = a_pre
    clear_A(net)
    net.neurons[2].A = a_post

    hebbian_update(net, prev_A, reward=1.0)

    if expect_increase:
        # ΔW = 0.05*1*1 - 0.01*0.2 = 0.048 -> 0.248
        assert net.edges[e.edge_id].weight == pytest.approx(0.248)
    else:
        # ΔW = 0 - 0.01*0.2 = -0.002 -> 0.198
        assert net.edges[e.edge_id].weight == pytest.approx(0.198)


def test_hebbian_reward_scales_update():
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=0.2)
    prev_A = zero_prev_A(net)
    prev_A[1] = 1
    clear_A(net)
    net.neurons[2].A = 1

    hebbian_update(net, prev_A, reward=2.0)
    # ΔW = 0.05*2*1*1 - 0.01*0.2 = 0.1 - 0.002 = 0.098 -> 0.298
    assert net.edges[e.edge_id].weight == pytest.approx(0.298)


def test_hebbian_w_max_cap():
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=1.99)
    prev_A = zero_prev_A(net)
    prev_A[1] = 1
    clear_A(net)
    net.neurons[2].A = 1

    hebbian_update(net, prev_A, reward=1.0)
    # ΔW = 0.05 - 0.0199 = 0.0301 -> 2.0201 -> cap 2.0
    assert net.edges[e.edge_id].weight == pytest.approx(2.0)


def test_hebbian_zero_floor_on_negative_delta():
    """음수 ΔW가 W를 0 미만으로 내리면 0으로 클립된다.

    W가 작고 η_u·R·A_j·A_i가 0이라면 ΔW = -η_d·W.
    η_d < 1이므로 이 경우 W는 0으로 정확히 수렴하지 않고 감쇠만 한다.
    클립이 실제로 발동하는 시나리오는 음의 reward 도입 시이므로,
    본 테스트는 명시적으로 큰 음의 reward를 넣어 클립을 검증한다.
    """
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=0.5)
    prev_A = zero_prev_A(net)
    prev_A[1] = 1
    clear_A(net)
    net.neurons[2].A = 1

    # 음의 reward: ΔW = 0.05*(-100)*1*1 - 0.01*0.5 = -5.005
    hebbian_update(net, prev_A, reward=-100.0)
    assert net.edges[e.edge_id].weight == pytest.approx(0.0)


def test_hebbian_duplicate_edges_independent():
    net = make_bare_network()
    e1 = net.add_edge(source=1, target=2, weight=0.2)
    e2 = net.add_edge(source=1, target=2, weight=0.4)
    prev_A = zero_prev_A(net)
    prev_A[1] = 1
    clear_A(net)
    net.neurons[2].A = 1

    hebbian_update(net, prev_A, reward=1.0)

    # e1: Δ = 0.05 - 0.01*0.2 = 0.048 -> 0.248
    # e2: Δ = 0.05 - 0.01*0.4 = 0.046 -> 0.446
    assert net.edges[e1.edge_id].weight == pytest.approx(0.248)
    assert net.edges[e2.edge_id].weight == pytest.approx(0.446)


# ----------------------------------------------------------------------
# Pruning
# ----------------------------------------------------------------------

def test_prune_removes_below_w_min():
    net = make_bare_network()
    e_low = net.add_edge(source=1, target=2, weight=0.005)
    e_keep = net.add_edge(source=1, target=2, weight=0.02)

    removed = prune_edges(net)

    assert e_low.edge_id in removed
    assert e_low.edge_id not in net.edges
    assert e_keep.edge_id in net.edges


def test_prune_boundary_w_equal_w_min_is_kept():
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=0.01)  # W_min == 0.01

    removed = prune_edges(net)

    assert e.edge_id not in removed
    assert e.edge_id in net.edges


def test_prune_updates_in_out_indices():
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=0.005)
    eid = e.edge_id

    prune_edges(net)

    assert eid not in net.out_edges[1]
    assert eid not in net.in_edges[2]


def test_prune_returns_empty_when_nothing_to_remove():
    net = make_bare_network()
    net.add_edge(source=1, target=2, weight=0.5)
    assert prune_edges(net) == []


# ----------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------

def test_generate_respects_stagger():
    """T_gen=10, N=4 -> t=1 에서는 source 1만 생성 검사 대상."""
    net = make_bare_network(T_gen=10)
    rng = random.Random(0)

    created = generate_edges(net, t=1, gen_rng=rng)
    # source 1만 stagger 조건 통과 (i mod 10 == 1 mod 10 == 1)
    assert len(created) == 1
    assert net.edges[created[0]].source == 1

    # t=2에서는 source 2만 통과
    created2 = generate_edges(net, t=2, gen_rng=rng)
    assert len(created2) == 1
    assert net.edges[created2[0]].source == 2

    # t=5에서는 어떤 source도 조건 미충족 (N=4)
    created3 = generate_edges(net, t=5, gen_rng=rng)
    assert created3 == []


def test_generate_target_is_non_input():
    net = make_bare_network(n_input=1, n_internal=2, n_output=1, T_gen=1)
    rng = random.Random(0)

    created = generate_edges(net, t=0, gen_rng=rng)
    for eid in created:
        assert net.edges[eid].target != 0  # input이 아님


def test_generate_respects_out_degree_cap():
    net = make_bare_network(T_gen=10, max_out_edges=2)
    # source 1의 out-degree를 2로 채운다.
    net.add_edge(source=1, target=2, weight=0.2)
    net.add_edge(source=1, target=2, weight=0.2)
    assert net.out_degree(1) == 2

    created = generate_edges(net, t=1, gen_rng=random.Random(0))
    # source 1은 캡 초과 -> 생성 없음
    assert created == []


def test_generate_allows_self_loop_and_duplicate():
    """source 1이 target 1을 고르는 경우 self-loop. 중복도 허용."""
    net = make_bare_network(n_input=1, n_internal=1, n_output=1, T_gen=10)
    rng = random.Random(0)

    # 후보 집합은 {1, 2} (input=0 제외). source 1이 self-loop을 뽑을 수도 있다.
    created = generate_edges(net, t=1, gen_rng=rng)
    assert len(created) == 1
    e = net.edges[created[0]]
    assert e.source == 1
    assert e.target in (1, 2)


def test_generate_deterministic_with_seed():
    net_a = make_bare_network(T_gen=1)
    net_b = make_bare_network(T_gen=1)
    created_a = generate_edges(net_a, t=0, gen_rng=random.Random(7))
    created_b = generate_edges(net_b, t=0, gen_rng=random.Random(7))

    pairs_a = sorted(
        (net_a.edges[eid].source, net_a.edges[eid].target) for eid in created_a
    )
    pairs_b = sorted(
        (net_b.edges[eid].source, net_b.edges[eid].target) for eid in created_b
    )
    assert pairs_a == pairs_b


# ----------------------------------------------------------------------
# Step 3 전체 순서 및 통합
# ----------------------------------------------------------------------

def test_step3_hebbian_then_prune():
    """Hebbian 감쇄로 W가 W_min 밑으로 내려가면 같은 틱에 prune된다."""
    net = make_bare_network(T_gen=10)
    # W_min 직상: Hebbian 후 W < W_min 이 되어야 함.
    # ΔW = -0.01·W -> W_new = 0.99·W.  W=0.01005 -> 0.0099495 < 0.01 ✓
    e = net.add_edge(source=1, target=2, weight=0.01005)
    prev_A = zero_prev_A(net)  # (0, 0)

    step3_apply_plasticity(
        net, t=0, prev_A=prev_A, gen_rng=random.Random(0)
    )

    assert e.edge_id not in net.edges


def test_step3_new_edge_not_updated_same_tick():
    """신규 엣지는 같은 틱 Hebbian에 소급되지 않는다 (5.2)."""
    net = make_bare_network(T_gen=10)
    # 기존 엣지 하나를 만들고, prev_A=(0,0)으로 감쇄시킨 뒤 생성까지 진행.
    old = net.add_edge(source=1, target=2, weight=0.2)
    prev_A = zero_prev_A(net)

    before_ids = set(net.edges.keys())
    step3_apply_plasticity(
        net, t=1, prev_A=prev_A, gen_rng=random.Random(0)
    )
    new_ids = set(net.edges.keys()) - before_ids

    # 기존 엣지는 감쇄 (0.2 -> 0.198)
    assert net.edges[old.edge_id].weight == pytest.approx(0.198)

    # 신규 엣지는 W_init 그대로 (감쇄 없음)
    assert len(new_ids) == 1
    new_eid = next(iter(new_ids))
    assert net.edges[new_eid].weight == pytest.approx(0.2)


def test_step3_uses_config_reward_by_default():
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=0.2)
    prev_A = zero_prev_A(net)
    prev_A[1] = 1
    clear_A(net)
    net.neurons[2].A = 1

    step3_apply_plasticity(
        net, t=0, prev_A=prev_A, gen_rng=random.Random(0)
    )
    # config.reward = 1.0 -> W = 0.248
    assert net.edges[e.edge_id].weight == pytest.approx(0.248)


def test_step3_explicit_reward_overrides_config():
    net = make_bare_network()
    e = net.add_edge(source=1, target=2, weight=0.2)
    prev_A = zero_prev_A(net)
    prev_A[1] = 1
    clear_A(net)
    net.neurons[2].A = 1

    step3_apply_plasticity(
        net, t=0, prev_A=prev_A, gen_rng=random.Random(0), reward=2.0
    )
    # ΔW = 0.05*2 - 0.002 = 0.098 -> 0.298
    assert net.edges[e.edge_id].weight == pytest.approx(0.298)

# ----------------------------------------------------------------------
# 수정 반영
# ----------------------------------------------------------------------

def test_hebbian_input_pre_uses_same_tick_activation():
    """input pre는 prev_A가 아니라 현재 A_j(t)를 사용한다 (4.1과 통일)."""
    net = make_bare_network(n_input=1, n_internal=2, n_output=1)
    # edge: 0(input) -> 1(internal)
    e = net.add_edge(source=0, target=1, weight=0.2)

    # prev_A에는 input의 이전 틱 값을 0으로 둔다.
    prev_A = zero_prev_A(net)
    prev_A[0] = 0

    # 하지만 현재 input A_j(t)는 1로 설정.
    clear_A(net)
    net.neurons[0].A = 1
    net.neurons[1].A = 1  # post 발화

    hebbian_update(net, prev_A, reward=1.0)

    # (b)라면: ΔW = 0.05*1*1*1 - 0.01*0.2 = 0.048 -> 0.248
    # (a)라면: ΔW = 0.05*1*0*1 - 0.002 = -0.002 -> 0.198
    assert net.edges[e.edge_id].weight == pytest.approx(0.248)


def test_hebbian_input_pre_no_update_when_current_a_zero():
    """input pre라도 현재 A_j(t)=0이면 co-fire 항이 0이 된다."""
    net = make_bare_network(n_input=1, n_internal=2, n_output=1)
    e = net.add_edge(source=0, target=1, weight=0.2)

    prev_A = zero_prev_A(net)
    prev_A[0] = 1  # 이전 틱에는 발화했었음

    clear_A(net)
    net.neurons[0].A = 0  # 하지만 현재는 0
    net.neurons[1].A = 1

    hebbian_update(net, prev_A, reward=1.0)

    # (b)라면: ΔW = 0 - 0.002 = -0.002 -> 0.198
    # (a)라면: ΔW = 0.05*1*1*1 - 0.002 = 0.048 -> 0.248
    assert net.edges[e.edge_id].weight == pytest.approx(0.198)