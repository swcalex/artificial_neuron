"""Network 자료구조 단위 테스트 (7.5-1, 7.5-2).

커버:
- ID 매핑 (2.3)
- E/I 비율 보존 (2.3)
- 입력 뉴런의 V/theta 미사용 (2.1)
- 전하 초기화 클립 (2.3)
- 초기 시냅스 개수 (2.3)
- edge_id 단조 증가 및 재사용 금지 (7.3-2)
- in/out 인덱스 일관성 (7.3-3)
- Input Neuron 타깃 금지 (2.1)
- seed 고정 시 완전 재현 (2.3)
"""

import random

import pytest

from artificial_neuron.config import Config
from artificial_neuron.network import Network


def make_network(config: Config | None = None) -> Network:
    cfg = config or Config()
    rng = random.Random(cfg.init_seed)
    return Network(cfg, rng)


# ----------------------------------------------------------------------
# 기본 구조
# ----------------------------------------------------------------------

def test_network_size():
    net = make_network()
    assert len(net.neurons) == 20
    assert set(net.neurons.keys()) == set(range(20))


def test_id_mapping():
    net = make_network()
    for nid in (0, 1):
        assert net.neurons[nid].is_input is True
    for nid in range(2, 19):
        assert net.neurons[nid].is_input is False
    assert net.neurons[19].is_input is False


def test_input_neurons_have_no_charge_or_theta():
    net = make_network()
    for nid in (0, 1):
        n = net.neurons[nid]
        assert n.V == 0.0
        assert n.theta == 0.0


def test_non_input_neurons_have_theta_default():
    net = make_network()
    for nid in range(2, 20):
        assert net.neurons[nid].theta == pytest.approx(1.0)


def test_all_charges_non_negative():
    net = make_network()
    for nid in range(2, 20):
        assert net.neurons[nid].V >= 0.0


# ----------------------------------------------------------------------
# E/I 비율
# ----------------------------------------------------------------------

def test_ei_ratio():
    net = make_network()
    excitatory = [n for n in net.neurons.values() if n.S == +1]
    inhibitory = [n for n in net.neurons.values() if n.S == -1]
    assert len(excitatory) == 16
    assert len(inhibitory) == 4


def test_ei_values_are_pm_one():
    net = make_network()
    for n in net.neurons.values():
        assert n.S in (+1, -1)


# ----------------------------------------------------------------------
# 초기 시냅스
# ----------------------------------------------------------------------

def test_initial_out_degree():
    net = make_network()
    for nid in range(20):
        assert net.out_degree(nid) == 3


def test_initial_edge_count():
    net = make_network()
    # 20 뉴런 x 3 = 60
    assert len(net.edges) == 60


def test_initial_edge_weights():
    net = make_network()
    for e in net.edges.values():
        assert e.weight == pytest.approx(0.2)


def test_no_input_neuron_as_target():
    net = make_network()
    for e in net.edges.values():
        assert e.target not in (0, 1)


# ----------------------------------------------------------------------
# edge_id
# ----------------------------------------------------------------------

def test_edge_id_uniqueness_and_monotonic():
    net = make_network()
    ids = sorted(net.edges.keys())
    assert ids == list(range(len(ids)))


def test_edge_id_not_reused_after_removal():
    net = make_network()
    before_counter = net.edge_id_counter

    # 첫 엣지 하나 제거
    some_eid = next(iter(net.edges))
    net.remove_edge(some_eid)
    assert some_eid not in net.edges

    # 새 엣지 추가 -> 기존 카운터에서 이어짐
    new_edge = net.add_edge(source=0, target=5, weight=0.2)
    assert new_edge.edge_id == before_counter
    assert new_edge.edge_id != some_eid


# ----------------------------------------------------------------------
# in/out 인덱스 일관성
# ----------------------------------------------------------------------

def test_in_out_consistency():
    net = make_network()
    for nid in range(20):
        for eid in net.out_edges[nid]:
            e = net.edges[eid]
            assert e.source == nid
            assert eid in net.in_edges[e.target]
        for eid in net.in_edges[nid]:
            e = net.edges[eid]
            assert e.target == nid
            assert eid in net.out_edges[e.source]


def test_in_out_consistency_after_removal():
    net = make_network()
    eid = next(iter(net.edges))
    e = net.edges[eid]
    src, tgt = e.source, e.target

    net.remove_edge(eid)

    assert eid not in net.out_edges[src]
    assert eid not in net.in_edges[tgt]
    assert eid not in net.edges


# ----------------------------------------------------------------------
# add_edge 제약
# ----------------------------------------------------------------------

def test_add_edge_rejects_input_target():
    net = make_network()
    with pytest.raises(ValueError):
        net.add_edge(source=5, target=0, weight=0.2)
    with pytest.raises(ValueError):
        net.add_edge(source=5, target=1, weight=0.2)


def test_add_edge_allows_self_loop():
    net = make_network()
    e = net.add_edge(source=5, target=5, weight=0.2)
    assert e.source == 5 and e.target == 5


def test_add_edge_allows_duplicate_target():
    net = make_network()
    a = net.add_edge(source=5, target=7, weight=0.2)
    b = net.add_edge(source=5, target=7, weight=0.2)
    assert a.edge_id != b.edge_id
    assert net.out_degree(5) == 5  # 3(초기) + 2


def test_add_edge_unknown_neuron():
    net = make_network()
    with pytest.raises(KeyError):
        net.add_edge(source=99, target=5, weight=0.2)
    with pytest.raises(KeyError):
        net.add_edge(source=5, target=99, weight=0.2)


# ----------------------------------------------------------------------
# 재현성 (2.3 RNG 호출 순서 고정)
# ----------------------------------------------------------------------

def test_seed_reproducibility():
    net_a = make_network()
    net_b = make_network()

    # 뉴런 속성
    for nid in range(20):
        a, b = net_a.neurons[nid], net_b.neurons[nid]
        assert a.S == b.S
        assert a.V == pytest.approx(b.V)

    # 엣지
    assert set(net_a.edges.keys()) == set(net_b.edges.keys())
    for eid in net_a.edges:
        ea = net_a.edges[eid]
        eb = net_b.edges[eid]
        assert (ea.source, ea.target) == (eb.source, eb.target)
        assert ea.weight == pytest.approx(eb.weight)


def test_different_seed_gives_different_network():
    cfg_a = Config(init_seed=42)
    cfg_b = Config(init_seed=9999)
    net_a = make_network(cfg_a)
    net_b = make_network(cfg_b)

    # 완전히 동일할 확률은 사실상 0.
    edge_pairs_a = sorted((e.source, e.target) for e in net_a.edges.values())
    edge_pairs_b = sorted((e.source, e.target) for e in net_b.edges.values())
    assert edge_pairs_a != edge_pairs_b