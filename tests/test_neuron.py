"""Neuron / Edge 자료구조 단위 테스트 (7.5-1)."""

import pytest

from artificial_neuron.neuron import Edge, Neuron


def test_edge_fields():
    e = Edge(edge_id=7, source=1, target=5, weight=0.3)
    assert e.edge_id == 7
    assert e.source == 1
    assert e.target == 5
    assert e.weight == pytest.approx(0.3)


def test_neuron_fields():
    n = Neuron(id=3, S=-1, V=0.12, A=1, theta=1.0, is_input=False)
    assert n.id == 3
    assert n.S == -1
    assert n.V == pytest.approx(0.12)
    assert n.A == 1
    assert n.theta == pytest.approx(1.0)
    assert n.is_input is False