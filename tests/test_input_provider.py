"""External Signal Provider 단위 테스트 (7.5-1, 7.4 Step 2).

커버:
- 인터페이스 계약 (길이, 이진값)
- 순환 주기 경계 (t=0, 199, 200, 799, 800)
- Config 연동 (from_config)
- 입력 검증 (period, patterns)
"""

import pytest

from artificial_neuron.config import Config
from artificial_neuron.input_provider import (
    CyclicPatternProvider,
    InputProvider,
)


# ----------------------------------------------------------------------
# 생성자 검증
# ----------------------------------------------------------------------

def test_rejects_non_positive_period():
    with pytest.raises(ValueError):
        CyclicPatternProvider(n_input=2, patterns=((0, 0),), period=0)
    with pytest.raises(ValueError):
        CyclicPatternProvider(n_input=2, patterns=((0, 0),), period=-1)


def test_rejects_empty_patterns():
    with pytest.raises(ValueError):
        CyclicPatternProvider(n_input=2, patterns=(), period=200)


def test_rejects_pattern_length_mismatch():
    with pytest.raises(ValueError):
        CyclicPatternProvider(
            n_input=2, patterns=((0, 0, 0),), period=200
        )


def test_rejects_non_binary_pattern():
    with pytest.raises(ValueError):
        CyclicPatternProvider(
            n_input=2, patterns=((0, 2),), period=200
        )


# ----------------------------------------------------------------------
# from_config
# ----------------------------------------------------------------------

def test_from_config_uses_config_values():
    cfg = Config()
    provider = CyclicPatternProvider.from_config(cfg)
    assert provider.n_input == cfg.n_input
    # 첫 패턴은 (0, 0)
    assert provider.sample(0) == (0, 0)


# ----------------------------------------------------------------------
# 순환 패턴 정확성 (2.4)
# ----------------------------------------------------------------------

def make_provider() -> CyclicPatternProvider:
    return CyclicPatternProvider.from_config(Config())


def test_first_cycle():
    p = make_provider()
    assert p.sample(0) == (0, 0)
    assert p.sample(199) == (0, 0)
    assert p.sample(200) == (0, 1)
    assert p.sample(399) == (0, 1)
    assert p.sample(400) == (1, 0)
    assert p.sample(599) == (1, 0)
    assert p.sample(600) == (1, 1)
    assert p.sample(799) == (1, 1)


def test_wraps_around_at_800():
    p = make_provider()
    # t=800은 (t//200)=4, 4 mod 4 = 0 -> (0,0)
    assert p.sample(800) == (0, 0)
    assert p.sample(801) == (0, 0)
    # t=1000은 (1000//200)=5, 5 mod 4 = 1 -> (0,1)
    assert p.sample(1000) == (0, 1)


def test_boundary_every_period():
    p = make_provider()
    expected = {
        0: (0, 0),
        200: (0, 1),
        400: (1, 0),
        600: (1, 1),
        800: (0, 0),
        1000: (0, 1),
        1200: (1, 0),
        1400: (1, 1),
    }
    for t, pattern in expected.items():
        assert p.sample(t) == pattern


# ----------------------------------------------------------------------
# 계약 (길이, 이진값)
# ----------------------------------------------------------------------

def test_output_length_and_binary():
    p = make_provider()
    for t in range(0, 2000, 37):
        vec = p.sample(t)
        assert len(vec) == p.n_input
        for v in vec:
            assert v in (0, 1)


def test_negative_t_rejected():
    p = make_provider()
    with pytest.raises(ValueError):
        p.sample(-1)


# ----------------------------------------------------------------------
# 결정성
# ----------------------------------------------------------------------

def test_sample_is_deterministic():
    p = make_provider()
    for t in range(0, 1000, 13):
        assert p.sample(t) == p.sample(t)


def test_no_internal_state_between_calls():
    """순서와 무관하게 동일 t는 동일 결과를 반환해야 한다."""
    p = make_provider()
    # 뒤죽박죽 순서로 호출해도 결과는 t에만 의존.
    assert p.sample(500) == (1, 0)   # 500//200=2, 2%4=2 -> patterns[2]
    assert p.sample(100) == (0, 0)   # 100//200=0, 0%4=0 -> patterns[0]
    assert p.sample(500) == (1, 0)   # 앞선 호출과 동일


# ----------------------------------------------------------------------
# ABC 계약
# ----------------------------------------------------------------------

def test_input_provider_is_abstract():
    with pytest.raises(TypeError):
        InputProvider(n_input=2)  # type: ignore[abstract]


def test_cyclic_is_input_provider():
    assert issubclass(CyclicPatternProvider, InputProvider)