"""External Signal Provider 인터페이스 및 기본 순환 공급자.

algorithm_spec.md 2.1, 2.4, 7.3-8 참조.

설계 의도:
- 입력 뉴런의 활성화 벡터를 매 틱 공급한다.
- 시뮬레이터 본체가 특정 공급자 구현에 결합되지 않도록
  `sample(t) -> Tuple[int, ...]` 인터페이스로 추상화한다.
- 기본 공급자는 결정적이므로 `input_rng`를 소비하지 않는다 (2.4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from .config import Config


class InputProvider(ABC):
    """External Signal Provider 인터페이스 (7.3-8).

    구현체는 반드시 다음 계약을 만족해야 한다.

    Contract:
        - `sample(t)`는 길이 `n_input`의 이진 벡터를 반환한다.
        - 각 원소는 0 또는 1이어야 한다.
        - 동일 `t`에 대해 반복 호출하면 동일한 값을 반환해야 한다
          (stateless 권장). 상태를 갖는 공급자는 재현성 보장을 위해
          내부 상태를 체크포인트에 저장할 책임이 있다.
    """

    def __init__(self, n_input: int) -> None:
        self._n_input = n_input

    @property
    def n_input(self) -> int:
        return self._n_input

    @abstractmethod
    def sample(self, t: int) -> Tuple[int, ...]:
        """틱 `t`에 공급할 입력 활성화 벡터를 반환한다."""
        raise NotImplementedError


class CyclicPatternProvider(InputProvider):
    """기본 순환 공급자 (2.4).

    매 `period` 틱마다 `patterns`의 다음 패턴으로 순환한다.

        pattern_index = (t // period) mod len(patterns)

    기본 설정 기준 (period=200, patterns 4개):
        t ∈ [0,   200): (0, 0)
        t ∈ [200, 400): (0, 1)
        t ∈ [400, 600): (1, 0)
        t ∈ [600, 800): (1, 1)
        t = 800 부터 다시 (0, 0)

    본 공급자는 결정적이며 상태를 갖지 않는다.
    """

    def __init__(
        self,
        n_input: int,
        patterns: Tuple[Tuple[int, ...], ...],
        period: int,
    ) -> None:
        super().__init__(n_input)

        if period <= 0:
            raise ValueError(f"period must be positive, got {period}")
        if not patterns:
            raise ValueError("patterns must be non-empty")
        for i, p in enumerate(patterns):
            if len(p) != n_input:
                raise ValueError(
                    f"pattern[{i}] length {len(p)} != n_input {n_input}"
                )
            for v in p:
                if v not in (0, 1):
                    raise ValueError(
                        f"pattern[{i}] contains non-binary value: {v}"
                    )

        self._patterns = patterns
        self._period = period

    @classmethod
    def from_config(cls, config: Config) -> "CyclicPatternProvider":
        return cls(
            n_input=config.n_input,
            patterns=config.input_patterns,
            period=config.input_pattern_period,
        )

    def sample(self, t: int) -> Tuple[int, ...]:
        if t < 0:
            raise ValueError(f"t must be non-negative, got {t}")
        idx = (t // self._period) % len(self._patterns)
        return self._patterns[idx]