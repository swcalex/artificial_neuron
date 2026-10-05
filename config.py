"""모든 하이퍼파라미터 및 네트워크 상수를 단일 dataclass로 관리한다.

algorithm_spec.md 2.2, 2.3, 4.1, 4.2, 5.1, 5.2, 7.3-7을 참조.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Config:
    """전역 설정.

    Spec 7.3-7: 매직 넘버는 모두 이곳에서 관리한다.
    본 dataclass는 불변(frozen)이므로 시뮬레이션 도중 실수로 변경될 수 없다.
    """

    # ---- 2.3 네트워크 규모 ----
    n_input: int = 2
    n_internal: int = 17
    n_output: int = 1

    # ---- 2.3 E/I 비율 ----
    # 20% 억제성. N=20 기준 4개.
    inhibitory_ratio: float = 0.2

    # ---- 4.2 임계값 ----
    # 모든 비입력 뉴런에 고정.
    theta: float = 1.0

    # ---- 4.1 전하 감쇄 ----
    gamma: float = 0.9

    # ---- 5.1 Hebbian ----
    eta_u: float = 0.05
    eta_d: float = 0.01
    W_max: float = 2.0
    # 5.1의 R(t). 비지도 환경 기본값 1.0.
    reward: float = 1.0

    # ---- 5.2 구조 가변성 ----
    W_min: float = 0.01
    W_init: float = 0.2
    max_out_edges: int = 5          # self-loop, 중복 엣지 포함
    init_out_edges: int = 3         # 초기 뉴런당 시냅스 수
    T_gen: int = 100                # 신생 주기 (틱)

    # ---- 2.3 전하 초기화 ----
    V_init_mean: float = 0.1
    V_init_std: float = 0.05

    # ---- 2.4 기본 입력 공급자 ----
    input_pattern_period: int = 200
    input_patterns: Tuple[Tuple[int, ...], ...] = (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    )

    # ---- 2.3 RNG 스트림별 seed ----
    init_seed: int = 42
    input_seed: int = 43
    gen_seed: int = 44

    # ---- 파생 속성 (계산값) ----

    @property
    def n_total(self) -> int:
        return self.n_input + self.n_internal + self.n_output

    @property
    def n_inhibitory(self) -> int:
        """억제성 뉴런 개수. round(20 * 0.2) = 4."""
        return int(round(self.n_total * self.inhibitory_ratio))

    # ---- ID 매핑 (2.3 고정) ----

    @property
    def input_ids(self) -> range:
        return range(0, self.n_input)

    @property
    def internal_ids(self) -> range:
        return range(self.n_input, self.n_input + self.n_internal)

    @property
    def output_ids(self) -> range:
        return range(self.n_input + self.n_internal, self.n_total)

    @property
    def non_input_ids(self) -> range:
        """Input Neuron을 제외한 모든 ID. 생성 타깃 후보 집합(5.2)."""
        return range(self.n_input, self.n_total)