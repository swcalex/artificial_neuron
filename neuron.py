"""Neuron, Edge 데이터 클래스.

algorithm_spec.md 2.2, 7.3-1, 7.3-2, 7.3-3 참조.

설계 메모 (7.3-1, 7.3-3):
- `Network`가 그래프의 유일한 소유자다.
- `Neuron`은 순수 상태 보유자이며, 자신의 인접 리스트를 직접 수정하지 않는다.
- in/out adjacency는 `Network`가 관리한다.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Edge:
    """개별 시냅스 엣지.

    중복 엣지가 허용되므로 (source, target)을 key로 쓸 수 없다.
    `edge_id`는 `Network`가 단조 증가 카운터로 발급하며 재사용하지 않는다.
    """

    edge_id: int
    source: int
    target: int
    weight: float


@dataclass
class Neuron:
    """뉴런 상태 보유자.

    Attributes:
        id: 고유번호 (0 ~ N-1).
        S: 고유 전달 강도 (+1 흥분성 / -1 억제성). 학습으로 변하지 않는다.
        V: 전하값. 입력 뉴런은 사용하지 않으므로 0.0으로 둔다.
           (semantic guard는 `is_input` 플래그로 수행한다.)
        A: 활성화 상태 (0 또는 1). 매 틱 Step 2에서 갱신된다.
        theta: 발화 임계값. 입력 뉴런은 0.0 (사용 안 함).
        is_input: 입력 뉴런 여부.
    """

    id: int
    S: int
    V: float
    A: int
    theta: float
    is_input: bool