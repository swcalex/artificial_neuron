"""Network: 뉴런 컬렉션, 엣지 관리, in/out 인접 인덱스, edge_id 발급.

algorithm_spec.md 2.3, 5.2, 7.3-1/2/3 참조.

소유권 원칙 (7.3-1):
- 뉴런·엣지의 생성·삭제·조회는 반드시 `Network`를 통한다.
- `Neuron`은 자신의 인접 리스트를 직접 수정하지 않는다.

in/out 일관성 (7.3-3):
- 엣지 생성·소멸 시 `out_edges`와 `in_edges`를 원자적으로 갱신한다.
- edge_id는 단조 증가 카운터로 발급하며, 소멸 후 재사용하지 않는다.
"""

from __future__ import annotations

import random
from typing import Dict, List

from .config import Config
from .neuron import Edge, Neuron


class Network:
    """뉴런 그래프의 유일한 소유자.

    Public read-only by convention:
        neurons: Dict[int, Neuron]
        edges:   Dict[int, Edge]

    Mutation API:
        add_edge(source, target, weight) -> Edge
        remove_edge(edge_id) -> None

    Lookup API:
        get_out_edges(nid) -> List[Edge]
        get_in_edges(nid)  -> List[Edge]
        out_degree(nid)    -> int
        in_degree(nid)     -> int
    """

    def __init__(self, config: Config, init_rng: random.Random) -> None:
        self.config = config

        self.neurons: Dict[int, Neuron] = {}
        self.edges: Dict[int, Edge] = {}
        self.out_edges: Dict[int, List[int]] = {}
        self.in_edges: Dict[int, List[int]] = {}
        self._edge_id_counter: int = 0

        self._init_neurons(init_rng)
        self._init_edges(init_rng)

    # ------------------------------------------------------------------
    # 초기화 (2.3)
    # ------------------------------------------------------------------

    def _init_neurons(self, init_rng: random.Random) -> None:
        """뉴런 생성 + E/I 배정 + 전하 초기화.

        RNG 호출 순서 (2.3 고정):
          1. rng.shuffle(ids)          -> E/I 배정
          2. rng.gauss(...) x (N-2)    -> V_i(0) 샘플링 (ID 오름차순)
        """
        cfg = self.config
        n = cfg.n_total

        # 1) E/I 배정: 전체 ID 셔플 후 앞 n_inhibitory 개를 억제성으로.
        ids = list(range(n))
        init_rng.shuffle(ids)
        inhibitory = set(ids[: cfg.n_inhibitory])

        # 2) 뉴런 생성 (ID 오름차순). 입력 뉴런 제외 V 샘플링.
        for nid in range(n):
            S = -1 if nid in inhibitory else +1
            is_input = nid in cfg.input_ids
            if is_input:
                # 입력 뉴런은 V, theta를 보유하지 않는다 (2.1).
                V = 0.0
                theta = 0.0
            else:
                sample = init_rng.gauss(cfg.V_init_mean, cfg.V_init_std)
                # 2.3: 음수는 재샘플링 없이 0으로 클립.
                V = max(0.0, sample)
                theta = cfg.theta

            self.neurons[nid] = Neuron(
                id=nid, S=S, V=V, A=0, theta=theta, is_input=is_input
            )
            self.out_edges[nid] = []
            self.in_edges[nid] = []

    def _init_edges(self, init_rng: random.Random) -> None:
        """초기 시냅스 생성 (2.3).

        RNG 호출 순서 (고정):
          - 소스 ID 오름차순 (0 ~ N-1)
          - 각 소스마다 init_out_edges(=3)회 rng.choice(non_input_ids)
        """
        cfg = self.config
        candidates = list(cfg.non_input_ids)
        for src in range(cfg.n_total):
            for _ in range(cfg.init_out_edges):
                tgt = init_rng.choice(candidates)
                self.add_edge(source=src, target=tgt, weight=cfg.W_init)

    # ------------------------------------------------------------------
    # 엣지 조작 (7.3-2, 7.3-3)
    # ------------------------------------------------------------------

    def add_edge(self, source: int, target: int, weight: float) -> Edge:
        """신규 엣지 생성. in/out 인덱스 모두 갱신.

        Raises:
            ValueError: target이 Input Neuron인 경우 (2.1 제약).
            KeyError:   source 또는 target이 미등록 뉴런인 경우.
        """
        if target in self.config.input_ids:
            raise ValueError(
                f"Input Neuron cannot be an edge target (target={target})."
            )
        if source not in self.neurons:
            raise KeyError(f"Unknown source neuron id: {source}")
        if target not in self.neurons:
            raise KeyError(f"Unknown target neuron id: {target}")

        eid = self._edge_id_counter
        self._edge_id_counter += 1

        edge = Edge(edge_id=eid, source=source, target=target, weight=weight)
        self.edges[eid] = edge
        self.out_edges[source].append(eid)
        self.in_edges[target].append(eid)
        return edge

    def remove_edge(self, edge_id: int) -> None:
        """엣지 소멸. out_edges, in_edges 양쪽에서 제거.

        edge_id는 재사용하지 않는다 (7.3-2).
        """
        edge = self.edges.pop(edge_id)  # KeyError if missing
        self.out_edges[edge.source].remove(edge_id)
        self.in_edges[edge.target].remove(edge_id)

    # ------------------------------------------------------------------
    # 조회
    # ------------------------------------------------------------------

    def get_out_edges(self, nid: int) -> List[Edge]:
        return [self.edges[eid] for eid in self.out_edges[nid]]

    def get_in_edges(self, nid: int) -> List[Edge]:
        return [self.edges[eid] for eid in self.in_edges[nid]]

    def out_degree(self, nid: int) -> int:
        return len(self.out_edges[nid])

    def in_degree(self, nid: int) -> int:
        return len(self.in_edges[nid])

    # ------------------------------------------------------------------
    # 속성
    # ------------------------------------------------------------------

    @property
    def edge_id_counter(self) -> int:
        """다음 발급될 edge_id (체크포인트 저장 대상, 7.7)."""
        return self._edge_id_counter

    # ------------------------------------------------------------------
    # step 8 추가
    # ------------------------------------------------------------------

    @classmethod
    def empty(cls, config: Config) -> "Network":
        """초기화 없이 빈 네트워크를 생성한다 (체크포인트 로드용).

        뉴런/엣지/카운터가 모두 비어 있는 상태로 시작하며, 이후
        `load_state()`로 상태를 채운다.
        """
        net = cls.__new__(cls)
        net.config = config
        net.neurons = {}
        net.edges = {}
        net.out_edges = {}
        net.in_edges = {}
        net._edge_id_counter = 0
        return net

    def load_state(
        self,
        neurons: List[Neuron],
        edges: List[Edge],
        edge_id_counter: int,
    ) -> None:
        """체크포인트로부터 상태를 원자적으로 복원한다.

        기존 상태를 완전히 덮어쓴다. `in_edges`/`out_edges` 인덱스도
        함께 재구축하여 일관성을 보장한다.

        Args:
            neurons: 복원할 뉴런 리스트.
            edges: 복원할 엣지 리스트.
            edge_id_counter: 다음 발급될 edge_id.
        """
        self.neurons = {n.id: n for n in neurons}
        self.edges = {e.edge_id: e for e in edges}
        self.out_edges = {nid: [] for nid in self.neurons}
        self.in_edges = {nid: [] for nid in self.neurons}
        for e in edges:
            self.out_edges[e.source].append(e.edge_id)
            self.in_edges[e.target].append(e.edge_id)
        self._edge_id_counter = edge_id_counter