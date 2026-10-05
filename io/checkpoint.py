"""Checkpoint 저장/로드.

algorithm_spec.md 7.7 참조.

저장 항목:
    - Config 스냅샷 (전체)
    - tick (현재 틱 인덱스)
    - 뉴런 속성 (id, S, V, A, theta, is_input)
    - 엣지 리스트 (edge_id, source, target, weight)
    - edge_id_counter
    - gen_rng 상태
    - edge_cofire 누적 카운트 (로그 일관성)

정책:
    - 로그 버퍼 (activation_history, tick_records 등)는 저장하지 않는다.
      재개 시 빈 상태에서 새로 시작한다.
    - init_rng / input_rng는 시뮬레이션 도중 소비되지 않으므로 저장하지
      않는다. 초기 상태는 체크포인트의 뉴런/엣지 정보로 완전히 복원된다.
    - 기본 provider (CyclicPatternProvider)는 stateless이므로 상태를
      저장하지 않고 Config로부터 재생성한다.
    - RNG 상태는 JSON 직렬화를 위해 (version, list(internal), gauss_next)
      형태로 변환한다.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Tuple

from ..config import Config
from ..input_provider import CyclicPatternProvider, InputProvider
from ..network import Network
from ..neuron import Edge, Neuron
from ..simulator import Simulator


CHECKPOINT_FORMAT_VERSION = 1


# ----------------------------------------------------------------------
# RNG 상태 직렬화
# ----------------------------------------------------------------------

def _serialize_rng_state(rng: random.Random) -> List[Any]:
    version, internal_state, gauss_next = rng.getstate()
    return [version, list(internal_state), gauss_next]


def _deserialize_rng_state(
    data: List[Any],
) -> Tuple[int, Tuple[int, ...], Any]:
    version, internal_state, gauss_next = data
    return (version, tuple(internal_state), gauss_next)


# ----------------------------------------------------------------------
# 저장
# ----------------------------------------------------------------------

def save_checkpoint(sim: Simulator, path: Path) -> None:
    """Simulator의 상태를 JSON 파일로 저장한다.

    Args:
        sim: 저장할 시뮬레이터.
        path: 저장 경로. 부모 디렉터리는 자동 생성된다.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    net = sim.network
    neurons = [
        {
            "id": n.id,
            "S": n.S,
            "V": n.V,
            "A": n.A,
            "theta": n.theta,
            "is_input": n.is_input,
        }
        for n in net.neurons.values()
    ]
    edges = [
        {
            "edge_id": e.edge_id,
            "source": e.source,
            "target": e.target,
            "weight": e.weight,
        }
        for e in net.edges.values()
    ]

    payload = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "config": asdict(sim.config),
        "tick": sim.tick,
        "neurons": neurons,
        "edges": edges,
        "edge_id_counter": net.edge_id_counter,
        "gen_rng_state": _serialize_rng_state(sim.gen_rng),
        "edge_cofire": {str(k): v for k, v in sim.edge_cofire.items()},
        "enable_plasticity": sim.enable_plasticity,
        "snapshot_interval": sim.snapshot_interval,
    }

    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# ----------------------------------------------------------------------
# 로드
# ----------------------------------------------------------------------

def load_checkpoint(path: Path) -> Simulator:
    """체크포인트로부터 Simulator를 재구성한다.

    Args:
        path: 체크포인트 파일 경로.

    Returns:
        복원된 Simulator. 로그 버퍼는 비어 있고, tick은 저장 시점 값이다.

    Raises:
        ValueError: 지원하지 않는 format_version.
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    fmt = data.get("format_version")
    if fmt != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format_version: {fmt} "
            f"(expected {CHECKPOINT_FORMAT_VERSION})"
        )

    # Config 복원 (tuple 필드 재구성)
    cfg_dict = dict(data["config"])
    for key in ("input_patterns",):
        if key in cfg_dict and isinstance(cfg_dict[key], list):
            cfg_dict[key] = tuple(
                tuple(p) if isinstance(p, list) else p
                for p in cfg_dict[key]
            )
    cfg = Config(**cfg_dict)

    # 빈 네트워크 생성 후 상태 복원
    net = Network.empty(cfg)

    neurons = [
        Neuron(
            id=rec["id"],
            S=rec["S"],
            V=rec["V"],
            A=rec["A"],
            theta=rec["theta"],
            is_input=rec["is_input"],
        )
        for rec in data["neurons"]
    ]
    edges = [
        Edge(
            edge_id=rec["edge_id"],
            source=rec["source"],
            target=rec["target"],
            weight=rec["weight"],
        )
        for rec in data["edges"]
    ]
    net.load_state(neurons, edges, data["edge_id_counter"])

    # Provider 재생성 (기본 provider만 지원)
    provider: InputProvider = CyclicPatternProvider.from_config(cfg)

    # gen_rng 복원
    gen_rng = random.Random()
    gen_rng.setstate(_deserialize_rng_state(data["gen_rng_state"]))

    sim = Simulator(
        cfg,
        net,
        provider,
        gen_rng=gen_rng,
        enable_plasticity=data.get("enable_plasticity", True),
        snapshot_interval=data.get("snapshot_interval", 100),
    )
    sim.tick = data["tick"]
    sim.edge_cofire = {
        int(k): int(v) for k, v in data.get("edge_cofire", {}).items()
    }

    return sim