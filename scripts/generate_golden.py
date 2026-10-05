"""Golden snapshot 생성 스크립트 (spec 7.5-3).

Config나 시뮬레이터 로직이 의도적으로 변경되었을 때 재실행하여
회귀 스냅샷을 갱신한다.

사용 (프로젝트 루트에서):
    python scripts/generate_golden.py
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

# 어디서 실행해도 `artificial_neuron` 패키지를 찾도록 sys.path 보정
def _find_project_root() -> Path:
    """artificial_neuron 패키지의 부모 디렉터리를 찾는다.

    이 스크립트가 `<project>/scripts/` 아래에 있든
    `<project>/artificial_neuron/scripts/` 아래에 있든 동작한다.
    """
    here = Path(__file__).resolve()
    for candidate in here.parents:
        if (candidate / "artificial_neuron" / "__init__.py").exists():
            return candidate
    raise RuntimeError(
        "artificial_neuron package not found. "
        "Ensure the script is inside the project tree."
    )


_PROJECT_ROOT = _find_project_root()
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from artificial_neuron.config import Config
from artificial_neuron.input_provider import CyclicPatternProvider
from artificial_neuron.network import Network
from artificial_neuron.simulator import Simulator


# Golden test와 반드시 동일해야 하는 설정
GOLDEN_CONFIG = Config(
    n_input=1,
    n_internal=3,
    n_output=1,
    input_patterns=((0,), (1,)),
    input_pattern_period=5,
    init_seed=12345,
    input_seed=23456,
    gen_seed=34567,
)

TICKS = 20
SNAPSHOT_INTERVAL = 10
PLASTICITY_ENABLED = True

GOLDEN_PATH = (
    _PROJECT_ROOT
    / "artificial_neuron"
    / "tests"
    / "golden"
    / "snapshot_n5_t20.json"
)


def main() -> None:
    cfg = GOLDEN_CONFIG
    init_rng = random.Random(cfg.init_seed)
    network = Network(cfg, init_rng)
    provider = CyclicPatternProvider.from_config(cfg)
    sim = Simulator(
        cfg,
        network,
        provider,
        gen_rng=random.Random(cfg.gen_seed),
        enable_plasticity=PLASTICITY_ENABLED,
        snapshot_interval=SNAPSHOT_INTERVAL,
    )
    sim.run(TICKS)

    final_edges = sorted(
        [
            [e.edge_id, e.source, e.target, round(e.weight, 10)]
            for e in network.edges.values()
        ],
        key=lambda row: row[0],
    )

    snapshot = {
        "description": (
            "Golden snapshot for regression detection (spec 7.5-3). "
            "Regenerate via `python scripts/generate_golden.py`."
        ),
        "ticks": TICKS,
        "snapshot_interval": SNAPSHOT_INTERVAL,
        "plasticity_enabled": PLASTICITY_ENABLED,
        "seed": {
            "init": cfg.init_seed,
            "input": cfg.input_seed,
            "gen": cfg.gen_seed,
        },
        "config": {
            "n_input": cfg.n_input,
            "n_internal": cfg.n_internal,
            "n_output": cfg.n_output,
            "input_patterns": [list(p) for p in cfg.input_patterns],
            "input_pattern_period": cfg.input_pattern_period,
            "inhibitory_ratio": cfg.inhibitory_ratio,
        },
        "activation_history": [
            [int(x) for x in row] for row in sim.activation_history
        ],
        "voltage_history": [
            [round(float(x), 10) for x in row] for row in sim.voltage_history
        ],
        "output_history": [
            [int(x) for x in row] for row in sim.output_history
        ],
        "final_edges": final_edges,
        "final_edge_id_counter": network.edge_id_counter,
    }

    GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GOLDEN_PATH, "w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)
    print(f"[golden] wrote {GOLDEN_PATH}")


if __name__ == "__main__":
    main()