"""Simulator: tick loop + profiling + 로깅.

algorithm_spec.md 3, 3.1, 5.1, 5.4, 6.4, 6.6, 6.8, 7.3-5 참조.

Step 1 -> Step 2 -> Step 3 -> Step 4의 완전 루프를 순회한다.

prev_A 캡처 (5.1 각주):
    Step 3 Hebbian의 co-fire 항은 A_j(t-1)을 사용한다. 그런데 Step 2가
    모든 비입력 뉴런의 A를 t 상태로 덮어쓰므로, t-1 값은 어디에도
    남아 있지 않다. 따라서 Simulator가 매 틱 Step 1 실행 이전에
    prev_A 스냅샷을 캡처하여 Step 3에 전달한다.

co-fire 누적 (6.6):
    Hebbian과 동일한 A_j^(pre)(t) 규칙을 사용하여, co-fire가 발생한
    엣지의 누적 횟수를 매 틱 갱신한다. 엣지 스냅샷 CSV에 cofire_count
    컬럼으로 포함된다.

로그 저장 (6.8, 7a):
    `save_run(output_dir)`는 다음을 생성한다.

        output_dir/
        ├── run_meta.json
        └── logs/
            ├── neuron_meta.csv           # 뉴런별 S_i, is_input, theta
            ├── activation_history.csv    # 매 틱 전체 뉴런 A(t)   (6.1)
            ├── voltage_history.csv       # 매 틱 전체 뉴런 V(t)   (6.3)
            ├── output_history.csv        # 매 틱 출력 벡터       (6.4)
            ├── activation_summary.csv    # 그룹별 발화율         (6.1)
            ├── tick_timing.csv           # Step별 소요 시간      (6.4)
            ├── weight_stats.csv          # 매 틱 가중치 통계     (6.6)
            ├── edge_events.csv           # prune/generate 이벤트 (6.2/6.6)
            └── edge_snapshots/
                ├── edges_t0000.csv       # 100틱 간격 스냅샷
                └── ...                   # 컬럼: edge_id, source,
                                          #       target, weight,
                                          #       cofire_count

    분석 모듈은 `run_meta.json`과 `logs/`만 읽어 독립 실행된다 (6.8).

설계 메모 (7.3-5):
    Simulator는 Step 순서를 조정하는 코디네이터다. 각 Step은
    dynamics/plasticity의 순수 함수이며, network를 in-place로 mutate한다.
    Simulator 자체는 다음 틱으로 넘어가기 위한 최소 상태만 보유한다.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import Config
from .dynamics import (
    step1_update_inputs,
    step2_update_dynamics,
    step4_collect_outputs,
)
from .input_provider import InputProvider
from .network import Network
from .plasticity import step3_apply_plasticity


# ----------------------------------------------------------------------
# 데이터 컨테이너
# ----------------------------------------------------------------------


@dataclass
class TickRecord:
    """tick_timing.csv 한 행에 대응 (6.4)."""

    tick: int
    step1_ms: float
    step2_ms: float
    step3_ms: float
    step4_ms: float
    total_ms: float


@dataclass
class EdgeEvent:
    """Step 3의 prune / generate 이벤트 (6.2, 6.6 원자료)."""

    tick: int
    event_type: str  # "prune" | "generate"
    edge_id: int
    source: int
    target: int
    weight: float


@dataclass
class WeightStats:
    """매 틱 가중치 분포 통계 (6.6)."""

    tick: int
    edge_count: int
    mean: float
    std: float
    min: float
    max: float
    ratio_W_le_0_01: float
    ratio_W_ge_1_99: float


class Simulator:
    """Step 1 -> 2 -> 3 -> 4 순회 코디네이터."""

    def __init__(
        self,
        config: Config,
        network: Network,
        provider: InputProvider,
        gen_rng: Optional[random.Random] = None,
        enable_plasticity: bool = True,
        snapshot_interval: int = 100,
    ) -> None:
        if snapshot_interval <= 0:
            raise ValueError(
                f"snapshot_interval must be positive, got {snapshot_interval}"
            )

        self.config = config
        self.network = network
        self.provider = provider

        if gen_rng is None:
            gen_rng = random.Random(config.gen_seed)
        self.gen_rng = gen_rng
        self.enable_plasticity = enable_plasticity
        self.snapshot_interval = snapshot_interval

        self.tick: int = 0
        self.start_time: datetime = datetime.now()

        # 로그 버퍼
        self.tick_records: List[TickRecord] = []
        self.activation_history: List[Tuple[int, ...]] = []
        self.voltage_history: List[Tuple[float, ...]] = []
        self.output_history: List[Tuple[int, ...]] = []
        self.prune_history: List[List[int]] = []
        self.generate_history: List[List[int]] = []
        self.weight_stats_history: List[WeightStats] = []
        self.edge_events: List[EdgeEvent] = []
        self.edge_cofire: Dict[int, int] = {}
        # edge_snapshots: 각 원소는
        #   (tick, ((edge_id, source, target, weight, cofire_count), ...))
        self.edge_snapshots: List[
            Tuple[int, Tuple[Tuple[int, int, int, float, int], ...]]
        ] = []

    # ------------------------------------------------------------------
    # 단일 틱
    # ------------------------------------------------------------------

    def step(self) -> Tuple[int, ...]:
        """1틱 실행. Step 1 -> Step 2 -> Step 3 -> Step 4."""
        t = self.tick

        # prev_A 스냅샷 (5.1 각주). Step 1이 입력 뉴런 A를 덮어쓰기 이전.
        prev_A: Dict[int, int] = {
            nid: n.A for nid, n in self.network.neurons.items()
        }

        t_start = time.perf_counter()

        step1_update_inputs(self.network, t, self.provider)
        t_after_s1 = time.perf_counter()

        step2_update_dynamics(self.network)
        t_after_s2 = time.perf_counter()

        # co-fire 누적 (6.6). Hebbian과 동일한 A_j^(pre)(t) 규칙.
        if self.enable_plasticity:
            for edge in self.network.edges.values():
                src_neuron = self.network.neurons[edge.source]
                if src_neuron.is_input:
                    a_pre = src_neuron.A
                else:
                    a_pre = prev_A[edge.source]
                a_post = self.network.neurons[edge.target].A
                if a_pre == 1 and a_post == 1:
                    self.edge_cofire[edge.edge_id] = (
                        self.edge_cofire.get(edge.edge_id, 0) + 1
                    )

        pruned: List[int] = []
        generated: List[int] = []
        if self.enable_plasticity:
            before_edges = {
                eid: (e.source, e.target, e.weight)
                for eid, e in self.network.edges.items()
            }
            step3_apply_plasticity(
                self.network, t, prev_A, self.gen_rng
            )
            after_edges = {
                eid: (e.source, e.target, e.weight)
                for eid, e in self.network.edges.items()
            }
            pruned = sorted(set(before_edges) - set(after_edges))
            generated = sorted(set(after_edges) - set(before_edges))

            for eid in pruned:
                src, tgt, w = before_edges[eid]
                self.edge_events.append(
                    EdgeEvent(
                        tick=t,
                        event_type="prune",
                        edge_id=eid,
                        source=src,
                        target=tgt,
                        weight=w,
                    )
                )
            for eid in generated:
                src, tgt, w = after_edges[eid]
                self.edge_events.append(
                    EdgeEvent(
                        tick=t,
                        event_type="generate",
                        edge_id=eid,
                        source=src,
                        target=tgt,
                        weight=w,
                    )
                )
        t_after_s3 = time.perf_counter()

        out = step4_collect_outputs(self.network)
        t_after_s4 = time.perf_counter()

        def ms(a: float, b: float) -> float:
            return (b - a) * 1000.0

        self.tick_records.append(
            TickRecord(
                tick=t,
                step1_ms=ms(t_start, t_after_s1),
                step2_ms=ms(t_after_s1, t_after_s2),
                step3_ms=ms(t_after_s2, t_after_s3),
                step4_ms=ms(t_after_s3, t_after_s4),
                total_ms=ms(t_start, t_after_s4),
            )
        )

        # 상태 캡처
        act = tuple(
            self.network.neurons[nid].A for nid in range(self.config.n_total)
        )
        volt = tuple(
            self.network.neurons[nid].V for nid in range(self.config.n_total)
        )
        self.activation_history.append(act)
        self.voltage_history.append(volt)
        self.output_history.append(out)
        self.prune_history.append(pruned)
        self.generate_history.append(generated)

        # 가중치 통계 (6.6)
        self.weight_stats_history.append(self._compute_weight_stats(t))

        # 엣지 스냅샷 (6.2/6.6, 100틱 간격)
        if t % self.snapshot_interval == 0:
            snap = tuple(
                (
                    e.edge_id,
                    e.source,
                    e.target,
                    e.weight,
                    self.edge_cofire.get(e.edge_id, 0),
                )
                for e in self.network.edges.values()
            )
            self.edge_snapshots.append((t, snap))

        self.tick += 1
        return out

    def run(self, n_ticks: int) -> None:
        """`n_ticks` 만큼 step을 반복."""
        if n_ticks < 0:
            raise ValueError(f"n_ticks must be non-negative, got {n_ticks}")
        for _ in range(n_ticks):
            self.step()

    # ------------------------------------------------------------------
    # 지표
    # ------------------------------------------------------------------

    def firing_rate_series(self) -> List[float]:
        n = self.config.n_total
        return [sum(a) / n for a in self.activation_history]

    def firing_rate_series_by_group(self) -> Dict[str, List[float]]:
        cfg = self.config
        input_ids = list(cfg.input_ids)
        internal_ids = list(cfg.internal_ids)
        output_ids = list(cfg.output_ids)

        series: Dict[str, List[float]] = {
            "input": [],
            "internal": [],
            "output": [],
            "total": [],
        }
        for act in self.activation_history:
            series["input"].append(
                sum(act[nid] for nid in input_ids) / len(input_ids)
            )
            series["internal"].append(
                sum(act[nid] for nid in internal_ids) / len(internal_ids)
            )
            series["output"].append(
                sum(act[nid] for nid in output_ids) / len(output_ids)
            )
            series["total"].append(sum(act) / cfg.n_total)
        return series

    def firing_rate_series_by_ei(self) -> Dict[str, List[float]]:
        e_ids = [nid for nid, n in self.network.neurons.items() if n.S == +1]
        i_ids = [nid for nid, n in self.network.neurons.items() if n.S == -1]

        exc_series: List[float] = []
        inh_series: List[float] = []
        for act in self.activation_history:
            exc_series.append(sum(act[nid] for nid in e_ids) / len(e_ids))
            inh_series.append(sum(act[nid] for nid in i_ids) / len(i_ids))
        return {"excitatory": exc_series, "inhibitory": inh_series}

    def mean_firing_rate(self) -> float:
        if not self.activation_history:
            return 0.0
        total = sum(sum(a) for a in self.activation_history)
        return total / (len(self.activation_history) * self.config.n_total)

    def mean_firing_rate_by_group(self) -> Dict[str, float]:
        series = self.firing_rate_series_by_group()
        return {
            k: (sum(v) / len(v) if v else 0.0) for k, v in series.items()
        }

    def windowed_firing_rate(
        self, window_size: int
    ) -> Dict[str, List[Tuple[int, float]]]:
        if window_size <= 0:
            raise ValueError(
                f"window_size must be positive, got {window_size}"
            )
        T = len(self.activation_history)
        series = self.firing_rate_series_by_group()
        result: Dict[str, List[Tuple[int, float]]] = {
            "total": [], "input": [], "internal": [], "output": []
        }
        for start in range(0, T, window_size):
            end = min(start + window_size, T)
            length = end - start
            for key in ("total", "input", "internal", "output"):
                chunk = series[key][start:end]
                mean = sum(chunk) / length if length > 0 else 0.0
                result[key].append((start, mean))
        return result

    def timing_summary(self) -> Dict[str, float]:
        if not self.tick_records:
            return {
                "mean_ms": 0.0, "p50_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0,
                "ticks_per_second": 0.0, "total_ticks": 0,
            }
        totals = sorted(rec.total_ms for rec in self.tick_records)
        n = len(totals)
        mean_ms = sum(totals) / n
        p50_ms = totals[n // 2]
        p99_ms = totals[min(n - 1, int(n * 0.99))]
        max_ms = totals[-1]
        tps = 1000.0 / mean_ms if mean_ms > 0 else float("inf")
        return {
            "mean_ms": mean_ms, "p50_ms": p50_ms, "p99_ms": p99_ms,
            "max_ms": max_ms, "ticks_per_second": tps, "total_ticks": n,
        }

    # ------------------------------------------------------------------
    # 내부: 가중치 통계
    # ------------------------------------------------------------------

    def _compute_weight_stats(self, t: int) -> WeightStats:
        weights = [e.weight for e in self.network.edges.values()]
        if not weights:
            return WeightStats(
                tick=t, edge_count=0, mean=0.0, std=0.0, min=0.0, max=0.0,
                ratio_W_le_0_01=0.0, ratio_W_ge_1_99=0.0,
            )
        n = len(weights)
        mean = sum(weights) / n
        var = sum((w - mean) ** 2 for w in weights) / n
        std = var ** 0.5
        ratio_le = sum(1 for w in weights if w <= 0.01) / n
        ratio_ge = sum(1 for w in weights if w >= 1.99) / n
        return WeightStats(
            tick=t, edge_count=n, mean=mean, std=std,
            min=min(weights), max=max(weights),
            ratio_W_le_0_01=ratio_le, ratio_W_ge_1_99=ratio_ge,
        )

    # ------------------------------------------------------------------
    # 저장 (7a, 7b)
    # ------------------------------------------------------------------

    def save_run(self, output_dir: Path) -> None:
        """모든 분석용 원자료를 output_dir에 저장 (6.8)."""
        output_dir = Path(output_dir)
        logs_dir = output_dir / "logs"
        snapshots_dir = logs_dir / "edge_snapshots"
        snapshots_dir.mkdir(parents=True, exist_ok=True)

        self._save_run_meta(output_dir)
        self._save_neuron_meta(logs_dir)
        self._save_activation_history(logs_dir)
        self._save_voltage_history(logs_dir)
        self._save_output_history(logs_dir)
        self._save_activation_summary(logs_dir)
        self._save_tick_timing(logs_dir)
        self._save_weight_stats(logs_dir)
        self._save_edge_events(logs_dir)
        self._save_edge_snapshots(snapshots_dir)

    def _save_run_meta(self, output_dir: Path) -> None:
        meta = {
            "spec_version": "algorithm_spec.md",
            "project_version": "v0.0.0",
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_ticks": self.tick,
            "snapshot_interval": self.snapshot_interval,
            "plasticity_enabled": self.enable_plasticity,
            "provider": type(self.provider).__name__,
            "seeds": {
                "init_seed": self.config.init_seed,
                "input_seed": self.config.input_seed,
                "gen_seed": self.config.gen_seed,
            },
            "config": asdict(self.config),
            "final_neuron_count": len(self.network.neurons),
            "final_edge_count": len(self.network.edges),
            "total_pruned_events": sum(len(p) for p in self.prune_history),
            "total_generated_events": sum(
                len(g) for g in self.generate_history
            ),
        }
        with open(output_dir / "run_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _save_neuron_meta(self, logs_dir: Path) -> None:
        """뉴런별 정적 속성 (분석 모듈이 E/I, 입력 여부를 알기 위함)."""
        path = logs_dir / "neuron_meta.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["neuron_id", "S", "is_input", "theta"])
            for nid in range(self.config.n_total):
                n = self.network.neurons[nid]
                w.writerow([nid, n.S, int(n.is_input), n.theta])

    def _save_activation_history(self, logs_dir: Path) -> None:
        n = self.config.n_total
        path = logs_dir / "activation_history.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tick"] + [f"n{i}" for i in range(n)])
            for t, act in enumerate(self.activation_history):
                w.writerow([t] + list(act))

    def _save_voltage_history(self, logs_dir: Path) -> None:
        n = self.config.n_total
        path = logs_dir / "voltage_history.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tick"] + [f"n{i}" for i in range(n)])
            for t, volt in enumerate(self.voltage_history):
                w.writerow([t] + list(volt))

    def _save_output_history(self, logs_dir: Path) -> None:
        n_out = self.config.n_output
        path = logs_dir / "output_history.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tick"] + [f"out{i}" for i in range(n_out)])
            for t, out in enumerate(self.output_history):
                w.writerow([t] + list(out))

    def _save_activation_summary(self, logs_dir: Path) -> None:
        e_ids = [
            nid for nid, n in self.network.neurons.items() if n.S == +1
        ]
        i_ids = [
            nid for nid, n in self.network.neurons.items() if n.S == -1
        ]
        path = logs_dir / "activation_summary.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "tick", "total_firing_rate",
                "exc_firing_rate", "inh_firing_rate",
            ])
            for t, act in enumerate(self.activation_history):
                total = sum(act) / self.config.n_total
                exc = sum(act[nid] for nid in e_ids) / len(e_ids)
                inh = sum(act[nid] for nid in i_ids) / len(i_ids)
                w.writerow([t, total, exc, inh])

    def _save_tick_timing(self, logs_dir: Path) -> None:
        path = logs_dir / "tick_timing.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "tick", "step1_ms", "step2_ms", "step3_ms", "step4_ms",
                "total_ms",
            ])
            for rec in self.tick_records:
                w.writerow([
                    rec.tick, rec.step1_ms, rec.step2_ms, rec.step3_ms,
                    rec.step4_ms, rec.total_ms,
                ])

    def _save_weight_stats(self, logs_dir: Path) -> None:
        path = logs_dir / "weight_stats.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "tick", "edge_count", "mean", "std", "min", "max",
                "ratio_W_le_0.01", "ratio_W_ge_1.99",
            ])
            for s in self.weight_stats_history:
                w.writerow([
                    s.tick, s.edge_count, s.mean, s.std, s.min, s.max,
                    s.ratio_W_le_0_01, s.ratio_W_ge_1_99,
                ])

    def _save_edge_events(self, logs_dir: Path) -> None:
        path = logs_dir / "edge_events.csv"
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "tick", "event_type", "edge_id", "source", "target", "weight",
            ])
            for ev in self.edge_events:
                w.writerow([
                    ev.tick, ev.event_type, ev.edge_id,
                    ev.source, ev.target, ev.weight,
                ])

    def _save_edge_snapshots(self, snapshots_dir: Path) -> None:
        for t, snap in self.edge_snapshots:
            path = snapshots_dir / f"edges_t{t:04d}.csv"
            with open(path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "edge_id", "source", "target", "weight", "cofire_count",
                ])
                for edge_id, src, tgt, weight, cofire in snap:
                    w.writerow([edge_id, src, tgt, weight, cofire])