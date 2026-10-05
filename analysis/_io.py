"""분석 모듈용 로그 I/O 헬퍼.

spec 6.8: 분석 모듈은 저장된 로그만 읽는다.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


def load_run_meta(output_dir: Path) -> Dict[str, Any]:
    path = Path(output_dir) / "run_meta.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_neuron_meta(logs_dir: Path) -> List[Dict[str, Any]]:
    """neuron_meta.csv -> [{neuron_id, S, is_input, theta}, ...]"""
    path = Path(logs_dir) / "neuron_meta.csv"
    rows: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "neuron_id": int(row["neuron_id"]),
                "S": int(row["S"]),
                "is_input": int(row["is_input"]),
                "theta": float(row["theta"]),
            })
    return rows


def load_activation_history(logs_dir: Path) -> np.ndarray:
    """activation_history.csv -> (T, N) int array. tick 컬럼은 제외."""
    path = Path(logs_dir) / "activation_history.csv"
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        rows = [list(map(int, row[1:])) for row in reader]
    return np.asarray(rows, dtype=np.int8)


def load_voltage_history(logs_dir: Path) -> np.ndarray:
    """voltage_history.csv -> (T, N) float array."""
    path = Path(logs_dir) / "voltage_history.csv"
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [list(map(float, row[1:])) for row in reader]
    return np.asarray(rows, dtype=np.float64)


def load_output_history(logs_dir: Path) -> np.ndarray:
    """output_history.csv -> (T, N_output) int array."""
    path = Path(logs_dir) / "output_history.csv"
    with open(path, "r", newline="") as f:
        reader = csv.reader(f)
        next(reader)
        rows = [list(map(int, row[1:])) for row in reader]
    return np.asarray(rows, dtype=np.int8)


def load_tick_timing(
    logs_dir: Path,
) -> List[Tuple[int, float, float, float, float, float]]:
    """tick_timing.csv -> [(tick, s1, s2, s3, s4, total), ...]"""
    path = Path(logs_dir) / "tick_timing.csv"
    out: List[Tuple[int, float, float, float, float, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((
                int(row["tick"]),
                float(row["step1_ms"]),
                float(row["step2_ms"]),
                float(row["step3_ms"]),
                float(row["step4_ms"]),
                float(row["total_ms"]),
            ))
    return out


def load_weight_stats(logs_dir: Path) -> List[Dict[str, float]]:
    path = Path(logs_dir) / "weight_stats.csv"
    out: List[Dict[str, float]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "tick": int(row["tick"]),
                "edge_count": int(row["edge_count"]),
                "mean": float(row["mean"]),
                "std": float(row["std"]),
                "min": float(row["min"]),
                "max": float(row["max"]),
                "ratio_W_le_0.01": float(row["ratio_W_le_0.01"]),
                "ratio_W_ge_1.99": float(row["ratio_W_ge_1.99"]),
            })
    return out


def load_edge_events(logs_dir: Path) -> List[Dict[str, Any]]:
    path = Path(logs_dir) / "edge_events.csv"
    out: List[Dict[str, Any]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append({
                "tick": int(row["tick"]),
                "event_type": row["event_type"],
                "edge_id": int(row["edge_id"]),
                "source": int(row["source"]),
                "target": int(row["target"]),
                "weight": float(row["weight"]),
            })
    return out


def load_edge_snapshot(
    path: Path,
) -> List[Tuple[int, int, int, float, int]]:
    """edges_tXXXX.csv -> [(edge_id, source, target, weight, cofire), ...]"""
    out: List[Tuple[int, int, int, float, int]] = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append((
                int(row["edge_id"]),
                int(row["source"]),
                int(row["target"]),
                float(row["weight"]),
                int(row["cofire_count"]),
            ))
    return out


def list_edge_snapshots(snapshots_dir: Path) -> List[Tuple[int, Path]]:
    """edge_snapshots 디렉터리의 파일을 (tick, path)로 정렬하여 반환."""
    result: List[Tuple[int, Path]] = []
    for p in sorted(Path(snapshots_dir).glob("edges_t*.csv")):
        stem = p.stem  # 예: "edges_t0000"
        tick = int(stem.split("_t")[1])
        result.append((tick, p))
    return result