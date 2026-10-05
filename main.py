"""CLI 진입점.

algorithm_spec.md 7.2 참조.

사용 예:
    # 시뮬레이터만 실행 (기본 T=1000)
    python main.py

    # 지정 틱 수 실행 + 모든 분석 모듈 실행
    python main.py --ticks 5000 --analyze --output-dir outputs/run_001

    # Step 3 (plasticity) 없이 baseline만
    python main.py --no-plasticity --ticks 1000

설계:
    - 기본은 시뮬레이터만 실행하고 로그를 저장.
    - --analyze 플래그가 있으면 analysis/ 7종을 순차 호출.
    - seed override로 재현성 실험을 편하게 수행.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

from .analysis.activation import analyze_activation
from .analysis.graph import analyze_graph
from .analysis.in_degree import analyze_in_degree
from .analysis.projection import analyze_projection
from .analysis.selfloop import analyze_selfloop
from .analysis.timing import analyze_timing
from .analysis.weight import analyze_weight
from .config import Config
from .input_provider import CyclicPatternProvider
from .network import Network
from .simulator import Simulator


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="artificial_neuron",
        description="artificial_neuron simulator CLI",
    )
    p.add_argument(
        "--ticks", type=int, default=1000,
        help="실행할 틱 수 (기본: 1000)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("outputs"),
        help="로그 저장 디렉터리 (기본: outputs)",
    )
    p.add_argument(
        "--no-plasticity", action="store_true",
        help="Step 3 (Hebbian + prune + generate) 비활성화",
    )
    p.add_argument(
        "--snapshot-interval", type=int, default=100,
        help="엣지 스냅샷 저장 주기 (기본: 100)",
    )
    p.add_argument(
        "--analyze", action="store_true",
        help="저장 후 analysis/ 7종 모듈을 실행",
    )
    p.add_argument(
        "--init-seed", type=int, default=None,
        help="init RNG seed override (기본: Config 값)",
    )
    p.add_argument(
        "--gen-seed", type=int, default=None,
        help="gen RNG seed override (기본: Config 값)",
    )
    p.add_argument(
        "--quiet", action="store_true",
        help="진행 로그를 최소화",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_argparser().parse_args(argv)

    if args.ticks < 0:
        print(f"error: --ticks must be non-negative, got {args.ticks}",
              file=sys.stderr)
        return 2
    if args.snapshot_interval <= 0:
        print(
            f"error: --snapshot-interval must be positive, "
            f"got {args.snapshot_interval}",
            file=sys.stderr,
        )
        return 2

    # Config 준비 (seed override는 dataclasses.replace 사용)
    cfg = Config()
    overrides = {}
    if args.init_seed is not None:
        overrides["init_seed"] = args.init_seed
    if args.gen_seed is not None:
        overrides["gen_seed"] = args.gen_seed
    if overrides:
        cfg = replace(cfg, **overrides)

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.quiet:
        print(f"[main] config: N={cfg.n_total}, "
              f"ticks={args.ticks}, "
              f"plasticity={'on' if not args.no_plasticity else 'off'}")
        print(f"[main] output_dir: {output_dir.resolve()}")

    # 파이프라인 구성
    init_rng = random.Random(cfg.init_seed)
    network = Network(cfg, init_rng)
    provider = CyclicPatternProvider.from_config(cfg)
    sim = Simulator(
        cfg,
        network,
        provider,
        gen_rng=random.Random(cfg.gen_seed),
        enable_plasticity=not args.no_plasticity,
        snapshot_interval=args.snapshot_interval,
    )

    # 실행
    t0 = time.perf_counter()
    sim.run(args.ticks)
    elapsed = time.perf_counter() - t0

    if not args.quiet:
        summary = sim.timing_summary()
        print(f"[main] ran {args.ticks} ticks in {elapsed:.3f} s "
              f"({summary['ticks_per_second']:.0f} ticks/s)")

    # 저장
    sim.save_run(output_dir)
    if not args.quiet:
        print(f"[main] logs saved to {output_dir / 'logs'}")

    # 분석
    if args.analyze:
        if not args.quiet:
            print("[main] running analysis modules...")
        analyze_activation(output_dir)
        analyze_graph(output_dir)
        analyze_projection(output_dir)
        analyze_timing(output_dir)
        analyze_in_degree(output_dir)
        analyze_weight(output_dir)
        analyze_selfloop(output_dir)
        if not args.quiet:
            print(f"[main] analysis outputs saved under {output_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())