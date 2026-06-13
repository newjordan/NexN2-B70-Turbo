#!/usr/bin/env python3
"""Summarize NX2 kernel release-gate llama-bench artifacts.

The expected inputs are llama-bench JSON files named like:

  baseline-q5-ctx0.json
  candidate-q5-ctx0.json
  baseline-q4-ctx0.json
  candidate-q4-ctx0.json
  baseline-q5-131k.json
  candidate-q5-131k.json
  baseline-q5-ctx0-sm-tensor.json
  candidate-q5-ctx0-sm-tensor.json

Only q5 ctx0 is a promotion gate. Q4 and long-context Q5 are guardrails.
The tensor split row is informational for multi-GPU claims and is not a gate.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


PROMOTION_MIN_GAIN_PCT = 2.0
Q4_MAX_REGRESSION_PCT = 1.0
LONGCTX_MAX_REGRESSION_PCT = 3.0
PPL_MAX_REGRESSION_PCT = 0.5
MIN_NOISE_MULTIPLE = 2.0


@dataclass(frozen=True)
class Bench:
    label: str
    path: Path
    avg_ts: float
    stddev_ts: float
    samples_ts: tuple[float, ...]
    n_depth: int
    n_prompt: int
    n_gen: int
    model_filename: str
    build_commit: str
    test_time: str

    @property
    def sem(self) -> float:
        if len(self.samples_ts) <= 1:
            return 0.0
        return statistics.stdev(self.samples_ts) / math.sqrt(len(self.samples_ts))


@dataclass(frozen=True)
class Ppl:
    label: str
    path: Path
    value: float
    stderr: float


@dataclass(frozen=True)
class Ops:
    path: Path
    passed: int
    total: int


def load_bench(path: Path, label: str | None = None) -> Bench:
    data = json.loads(path.read_text())
    if not isinstance(data, list) or not data:
        raise ValueError(f"{path}: expected non-empty llama-bench JSON array")
    row = data[0]
    samples = tuple(float(v) for v in row.get("samples_ts", []))
    avg_ts = float(row["avg_ts"])
    stddev_ts = float(row.get("stddev_ts", 0.0))
    if not samples:
        samples = (avg_ts,)
    return Bench(
        label=label or path.stem,
        path=path,
        avg_ts=avg_ts,
        stddev_ts=stddev_ts,
        samples_ts=samples,
        n_depth=int(row.get("n_depth", 0)),
        n_prompt=int(row.get("n_prompt", 0)),
        n_gen=int(row.get("n_gen", 0)),
        model_filename=str(row.get("model_filename", "")),
        build_commit=str(row.get("build_commit", "")),
        test_time=str(row.get("test_time", "")),
    )


def pct_delta(candidate: Bench, baseline: Bench) -> float:
    return 100.0 * (candidate.avg_ts - baseline.avg_ts) / baseline.avg_ts


def combined_sem_pct(candidate: Bench, baseline: Bench) -> float:
    combined = math.sqrt(candidate.sem * candidate.sem + baseline.sem * baseline.sem)
    return 100.0 * combined / baseline.avg_ts


def find_pair(run_dir: Path, stem: str) -> tuple[Bench, Bench] | None:
    baseline_path = run_dir / f"baseline-{stem}.json"
    candidate_path = run_dir / f"candidate-{stem}.json"
    if not baseline_path.exists() or not candidate_path.exists():
        return None
    return load_bench(baseline_path, f"baseline-{stem}"), load_bench(candidate_path, f"candidate-{stem}")


def load_ppl(path: Path, label: str | None = None) -> Ppl:
    text = path.read_text(errors="replace")
    match = re.search(r"Final estimate:\s+PPL\s*=\s*([0-9.]+)\s*\+/-\s*([0-9.]+)", text)
    if not match:
        raise ValueError(f"{path}: could not find final PPL estimate")
    return Ppl(
        label=label or path.stem,
        path=path,
        value=float(match.group(1)),
        stderr=float(match.group(2)),
    )


def find_ppl_pair(run_dir: Path) -> tuple[Ppl, Ppl] | None:
    baseline_path = run_dir / "baseline-q5-ppl.log"
    candidate_path = run_dir / "candidate-q5-ppl.log"
    if not baseline_path.exists() or not candidate_path.exists():
        return None
    return load_ppl(baseline_path, "baseline-q5-ppl"), load_ppl(candidate_path, "candidate-q5-ppl")


def load_ops(path: Path) -> Ops:
    text = path.read_text(errors="replace")
    matches = re.findall(r"(\d+)/(\d+) tests passed", text)
    if not matches:
        raise ValueError(f"{path}: could not find backend-op pass count")
    passed, total = matches[-1]
    return Ops(path=path, passed=int(passed), total=int(total))


def find_ops(run_dir: Path) -> Ops | None:
    path = run_dir / "candidate-test-backend-ops-mul-mat-id.log"
    if not path.exists():
        return None
    return load_ops(path)


def decision(
    pairs: dict[str, tuple[Bench, Bench]],
    ppl_pair: tuple[Ppl, Ppl] | None,
    ops: Ops | None,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    q5 = pairs.get("q5-ctx0")
    if q5 is None:
        return "INCOMPLETE", ["missing q5 ctx0 baseline/candidate pair"]

    q5_base, q5_cand = q5
    q5_gain = pct_delta(q5_cand, q5_base)
    q5_noise = combined_sem_pct(q5_cand, q5_base)

    if q5_gain < PROMOTION_MIN_GAIN_PCT:
        reasons.append(
            f"q5 ctx0 gain {q5_gain:.2f}% is below {PROMOTION_MIN_GAIN_PCT:.1f}% promotion bar"
        )
    if q5_noise > 0 and q5_gain < MIN_NOISE_MULTIPLE * q5_noise:
        reasons.append(
            f"q5 ctx0 gain {q5_gain:.2f}% is below {MIN_NOISE_MULTIPLE:.0f}x combined SEM ({q5_noise:.2f}%)"
        )

    q4 = pairs.get("q4-ctx0")
    if q4 is not None:
        q4_gain = pct_delta(q4[1], q4[0])
        if q4_gain < -Q4_MAX_REGRESSION_PCT:
            reasons.append(
                f"q4 ctx0 regression {q4_gain:.2f}% exceeds {Q4_MAX_REGRESSION_PCT:.1f}% guardrail"
            )

    longctx = pairs.get("q5-131k")
    if longctx is not None:
        long_gain = pct_delta(longctx[1], longctx[0])
        if long_gain < -LONGCTX_MAX_REGRESSION_PCT:
            reasons.append(
                f"q5 131k regression {long_gain:.2f}% exceeds {LONGCTX_MAX_REGRESSION_PCT:.1f}% guardrail"
            )

    if ppl_pair is not None:
        base_ppl, cand_ppl = ppl_pair
        ppl_delta = 100.0 * (cand_ppl.value - base_ppl.value) / base_ppl.value
        if ppl_delta > PPL_MAX_REGRESSION_PCT:
            reasons.append(
                f"q5 PPL regression {ppl_delta:.2f}% exceeds {PPL_MAX_REGRESSION_PCT:.1f}% guardrail"
            )

    if ops is not None and ops.passed != ops.total:
        reasons.append(f"candidate backend-op check only passed {ops.passed}/{ops.total}")

    if reasons:
        return "NO_HF_UPDATE", reasons
    return "HF_UPDATE_WORTHY", ["q5 ctx0 clears promotion bar and guardrails pass"]


def format_pair(stem: str, baseline: Bench, candidate: Bench) -> str:
    delta = pct_delta(candidate, baseline)
    noise = combined_sem_pct(candidate, baseline)
    return (
        f"| {stem} | {baseline.avg_ts:.4f} | {candidate.avg_ts:.4f} | "
        f"{delta:+.2f}% | {noise:.2f}% | {len(baseline.samples_ts)}/{len(candidate.samples_ts)} |"
    )


def emit_markdown(
    run_dir: Path,
    pairs: dict[str, tuple[Bench, Bench]],
    ppl_pair: tuple[Ppl, Ppl] | None,
    ops: Ops | None,
) -> str:
    status, reasons = decision(pairs, ppl_pair, ops)
    lines: list[str] = []
    lines.append(f"# NX2 Kernel Release Gate: {run_dir.name}")
    lines.append("")
    lines.append(f"Decision: **{status}**")
    lines.append("")
    lines.append("## Why")
    lines.append("")
    for reason in reasons:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append("## Throughput")
    lines.append("")
    lines.append("| case | baseline t/s | candidate t/s | delta | combined SEM | samples |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for stem in ("q5-ctx0", "q4-ctx0", "q5-131k", "q5-ctx0-sm-tensor"):
        pair = pairs.get(stem)
        if pair is not None:
            lines.append(format_pair(stem, pair[0], pair[1]))
    lines.append("")
    if ppl_pair is not None:
        base_ppl, cand_ppl = ppl_pair
        ppl_delta = 100.0 * (cand_ppl.value - base_ppl.value) / base_ppl.value
        lines.append("## Accuracy")
        lines.append("")
        lines.append("| case | baseline PPL | candidate PPL | delta | stderr baseline/candidate |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(
            f"| q5-30chunk-ppl | {base_ppl.value:.4f} | {cand_ppl.value:.4f} | "
            f"{ppl_delta:+.2f}% | {base_ppl.stderr:.5f}/{cand_ppl.stderr:.5f} |"
        )
        lines.append("")
    if ops is not None:
        lines.append("## Correctness")
        lines.append("")
        lines.append(f"- candidate `MUL_MAT_ID` backend-op check: {ops.passed}/{ops.total}")
        lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for stem in sorted(pairs):
        base, cand = pairs[stem]
        lines.append(
            f"- {stem}: `{base.path.name}` ({base.build_commit}) vs "
            f"`{cand.path.name}` ({cand.build_commit})"
        )
    if ppl_pair is not None:
        base_ppl, cand_ppl = ppl_pair
        lines.append(f"- q5-ppl: `{base_ppl.path.name}` vs `{cand_ppl.path.name}`")
    if ops is not None:
        lines.append(f"- backend-ops: `{ops.path.name}`")
    lines.append("")
    return "\n".join(lines)


def collect_pairs(run_dir: Path) -> dict[str, tuple[Bench, Bench]]:
    pairs: dict[str, tuple[Bench, Bench]] = {}
    for stem in ("q5-ctx0", "q4-ctx0", "q5-131k", "q5-ctx0-sm-tensor"):
        pair = find_pair(run_dir, stem)
        if pair is not None:
            pairs[stem] = pair
    return pairs


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path, help="directory containing release-gate JSON files")
    parser.add_argument("--write", action="store_true", help="write SUMMARY.md in the run directory")
    args = parser.parse_args(argv)

    run_dir = args.run_dir.resolve()
    pairs = collect_pairs(run_dir)
    ppl_pair = find_ppl_pair(run_dir)
    ops = find_ops(run_dir)
    if not pairs:
        raise SystemExit(f"no baseline/candidate pairs found in {run_dir}")

    text = emit_markdown(run_dir, pairs, ppl_pair, ops)
    if args.write:
        (run_dir / "SUMMARY.md").write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
