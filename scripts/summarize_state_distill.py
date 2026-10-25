"""Summarize a streaming-state-distillation run from its JSONL training log.

Reads the ``{"step": ...}`` records and ``STREAM``/``EVAL`` lines, and reports
loss components, gradient magnitudes, state-term statistics and the guardrail
series. Guardrails are diagnostics only; they are never promotion numbers.

Usage::

    PYTHONPATH=src python scripts/summarize_state_distill.py \
        --arms b05 c01 d20 e08 f20 g_noLN --out artifacts/stage3/sd_summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "artifacts" / "stage3"


def parse_log(path: Path) -> dict:
    records: list[dict] = []
    streams: list[dict] = []
    evals: list[dict] = []
    if not path.exists():
        return {"status": "MISSING", "records": records, "streams": streams, "evals": evals}
    for line in path.read_text(errors="ignore").splitlines():
        line = line.strip()
        if line.startswith('{"step"'):
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        elif line.startswith("STREAM "):
            try:
                streams.append(json.loads(line[len("STREAM "):]))
            except json.JSONDecodeError:
                pass
        elif line.startswith("EVAL "):
            try:
                evals.append(json.loads(line[len("EVAL "):]))
            except json.JSONDecodeError:
                pass
    return {"records": records, "streams": streams, "evals": evals}


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
    }


def summarize(arm: str) -> dict:
    parsed = parse_log(LOG_DIR / f"qat_stream_sd_{arm}.log")
    records = parsed["records"]
    states = [float(r["state"]) for r in records if r.get("state") is not None]
    nonzero_states = [value for value in states if value > 0]
    ce = [float(r["ce"]) for r in records if r.get("ce") is not None]
    kl = [float(r["kl"]) for r in records if r.get("kl") is not None]
    grad = [float(r["max_grad_norm"]) for r in records if r.get("max_grad_norm") is not None]
    loss = [float(r["loss"]) for r in records if r.get("loss") is not None]
    nonfinite = sum(
        1 for value in ce + kl + states + grad + loss if not math.isfinite(value)
    )
    final_step = int(records[-1]["step"]) if records else None
    return {
        "arm": arm,
        "status": "TRAINED" if final_step else parsed["status"],
        "final_step": final_step,
        "loss": _stats(loss),
        "ce": _stats(ce),
        "kl": _stats(kl),
        "state_nonzero": _stats(nonzero_states),
        "state_all": _stats(states),
        "max_grad_norm": _stats(grad),
        "nonfinite_values": nonfinite,
        "guardrails": parsed["streams"],
        "offline_evals": parsed["evals"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arms", nargs="+", default=["b05", "c01", "d20", "e08", "f20", "g_noLN"])
    parser.add_argument("--out", default="artifacts/stage3/sd_summary.json")
    args = parser.parse_args()
    summary = {arm: summarize(arm) for arm in args.arms}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for arm, row in summary.items():
        state = row["state_nonzero"]
        print(
            f"{arm:7s} step={row['final_step']} "
            f"ce={row['ce'].get('mean', float('nan')):.3f} "
            f"kl={row['kl'].get('mean', float('nan')):.3f} "
            f"state={state.get('mean', 0.0):.3f} "
            f"maxgrad={row['max_grad_norm'].get('max', 0.0):.1f} "
            f"nonfinite={row['nonfinite_values']}"
        )
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
