"""Generate project status figures from the Stage 1 artifacts.

Reads the run logs (``EVAL`` lines) and final ablation JSONs, then writes:

- ``docs/figures/wer_trajectories.png``  WER vs step for the main runs
- ``docs/figures/final_wer.png``         final WER per configuration
- ``docs/figures/quantizer_error.png``   quantizer relative error by config
- ``docs/figures/fp16_drift.png``        fp16 readout vs step (shadow drift)
- ``docs/figures/roadmap.png``           stage status overview
- ``artifacts/stage1/wer_summary.json``  the extracted numbers

Run:  PYTHONPATH=src python scripts/make_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "stage1"
FIG = ROOT / "docs" / "figures"
FP16_BASELINE = 2.4364406779661016  # pretrained, 50-utt test-clean (seed 0)


def parse_evals(name: str) -> list[dict]:
    """Return the EVAL records from ``artifacts/stage1/<name>.log``."""
    path = ART / f"{name}.log"
    if not path.exists():
        return []
    records = []
    with path.open(errors="ignore") as handle:
        for line in handle:
            if line.startswith("EVAL "):
                records.append(json.loads(line[len("EVAL ") :]))
    return records


def series(records: list[dict], key: str) -> tuple[list[int], list[float]]:
    steps, values = [], []
    for record in records:
        value = record.get(key)
        if value is not None:
            steps.append(int(record["step"]))
            values.append(float(value))
    return steps, values


def load_final(name: str) -> dict:
    path = ART / name
    return json.loads(path.read_text()) if path.exists() else {}


def fig_wer_trajectories() -> dict:
    runs = {
        "Naive attn+FFN (ternary)": ("qat_ternary_attnffn_ls2", "ternary_wer", "#d62728", "-"),
        "FFN-only (ternary)": ("qat_ternary_ffn", "ternary_wer", "#ff7f0e", "-"),
        "Stage 2a attn+FFN rollout (ternary)": (
            ["qat_attn_rollout", "qat_attn_rollout_resume", "qat_attn_rollout_ext", "qat_attn_rollout_ext2"],
            "ternary_wer", "#1f77b4", "-",
        ),
        "Stage 2b attn+FFN rollout (binary)": ("qat_binary_rollout", "binary_wer", "#2ca02c", "-"),
        "Ternary long (8k data, 8000 steps)": ("qat_ternary_long", "ternary_wer", "#9467bd", "-"),
        "Binary long (8k data, 6000 steps)": ("qat_binary_long", "binary_wer", "#8c564b", "-"),
        "Ternary long2 (8k, lr 1e-5)": ("qat_ternary_long2", "ternary_wer", "#e377c2", "-"),
        "Binary long3 (8k, lr 5e-6)": ("qat_binary_long3", "binary_wer", "#7f7f7f", "-"),
        # --- recovery recipe era (2026-10-11): full 28 539-utt cache ---
        "Recovery recipe — ternary (full cache)": ("qat_ternary_full", "ternary_wer", "#000000", "-"),
        "Full-cache anneal — ternary (2nd cycle)": ("qat_ternary_full2", "ternary_wer", "#17becf", "--"),
        "Recovery recipe — binary (full cache)": ("qat_binary_full", "binary_wer", "#2ca02c", "--"),
        # --- streaming era (2026-10-12/15): STT recipe on the aligned path ---
        "Aligned streaming SFT — ternary": ("qat_stream_aligned2", "ternary_wer", "#bcbd22", "-"),
    }
    ffn_resume = ("qat_ternary_ffn_resume", "ternary_wer", "#ff7f0e")

    plt.figure(figsize=(11, 6))
    data = {}
    for label, (names, key, color, style) in runs.items():
        if isinstance(names, str):
            names = [names]
        steps, values = [], []
        for name in names:
            s, v = series(parse_evals(name), key)
            steps += s
            values += v
        order = sorted(range(len(steps)), key=lambda i: steps[i])
        steps = [steps[i] for i in order]
        values = [values[i] for i in order]
        data[label] = {"steps": steps, "wer": values}
        plt.plot(steps, values, marker="o", color=color, linestyle=style, label=label, linewidth=2)

    # FFN-only resume (affected by the LR-restart bug) as a dotted continuation.
    s, v = series(parse_evals(ffn_resume[0]), ffn_resume[1])
    data["FFN-only resume (LR-restart artifact)"] = {"steps": s, "wer": v}
    plt.plot(s, v, marker="o", color=ffn_resume[2], linestyle=":", linewidth=2,
             label="FFN-only resume (LR-restart artifact)")

    plt.axhline(FP16_BASELINE, color="black", linestyle=":", linewidth=1.5,
                label=f"fp16 pretrained baseline ({FP16_BASELINE:.2f})")

    plt.yscale("log")
    plt.xlabel("QAT step")
    plt.ylabel("WER % (log scale, 50-utt test-clean)")
    plt.title("Stage 1/2 QAT: WER trajectories (Qwen3-ASR-0.6B, ternary/binary decoder projections)")
    plt.grid(True, which="both", alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "wer_trajectories.png", dpi=150)
    plt.close()
    return data


def fig_final_wer() -> dict:
    finals = {
        "Naive attn+FFN\n(ternary)": {"ternary": 107.84, "binary": 99.79},
        "Self-distill attn+FFN\n(ternary)": {"ternary": 98.09, "binary": 118.33},
        "FFN-only\n(ternary)": {"ternary": 32.84, "binary": 70.97},
        "FFN-only\n(binary)": {"ternary": 106.78, "binary": 97.88},
        "Stage 2a attn+FFN\n(ternary, 4k)": {"ternary": 16.21, "binary": 99.89},
        "Stage 2b attn+FFN\n(binary, 4k)": {"ternary": 30.51, "binary": 17.48},
        "Ternary long\n(8k, 8000 steps)": {"ternary": 11.44, "binary": 100.0},
        "Binary long\n(8k, 6000 steps)": {"ternary": 21.29, "binary": 12.92},
        "Ternary long2\n(8k, 4000 steps)": {"ternary": 9.22, "binary": 100.42},
        "Binary long3\n(8k, 3000 steps)": {"ternary": 22.35, "binary": 10.70},
        # recovery recipe era: the full 28 539-utt cache (best checkpoints)
        "Ternary full cache\n(28.5k, recovery recipe)": {"ternary": 7.52, "binary": 100.0},
        "Ternary full cache\n(28.5k, 2nd cycle anneal)": {"ternary": 7.63, "binary": 100.0},
        "Binary full cache\n(28.5k, recovery recipe)": {"ternary": 100.0, "binary": 10.70},
    }
    labels = list(finals)
    ternary = [finals[k].get("ternary", 0) for k in labels]
    binary = [finals[k].get("binary", 0) for k in labels]

    x = range(len(labels))
    width = 0.38
    plt.figure(figsize=(11, 5.5))
    bars_t = plt.bar([i - width / 2 for i in x], ternary, width, label="ternary readout", color="#1f77b4")
    bars_b = plt.bar([i + width / 2 for i in x], binary, width, label="binary readout", color="#2ca02c")
    plt.axhline(FP16_BASELINE, color="black", linestyle=":", linewidth=1.5,
                label=f"fp16 pretrained ({FP16_BASELINE:.2f})")
    plt.xticks(list(x), labels, fontsize=9)
    plt.ylabel("WER % (50-utt test-clean)")
    plt.title("Final WER by configuration (best checkpoint; lower is better)")
    plt.grid(axis="y", alpha=0.25)
    plt.legend(fontsize=9)
    for bars in (bars_t, bars_b):
        for bar in bars:
            if bar.get_height() > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                         f"{bar.get_height():.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "final_wer.png", dpi=150)
    plt.close()
    return finals


def fig_quantizer_error() -> dict:
    path = ART / "diag_quant_error.json"
    payload = json.loads(path.read_text())
    overall = payload["overall"]
    names = ["binary\ntensor", "binary\ng64", "binary\ng128", "ternary\ntensor", "ternary\ng64", "ternary\ng128"]
    keys = ["binary_tensor", "binary_g64", "binary_g128", "ternary_tensor", "ternary_g64", "ternary_g128"]
    values = [overall[k] for k in keys]
    colors = ["#d62728", "#e377c2", "#9467bd", "#1f77b4", "#17becf", "#2ca02c"]

    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, values, color=colors)
    plt.axhline(0.60, color="black", linestyle=":", linewidth=1.2,
                label="Gaussian closed form (binary) ~0.60")
    plt.ylabel("relative Frobenius error ||Q(W)-W|| / ||W||")
    plt.title("Quantizer error on real Qwen3-ASR-0.6B decoder weights (196 projections)")
    plt.legend(fontsize=9)
    plt.grid(axis="y", alpha=0.25)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "quantizer_error.png", dpi=150)
    plt.close()
    return {"names": names, "values": values}


def fig_fp16_drift() -> dict:
    plt.figure(figsize=(10, 5))
    data = {}
    for label, name, color in (
        ("Stage 2a (ternary rollout)", "qat_attn_rollout_resume", "#1f77b4"),
        ("Stage 2a ext2", "qat_attn_rollout_ext2", "#17becf"),
        ("Stage 2b (binary rollout)", "qat_binary_rollout", "#2ca02c"),
        ("Ternary long (8k)", "qat_ternary_long", "#9467bd"),
        ("Binary long (8k)", "qat_binary_long", "#8c564b"),
        ("Ternary full cache (28.5k)", "qat_ternary_full", "#000000"),
        ("Binary full cache (28.5k)", "qat_binary_full", "#2ca02c"),
    ):
        records = parse_evals(name)
        steps, values = series(records, "fp16_wer")
        if label == "Stage 2a (ternary rollout)":
            extra = series(parse_evals("qat_attn_rollout"), "fp16_wer")
            steps = extra[0] + steps
            values = extra[1] + values
        data[label] = {"steps": steps, "fp16_wer": values}
        plt.plot(steps, values, marker="s", color=color, label=label, linewidth=2)
    plt.axhline(FP16_BASELINE, color="black", linestyle=":", linewidth=1.5,
                label=f"pretrained ({FP16_BASELINE:.2f})")
    plt.xlabel("QAT step")
    plt.ylabel("fp16 (alpha=0) readout WER %")
    plt.title("Shadow drift: the fp16 readout of the trained weights degrades during QAT")
    plt.grid(alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG / "fp16_drift.png", dpi=150)
    plt.close()
    return data


def fig_roadmap() -> dict:
    stages = [
        ("Stage 0: baselines & harness", 1.0, "#2ca02c",
         "gate passed 0.2.0 (fp16 anchors locked: streaming 320 ms 2.24/4.61; X=+1.5)"),
        ("Stage 1: BitLinear + QAT", 1.0, "#2ca02c",
         "machinery written, tested, audited; loss controllable; naive run's offline gate missed -> Stage 2"),
        ("Stage 2: accuracy recovery", 0.9, "#2ca02c",
         "recipe scaled on the full 28 539-utt cache: ternary 7.52 / binary 10.70 offline (best 9.1/10.7); +1.5 offline bar open"),
        ("Stage 3: scale to 1.7B", 0.35, "#ff7f0e",
         "FFN warm start 22.03 -> attention rollout 20.55; aligned recipe paused on cluster capacity (needs >=45 GB stable)"),
        ("Stage 4: streaming parity (LSP)", 0.75, "#ff7f0e",
         "official algorithm fixed at the recipe level: 203.5 -> 12.92 clean / 20.00 other, zero loops; gate (~3.7/6.1) open - gap is quantization damage"),
        ("Stage 5: edge deployment", 0.0, "#bbbbbb", "not started"),
        ("Stage 6: evaluation & writeup", 0.15, "#1f77b4",
         "started: status/handoff/results docs + figures regenerated from artifacts"),
    ]
    labels = [s[0] for s in stages][::-1]
    values = [s[1] for s in stages][::-1]
    colors = [s[2] for s in stages][::-1]
    notes = [s[3] for s in stages][::-1]

    plt.figure(figsize=(11, 4.8))
    bars = plt.barh(labels, values, color=colors, height=0.6)
    for bar, value, note in zip(bars, values, notes):
        if value >= 0.5:
            plt.text(value - 0.02, bar.get_y() + bar.get_height() / 2, note,
                     va="center", ha="right", fontsize=8, color="white")
        else:
            plt.text(0.02, bar.get_y() + bar.get_height() / 2, note,
                     va="center", ha="left", fontsize=8)
    plt.xlim(0, 1.25)
    plt.xticks([0, 0.25, 0.5, 0.75, 1.0], ["0 %", "25 %", "50 %", "75 %", "100 %"])
    plt.title("Project roadmap status (2026-10-15)")
    plt.grid(axis="x", alpha=0.25)
    plt.tight_layout()
    plt.savefig(FIG / "roadmap.png", dpi=150)
    plt.close()
    return {"stages": stages}


def load_streaming() -> dict:
    """Every quantized streaming evaluation on disk, keyed by artifact stem."""
    out = {}
    for path in sorted(ART.glob("streaming_*.json")):
        payload = json.loads(path.read_text())
        quant = (payload.get("conditions") or {}).get("quantized")
        if not quant:
            continue
        stem = path.name[len("streaming_") : -len(".json")]
        for suffix in ("_quantized", "_50"):
            stem = stem.removesuffix(suffix)
        out[stem] = {
            "config": payload.get("config") or "clean",
            "chunk_ms": payload.get("chunk_ms"),
            "wer": float(quant["wer"]),
            "cer": float(quant["cer"]),
            "loop_truncations": (quant.get("lsp") or {}).get("loop_truncations"),
            "wer_raw": (quant.get("lsp") or {}).get("wer_raw"),
        }
    return out


def fig_streaming_results() -> dict:
    """Stage-4 figure: the collapse -> fix history and the promoted matrix.

    Left panel: full-sample (50-utt test-clean) WER of the official prefix
    algorithm at 320 ms, one bar per recipe stage, log scale.  Right panel: the
    promoted aligned-targets recipe across chunk sizes and splits, with the
    binary arm and the fp16-1.7B/gate reference lines.
    """
    streams = load_streaming()
    evolution = [
        ("ternary_prefix", "ternary prefix\n(first attempt)", "collapse"),
        ("opd", "+ DAgger", "fail"),
        ("opd_distill", "+ distill v1", "fail"),
        ("opd_distill2", "distill v2", "fail"),
        ("opd_distill3", "distill v3", "fail"),
        ("opd_adapt5", "adapt5", "fail"),
        ("opd_lspdistill3", "+ LSP loop\npolicy", "fail"),
        ("ternary_long2", "long2 state", "fail"),
        ("opd_aligned2bin", "binary +\naligned", "promoted-binary"),
        ("opd_aligned2", "ternary +\naligned", "promoted"),
    ]
    labels, values, colors = [], [], []
    for stem, label, kind in evolution:
        if stem not in streams:
            continue
        labels.append(label)
        values.append(streams[stem]["wer"])
        colors.append({"collapse": "#8c564b", "fail": "#c7c7c7",
                       "promoted-binary": "#2ca02c", "promoted": "#1f77b4"}[kind])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6), width_ratios=[1.25, 1])

    # --- left: recipe history (320 ms, clean, log scale) ---
    ax1.bar(range(len(labels)), values, color=colors)
    for i, value in enumerate(values):
        ax1.text(i, value * 1.15, f"{value:.0f}" if value >= 10 else f"{value:.1f}",
                 ha="center", fontsize=8)
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(labels)))
    ax1.set_xticklabels(labels, fontsize=7.5, rotation=30, ha="right")
    ax1.set_ylabel("WER % (log scale)")
    ax1.set_ylim(8, 2000)
    ax1.set_title("Official prefix algorithm, 320 ms, 50-utt test-clean:\n"
                  "every recipe stage (lower is better)", fontsize=10)
    ax1.grid(axis="y", which="both", alpha=0.25)

    # --- right: the promoted matrix (chunk x split) + binary + references ---
    def cell(stem: str) -> float:
        return streams[stem]["wer"]

    groups = ["160 ms", "320 ms", "1000 ms"]
    clean = [cell("opd_aligned2_c160"), cell("opd_aligned2"), cell("opd_aligned2_c1000")]
    other = [cell("opd_aligned2_other_c160"), cell("opd_aligned2_other"),
             cell("opd_aligned2_other_c1000")]
    groups.append("binary 320 ms")
    clean.append(cell("opd_aligned2bin"))
    other.append(cell("opd_aligned2bin_other"))

    x = range(len(groups))
    width = 0.36
    bars_c = ax2.bar([i - width / 2 for i in x], clean, width, label="test-clean", color="#1f77b4")
    bars_o = ax2.bar([i + width / 2 for i in x], other, width, label="test-other", color="#ff7f0e")
    for bars in (bars_c, bars_o):
        for bar in bars:
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.4,
                     f"{bar.get_height():.2f}", ha="center", fontsize=8)
    for y, label, side in ((2.24, "fp16 1.7B clean 2.24", "right"),
                           (4.61, "fp16 1.7B other 4.61", "right"),
                           (3.74, "gate clean = 2.24 + 1.5", "left"),
                           (6.11, "gate other = 4.61 + 1.5", "left")):
        ax2.axhline(y, color="black", linestyle=":" if "fp16" in label else "--",
                    linewidth=1.3, alpha=0.8)
        if side == "right":
            ax2.text(len(groups) - 0.45, y + 0.35, label, fontsize=7, ha="right")
        else:
            ax2.text(-0.45, y + 0.35, label, fontsize=7, ha="left")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(groups, fontsize=9)
    ax2.set_ylabel("WER %")
    ax2.set_ylim(0, max(other) * 1.25)
    ax2.set_title("Promoted recipe (aligned targets + LSP policy):\n"
                  "mean WER per chunk and split", fontsize=10)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Stage 4 — streaming with the *official* Qwen3-ASR prefix algorithm "
                 "(Qwen3-ASR-0.6B, quantized readouts)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(FIG / "streaming_results.png", dpi=150)
    plt.close(fig)
    return {"evolution": {"labels": labels, "wer": values},
            "promoted": {"groups": groups, "clean": clean, "other": other}}


def main() -> int:
    FIG.mkdir(parents=True, exist_ok=True)
    summary = {
        "fp16_baseline": FP16_BASELINE,
        "trajectories": fig_wer_trajectories(),
        "final_wer": fig_final_wer(),
        "quantizer_error": fig_quantizer_error(),
        "fp16_drift": fig_fp16_drift(),
        "roadmap": fig_roadmap(),
        "streaming": fig_streaming_results(),
    }
    out = ART / "wer_summary.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote figures to", FIG)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
