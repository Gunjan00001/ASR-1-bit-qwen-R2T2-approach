# Bonsai experiment tracker

_Updated 2026-10-23. Status labels: NOT_STARTED / RUNNING / QUEUED / COMPLETED /
FAILED / BLOCKED. Every queued or running job carries its PBS ID and output
path._

| # | experiment | status | job / location | notes |
|---|---|---|---|---|
| 0 | Baseline audit (scale granularity, STE, group layout, conventions) | **COMPLETED** | `docs/bonsai_inspired_quantization_report.md` §Phase A | baseline is per-tensor; STE confirmed; decoder-only policies |
| 1 | b17 official evaluations, `POLICY=decoder_ffn` | **RUNNING (2/6 numbers)** | PBS `12466`, `12467`, `12468` | clean 119.17 @1600 and 69.28 @2000 already written; step 1200 and the `other` splits outstanding |
| 2 | 0.6B group-wise reconstruction (G0/G128/G64/G32) | **COMPLETED** | `artifacts/stage1/diag_quant_error.json` | mean 0.5654 → 0.5220 / 0.5190 / 0.5138 |
| 3 | Rotation prototype + correctness gates | **COMPLETED** | `src/asr1bit/rotate.py`, `tests/test_rotate.py` (6 gates) | caught an orientation bug; equivalence incl. remainders |
| 4 | Real-weight diagnostic on 1.7B (groups + rotation + combination, per projection/layer) | **COMPLETED** | local CPU task `s1hq4uxy`; output `artifacts/stage1/bonsai_17b/` (worktree) | weights obtained via HF snapshot into `/tmp/asr1bit/hf`; 28 decoder layers, sampling 0/7/14/21/27 |
| 5 | PBS CPU copy of #4 (workq) | **FAILED** | PBS `12477` | node had no model weights; superseded by #4 after the snapshot download |
| 6 | Group-wise ASR pilot (g64, b12 recipe) | **QUEUED** | PBS `12479`; `scripts/qat_stream_aligned17b21.pbs`; out `artifacts/stage3/qat_stream_aligned17b21.log` | one variable vs b12; starts when a GPU frees |
| 7 | Rotation ASR pilot | **NOT_STARTED** | — | rotation gains only 2.1% alone and 7.4% combined; queued behind #6/#10 on evidence priority |
| 8 | Combined (rotation × group) ASR pilot | **NOT_STARTED** | — | blocked on #6/#7 evidence |
| 9 | 2-bit quantizer implementation + tests | **COMPLETED** | `src/asr1bit/quant.py`, `tests/test_quant_2bit.py` | lower reconstruction error than ternary on synthetic weights |
| 10 | 2-bit ASR training (per-tensor, b12 recipe) | **QUEUED** | PBS `12478`; `scripts/qat_stream_aligned17b20.pbs`; out `artifacts/stage3/qat_stream_aligned17b20.log` | guardrail made width-aware; evaluator takes `BIT_WIDTH=2.0` |
| 11 | Streaming-state distillation implementation | **NOT_STARTED** | audit: `docs/distillation_audit.md` | output-logit KL exists; no state supervision; insertion point documented |
| 12 | Deployment measurement (packed size, latency) | **NOT_STARTED** | — | no packed runtime exists; size figures are analytic only |

## Dependencies

- #1 blocks any new GPU job (must not compete); results are collected as they land.
- #6–#8 depend on #4; #8 additionally on #6 and #7.
- #10–#12 are independent of the Bonsai questions and blocked only by GPU
  availability / implementation time.

## Resources

- Scheduler: PBS. `gpu` queue (jobs need `ngpus=1`, 12 h typical), `workq`
  (CPU). At 18:50 the GPU queue had no free card ≥45 GB.
- Model weights for #4: `/tmp/asr1bit/hf/hub/models--Qwen--Qwen3-ASR-1.7B/
  snapshots/7278e1e…/model-0000{1,2}-of-00002.safetensors` (4.4 GB, two shards).
