# Experiment manifest (Phase 0 baseline)

_Generated 2026-10-23 from the repository and live scheduler state. Branch
`stage-1`, commit `fd36121`._

## Environment

| item | value |
|---|---|
| scheduler | PBS (`/opt/pbs/bin/qsub`), queue `gpu` |
| GPU policy | `scripts/pick_gpu.sh` requires **≥45 000 MiB** free for 1.7B; `pick_gpu_wait.sh` polls inside the job |
| job resources | `select=1:ncpus=4:ngpus=1:mem=64gb`, walltime 12 h |
| model | `/tmp/asr1bit/models/Qwen3-ASR-1.7B` (fallback `Qwen/Qwen3-ASR-1.7B`) |
| train data | `data/qat_clean_train.100_n8000.pt` — **8 000 utterances** (this is the aligned-set ceiling) |
| aligned cache | `data/aligned_cache/` keys for 1 200 / 2 400 / 6 000 / 8 000 |
| evaluator | `scripts/eval_streaming_17b.pbs` → `scripts/eval_streaming.py`; official Qwen3-ASR prefix algorithm, 320 ms chunks, LSP loop policy, 50 utts per split |
| figures | `scripts/make_plots.py` → `docs/figures/`, `artifacts/stage1/wer_summary.json` |

## Reference points (verified against committed records)

| reference | clean | other | source |
|---|---:|---:|---|
| fp16 1.7B (Stage-0 anchor) | 2.24 | 4.61 | README, `docs/stage0_results.md` |
| gate | **≤3.74** | **≤6.11** | README (+1.5 rule) |
| **b12 ternary line-best pair** | **15.15** | **18.98** | `docs/stage3_results.md`, JSONs below |
| b16 clean record | 14.94 | 21.59 | `stage3_results.md` |
| b13 other record | 21.72 | 17.95 | `stage3_results.md` |
| 0.6B promoted | 9.96 | 18.98 | README |

## Baseline run (the reference to beat)

| field | value |
|---|---|
| launcher | `scripts/qat_stream_aligned17b12.pbs` |
| init | `data/checkpoints/qat_17b_recover7/last.pt` (offline 9.85) |
| policy / bit width | `decoder_attn_ffn` / 1.58 |
| aligned targets | 6 000 (cached) |
| mix frac | 0.5 |
| lr / warmup / schedule | 2.5e-6 / 100 / cosine |
| steps | 2 000 |
| checkpoints | `data/checkpoints/qat_stream_aligned17b12/step{400..2000}.pt` |
| selected state | `step1600.pt` |
| official results | `artifacts/stage1/streaming_qat_stream_aligned17b12_step1600_{clean,other}.json` |

## b17 (Phase 1/2 subject)

| field | value |
|---|---|
| job | **12342** (running, host `jiit-gpu01`, started 05:24, 12 h limit) |
| launcher | `scripts/qat_stream_aligned17b17.pbs` |
| difference from b12 | `--policy decoder_ffn` (FFN ternary, attention **fp16**) |
| checkpoints | `data/checkpoints/qat_stream_aligned17b17/{last.pt,step400.pt,...}` |
| log | `artifacts/stage3/qat_stream_aligned17b17.log` |

## Evaluation protocol (unchanged; do not tune against it)

`qsub -v CKPT=<state>.pt scripts/eval_streaming_17b.pbs` — writes
`artifacts/stage1/streaming_qat_stream_<arm>_<state>_{clean,other}.json` with
the checkpoint name in the TAG. **A number is only final when its JSON's
`checkpoint` field matches the state** (LAB_NOTES 2026-10-19).

## Run inventory (1.7B Stage-B lineage)

| arm | change vs b12 | clean | other |
|---|---|---:|---:|
| b7 | 10.81 base, lr 5e-6, 1 200 aligned | 16.63 | 23.41 |
| b8 | 9.85 base, lr 5e-6 | 18.43 | 25.34 |
| b10 | warmup 500 | 19.17 | 22.39 |
| b11 | lr 2.5e-6, 1 200 aligned | 16.53 | 22.05 |
| **b12** | lr 2.5e-6, 6 000 aligned | **15.15** | **18.98** |
| b13 | 8 000 aligned | 21.72 | **17.95** |
| b14 | mix 0.4 | 18.01 | 27.73 |
| b15 | 6 000 steps | 15.15 | 19.55 |
| b16 | lr 1.25e-6 | **14.94** | 21.59 |
| b17 | FFN-only quantization | _running_ | _running_ |
