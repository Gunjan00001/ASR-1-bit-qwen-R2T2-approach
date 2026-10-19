# Changelog / checkpoints

Versioning is **`MAJOR.MINOR.BUGS`**:

- **MAJOR** — an end-to-end milestone (e.g. `1.0.0` = 1.7B binary streaming
  pipeline works end-to-end).
- **MINOR** — a **completed stage**: its `docs/PLANNING.md` gate met *and* an
  outcome the project stands behind (the fixed bars in `PLANNING.md` §3 are the
  standing guard). Clarified 2026-10-15 to match practice: **Stage 1's** gate
  (*binary/ternary loss controllable; WER table emitted*) and **Stage 2's**
  (*measurable WER recovery vs Stage 1*) are both **met**, yet `0.3.0` stays
  untagged because the `+1.5` accuracy bars are still open — work continues on
  `stage-1` under the `0.2.x` series.
- **BUGS** — fixes and non-stage checkpoints.

Stage work lands on a per-stage branch; `main` only advances on a passed stage
gate. `stage-0` was merged to `main` at `0.2.0`.

## Checkpoint map

| Tag | Date | Branch | Checkpoint | Gate |
|---|---|---|---|---|
| `0.1.0` | 2026-09-20 | `main` | Repo scaffold + project plan | — |
| `0.1.1` | 2026-09-20 | `stage-0` | Stage 0 plan corrections (eval protocol, append-only scope, feasibility) | — |
| `0.1.2`–`0.1.17` | 2026-09-20 | `stage-0` | Stage 0 implementation + gate-runner hardening (17 checkpoints, summarized below) | — |
| `0.2.0` | 2026-09-20 | `stage-0` → `main` | **Stage 0 gate PASSED** — fp16 baselines reproduced; `X = +1.5` locked | **passed** |
| `0.2.1`–`0.2.10` | 2026-09-20 | `stage-1` | Stage 1 T1.1–T1.4 implementation; first QAT run invalid (shadow corruption) | — |
| `0.2.11`–`0.2.28` | 2026-10-10 | `stage-1` | Quantizer audit (math correct), A100 tooling, log-scaled learnable scales, distillation QAT, policy ablations | — |
| `0.2.29`–`0.2.40` | 2026-10-11 | `stage-1` | Recovery recipe (FFN warm start → attention rollout + frozen anchor → warm restarts; progressive precision for binary); offline 9.2 ternary / 10.7 binary; streaming diagnosed as the critical path | — |
| `0.2.48` | 2026-10-19 | `stage-1` | **Best clean streaming yet**: `aligned8/step1000` (6 000-aligned recipe, selected early state) at **9.96 clean** (was 11.55) with zero loops; `other` is 20.11 vs the previous best 18.98 — a split result, both rows kept | — |
| `0.2.47` | 2026-10-15 | `stage-1` | **Streaming best improved**: the longer aligned cycle's step-1 000 state streams at **11.55 clean / 18.98 other** (was 12.92 / 20.00) with zero loops; the 1.7B anchored fp32-AdamW retry completed 1 200 steps (no permanent dip) | — |
| `0.2.41`–`0.2.46` | 2026-10-12..15 | `stage-1` | **Stage 4: the streaming algorithm is freed** — LSP emission policy (loop truncation, append-only), on-policy SFT + streaming-state distillation + filtered-DAgger guard, **aligned targets (forced aligner)**; official-algorithm WER 203.5 → **12.92 clean / 20.00 other** (zero loops); offline scale-up to 7.52 ternary (binary at a 10.70 floor) | — (Stage-4 gate open: the `+1.5` target is not met; the 1.7B path is paused on cluster capacity) |

## 0.2.0 — Stage 0 gate passed

T0.0 vLLM streaming spike **PASS**; published streaming WER reproduced within
tolerance on **Tesla T4, fp16** (torch 2.9.1+cu128, transformers 4.57.6,
vllm 0.14.0; Triton attention). Pinned protocol: greedy, `language=None`,
`max_new_tokens` 1024 offline / 32 streaming, `max_model_len=16384`,
`gpu_util=0.85`.

Results (`artifacts/stage0/gate_results.csv`; subsampled rows = fixed 500-utt
set, seed 0):

| Model | Mode | Chunk | Config | n | sub | WER | Published | Δ |
|---|---|---|---|---|---|---|---|---|
| Qwen3-ASR-1.7B | streaming | 2.0 s | clean | 2620 | full | **1.96** | 1.95 | **+0.01** |
| Qwen3-ASR-1.7B | streaming | 2.0 s | other | 500 | sub | **3.47** | 4.51 | −1.04 |
| Qwen3-ASR-1.7B | streaming | 320 ms | other | 500 | sub | **4.61** | — | — |
| Qwen3-ASR-1.7B | offline | — | clean | 500 | sub | **2.10** | 1.63 | +0.47 |
| Qwen3-ASR-1.7B | offline | — | other | 500 | sub | **3.47** | 3.38 | +0.09 |
| Qwen3-ASR-1.7B | streaming | 320 ms | clean | 500 | sub | **2.24** | — | — |

- **Primary metric** = streaming 320 ms, clean/other. fp16 baselines:
  clean **2.24**, other **4.61**.
- **`X = +1.5` absolute** locked (quantized model must stay within +1.5 WER of
  fp16 at the primary metric). Ternary fallback remains acceptable per the plan.
- Full-split offline and 0.6B confirmation are deferred (opportunistic).

## 0.1.2 – 0.1.17 — Stage 0 implementation & gate-runner hardening

Collapsed summary of 17 `stage-0` checkpoints (individual commits/tags retained).

**Implementation (T0.0–T0.4):**
- T0.0 vLLM go/no-go spike (`scripts/spike_vllm_streaming.py`) + Kaggle/Colab
  setup docs.
- T0.1 Python 3.12+ dev env, dependency split, `tests/conftest.py`.
- T0.2 LibriSpeech loader (`data/load.py`) + shared normalizer (`text.py`);
  hyphen-split normalizer (`librispeech_hyph`) is the default.
- T0.3 portable streaming engine (`qwen/stream.py`, official re-feed +
  prefix-rollback) and Qwen adapter (`qwen/backends.py`).
- T0.4 WER/CER, latency/RTF, matrix harness, Stage 0 runner, throughput probe.

**Gate-runner fixes found by running the gate (each a BUGS checkpoint):**
- `pip install -U pip` before the vLLM install; then install `qwen-asr[vllm]`
  from PyPI (nightly fallback without `--index-strategy`).
- vLLM `max_model_len=16384` / `gpu_memory_utilization=0.85` (KV cache did not
  fit the 65536 default on a T4).
- `VLLM_ATTENTION_BACKEND=TRITON_ATTN` (FlashInfer JIT build fails on sm75).
- `PYTHONPATH=$REPO_ROOT/src` so `asr1bit.*` imports without relying on the
  editable install.
- **`.gitignore` bug**: an unanchored `data/` had hidden `src/asr1bit/data/`
  from git (loader missing on the remote). Anchored to `/data/`.
- Offline-mode heuristic: only set `HF_HUB_OFFLINE=1` for local weight dirs.
- Matrix slicing: `--only-mode`, `--only-config`, `--only-chunk-ms`,
  `--offline-sample-n` (labelled subsamples so sessions fit).
- Load only the requested split parquet (the named builder materialized the
  ~104k-example `train.360` and failed); cast `Audio(decode=False)` to avoid
  `torchcodec`.
- Forward `max_new_tokens` to vLLM (offline 1024 / streaming 32 per mode;
  offline was truncated at 32).
- Skip artifact commit when `--artifacts` is outside the repo.
- EXIT/INT/TERM trap collects artifacts on any outcome; token-scoped `--push`
  warns and skips when `GITHUB_TOKEN` is absent.

## 0.2.1 – 0.2.10 — Stage 1 (BitLinear + QAT) work-in-progress

Tags `0.2.1`–`0.2.10` on branch `stage-1` (off `main` `0.2.0`). Stage 1 is **not**
complete; the gate is **not met** and `0.3.0` is intentionally untagged.

- T1.1 `quant.py` (binary/ternary + group-wise), T1.2 `bitlinear.py` (fp32
  shadow, STE, INT8 acts, progressive alpha), T1.3 `replace.py`
  (`apply_layer_policy`), T1.4 `train/qat.py` + `scripts/run_stage1_qat.py`.
- First QAT run **invalid** (post-QAT fp16 WER 256% — shadow corruption; see
  `docs/LAB_NOTES.md` and `docs/stage1_results.md`).
- Debug phase: STE correctness tests (overfit-one-batch at α=1), `grad_norms` /
  `check_finite_grads` diagnostics, and a stabilized runner (fp32 AdamW, α=1,
  no grad-ckpt/act-quant/autocast, QAT-only freeze). 171 tests pass.

## 0.2.11 – 0.2.28 — Stage 1 continued on the JIIT A100 cluster (2026-10-10)

- **Quantizer audit** (`scripts/diag_quant_error.py`): per-tensor binary
  relative Frobenius error 0.648 ≈ the closed-form 0.60 for Gaussian weights;
  group-wise scaling barely changes it. Verdict: quantizer math correct; the
  earlier single-projection collapse is inherent naive 1-bit behaviour.
- **Loss controllability proven** (`scripts/diag_generation.py`): single-utterance
  QAT memorizes **exactly** at ternary (loss 0.000, greedy == label) and
  partially at binary (loss 2.8). Earlier "bug" verdict retracted.
- **Learnable scales** (`BitLinear(learnable_scale=True)`): BitNet b1.58-style
  per-tensor gain, **log-parameterized** after the first run showed gamma
  collapsing to ~0 and going negative (`gamma()` stays positive; stats logged).
- **Training tooling**: `--lr-schedule cosine`, `--train-cache`, shadow
  checkpoints + resume, `--lazy-collate`, `--eval-both`,
  `scripts/build_qat_cache.py` (streaming; 4k utts train-clean-100, 14.5 h),
  `scripts/install_fast_tokenizer.py` (fixes multi-minute slow-tokenizer stalls),
  PBS launchers (`scripts/qat_ternary*.pbs`).
- **Self-distillation QAT** (`--distill-weight`, teacher = same shadows at
  α=0; `distillation_loss`): KL ~0.4–0.6, faster CE descent than naive QAT.
- **Policies**: added `decoder_ffn` (FFN-only) for the attention-bottleneck
  ablation.
- **Runs on A100**: naive ternary `decoder_attn_ffn` plateaus at WER ~96–97
  (test-clean 50-utt); distillation ~114 @1k steps; FFN-only queued. fp16
  readouts stay clean. Full WER table and analysis in
  `docs/stage1_results.md` (updated 2026-10-10). `0.3.0` stays untagged.

## 0.2.49 — aligned-words cache, checkpoint-cadence fix (2026-10-19)

- **Aligned-words cache** (`train/aligned_cache.py`): the forced aligner's
  output is persisted (JSON, keyed by the utterance set + aligner model) so a
  retry or re-run never pays the 20–40 min alignment pass twice; a corrupt or
  missing cache is a miss, never an error. Tested (a hit must not call the
  aligner again).
- **Step-checkpoint cadence fix**: `--keep-step-checkpoints` had been nested
  inside the `--save-every` branch, so with 250/400 only multiples of 2 000 were
  ever written. The cadences are now independent.
- Stage-3 result of record: two-stage pipeline proven single-variable
  (base 16.53 → 10.81 improved streaming 19.39 → **16.63/23.41**; a deeper
  9.85 base regressed to 18.43/25.34, and the runs are bit-reproducible).

## 0.2.41 – 0.2.46 — Stage 4: the streaming algorithm is freed (2026-10-12..15)

**Headline.** The official Qwen3-ASR prefix algorithm used to *collapse* for
quantized models (WER **203.5** with repetition loops). It now runs at
**12.92 clean / 20.00 other** (320 ms; 10.59 / 16.59 at 1 000 ms) with **zero
repetition loops**, and the binary arm streams at **18.43 / 25.45** (best
measured). The recipe: **forced-aligner chunk targets** ("emit the words whose
end time falls in this chunk, then stop"; Qwen3-ForcedAligner-0.6B) + the **LSP
loop-emission policy**, from the best offline checkpoint, evaluated with the
official algorithm at 160/320/1 000 ms on clean/other.

**New code.** `data/align.py` (T4.1 aligner adapter), `data/prefix.py` (T4.2
chunk targets), `qwen/stream.py` LSP state (`LSPState`, `update_lsp`,
`emit_stable`, `flush_lsp`, `truncate_trailing_loop`), `train/sft.py`
(batched on-policy rollout, gold alignment, distillation targets, degenerate
filter), `run_stage1_qat.py` flags `--op-rollout*`, `--op-distill`,
`--aligned-targets*`, `--stream-eval-every` (in-training `STREAM` guardrails),
`eval_streaming.py` `--lsp-margin-words`, `--lsp-truncate-loops`, `--config`;
**unbiased stochastic rounding** (`--quant-stochastic 1`): ``quantize_binary`` /
``quantize_ternary`` take ``stochastic=True``, ``BitLinear`` applies it in
training mode only (evals stay deterministic), the layer policy and trainer
thread it through, and the property is tested (E[q] = W/scale in range,
integral targets never flip, the all-zero group stays zero).
Launchers: `qat_stream_{opd,distill,anneal,aligned*}.pbs`, `eval_streaming_opd.pbs`
(`TRUNC`, `CHUNK`, `CFG` overrides). Tests: 325 passed.

**Research trail (all failures kept, `docs/LAB_NOTES.md`).** DAgger on the
student's own prefixes *poisons* the model (920 WER) → streaming-state
distillation (448, metadata conflict; 350 after stripping; 278 with the
three-way mix, policy correct) → the LSP loop policy halves it (144.6) →
**aligned targets take it to ~12** (the target no longer has to be inferred from
the audio). The content/stop balance, the fixed-20-utt guardrail's divergence
from the full metric, and the "snapshot before resuming" rule are documented.

**Also this window.** Offline scale-up on the full 28 539-utt cache (ternary
**7.52**, binary at a **10.70 quantizer floor**); Stage 3 began (1.7B: FFN warm
start 22.03, anchor-free attention rollout 20.55; the aligned recipe there is
paused on cluster capacity — it needs the anchored fp32-AdamW path and a quiet
GPU). `0.3.0` stays untagged (the Stage-4 gate's `+1.5` target is not met).

## 0.1.1 — Stage 0 plan corrections

- Locked the evaluation protocol (greedy, `language=None`, `max_new_tokens`
  1024/32, record versions).
- Scoped the append-only invariant to Stage 4 / T4.4 (Qwen baseline is
  pseudo-streaming).
- Streaming feasibility policy (offline full; streaming full clean + fixed
  500-utt other subsample) and the T0.0 spike.
- Noted `configs/base.yaml` `unfixed_token_num: 1` is intentional.

## 0.1.0 — Repo scaffold + project plan

- Package layout (`src/asr1bit/{quant,bitlinear,replace,data,qwen,train,eval,deploy}`),
  `tests/`, `configs/`, `docs/PLANNING.md`, initial `README.md`, `pyproject.toml`.
