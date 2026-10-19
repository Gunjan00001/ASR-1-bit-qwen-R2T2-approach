# Code guide — how this repository is organised and how to keep it readable

> Read this before adding or changing code. It exists because the first version
> of this project had a **900-line trainer script** with almost no comments; it
> was split into focused modules on 2026-10-15, and this file records the rules
> that keep it that way. If you are looking for the *science*, read
> [`DEEPDIVE.md`](DEEPDIVE.md); this file is about the *code*.

## 1. The five rules

1. **One concern per file.** If a file has two reasons to change, it is two
   files. Practical size guide: *a file over ~400 lines is almost always two
   modules wearing one name*. (History: the trainer script 897 → 571, and
   `train/sft.py` 476 → `samples/gold/filters/rollout/examples` — the largest
   src file today is `qwen/stream.py` (427), which holds three closely-related
   concerns — a known, recorded exception pending its own split.)
2. **The CLI stays thin.** `scripts/*.py` parse arguments and orchestrate; the
   *logic* lives in `src/asr1bit/**` where it can be imported and unit-tested.
   `scripts/run_stage1_qat.py` is the model for this: its docstring is the run's
   reading guide, and every phase of `main()` has a comment.
3. **Docstrings say *what*; comments say *why*.** A comment that restates the
   code is noise. A comment that records a reason, a failure, or a trap is
   gold. Compare:
   ```python
   # ✗ restates the code
   x = x + 1                                    # increment x

   # ✓ records a reason you cannot recover from the code
   # adam8bit trades ~4x optimizer memory for a little precision; the fp32 path
   # is the conservative choice for the 1.7B, where a bad update is permanent.
   optimizer = build_optimizer(model, lr=args.lr, use_8bit=(args.optimizer == "adam8bit"))
   ```
4. **Delete dead code.** A stub module that only holds a docstring
   (`train/opd.py`, removed 2026-10-15) is worse than nothing: it implies work
   that does not exist. Unused imports, flags and helpers go the same way.
5. **Every new behaviour gets a test, and the suite stays green.** `pytest -q`
   (337 tests as of 2026-10-16) plus `ruff check .` are the contract; a refactor that has not run
   both is not finished. Tests import from `src/asr1bit/**`, not from the CLI
   scripts.

## 2. Where everything lives

```
src/asr1bit/
  quant.py            binary/ternary quantizers (+ unbiased stochastic rounding)
  bitlinear.py        the QAT layer: shadow weights, STE, alpha, learnable scales
  replace.py          swap nn.Linear -> BitLinear by layer policy
  data/
    load.py           LibriSpeech iteration, audio cache, utterance dataclass
    align.py          Qwen3-ForcedAligner adapter (word timestamps)
    prefix.py         chunk arithmetic: words -> per-chunk targets
  qwen/
    backends.py       model loading, generation, the portable streaming path
    stream.py         LSP policy: append-only emission, loop truncation
  train/
    qat.py            losses (CE, KL distillation), label construction
    samples.py        the streaming-sample record + prompt/continuation + mixing
    gold.py           aligning the transcript against what the model already said
    filters.py        degenerate-output detection (the DAgger guard)
    rollout.py        batched replay of the official streaming algorithm
    examples.py       on-policy + distilled example builders
    dataset.py        ordinary transcription examples, collation, batching
    aligned.py        aligned-target examples (the streaming fix)
    aligned_cache.py  caches the aligner's words (retries skip the expensive pass)
    onpolicy.py       rollouts, distillation targets, degenerate filtering
    inloop_eval.py    cheap in-training evaluations (guardrails, not metrics)
    checkpoints.py    save / load / selectable step checkpoints
    schedules.py      warmup + cosine LR
    model_prep.py     small model-patching helpers
  eval/
    wer.py            WER/CER
    harness.py        evaluation harness shared by scripts
scripts/
  run_stage1_qat.py   trainer CLI (orchestration only — see its docstring)
  eval_streaming.py   streaming evaluator (official algorithm + LSP policy)
  eval_bin_es.pbs     examples of focused PBS launchers, one per experiment
  ...                 ~30 more launchers (.pbs/.sh), all committed with the runs
tests/                one test file per module, named after it
docs/                 planning, results, lab notes, this guide
```

## 3. Commenting style used across the codebase

- Module docstrings explain **why the module exists** and the trap it encodes
  (e.g. `train/inloop_eval.py`: *"the streaming guardrail uses 20 utterances and
  has diverged from the full metric four times — never promote on it"*).
- Function docstrings give arguments, returns, and **the failure they prevent**
  when relevant.
- Inline comments mark **phases** in long functions (see `main()` in the trainer)
  and **gotchas** (`# frozen modules must not require grad, or the optimizer
  silently allocates huge buffers`).

## 4. How to split a too-large file safely (the recipe used here)

1. Group functions by *concern* and write down a one-line purpose per group —
   that list is your new file list.
2. Move the bodies **verbatim** with an AST extraction (`ast.get_source_segment`)
   rather than retyping; retyping introduces silent typos.
   *Decorators live **outside** the AST node* — re-attach `@dataclass`,
   `@property` etc. by hand, or a dataclass silently becomes a plain class that
   "takes no arguments" (exactly what happened to `StreamingSample`).
3. Give each new module a docstring (why it exists) and a minimal import header.
4. Run `ruff check . --fix` and `pytest -q` and read the failures — missing
   imports (F821) and unused ones (F401) locate themselves.
5. If tests reached into the old file's namespace, either update them to import
   the new modules, or add a *documented* shim (see `tests/test_trainer_rollout.py`)
   rather than weakening the tests.
   **Grep-verify every patch**: a `str.replace` whose target does not match fails
   *silently*, and a later `ruff --fix` will then "helpfully" delete the import
   you thought you had added (this cost three round-trips during the sft.py
   split). Run `ruff check` (read-only) after each patch, and only `--fix` once
   the code is coherent.
6. Verify the CLI still runs: `python scripts/<script>.py --help` (launchers
   depend on it) and re-run the suite once more.

## 5. Telemetry every run must emit

- `EVAL {...}` lines: offline WER per precision, one JSON object per evaluation.
- `STREAM {...}` lines: the cheap in-training streaming guardrail.
- a results JSON next to the checkpoints with the full config, the history and
  the timings — this is what `docs/` and the figures are generated from.

## 6. Repo hygiene — what must never be committed

- **Secrets** (tokens, API keys, SSH material) live in the environment or in
  `~/.commandcode/` / `~/.ssh/`, never here. The repo's collected evidence:
  `settings.json` is a *permissions allowlist* (no secrets), the git-push helper
  reads `GITHUB_TOKEN` from the env and redacts it from output, and the Kaggle
  doc *instructs* how to store tokens as Kaggle Secrets. If a real secret ever
  lands in a commit: **rotate/revoke it first** — deleting the file afterwards
  does not undo the exposure, because the history still holds it; history
  rewrites are a last resort and require a forced push.
- **Tool / session state.** `.commandcode/` (the agent assistant's permissions
  and learned-taste files) was swept into a refactor commit by `git add -A`;
  it is now untracked and `.gitignore`d (the files stay on disk so the tool keeps
  its state). Before any broad `git add`, run `git status` and *see* what would
  be swept.
- **Caches and models.** `data/` (checkpoints, several GB) is ignored;
  `artifacts/**` are tracked *deliberately* — the docs and figures are generated
  from those JSONs. Keep result JSONs; never model weights.
- **No network services** exist in this codebase (no HTTP servers), and model
  downloads are anonymous hub fetches of Apache-2.0 weights, i.e. there is no
  API key to leak.

## 7. Documentation duties (non-negotiable)

- `docs/LAB_NOTES.md` — **append-only**: every attempt, successful or failed,
  with its number and root cause. Never rewrite history; add a correction.
- `docs/ACTIVITY_LOG.md` — the live status table (measured rate + ETA per job)
  and a session log entry per working round.
- On a promotion: update `README.md`, `PROJECT_STATUS.md`, `HANDOFF.md` and
  `stage*_results.md`, and tag `MAJOR.MINOR.BUGS` (no leading `v`).
- This guide: when you split a file or change a convention, update §1–§2 in the
  same commit.
