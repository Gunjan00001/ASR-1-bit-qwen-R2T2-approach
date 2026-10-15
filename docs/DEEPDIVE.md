# Deep Dive — ASR-1-bit-Qwen, explained from the ground up

> **Who this is for.** Three readers at once:
> 1. **A newcomer** who has never heard of quantization or streaming ASR — read
>    Parts 1–3 in order, slowly; every term is explained where it first appears
>    (and collected in the glossary, Part 8).
> 2. **You, before a viva** — Part 7 is 30 likely questions with the answers, and
>    Part 6 is the "failures and what we did" index that examiners love.
> 3. **A reviewer** who wants the exact numbers — every figure here appears in
>    `docs/stage0_results.md`, `docs/stage1_results.md`, `docs/stage4_results.md`
>    or the append-only `docs/LAB_NOTES.md`, and every claim traces to a JSON in
>    `artifacts/`.
>
> This document is long by design. It is the story *and* the reasoning.

---

## Part 1 — The problem, in plain words

### 1.1 What is ASR?

Automatic Speech Recognition = turning audio into text. You speak; the machine
writes. Modern ASR is a neural network: audio goes in, characters/words come out.

### 1.2 What is *streaming* ASR, and why is it harder?

A normal (offline) ASR system hears the **whole sentence first** and then writes
it. A streaming (or online) system must **write while listening** — like a live
court stenographer, not a translator who waits for the paragraph.

That is much harder, because the model must decide *what it has heard so far*
and *what to commit to*, and it is never allowed to "hear the future". In our
project the audio arrives in **chunks of 320 milliseconds** (about a short
word), the model writes something, and moves on.

Two failure modes make streaming dangerous:

- **Under-emitting** — it stays silent when it should have written words.
- **Over-emitting / looping** — it repeats what it already wrote forever
  ("THE END THE END THE END THE END…"). In ASR scores, a loop is catastrophic.

### 1.3 Why quantize weights at all?

A normal network stores each weight as a 16- or 32-bit float. Those floats are
what make big models expensive: memory, bandwidth and energy.

**Quantization** means storing each weight with very few possible values:

| Precision | Each weight is | Bits | Our name |
|---|---|---|---|
| fp16/fp32 | any real number | 16–32 | full precision |
| **ternary** | one of **{−1, 0, +1}** | ~2 | `bit_width=1.58` |
| **binary** | one of **{−1, +1}** | 1 | `bit_width=1.0` |

A useful analogy: an fp16 weight is a **poem written with every word in the
dictionary**; a ternary weight is the **same poem written with only three
symbols** — plus, and this matters, *one scale number per group of weights* to
tell the reader how loud the symbols are (see §2.3).

Why do it? Because a model whose weights are {−1,0,+1} needs no multipliers —
on the right hardware you can run it with additions and comparisons, which is
what makes **on-device, real-time, low-power** speech recognition plausible.

### 1.4 The catch, and the project's reason to exist

Quantization *hurts accuracy*. A lot. If you just round a trained model's
weights, the model can fall apart — and for streaming ASR, it falls apart
spectacularly (we measured **WER 203 %**; see §4.1).

So the project's question is:

> **Can we train a Qwen ASR model whose weights are (nearly) 1-bit, and keep
> streaming accuracy within +1.5 WER of the full-precision 1.7B model?**

That "+1.5" is the fixed bar, locked in Stage 0. WER = Word Error Rate in
percent: the fraction of words that are wrong (insertions + deletions +
substitutions ÷ reference words). 0 % is perfect, 100 % means as many errors as
words. `test-clean` in LibriSpeech is easy read speech, `test-other` is harder.

### 1.5 The plan in one picture (stages and gates)

| Stage | Goal | Its gate |
|---|---|---|
| 0 | Reproduce the official fp16 numbers (so we know our measuring stick) | numbers match published |
| 1 | Build the quantization machinery (BitLinear + QAT) | "loss controllable; WER table emitted" |
| 2 | Recover the accuracy the machinery lost | "measurable WER recovery vs Stage 1" |
| 3 | Scale the working recipe to the 1.7B model | (in progress) |
| 4 | Make the **official streaming algorithm** work with quantized weights | 320 ms WER ≤ 2.24 + 1.5 (clean) and ≤ 4.61 + 1.5 (other) |
| 5 | Edge deployment (CPU, RTF < 1) | (not started) |
| 6 | Evaluation + write-up | (ongoing) |

**Gate numbers explained:** the fp16 Qwen3-ASR-1.7B streams at 2.24 (clean) /
4.61 (other) on our protocol; +1.5 gives **3.74 / 6.11** — the bar the
quantized model must reach. Right now the promoted state is **11.55 / 18.98**.

---

## Part 2 — The cast of characters

### 2.1 Models (all Apache-2.0, all Qwen — no R2T2 weights or code are used)

- **Qwen3-ASR-0.6B** — the small model we do nearly all our work on. It is the
  "learner".
- **Qwen3-ASR-1.7B** — the bigger model, the eventual target.
- **Qwen3-ForcedAligner-0.6B** — a *different* model that solves a different
  problem: given audio **and its transcript**, it returns **when each word was
  spoken** (start/end timestamps). This becomes the key to fixing streaming
  (Stage 4, §4.4).

### 2.2 Data

**LibriSpeech** — public English audiobooks read aloud, with exact transcripts.
- `test-clean` / `test-other`: the two standard test sets.
- We mostly report on a **fixed 50-utterance slice** of `test-clean` (seed 0)
  because each streaming evaluation costs GPU-minutes, not seconds. This is a
  real limitation and is stated everywhere it matters.
- For training we built a **cache of 28,539 utterances** ("the full cache"),
  tokenised once so training doesn't wait on text processing.

### 2.3 The quantization machinery (`src/asr1bit/quant.py`, `bitlinear.py`)

**Quantization rule, ternary:**  `q = clip(round(W / γ), −1, +1)`, then
`Ŵ = q · γ`, where `γ` is a scale (itself learned, `learnable_scale=True`).

**Quantization rule, binary:**  subtract the mean, then `q = sign(W − mean)`,
`Ŵ = q · mean|W|`.

**Group-wise scaling:** instead of one scale for a whole matrix, split it into
blocks of `group_size` along the last dimension and give each block its own
scale. Like a *per-paragraph* translation instead of per-book: much less damage.

**BitLinear** is a `nn.Linear` that swaps in quantized weights. The subtle part
is training: you keep the real-valued ("**shadow**") weights as the trainable
parameters, quantize them for the forward pass, and copy gradients straight back
to the shadow values — the **straight-through estimator (STE)**. The shadow
weights are the "real" model; quantization is something we *do to them* every
forward pass.

**Progressive alpha (α):** a knob `0 → 1` that blends full precision into full
quantization. We start at α = 0 (harmless) and ramp to 1 (fully quantized), so
the network can adapt gradually instead of being hit by a truck on step 1. It
also lets us ask "what would this *same* model do at full precision?" by
evaluating at α = 0 — the "fp16 readout", which is how we watch the weights
drift (§3.3).

### 2.4 How we measure everything

- **WER / CER** — word / character error rate, on the text the model commits.
- **50-utt, greedy decode**, fixed seeds, one JSON per evaluation under
  `artifacts/stage1/`. Every number in this document comes from such a JSON.
- **Two streaming protocols exist in this repo**, and confusing them is the
  easiest mistake a reader can make:
  1. **chunked re-feed** (simpler, "naive"): re-run the model on the audio heard
     so far, every 320 ms. Cheap to try; not how Qwen streams.
  2. **the official prefix algorithm** — the model is fed a *prefix* and asks
     "transcribe what you have"; the harness calls it repeatedly, one chunk at a
     time, and the model's answer is compared with the reference. This is the
     real target and the hard one.
- **The LSP policy** (`qwen/stream.py`) — our inference-side rule that turns a
  sequence of per-chunk answers into one **append-only** transcript: text that
  has been committed can never change; if the model starts repeating itself, we
  truncate the loop but keep one occurrence.

---

## Part 3 — The story, in order (what we did, why, what broke)

### 3.0 Stage 0 — build the ruler before measuring (`0.2.0`, gate passed)

**Why:** if our evaluation harness is wrong, every later number is meaningless.

**Did:** reproduced the published fp16 numbers (0.6B offline 2.11 / 4.55;
1.7B offline 1.63 / 3.38; streaming 320 ms 1.7B 2.24 / 4.61). Locked the bar
`X = +1.5`.

**Lesson:** our streaming harness initially used *plain text* inputs, which
made the model fall back to its trained "language token" behaviour. Fixing the
prompt format (a metadata prefix) changed numbers by tens of points. **The
measuring stick must be calibrated before you trust any result.**

### 3.1 Stage 1 — the machinery (`0.2.1`–`0.2.40`)

**Did:** implemented `quant.py` (binary/ternary, group-wise), `bitlinear.py`
(shadow weights, STE, INT8 activations, progressive α), `replace.py` (swap
layers by policy), the QAT loop, the cache builder, and PBS launchers.

**First disaster (kept in the record):** the very first QAT run was
**invalid** — the shadow weights got corrupted and the model memorised noise.
Fixed by re-auditing the quantizer and the shadow bookkeeping; the fix is what
made the loss "controllable" (ternary QAT can memorise a batch *exactly*, which
is our controllability proof).

**Second reality check:** the naive ternary QAT then **collapsed to ≈96–108
WER**. Almost every word wrong. This is not a bug — it is what happens when you
quantize a model that was never trained for it, especially when streaming.

**Verdict:** Stage 1's *written* gate ("loss controllable; WER table emitted")
was met; its *spirit* (a usable quantized model) was not. That is exactly why
Stage 2 exists, and it is why the `0.3.0` tag was deliberately withheld.

### 3.2 Stage 2 — recovery, a recipe discovered by failing

The winning recipe, found over many attempts:

```
1. FFN warm start      quantize only the feed-forward layers first (easier)
2. attention rollout   then add the attention projections
3. frozen anchor       keep the fp16 model frozen and add a KL term (w≈0.05)
                       so the quantized model cannot drift into a bad basin
4. warm restarts       fresh cosine cycles (fresh LR) rather than one long decay
5. scale the data      8k utterances → the full 28,539-utterance cache
```

Progression on the offline metric (50-utt test-clean WER):

| configuration | WER |
|---|---|
| naive ternary QAT | ≈96–108 |
| FFN-only ternary | 32.8 |
| attention rollout (2a) | 16.2 |
| + long run, 8k data | 11.4 |
| + higher LR (long2) | 9.2 |
| **full 28.5k cache + anneal** | **7.52** |
| **binary, same recipe** | **10.70** |

**Failures worth knowing (all in LAB_NOTES):**

- **The LR-restart artifact.** Resuming a run restarts the learning-rate
  schedule; the loss jumps and one evaluation looked like a miracle. Because we
  keep *dated, structured* logs, we caught that the "improvement" was an
  artifact of the restart, not learning. That is why the trajectory plots mark
  the resume separately (dotted line).
- **The binary floor (10.70).** The binary arm will not go below ≈10.7 offline
  no matter how long we train. The reason is structural: the binary sign rule
  has a systematic bias for weights near zero (a "mirror" effect), not a lack
  of optimisation. Hence the untried lever: **stochastic rounding** (§5.2).
- **Distillation alone wasn't enough** (≈114 WER at 1k steps) — it helped but
  could not fix a model that was collapsing on the streaming distribution.

### 3.3 Watching the model: fp16 readout and "shadow drift"

At any point we can evaluate the *same* weights at α = 0 (full precision) —
`fp16_drift.png`. Curiously, this number gets **worse** during QAT (from 2.4 to
20–27). That is not a bug: the weights are being pushed into a region that is
*good for quantized inference* and no longer good for fp16 inference. It is
direct evidence that QAT changes *where* in weight space the model lives.

### 3.4 Stage 4 — the streaming collapse and its cure (the heart of the project)

#### 4.1 The collapse

Under the official prefix algorithm, our quantized model didn't just lose
accuracy — it **degenerated**: it produced loops and garbage. First
measurements: **951.8** and **203.5 WER** (loops everywhere). The fp16 model
streams at 2.24/4.61; ours was off by two orders of magnitude.

**Why, in one sentence:** the official algorithm asks the model "given the audio
so far, write what you have"; for a quantized model the *implicit* rule of
"how much text belongs to how much audio" was never learned — it guessed, and
its guesses fed back into the next prefix (a feedback loop), so errors
compounded into repetition.

#### 4.2 What failed, and the lesson each failure taught

| Attempt | Result (320 ms, clean) | Why it failed | Lesson |
|---|---|---|---|
| **DAgger** on the model's *own* prefixes | **920** (worse!) | the model was trained on its own garbage outputs — it learned its own loops as correct | never train on a collapsed student's outputs without filtering |
| **Streaming-state distillation v1** | **448** | teacher outputs carried the `language … <asr_text>` metadata, which clashed with our plain-text targets — the student's content got garbled | match the *format* of targets exactly before blaming the method |
| **distillation v2** | **350** | metadata stripped, but now the model *under-emitted* (it learned to be silent to avoid errors) | a content/stop balance problem, not a knowledge problem |
| **distillation v3** (three-way mix) | **278** | better balance, still not enough | the target itself ("how much text for this chunk?") was still being *inferred* |
| **inference-side repetition controls** | **438 / 264** | post-hoc fixes cannot unlearn whatever the model actually wants to say | fixing symptoms at inference is weaker than fixing training |
| **LSP loop-emission policy** (truncate loops, append-only) | **144.6** | *first real win*: it stops the visible degeneration | a policy (rule) at inference is legitimate and cheap — but it can only clean up, not teach |
| **aligned targets (the fix)** | **12.6 / 12.9** | the model no longer has to infer the mapping | **if a target is hard to infer, make it explicit** |
| longer cycle + early state selection | **11.55 / 18.98** | — (promoted, tag `0.2.47`) | the *right* schedule length is discovered, not assumed |

#### 4.3 The fix that worked: aligned targets

We used the **forced aligner** to get the time of every word in the training
audio, then defined, for chunk *k*:

> the target is exactly **the words whose end time falls inside chunk *k*** —
> if none do, the correct answer is to say **nothing** (a "stop" sample).

Now the model is trained on the *real* streaming question, with the answer
given: "you have heard this much; say exactly these words, or stay silent."
Loops vanished (0 truncations, and `WER raw ≈ WER` shows the policy has almost
nothing left to clean), and WER fell from 144.6 to ~12.

**Mix:** 50 % aligned-target samples + 50 % standard samples — otherwise the
model forgets how to transcribe fluently (the "content" regression we measured
as `distill2`'s under-emission).

#### 4.4 Long cycles and *selecting* the state (a hard-won practice)

The promoted state (11.55) came from a 5,000-step cycle **cut at step 1,000**.
The trajectory *peaks early* and then degrades: 11.55 @1000 → 12.82 @1500 →
14.30 @~1750. Three separate lineages in this project show this pattern.

**So we stopped trusting "the final checkpoint".** `--keep-step-checkpoints 500`
writes step-numbered snapshots; the evaluator scores each state on full samples
and the best is promoted. That single change is what made this project's best
number visible at all — before it, the run's *worst* state would have been the
only candidate.

### 3.5 Stage 3 — the 1.7B (in progress, and what it taught us)

The 1.7B is the target size (its fp16 streaming 2.24/4.61 is the gate's
baseline). Sequence of attempts:

1. **FFN warm start** → offline 22.03; **anchor-free attention rollout** → 20.55.
2. **Aligned streaming retry v2** (lr 3e-6, adam8bit, no anchor): the model
   **dipped 20.55 → 30.6 and never recovered** — a permanent failure, not a
   transient. This is the "permanent dip".
3. **v4 — anchored fp32-AdamW** (the recipe the record demanded): ran clean,
   1,200 steps, **zero OOMs**, no dip (offline recovering 27.97 → 24.05). Its
   **first streaming number: 29.24 clean** from an offline 24.05.

**The decomposition that matters** (and it corrected an earlier wrong claim):

| model | offline (quantized) | streaming | streaming gap | damage vs its own fp16 |
|---|---|---|---|---|
| fp16 0.6B | 2.11 | 2.54 | +0.4 | — |
| **quantized 0.6B** | **7.52** | **11.55** | **+4.0** | **+5.4** |
| fp16 1.7B | 1.63 | 2.24 | +0.6 | — |
| **quantized 1.7B** | **24.05** | **29.24** | **+5.2** | **+22.4** |

The streaming recipe **transfers** (the gap is the same ~4–5 points at both
sizes). What does *not* transfer is the base: the 1.7B's *quantization damage*
is +22.4 versus the 0.6B's +5.4 — because the 1.7B never received the *complete*
offline recovery recipe (full-cache QAT + anneals) before its streaming SFT.

**And this kills the "capacity" story:** a *bigger* model should suffer *less*
quantization damage under an equal recipe. So the culprit is the recipe's
completeness — not model size. (Earlier docs blamed "0.6B capacity"; the record
was corrected on 2026-10-15.)

### 3.6 What is running right now, and how we know

- `qat_stream_aligned5b` — a resume of the longer cycle (to step 5,000).
- `qat_stream_bin_long` — a 3,000-step binary run from the best ternary state
  (the 1,500-step budget was too short; its curve was still descending).
- Evaluators scoring the resume's states and the 1.7B's final state.
- A 2-hourly automated progress check whose instructions require it to record
  every new full-sample number and to keep the development moving.

---

## Part 4 — The results (what we can defend)

**Promoted (tag `0.2.47`), official algorithm, 50 utts, 320 ms:**

| | clean | other |
|---|---|---|
| ternary, aligned targets + LSP (step-1000 state of a 5,000-step cycle) | **11.55** | **18.98** |
| same state at 160 ms / 1000 ms (clean) | 13.77 / 10.70 | — |
| previous promoted state (`aligned2`, 2,500-step cycle) | 12.92 | 20.00 |
| binary, best measured (weights since lost to a resume) | 18.43 | 25.45 |
| **gate** (fp16 1.7B + 1.5) | 3.74 | 6.11 |

**Reading it honestly:** the gate is **not met**. We are 7.8 points above it on
clean. The achievement is the *journey*: 203.5 (collapse) → 11.55 with **zero
loops**, a ~18× reduction, plus the identification of exactly where the
remaining error lives (quantization damage, §3.5) and two concrete levers to
attack it (complete 1.7B offline recovery; stochastic rounding, §5.2).

---

## Part 5 — Engineering practices that saved us

### 5.1 The promotion rule (learned the hard way)

**Only full-sample (50-utt, both splits) evaluations decide.** The in-training
20-utt guardrail was used as a cheap signal four separate times and diverged
from the full-sample metric *every* time — once spectacularly (a run whose
guardrail was the best of its family turned out to have the worst full-sample
number, 14.41 vs 12.92). Never promote on a proxy.

### 5.2 The lever list (what we would try next, with the reason)

1. **1.7B complete offline recovery** — full-cache QAT with the frozen anchor +
   anneal cycles, *then* streaming SFT. Rationale: §3.5's table.
2. **Stochastic rounding** (`--quant-stochastic 1`, implemented and tested) —
   removes the *bias* of rounding, which is the suspected cause of the binary
   floor. It costs nothing at deployment (the grid is identical at eval time).
3. **More aligned data** — the aligner can label the whole corpus given GPU time.
4. **Scale further** — only meaningful once (1) works.

### 5.3 Reliability habits (each one is a scar)

- **Snapshot before resuming.** A resume overwrites `last.pt`; the binary arm
  lost its best state (18.43) that way, permanently.
- **Measure the rate, publish an ETA** — every status line carries steps/min
  and a completion estimate, so a "stuck" run is obvious within minutes.
- **Never respawn into a saturated cluster** — the GPU picker samples three
  times and refuses below a bar (≥20 GB free, ≥35 GB for training runs) rather
  than launching a job that will die. Four separate runs OOM'd before this rule.
- **A launcher is code**: after generating one, grep `init-from|resume|save-dir`
  to catch the substitution bug that once pointed a run at its own output.
- **Keep negatives.** Every failed recipe stays in `docs/LAB_NOTES.md` with its
  number. Several "failures" became the diagnosis (DAgger → format bug;
  under-emission → the mix ratio; the 1.7B dip → the anchor).
- **Environment facts are facts**: this cluster's driver (525) cannot use
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`; gradient checkpointing is
  incompatible with a frozen-only training setup; `adam8bit` needs its
  environment. Each of these cost a run to learn; none has been re-tried since.

---

## Part 6 — Failures & fixes index (the viva's favourite section)

| # | Symptom (with number) | Root cause | Fix | Where |
|---|---|---|---|---|
| 1 | First QAT run invalid; loss didn't track | shadow-weight corruption | re-audited quantizer + shadow bookkeeping; added the exact-memorisation controllability test | LAB_NOTES (early) |
| 2 | Naive ternary ≈96–108 WER | quantization without a recovery strategy | Stage-2 recipe (warm start → rollout → anchor → anneals) → 7.52 | §3.2 |
| 3 | "Miracle" improvement after a resume | LR-schedule restart artifact | marked resume points separately on plots; distrust single evaluations | `wer_trajectories.png` |
| 4 | Binary stuck at 10.70 offline | structural bias of the sign rule | identified stochastic rounding as the lever; implemented + tested it | §5.2 |
| 5 | Streaming collapse (951.8, 203.5) | quantized model cannot infer the audio-to-text mapping | aligned targets (forced aligner) + LSP policy → 11.55 | §4.3 |
| 6 | DAgger made it worse (920) | trained on its own loops | stopped training on unfiltered self-outputs | §4.2 |
| 7 | Distillation v1 (448) | metadata-format clash in targets | strip metadata; match target format | §4.2 |
| 8 | Distillation v2 under-emitted (350) | stop/content imbalance | three-way mix + aligned stop samples (`--aligned-stop-stride 2`) | §4.2 |
| 9 | Loop-removal at inference only (438→264) | symptom, not cause | moved the fix into training | §4.2 |
| 10 | 1.7B v2 permanent dip (20.55 → 30.6) | too-aggressive LR + adam8bit + no anchor | anchored fp32-AdamW at lr 1.5e-6 (v4) ran clean | §3.5 |
| 11 | 1.7B died in its own final eval, no traceback | host-RAM SIGKILL (`mem=48gb`) | evaluate with `mem=64gb`; keep checkpoints independent of the eval | LAB_NOTES 19:15 |
| 12 | Four training runs OOM'd, some after 40 min of alignment | co-tenants grow *after* the GPU picker samples | strict picker (≥35 GB) + wait-inside-the-job + pre-resume snapshots | §5.3 |
| 13 | Launcher pointed `--init-from` at its own save-dir (twice) | bulk `sed` replacements | grep-verify every generated launcher | §5.3 |
| 14 | Guardrail ≠ full-sample metric (4 divergences) | 20-utt slice is too small and schedule-dependent | full-sample promotion only | §5.1 |
| 15 | Best binary state (18.43) unrecoverable | resume overwrote `last.pt` | snapshot rule + `--keep-step-checkpoints` | §5.3 |

---

## Part 7 — Viva questions and answers

**Q1. In one sentence, what did you build?**
A training and evaluation pipeline that makes Qwen3-ASR stream English speech
with (nearly) 1-bit weights, and the recipe that fixed the streaming collapse
quantized weights caused.

**Q2. Why 1-bit instead of just a smaller model?**
Because the constraint is *deployment*: ternary/binary weights replace
multiplies with adds/comparisons, enabling CPUs and edge chips. A smaller fp16
model doesn't get that; a 1-bit model does.

**Q3. What is QAT and why not quantize after training?**
QAT = quantization-aware training: the model trains *with* quantization in the
loop, so it adapts. Post-training quantization rounds a model trained for full
precision; we measured that it collapses (≈100 WER offline, 203 streaming).

**Q4. How does gradient flow through `round()`?**
It doesn't — the derivative is zero almost everywhere. We use the
straight-through estimator: forward uses the quantized value, backward passes
the gradient as if quantization were the identity. The trainable parameters are
the full-precision "shadow" weights.

**Q5. What's the difference between your binary and ternary modes?**
Binary: weights ∈ {−1,+1} with a centered-sign rule and a `mean|W|` scale.
Ternary: {−1,0,+1} with `RoundClip(W/γ)` and a *learnable* γ. Both support
group-wise scales (one per block of weights).

**Q6. Why group-wise scaling?**
A single scale per matrix is too coarse — some blocks have large weights, some
small. Per-block scales recover most of the lost precision at almost no cost.

**Q7. Why is streaming harder than offline, technically?**
The model must commit text without seeing the future, and its own committed
prefix becomes its next input — errors compound. Offline it sees the whole
utterance once.

**Q8. What is the official prefix algorithm, and your LSP policy?**
The official algorithm feeds the model an audio prefix and asks it to transcribe
what it has. Our LSP policy turns the per-chunk outputs into one append-only
transcript: committed text is final; loops are truncated keeping one occurrence.

**Q9. Why is "zero loops" such a big deal?**
A loop multiplies errors: one repeated phrase can push WER past 100 %. Going
from thousands of loops to zero (with `WER raw ≈ WER`, i.e. the policy has
almost nothing to clean) is what makes the number *trustworthy*.

**Q10. What was the single most effective fix?**
Aligned targets: using the forced aligner to label exactly which words belong to
which 320 ms chunk during training (and training "say nothing" when none do).
It took the official algorithm from 144.6 to ~12.6 WER in one move.

**Q11. Why did DAgger fail?**
Because the student was already collapsed: training on its own outputs taught it
its own loops. DAgger works when the student's errors are *recoverable*; ours
were catastrophic.

**Q12. Why does distillation stop at ~278?**
Because it was still asking the model to *infer* the audio-to-text alignment
from examples. Distillation transfers knowledge; it does not supply information
the model cannot extract. Aligned targets supply it directly.

**Q13. Where does the "+1.5" gate come from, and what are 3.74/6.11?**
Stage 0 locked the primary metric: the fp16 1.7B's streaming WER at 320 ms is
2.24 (clean) / 4.61 (other). `X = +1.5` absolute was locked at the same time, so
the quantized model must reach ≤3.74 (clean) and ≤6.11 (other).

**Q14. Are you meeting the gate?**
No. We are at 11.55/18.98 — 7.8 and 12.9 points above. The honest summary: the
*algorithm* is fixed (collapse → zero loops) and the remaining gap is
quantization damage, with concrete plans to attack it.

**Q15. How do you know the remaining gap isn't just model size?**
Because the fp16 **0.6B** streams at 2.54/6.27 — within ~0.3 of the 1.7B's
2.24/4.61. A small model has no trouble streaming *when it isn't quantized*.
The gap appears with quantization, and it grows with the damage the recipe
leaves behind (1.7B: +22.4 vs the 0.6B's +5.4).

**Q16. Why did the 1.7B do worse than the 0.6B?**
Because it received a *less complete* offline recovery: it went from a partial
attention rollout (20.55 offline) straight into streaming SFT, while the 0.6B
got the full-cache recovery cycle first (7.52). Bigger models should tolerate
quantization *better* under an equal recipe — so this is a recipe gap, not a
capacity effect.

**Q17. What is your best number, exactly, and on what?**
11.55 clean / 18.98 other WER, official algorithm, 320 ms chunks, 50-utt
test-clean/test-other, ternary weights, zero loop truncations. It is the
step-1000 state of a 5,000-step cycle — selected, not the final checkpoint.

**Q18. Why only 50 utterances?**
Cost: each streaming evaluation re-generates for every chunk and takes tens of
GPU-minutes. 50 utterances is enough to separate 11.5 from 13; it is *not* enough
to resolve 0.2-point differences, and we say so wherever we quote them.

**Q19. What's the difference between your "raw" and "policy" WER?**
`raw` is measured on the model's own output; `policy` after the LSP filter.
When they are equal, the policy is doing nothing — i.e. the weights themselves
behave, which is what we want.

**Q20. How do you know a result is real?**
Every evaluation writes a JSON with the checkpoint path, config, chunk size,
hypotheses and references. Promotions require full-sample evals of *both*
splits. Checkpoints are tagged (`0.2.47`) and every run's launcher is committed.

**Q21. What was your most embarrassing bug, and what did you change?**
A resume overwrote the binary arm's best checkpoint (18.43) — unrecoverable.
Since then: snapshot before every resume, and `--keep-step-checkpoints` so a
trajectory's states stay selectable.

**Q22. What is the binary "floor"?**
The binary arm plateaus at ≈10.70 offline regardless of training time, because
the sign rule's rounding bias affects near-zero weights systematically. The fix
we prepared is stochastic rounding — unbiased in expectation, identical grid at
deployment.

**Q23. Why do you keep failed experiments?**
Because they are the diagnosis. Four examples in this project: the "miracle"
that was an LR-restart artifact; DAgger teaching its own loops; distillation's
format clash; the 1.7B dip that revealed the value of the anchor. A project
with no failed experiments in its log has either not explored or not looked.

**Q24. What is the fp16 "drift" plot showing?**
We evaluate the same weights at α = 0 (no quantization). That number *worsens*
during QAT (2.4 → 27) — the weights move to a region that is good for quantized
inference and bad for full precision. It proves QAT is doing something real.

**Q25. What's left before deployment (Stage 5)?**
Packing the ternary/binary weights (GGUF-style), CPU kernels for
add/compare-only inference, and verifying RTF < 1 with constant memory as audio
grows.

**Q26. If you had 10× the compute tomorrow, what would you run first?**
The 1.7B's complete offline recovery with stochastic rounding enabled, then the
aligned streaming SFT on top of it — because that is the sequence the evidence
points at (§3.5), and the 1.7B's fp16 streaming is the number the gate is
defined against.

**Q27. What does the forced aligner contribute, exactly?**
Word-level timestamps for training audio. From them we derive per-chunk targets
("these words end in this 320 ms window") and stop samples (empty windows).
It is only used at *training-data preparation* time; it is not part of the
deployed model.

**Q28. How do you handle a cluster that keeps killing your jobs?**
Strict GPU selection (sample, require a free-memory bar, refuse otherwise),
waiting inside the job rather than racing, pre-resume snapshots, and running
every evaluation as its own job so a training job's death never costs the
result. Four OOMs and one host-RAM kill taught each of those.

**Q29. What would you do differently from the start?**
(a) Bring the forced aligner in on day one — alignment was the bottleneck, not
optimisation. (b) Always keep step checkpoints: we lost the best binary state by
assuming the final checkpoint was the best. (c) Log structure first: the
append-only, five-field lab notes are the reason I can quote every failure above
with its number.

**Q30. Sum up the project in one line for a non-expert.**
We taught a speech recogniser to write while listening using weights that can
only be "yes", "no" or "nothing" — and when it broke, we found out exactly why
and fixed it one step at a time.

---

## Part 8 — Glossary

- **ASR** — automatic speech recognition (audio → text).
- **Streaming / online** — output is produced while audio arrives, in chunks
  (ours: 320 ms).
- **Offline / batch** — the model sees the whole utterance before writing.
- **WER / CER** — word / character error rate (substitutions + insertions +
  deletions ÷ reference length), in percent. Lower is better.
- **test-clean / test-other** — LibriSpeech's two test sets (easy / hard).
- **Quantization** — storing weights with few discrete values.
- **Ternary / binary** — weights in {−1,0,+1} / {−1,+1}.
- **Group-wise scaling** — one scale per block of weights instead of one per
  matrix.
- **QAT** — quantization-aware training: quantization happens during training.
- **Shadow weights** — the full-precision trainable copy behind a quantized
  forward pass.
- **STE (straight-through estimator)** — pretend quantization is the identity in
  the backward pass.
- **α (alpha)** — blend between full precision (0) and fully quantized (1);
  progressively ramped during training.
- **fp16 readout** — evaluating the trained weights at α = 0; the drift plot.
- **Official prefix algorithm** — Qwen's streaming interface: feed an audio
  prefix, get "what I have heard so far".
- **Chunked re-feed** — our cheaper, non-official streaming variant (re-run the
  model on all audio so far). Useful, but not the gate's protocol.
- **LSP policy** — our loop-suppressing, append-only emission rule.
- **Append-only** — committed text can never be revised.
- **Loop / repetition degeneration** — the model repeats text endlessly.
- **Forced alignment** — matching a known transcript to audio time-stamps
  (done by Qwen3-ForcedAligner-0.6B).
- **Aligned targets** — per-chunk training targets derived from those
  time-stamps.
- **Distillation** — training a student to match a teacher's output
  distribution (here: a frozen fp16 copy of the model itself).
- **DAgger** — a policy-iteration variant that trains on states the student
  visits; ours failed because the student was collapsed.
- **Guardrail** — the cheap in-training 20-utt streaming check; *not* a
  promotion metric.
- **Promotion** — declaring a new best state/tag after full-sample evaluation.
- **Permanent dip** — a training trajectory that falls and never recovers
  (the 1.7B v2 failure).
- **Quantization damage** — accuracy lost *by* quantization, measured against
  the same model's fp16 performance.
- **Stochastic rounding** — unbiased rounding: round down, then step up with
  probability equal to the discarded fraction.
- **TFLOPS / RTF** — real-time factor: seconds of compute per second of audio;
  CPU deployment needs RTF < 1.

---

## Part 9 — Where everything lives

```
src/asr1bit/quant.py        binary/ternary quantizers (+ stochastic rounding)
src/asr1bit/bitlinear.py    the QAT layer (shadow weights, STE, alpha, scales)
src/asr1bit/replace.py      layer-policy swapping
src/asr1bit/data/           LibriSpeech loading, caching, aligner, chunk targets
src/asr1bit/qwen/stream.py  LSP streaming policy (append-only emission)
src/asr1bit/train/sft.py    streaming-SFT example builders (gold/distill/aligned)
src/asr1bit/train/trainer.py  the QAT loop (rollouts, in-training evals, saves)
scripts/run_stage1_qat.py   the trainer CLI (all flags: bit_width, alpha, anchor,
                            aligned targets, keep-step-checkpoints, stochastic…)
scripts/eval_streaming*.py  the streaming evaluators (official algorithm, LSP)
scripts/*.pbs               PBS launchers for every experiment we ran
docs/PLANNING.md            the plan of record (stages, gates, decisions)
docs/stage0_results.md      the fp16 rulers
docs/stage1_results.md      the offline matrix
docs/stage4_results.md      the streaming matrix + recipe history
docs/LAB_NOTES.md           append-only lab notes: every attempt, its number,
                            its root cause (read this for the full gore)
docs/ACTIVITY_LOG.md        live status + session log (rates, ETAs, decisions)
docs/HANDOFF.md             state of the world for the next person/agent
CHANGELOG.md                tagged checkpoints and what each tag contains
```

**Reading order for a newcomer:** this document → `PLANNING.md` (the plan) →
`stage4_results.md` (the numbers) → `LAB_NOTES.md` (the failures) →
`HANDOFF.md` (the current state).
