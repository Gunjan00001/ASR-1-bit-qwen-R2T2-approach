# Dream-RSI — one-page reference note

**Source:** Zheng et al., *Dream-RSI: Recursive Self-Improvement through Evolving Worlds*
(Google / DeepMind / UMD / UVA, Sep 2026) — `github.com/zhengkid/Dream-RSI`, `dream-rsi.com`.
This is a **general methodology reference**, not part of the ASR plan. Code is unreleased;
only the paper and project page exist.

## The core idea

A long-horizon discovery loop spends most of its compute on *exploration* — which attempt to
continue, what to run in parallel, when to stop. That policy is usually hand-written and
frozen. Dream-RSI instead optimizes it, cheaply.

Key insight: **a completed discovery run is already an exact replay simulator.** It records a
structured tree of exploration decisions and their realized outcomes. An alternative policy
can traverse that same tree differently (different branches, order, parallel grouping, stop
points) — and because every outcome is already saved, evaluating it costs **zero executions**.

## The loop (recursive)

1. **Online explore** — the current exploration *policy* (executable code) guides a **fixed**
   discovery agent to expand a discovery tree. Each node stores parent, workspace/artifact,
   config, score, cost, diagnostics.
2. **Construct replay simulator** — the finished tree becomes a reusable "world".
3. **Dreaming-based policy improvement** — an offline policy-development agent writes `M`
   revisions of the policy code; each is scored by **replaying over the recorded history at
   zero executions**; the best is redeployed online, which records a new tree and grows the
   pool. Repeat.

## Mechanics worth stealing

- **Separate three things:** the *exploration policy* (where/when/how-parallel/stop), the
  *candidate generator* (produces attempts), and the *evaluator* (fixed). Only the policy code
  changes; the rest stays frozen.
- **Shared decision interface** online and offline: at each round the policy selects a **batch
  of nodes** (start points for attempts). Replay terminates on an explicit stop, a round limit,
  or when all recorded nodes are revealed.
- **Reward = quality + cost + parallelism.** The third term rewards batching useful
  continuations rather than serial probing.
- **Monotonic selection:** the currently deployed policy is itself a candidate, so the chosen
  `π_{t+1}` satisfies `score(π_{t+1}) ≥ score(π_t)` on the fixed history — never worse by
  construction.
- **Prefix-only discipline:** decide from observed cells only; do not peek at unrevealed scores
  and do not use global bookkeeping (`best_so_far`, `budget_spent`) to choose.

## Empirical findings

- The learned policy is **adaptive**: it conserves compute when improving (110 → 50 attempts)
  and widens again when progress plateaus.
- **Explicit semantic guidance underperforms replay.** Injecting high-level directional insights
  into the prompt consistently did worse than letting the policy replay history — it
  over-constrains the search and suppresses diversity.
- Reported gains vs a fixed-exploration baseline: up to ~2.4x fewer generations / ~2.1x higher
  score on kernel tasks, and up to 162x fewer discovery-agent calls than SimpleTES on one
  algorithm task.

## When it applies / when it doesn't

**Applies:** large search spaces (thousands of proposal–evaluation cycles), expensive and
delayed feedback, a fixed evaluator, and outcomes that are **deterministic and cacheable**
(algorithm/kernel discovery, evolutionary code search).

**Doesn't:** one-shot tasks, small searches, or evaluations that are stochastic and not
replayable — then "exact replay" degrades to an approximation and the never-worse guarantee
weakens.

## Practical takeaway (even without the full loop)

1. Log every attempt as a structured record (config → outcome → cost), not prose.
2. Treat that log as a replay table: score candidate "next-experiment" strategies against it
   before spending real compute.
3. Keep the incumbent strategy in the candidate set so changes are never a regression.
4. Widen exploration on plateaus, conserve when improving.
5. Prefer replaying history over injecting confident semantic guidance.

## Limits

- Code not released — any use is a reimplementation.
- Requires real orchestration (tree log, replay harness, parallel driver); overhead only pays
  off for genuinely expensive, repeated searches.
- The "exact simulator" property holds only over the realized search space: a policy can only
  be dreamt where history actually went.
