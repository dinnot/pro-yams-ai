# Fable Review — 2v2 Variant (tensor, correctness, performance)

Scope: full read of `src/engine` (tensor, duel, scoring, placement, game flow),
`src/solver`, `src/heuristic`, `src/self_play`, `src/training`, `src/model`,
`src/eval`, the 2v2 configs, and the 2v2 rules doc (`pro_yams_2v2.md`). All test
suites were run and pass (engine 158, tensor 42, self-play 38), so everything
below is a design-level finding, not a test-breaking regression. Each item has
a confidence tag and file:line references.

---

## 1. Tensor improvements (2v2) that should improve final model quality

### 1.1 Group G is blind to teammate-driven SS/LS poison — HIGH confidence, HIGH impact

`tensor.cc:547-587` restricts both G0 (defensive exposure) and G1 (offensive
leverage) to true opponents via `Traits::are_teammates`. But by the rules
(`pro_yams_2v2.md` §"SS/LS Interlock": *"LS must be strictly higher than the
highest SS recorded by anyone on the board (opponents **and, in 2v2, your
teammate**)"*), the golden_max for SS is global — a **teammate's** SS ≥ my
committed LS poisons my column exactly the way an opponent's does. Teammates
do this involuntarily all the time: in Down/Up/Mid columns the SS cell is
often the *forced* next placement, and a forced high SS placement by the
teammate is a real, common threat that G0 currently reports as "no capable
opponent → 0".

Symmetrically, the network gets no advance signal that pushing its own SS high
will poison its **teammate's** committed LS (the damage only becomes visible
post-hoc via Group C `p_one` collapsing to 0 after the placement). Since the
NN evaluates *afterstates*, the executed poison is visible — but threat
anticipation one or more turns ahead is exactly what Group G was added for
(see the SS/LS blind-spot analysis), and in 2v2 a quarter of the interlock
graph (teammate↔me) is missing.

Suggested fix (append-only, fits the V3 versioning contract in
`game_traits.h:109-128`): add 2 features per (player, column) — "teammate
poison exposure" (same definition as G0 but with `q` restricted to teammates)
and "I endanger teammate" (G1 over teammates). Cost: +48 floats in 2v2.

### 1.2 Group G magnitude encodes window width but not execution probability or damage — MEDIUM confidence

The G0/G1 magnitude `(30 - max(20, L))/10` (`tensor.cc:557,579`) measures how
easy the poison is (low LS = wide window) but nothing else:

- **Execution probability**: an opponent must actually roll a sum ≥ L and choose
  to commit it. `tables.prob_tables` already has compound one-turn
  P(sum ≥ threshold) — multiplying (or appending) the opponent's best one-turn
  probability of reaching ≥ L would separate "theoretically poisonable" from
  "will realistically be poisoned" much better than window width alone.
- **Damage**: the value at risk is the committed LS itself (up to 30 raw points
  before coefficient/crush) plus the clean-bonus equity. Currently the net must
  cross-reference Group A to recover L. A `L/30`-style companion feature is
  nearly free.
- **Reachability/timing**: `max_placeable_ss` (`tensor.cc:160-170`) checks only
  that the SS cell is empty, not whether the opponent can *reach* it under the
  column's fill-order constraint (Down/Up/Mid/UpDown). An opponent 5 cells away
  from SS in a Down column is a much weaker threat than one whose SS is
  immediately placeable. A cheap proxy: number of cells the opponent must fill
  before SS becomes legal in that column.

### 1.3 Group G G0 gates out realized and "unclean" exposure — MEDIUM confidence

Two gates at `tensor.cc:555-556` understate exposure:

- `!ctx.lower_has_scratch[p][col]`: if my column already has a lower scratch,
  the clean bonus is gone but the committed LS **points** (up to 30 × coeff ×
  crush leverage) are still destroyed by mutual destruction. The feature
  conflates clean-equity loss with LS-value loss; exposure should not drop to 0
  just because the column is already unclean.
- When the poison is already *realized* (`golden_max[col][SS] ≥ my LS`, i.e. my
  SS scratch is now inevitable), G0 reads 0 unless a "capable opponent" still
  exists. The doomed state is only visible indirectly through `p_one[SS] == 0`
  (Group C). Emitting the realized case explicitly (e.g. magnitude 1.0 or a
  third state) would let the value head price the inevitable LS wipe instead of
  inferring it from a 0 in an unrelated group.

### 1.4 Group D team-margin tanh scales mis-calibrated for 4 pairings — MEDIUM confidence

`tensor.cc:462-465` normalizes the global margin sums with
`80000 * kPlayersPerTeam` / `100000 * kPlayersPerTeam` (= 2× the 1v1 scale).
But the 2v2 sums accumulate **4** pairing margins (`kNumPairings = 4`), not 2.
In the typical correlated case (one team ahead in most duels) the sum runs ~4×
the 1v1 magnitude, so the tanh saturates ~2× earlier than the 1v1-calibrated
feature — exactly in the decisive mid/late game where margin discrimination
matters most. Suggest scaling by `Traits::kNumPairings` (or, better, histogram
the feature over real 2v2 self-play and pick the scale that keeps the bulk of
mass in tanh's linear region). Same check applies to the per-pairing
`tanh(margin/15000)` at `tensor.cc:404,420` — per-pairing magnitudes should be
1v1-like, so those are probably fine, but verify empirically.

### 1.5 Per-column team-margin aggregates are missing — LOW/MEDIUM confidence

Group B.2 gives 14 features per pairing per column (56/column in 2v2), and
Group D gives one *global* team margin. There is no per-column **team** margin
(sum of the 4 pairing margins for that column, now and expected). The net must
learn to linearly combine 4 scattered tanh features per column to answer the
core 2v2 question "which column is my team winning/losing and by how much" —
the quantity that drives column targeting against the coefficient. A 12-float
addition (6 columns × {now, expected}) is cheap and directly matches the game's
reward structure. (In 1v1 this was unnecessary because the single pairing *is*
the team margin.)

---

## 2. Bugs in 2v2 code whose fix should improve final model quality

### 2.1 TD targets bootstrap through past-opponent values from the OLD network — HIGH confidence, the most impactful item in this section

`training_data.cc:51-67` builds the TD(λ) recursion over **every** trajectory
step, including steps played by the past-opponent team (whose `step.value`
came from the *old checkpoint's* network via the opponent batch manager,
`worker.cc:113-119`). The samples for those steps are correctly excluded
(`training_data.cc:77-78`), but their value estimates still enter the kept
samples' targets through `v_next` — both in the `(1-λ)·v_{i+1}` term and as
the full target whenever the next step is exploratory (trace cut), and in
TD(0) mode (`training_data.cc:91-95`) it's worse: in 2v2 the next trajectory
step is *always* an opposing-team seat, so **every kept sample's TD(0) target
is exactly one old-network evaluation**, sign-flipped.

With `past_opponent_probability: 0.2` (`config/2v2/mlp-512/selfplay3.yaml:45`)
this injects stale-network value calibration into ~20% of games' targets. The
old checkpoint is at a different training stage and (after LR back-off /
output-head drift) systematically mis-calibrated relative to the current
network — a classic source of value-target bias that caps final strength.

Fix: when `exclude_player >= 0`, run the recursion only over the kept team's
steps — i.e. bootstrap from the next **same-team** step's value (treating the
excluded steps as environment transitions), and use the terminal for the tail.
This is also the semantically right multi-agent TD formulation when the other
team is a fixed (non-learning) opponent.

### 2.2 PBRS is not potential-based and is inconsistent under TD(λ) — MEDIUM confidence (currently off by default)

`training_data.cc:110-112` adds `step.pbrs_reward` flatly to whatever target
the TD mode produced. Two problems if `use_pbrs` is ever re-enabled in 2v2:
(a) it's a one-sided bonus, not a potential difference Φ(s')−Φ(s), so it
permanently inflates value estimates for states *preceding* milestones rather
than vanishing in the limit (Ng et al. shaping guarantee doesn't hold);
(b) under TD(λ) the shaping reward of step *i* is not propagated into the
λ-returns of earlier steps — only the step that earned it sees it — so the
shaped target function is inconsistent across the trajectory, and in 2v2 the
teammate never sees credit for milestones their play enabled. If you want
shaping in 2v2, fold Φ into the backward recursion (add
`r_{i+1} = Φ(s_{i+1}) − Φ(s_i)` inside the g recursion) or keep it off.

### 2.3 Latent: `InferenceEngine` tensor path never narrows input to the model's `input_size` — LOW confidence as a live bug, worth hardening

`inference.cc:35-72` (the `torch::Tensor` overload used by the coordinators)
trusts the caller's width — batch storage is always `Traits::kTensorSize`
(`batch_manager.h:34`). Today this is safe only because `load_checkpoint`
refuses tensor-version mismatches (`trainer.cc:344-349`), so a V1 (2126-wide)
past-opponent can't be loaded into a V2 (2174) run. But the distillation
machinery explicitly supports "teacher reads the V1 prefix of the V2 layout"
(`teacher_nn.cc:114-128`), and the moment someone enables the same for
self-play past-opponents (an obvious next step after the Group-G migration),
the coordinator will feed 2174-wide rows into a 2126-input model and crash —
or worse, if a future model tolerates the shape, silently mis-index features.
Add `input.narrow(1, 0, model->config().input_size)` (no-op when equal) to the
tensor overload.

### 2.4 Stale documentation that will mislead future feature work — HIGH confidence, trivial

`game_traits.h:44-45` still says Group G is "Placeholder zeros until the
features land" — they landed in `tensor.cc:547-587`. Also `game_flow.cc:104-106`
claims `get_game_result` is 1v1-only ("the 2v2 instantiation lands in Task 3")
while it is instantiated for 2v2 at line 127. Cheap to fix, and this kind of
stale comment is exactly what causes the next tensor version to be built on a
wrong assumption. (Likewise the stale `input_size:` keys in
`config/2v2/**.yaml` — e.g. `seed.yaml:51` says `986`, the *1v1* size — are
currently harmless because `config_loader.cc:127-130` derives the real value,
but they should be deleted to stop them being copy-pasted into tools that do
trust them.)

### 2.5 Verified-clean areas (no bug found, listing so you don't re-audit)

- `compute_duel<Yams2v2>` (`duel.cc:43-100`): per-pairing crush and per-pairing
  clean-bonus value — matches rules §4–5 including the "same clean column worth
  +100 in one duel and +200 in the other" case.
- Canonical rotation (`tensor.cc:335-340`, pairing tables
  `game_traits.h:88-89`): pairings cover exactly each player's two neighbor
  duels; Group D's `ci % 2` team split is correct in canonical order.
- Team-aware TD flips (`training_data.cc:24-33, 57-59, 93-95`): uses
  `are_teammates`, not `!=` — Pattern 8 from the migration notes is intact.
- `extract_training_samples` exclusion drops the whole past-opponent **team**.
- Golden-rule / SS-LS interlock engine paths (`scoring.cc:46-81`,
  `placement.cc:48-69` incl. mutual-destruction `recalc_golden_max`) match the
  rules doc, teammate included.
- `generate_tensor_batch` recomputes the placed column for **all** players
  (golden_max is shared) — Pattern 10 honored (`tensor.cc:663-674`).
- Worker→opponent-batch routing and eval seat assignment (`worker.cc:113-119`,
  `evaluator.cc:127-146`) are team-correct.

---

## 3. Performance improvements for 2v2 training speed

### 3.1 Heuristic evaluators recompute all 24 (player, column) DP blocks per request — HIGH confidence, biggest CPU win where heuristics run

`heuristic_evaluate_v2` (`heuristic_bot.cc:142-192`) and
`heuristic_evaluate_research` (`heuristic_bot.cc:345-448`) clone the board and
re-run `get_E_raw`/`get_P_clean` (each a full `build_Sc` + DP table walk) for
**all 4 players × all 6 columns** for *every* afterstate request. A placement
only changes the placed column (golden_max is per-column), so 5 of 6 columns ×
4 players are identical to the pre-placement base for every request. Computing
the 24 base blocks once and recomputing only the placed column (4 blocks) per
request is a ~5–6× cut in DP work — the same trick `generate_tensor_batch`
already uses (`tensor.cc:620-683`). This path runs:
- every first resolve in seeded self-play runs (`heuristic_weight > 0`),
- the heuristic seats of **every in-training eval game** (10,000 games per
  eval at `eval_interval: 25000`, 2v2 games being 312 turns each),
- the distillation heuristic teacher.

Bonus: both evaluators copy the full `GameContextT` per request, which includes
the two `LegalPlacementCache` arrays per player (~2 KB in 2v2) that they never
read — copy only the fields `copy_tensor_context_fields` copies
(`tensor.cc:143-152`).

### 3.2 `generate_tensor_batch`: skip the 3 non-active recomputes when the placement can't change shared state — HIGH confidence

`tensor.cc:667-674` recomputes the placed column's `PCData` for all 4 players
because the placement *may* raise `ctx.golden_max[col][row]`. But that only
happens when `req.score > ctx.golden_max[col][row]`, or on a mutual-destruction
scratch (`score == 0` on SS/LS with the counterpart filled positive). Scratches
and exactly-meet-the-threshold scores are very common, and in those cases the
other 3 players' PCData is bit-identical to base — recompute only the active
player's column. `compute_pc_data` (DP lookups ×3 horizon tiers + 13 p_one
lookups) dominates `tensor_batch_us` in the worker profile, and 2v2 pays 4
players per request where 1v1 paid 2, so this directly attacks the variant's
extra cost. Guard:

```cpp
bool shared_state_changed =
    req.score > ctx.golden_max[col][row] ||
    (req.score == 0 && (row == kRowSS || row == kRowLS));
```

(One subtlety: SS-row fills also tighten the LS threshold via golden_max[SS],
which the condition above already covers since that requires score > gmax.)

### 3.3 Don't generate/store trajectory tensors for past-opponent team steps — MEDIUM confidence (couples with finding 2.1)

`worker.cc:359-370` runs a dedicated `generate_tensor_batch` (1 request) for
the chosen placement of **every** seat, including past-opponent-team seats
whose steps are then discarded at extraction (`training_data.cc:77-78`). In
~20% of games, half of all per-placement chosen-tensor generations (the
`chosen_tensor_us` bucket) plus ~1.4 MB/game of trajectory writes are pure
waste. If you adopt the fix in 2.1 (bootstrap across same-team steps only),
the excluded steps' tensors *and* values become entirely unused and the worker
can skip recording them outright (keep `player` so indexing stays simple).

### 3.4 Replay buffer: 8.7 GB working set and a mutex held across ~70 MB of scattered copies — MEDIUM confidence

A 2v2 sample is `2174*4 + 8 ≈ 8.7 KB` (`training_data.h:20-24`), so
`replay_capacity: 1000000` ⇒ ~8.7 GB resident, and
`sample_batch_arrays` (`replay_buffer.h:72-86`) holds the buffer mutex while
doing 8192 random-index memcpys (~70 MB, all cache/TLB misses) per train step —
during which `add_batch` from game collection blocks. Options, in increasing
effort: (a) draw the 8192 indices under the lock, copy outside it (ring
overwrite of a just-sampled slot is statistically harmless here); (b) store
states as fp16 and widen on sample (halves footprint and copy bandwidth;
features are all in [-1, 1]); (c) shard the buffer N ways with per-shard locks.

### 3.5 Self-play fleet memory: 2048 × ~2.8 MB of inline trajectories — LOW/MEDIUM confidence

`GameInstanceT<Yams2v2>` embeds `312 × (2174 floats + …) ≈ 2.8 MB` of
trajectory inline (`game_instance.h:64-66`), so `num_games: 2048` keeps
~5.7 GB hot. 2v2 games are twice as long as 1v1, so per-game inference demand
is higher and batch occupancy can likely be sustained with fewer concurrent
games (e.g. 1024) — worth an A/B on `coordinator_pop_wait_us`/batch-fill
stats before paying for the memory. Alternatively make `trajectory` a
heap-allocated, lazily grown vector so unused tails don't occupy RSS, and
combine with 3.3.

### 3.6 In-training eval is synchronous, small-batch, per-turn inference — MEDIUM confidence

`run_evaluation` (`evaluator.cc:205-238`) blocks the training loop and plays
`eval_games: 10000` 2v2 games on 16 threads, where each NN turn does its own
`forward` of ~30–90 rows (`evaluator.cc:63-80`) — in 2v2 that's ~156 NN turns
per game ⇒ ~1.5 M tiny GPU launches per eval, competing with nothing (training
is paused). Either route eval through the existing async
orchestrator/batch-manager machinery with greedy config (big batches, same
code path as self-play), or cut `eval_games` for in-training evals (the
LR-backoff signal doesn't need ±0.5% precision) and keep large counts for
offline checkpoints. Heuristic-seat cost in these games is attacked by 3.1.

### 3.7 Minor

- `solver_get_requests` clears `req_map` with a 0..100 loop per legal cell
  (`solver.cc:34`); a `memset` of the row is ~4× less code-gen, though this
  path is only ~µs-level.
- `count_filled_cells` is an O(78) scan per player per tensor/heuristic call;
  a per-player filled counter maintained in `GameContextT` next to
  `non_turbo_cells_remaining` would remove it from all hot paths.

---

## Suggested priority order

1. **2.1** TD bootstrap through past-opponent values (model quality, contained change in `training_data.cc`).
2. **1.1 + 1.2/1.3** Group G teammate coverage + probability/damage refinement (tensor V3, append-only).
3. **3.1 + 3.2** base-plus-delta recomputation in heuristic evaluators and tensor batch (pure speed, no behavior change — verify with existing batch-equivalence tests).
4. **1.4 / 1.5** margin-scale recalibration + per-column team margins (cheap, measure feature histograms first).
5. **3.4 / 3.6** replay-buffer and eval throughput once the above land.
