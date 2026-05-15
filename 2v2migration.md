# 2v2 Migration Plan

This plan extends the engine, RL training, and UI from 1v1-only to support both **1v1** and **2v2** (Team 0 = {P0, P2}, Team 1 = {P1, P3}, clockwise seating A→B→C→D), selectable per run via config.

The plan is grounded in the current code (`src/engine/`, `src/self_play/`, `src/model/`, `src/training/`, `src/ui/`) and the rules in `pro_yams_2v2.md`. All tasks are stepwise: each ends with a runnable validation gate so 1v1 never regresses.

---

## Guiding decisions (locked in)

1. **Compile-time templating on a trait type** (`Yams1v1`, `Yams2v2`). No runtime player count. Two binaries are *not* required — a single binary dispatches at startup via `app_config.game_variant`.
2. **No Sec/Joker bonus.** Removed from spec; not implemented.
3. **No PBRS.** Treated as out-of-scope; not extended for 2v2. (Optional cleanup task at the end.)
4. **Heuristic in 2v2:** the *scalar* heuristic EV used for move selection computes all four pair duels (0v1, 0v3, 2v1, 2v3) with proper crush + clean-bonus math, summed into a Team 0 vs Team 1 margin. The *tensor* keeps per-player heuristic features separate so the network learns its own combination.
5. **`compute_duel` is rewritten**, not just templated — per-pairing crush multiplier and per-pairing clean-column-bonus value (100/200) are required by the rules.
6. **TD targets are team-based.** Terminal value, TD(0) bootstrap, and TD(λ) bootstrap all consult `T::are_teammates(future_player, my_player)`.
7. **Tensor rotational invariance:** all 4 sheets are emitted in canonical relative order `[Active, NextOpp, Teammate, PrevOpp]`. For 1v1 this collapses to `[Active, Opp]`.
8. **Checkpoint files carry a variant tag.** Loading a 1v1 checkpoint into a 2v2 run (or vice versa) fails fast at load time.

---

## Task 0 — Pre-work: spec & code alignment

**Goal:** Land the small prerequisites so Task 1 starts from a clean baseline.

- **0.1** Remove the Sec bonus section from `pro_yams_2v2.md`. *(Done.)*
- **0.2** (Optional, separate commit) Delete PBRS from `self_play/training_data.cc` (the `use_pbrs` branch and `step.pbrs_reward`) and from `TrainingSample`/`TrajectoryStep` if it isn't used by anything else. Skip if it complicates the diff — it is otherwise a no-op for 2v2.
- **0.3** Read `engine_bench` baseline numbers for 1v1 once and record them in the migration commit message. We compare against these numbers in Task 2 and Task 7.

**Validation:** existing `engine_tests`, `training_tests`, `self_play_tests` all green.

---

## Task 1 — Foundation: traits and templated types

**Goal:** Replace hardcoded `kNumPlayers = 2` with a compile-time trait without changing engine behavior.

- **1.1 Create `src/engine/game_traits.h`** with two trait types:
  ```cpp
  struct Yams1v1 {
      static constexpr int kNumPlayers = 2;
      static constexpr int kNumTeams   = 2;
      static constexpr int kCellsPerSheet = kNumColumns * kNumRows; // 78
      static constexpr int kTotalCells = kNumPlayers * kCellsPerSheet; // 156
      static constexpr bool are_teammates(int p1, int p2) { return p1 == p2; }
  };
  struct Yams2v2 {
      static constexpr int kNumPlayers = 4;
      static constexpr int kNumTeams   = 2;
      static constexpr int kCellsPerSheet = kNumColumns * kNumRows; // 78
      static constexpr int kTotalCells = kNumPlayers * kCellsPerSheet; // 312
      static constexpr bool are_teammates(int p1, int p2) {
          // Teams: {0, 2} and {1, 3}. Same parity ⇒ same team.
          return (p1 & 1) == (p2 & 1);
      }
  };
  ```
  Contract: `are_teammates(p, p) == true` for both traits. *(Add a static_assert.)*

- **1.2 Templatize `BoardState`, `GameContext`, `GameState`** on `T`. Per-player arrays become `[T::kNumPlayers]`. Add type aliases at the bottom of each header:
  ```cpp
  using BoardState1v1   = BoardState<Yams1v1>;
  using GameContext1v1  = GameContext<Yams1v1>;
  using GameState1v1    = GameState<Yams1v1>;
  ```
  Existing `using BoardState   = BoardState1v1;` etc. (in `constants.h` or wherever) keeps `engine_tests` compiling unchanged.

- **1.3 Widen `cells_filled` to `uint16_t`** in `BoardState`. 2v2 has 312 cells; `uint8_t` overflows silently and the game never terminates.

- **1.4 Keep `kNumPlayers` and `kTotalCells` in `constants.h`** as aliases (`constexpr int kNumPlayers = Yams1v1::kNumPlayers;`) so existing call sites compile while we migrate them in Task 2. They get removed at the end of Task 7.

**Validation:**
- `engine_tests` and `self_play_tests` build and pass with zero changes.
- New test `game_traits_test.cc`: asserts the `are_teammates` contract for self and all pairings.

---

## Task 2 — Engine: rules and turn flow

**Goal:** Move every player-count-dependent loop in the engine to templated code, with 1v1 perf unchanged.

- **2.1 Templatize `apply_placement<T>`, `calculate_score<T>`, `get_legal_placements<T>`** and the helpers they call. Where the original signatures take `BoardState&` / `GameContext&`, switch to `BoardState<T>&` / `GameContext<T>&`. The Golden Rule check in `scoring.cc` is already correct — it reads `ctx.golden_max[col][row]`, which is populated from all players in `recalc_golden_max`.

- **2.2 Turn rotation in `game_flow.cc`:** `gs.board.current_player = (p + 1) % T::kNumPlayers;`. This is the clockwise A→B→C→D order, which matches the seating spec.

- **2.3 Verify `recalc_golden_max` iterates `T::kNumPlayers`.** The Golden Rule in 2v2 **includes the teammate's scores** — no special-casing required. The loop just becomes `for (int p = 0; p < T::kNumPlayers; ++p)`.

- **2.4 LS interlock vs "highest SS by anyone."** `scoring.cc:73` already reads `ctx.golden_max[col][kRowSS]`, which after 2.3 covers all players including the teammate. No code change beyond 2.3, but **add a regression test** that fails if a future refactor narrows it.

**Validation:**
- All existing `engine_tests` pass.
- `engine_bench` 1v1 numbers are within 2% of the Task 0.3 baseline.
- New test `golden_rule_teammate_test.cc` (2v2 only): teammate scoring 24 in Free's "6" raises my own Free "6" threshold to 24.

---

## Task 3 — Duel computation (rewrite, not a templatization)

**Goal:** Make `compute_duel` produce the correct Team 0 − Team 1 margin under per-pairing crush + clean-bonus rules.

- **3.1 New shape:** `compute_duel<T>(board, ctx)` returns a signed `int` representing the Team 0 margin in **all** of Team 0's column duels summed across all 6 columns. For 1v1 this is exactly the current behavior; for 2v2 it's the sum of the four cross-matchups per column.

- **3.2 Per-pairing math.** Inside the loop over columns, compute pairings:
  - 1v1: `{(0, 1)}`.
  - 2v2: `{(0, 1), (0, 3), (2, 1), (2, 3)}`.
  
  For each pairing `(team0_player, team1_player)`:
  1. `raw0 = raw[team0_player]`, `raw1 = raw[team1_player]` (raw includes upper bonus, not clean bonus).
  2. `crush_t0 = crush_multiplier(raw0, raw1)` and `crush_t1 = crush_multiplier(raw1, raw0)`. Only one can exceed 1× (a player crushes only if `their_raw ≥ 2 × opp_raw` — these conditions are mutually exclusive for raw > 0).
  3. **Per-side clean-bonus value:** if `crush_t0 > 1` *or* `crush_t1 > 1` was active in this pairing, both sides' clean bonus is +100; otherwise +200. *(Both sides see the same bonus value within a pairing — but the value can still differ between different pairings of the same player.)*
  4. `adj0 = raw0 + clean(team0_player) * bonus_value`; same for `adj1`.
  5. `diff = adj0 - adj1`; `active_crush = max(crush_t0, crush_t1)`; pairing points = `diff * active_crush * coeff[col]`.
  
  Sum the four pairing points into the column's contribution to Team 0.

- **3.3 Delete the `active_crush = max(crush0, crush1)` shortcut** at `duel.cc:41` and the single shared `clean_bonus` variable at `duel.cc:50`. They are correct for 1v1 but wrong for 2v2 because the bonus value belongs to a *pairing*, not a *column*.

**Validation:**
- New `duel_2v2_test.cc`:
  - Scenario A: P0 crushes P1 (2×), P2 does not crush P3 (1×). Assert column points = `[(adj0+100 − adj1)·2 + (adj0+200 − adj3)·1 + (adj2+100 − adj1_again_against_p2)·crush + (adj2+200 − adj3)·1] × coeff` (substitute actuals). The point: pairing-specific bonus values are visible in the assertion.
  - Scenario B: P0 clean column, P1 zero raw → 5× crush. Assert correct +100 bonus value.
  - Scenario C: degenerate — all four players identical scores → margin = 0 in every column.
- Existing 1v1 duel tests still pass bit-for-bit (the 1v1 code path collapses to one pairing).

---

## Task 4 — Heuristic for 2v2

**Goal:** Make the heuristic evaluator return a team margin in 2v2 while keeping the DP tables themselves untouched.

- **4.1 Templatize `heuristic_evaluate_research<T>`** so its inputs are `BoardState<T>` / `GameContext<T>`. The per-cell EV queries against the existing DP tables don't change — they take a `(player, col, row, ctx)` and return that player's expected raw score contribution. The DP tables already handle multi-opponent Golden-Rule thresholds because they read `ctx.golden_max`.

- **4.2 Scalar heuristic EV aggregation.** Build expected raw scores per `(player, column)` from DP queries, then run them through `compute_duel<T>` semantics in expectation:
  - For each of the four pairings, compute an expected crush multiplier and expected clean-bonus value from the expected raws *(a simple approximation: use the expected raws as point estimates and apply Task 3's pairing logic to them)*.
  - Sum into Team 0 margin.
  
  This is an EV-of-margins approximation rather than a true E[max], same level of fidelity as the current 1v1 heuristic.

- **4.3 Tensor features (per-player, not pre-summed).** Update `tensor.cc` Groups B and D to expose **per-player** expected duel contributions and expected crush indicators, in canonical rotational order (see Task 5). The network learns its own combination instead of being handed a pre-summed team margin.

**Validation:**
- `heuristic_2v2_test.cc`: synth board where DP returns known EVs per player; assert that scalar `heuristic_evaluate<T=Yams2v2>` produces the expected Team 0 margin.
- 1v1 heuristic outputs unchanged on the existing fixture set.

---

## Task 5 — Tensor with rotational symmetry

**Goal:** Make the tensor invariant under cyclic seat rotation so the network generalizes across all 4 seats.

- **5.1 `kTensorSize` becomes a trait.** Add `T::kTensorSize` and route every consumer (`tensor.h`, `model_config.h`, `trainer.cc`, `inference.h`, `GameInstance::TrajectoryStep::tensor`) through `T::kTensorSize`.

- **5.2 Canonical view in `generate_tensor<T>`.** Instead of iterating sheets `0..N`, emit them in this order:
  - 1v1: `[Active, Opp]`.
  - 2v2: `[Active, NextOpp = (active+1)%4, Teammate = (active+2)%4, PrevOpp = (active+3)%4]`.
  
  Apply the same reordering to every per-player feature group, including the heuristic-EV features from Task 4.3. No raw seat indices in the tensor.

- **5.3 Two new feature slots in 2v2** (Groups B/D extension): one feature per neighbor pairing indicating the expected crush direction (3-valued: I crush, opp crushes, no crush) — gives the network a structured prior on which pairings are dominant.

**Validation:**
- `tensor_rotation_test.cc`:
  - Build a 2v2 board. Compute tensor with `active = 0`. Cyclic-shift the four sheets by one (and all per-player ctx state) and compute tensor with `active = 1`. Assert the float arrays are **bit-identical**.
- `tensor_team_swap_test.cc`:
  - Swap P0↔P2 sheets (within-team permutation). Tensor for `active = 0` after the swap should equal the tensor before the swap with `active = 2` (verifies the canonical view sees the teammate correctly).
- 1v1 tensor stays bit-equal to current output on existing fixtures (the canonical view collapses to `[Active, Opp]`).

---

## Task 6 — RL bootstrapping for team credit assignment

**Goal:** Make TD targets credit a team, not an individual.

This is the **single most important correctness fix**. Without it, teammates train antagonistically.

- **6.1 Trajectory bound:** in `game_instance.h:62`, change `kMaxTrajectorySteps` to depend on the trait (currently it already multiplies `kNumPlayers * kNumColumns * kNumRows`, so it naturally becomes `T::kTotalCells` once `GameInstance` is templated in Task 7).

- **6.2 Terminal target — team-based** (`training_data.cc:21`):
  ```cpp
  auto terminal_for = [&](int8_t my_player) -> double {
      bool same_team_as_p0 = T::are_teammates(my_player, 0);
      if (use_margin) return same_team_as_p0 ? terminal_p0 : -terminal_p0;
      return same_team_as_p0 ? terminal_p0 : 1.0 - terminal_p0;
  };
  ```

- **6.3 TD(0) bootstrap — team-aware** (`training_data.cc:50`):
  ```cpp
  const TrajectoryStep& next = game.trajectory[i + 1];
  target = T::are_teammates(next.player, my_player)
      ? next.value
      : flip(next.value);
  ```

- **6.4 TD(λ) bootstrap — team-aware** (`training_data.cc:64`):
  ```cpp
  double bootstrap = T::are_teammates(future.player, my_player)
      ? future.value
      : flip(future.value);
  ```

- **6.5 `game.result` and `game.final_duel_margin`** stay as **Team-0 perspective scalars** (no per-player array). `terminal_for` already handles the perspective flip via `are_teammates(my_player, 0)`. This keeps the data structures minimal and matches the 1v1 convention.

- **6.6 Hyperparameter note in `training_config.h`:** add a one-line comment pointing readers to `config/mlp-512/` (2v2 configs, added in Task 8) for tuned defaults. **Do not** change `td_lambda` for 2v2: at fixed λ the per-sample variance coefficient is `(1−λ)/(1+λ)`, a function of λ alone — not of trajectory length. The 1v1 `td_lambda = 0.95` carries over unchanged.

**Validation:**
- `trajectory_2v2_test.cc`:
  - Synthesize a 2v2 trajectory where Team 0 wins (`game.result = 1.0`). Extract MC samples. Assert P0 and P2 samples both have `target = 1.0`; P1 and P3 both have `target = 0.0`.
  - Same trajectory in TD(0) mode with mocked `value` fields per step. Assert that a P0 sample bootstrapping off the next step (P1's value) gets `flip(v)`, while bootstrapping off a P2 step (teammate) gets `v` directly.
  - Same in TD(λ) mode. Verify Watkin's cut still fires on exploratory steps.
- All existing 1v1 trajectory tests pass unchanged (1v1's `are_teammates(p, q) == (p == q)` reproduces the old `(p != q) ? flip : noflip` semantics exactly).

---

## Task 7 — Orchestrator, training loop, model config

**Goal:** Carry `T` through self-play and training, and tag checkpoints with the variant.

- **7.1 Templatize on `T`:**
  - `GameInstance<T>` (uses `T::kTensorSize` for `TrajectoryStep::tensor`, `T::kTotalCells` for `kMaxTrajectorySteps`).
  - `Worker<T>`, `SelfPlayOrchestrator<T>`, `BatchManager<T>` (only where they reference player count or tensor size).
  - `TrainingLoop<T>`.
- **7.2 `ModelConfig::input_size`** comes from `T::kTensorSize` at construction time. Drop the hardcoded `986`.
- **7.3 Checkpoint variant tag.** In `trainer.cc:135`, write an additional `"game_variant"` tensor (encode `1 = 1v1`, `2 = 2v2`). On load (`trainer.cc:280`), assert it matches the current binary's trait — fail loudly otherwise. This prevents silent shape mismatches.
- **7.4 Remove `kNumPlayers`/`kTotalCells`/`kTensorSize` from `constants.h`** once no call site references them directly. (Optional cleanup; can defer.)

**Validation:**
- 1v1 short training smoke (e.g., 100 self-play games + 1 train step) runs and produces an identical loss curve to the pre-migration baseline (within numerical noise). This proves templating didn't change semantics.
- 2v2 short training smoke: completes a full game (312 cells filled, terminal), produces non-degenerate loss.
- `engine_bench` 1v1 throughput within 2% of Task 0.3 baseline (re-check after orchestrator templatization, in case of inlining regressions).

---

## Task 8 — CLI, UI, tooling

**Goal:** Expose the variant to users and render 4-player boards.

- **8.1 Config:** add `game_variant: "1v1" | "2v2"` to `AppConfig`. In `main.cpp`, dispatch once at startup:
  ```cpp
  if (cfg.game_variant == "2v2") run_train<Yams2v2>(cfg);
  else                            run_train<Yams1v1>(cfg);
  ```
  Same pattern for `run_eval`, `run_play`, etc. Each entry point's body is template-instantiated for both traits.
- **8.2 JSON export** (`json_serialization.cc`): loop over `T::kNumPlayers` instead of `2`. Emit the variant in the JSON header so the UI can render it correctly.
- **8.3 UI (`board.js`, `play.js`, `play_static/`):** when the JSON reports 4 sheets, render them in a **2×2 grid**:
  - Top-left P0 (Team 0), top-right P1 (Team 1).
  - Bottom-left P2 (Team 0), bottom-right P3 (Team 1).
  - Team 0 column shaded one color, Team 1 the other.
- **8.4 Manual control:** in human-play mode, the human controls one seat (configurable, default P0), the other three are heuristic or NN bots.

**Validation:**
- `play_main.cpp` (or however the UI binary is wired) accepts `--variant 2v2` and launches.
- Manual Human vs 3× Heuristic game in the browser: turn order rotates A→B→C→D; Golden Rule blocking shows teammate's score as constraining; clean-column highlighting works on all four sheets; final scoring matches `compute_duel<Yams2v2>` output.
- 1v1 UI flow visually unchanged.

---

## Task 9 — 2v2 training configs (`config/mlp-512/`)

**Goal:** Land working YAML configs for the 2v2 training pipeline, mirroring the three-phase `config/mlp-256/` structure (`seed.yaml` → `selfplay.yaml` → `selfplay2.yaml`) with deltas tuned for the 2v2 setting.

### Reasoning for the deltas

The 2v2 differences from 1v1 that matter for hyperparameters are:
1. **Larger state space + harder credit assignment** ⇒ bigger model.
2. **2× samples per game** ⇒ a fixed-size buffer holds half as many *independent games*.
3. **Diluted credit per decision** (three other agents act between your turns) ⇒ slightly lower learning rate as insurance; lean on the heuristic longer during seeding.

What is *not* a 2v2 issue:
- TD(λ) per-sample variance — the coefficient `(1−λ)/(1+λ)` depends only on λ, not on trajectory length. Keep `td_lambda = 0.95`.
- Bias from long on-policy traces — there is none; all four agents share the policy.

### Concrete config deltas (vs `config/mlp-256/seed.yaml`)

| Knob | mlp-256 (1v1) | mlp-512 (2v2) | Rationale |
| :---- | :---- | :---- | :---- |
| `model.architecture` | mlp | mlp | unchanged |
| `model.hidden_width` | 256 | **512** | ~2× input + harder problem |
| `model.hidden_layers` | 3 | 3 | unchanged |
| `model.learning_rate` | 0.00045 | **0.0003** | noisier per-sample credit |
| `model.input_size` | 986 | **~2000** (set from `Yams2v2::kTensorSize`) | from Task 5.1 trait |
| `td_mode` | tdlambda | tdlambda | unchanged |
| `td_lambda` | 0.95 | 0.95 | per-sample variance coefficient is λ-only |
| `train_batch_size` | 8192 | 8192 | unchanged (revisit if GPU memory permits 16384) |
| `train_steps_per_collect` | 0.041 | 0.041 | unchanged — already game-heavy |
| `replay_capacity` | 1,000,000 | **2,000,000** | preserves independent-game count in buffer |
| `min_buffer_size` | 100,000 | **150,000** | same reasoning, scaled to per-game sample count |
| `heuristic_decay_steps` (seed only) | 80,000 | **120,000** | low-variance bootstrap target; 2v2 has more to learn |
| `initial_heuristic_weight` (seed) | 1.0 | 1.0 | unchanged |
| `num_steps` (seed) | 100,000 | **150,000** | matches the longer heuristic-decay window |
| temperature schedule | unchanged | unchanged | |
| `past_opponent_probability` (selfplay/2) | 0.2 | 0.2 | unchanged |
| `self_play.num_workers` | 24 | 24 | unchanged (revisit if 2v2 game generation is RAM-bound) |
| `self_play.num_games` | 2048 | 2048 | unchanged |

### Sub-tasks

- **9.1 Add `config/mlp-512/seed.yaml`** as a copy of `config/mlp-256/seed.yaml` with the deltas above and `game_variant: "2v2"` set.
- **9.2 Add `config/mlp-512/selfplay.yaml`** mirroring `mlp-256/selfplay.yaml` (no heuristic, temperature 0.5 → 0.001, past-opponent rotation 0.2) with the same width/buffer/lr deltas.
- **9.3 Add `config/mlp-512/selfplay2.yaml`** mirroring `mlp-256/selfplay2.yaml` (greedy, longer run).
- **9.4 No within-team augmentation.** Considered and rejected: it's a single 2-element symmetry per team (Active sheet ↔ Teammate sheet), not a rich symmetry group like Go's dihedral 8. The canonical-view tensor (Task 5.2) already bakes in the larger cyclic symmetry. What's left for the network to learn is one binary swap, well within the reach of an MLP-512 given the data volume. Cost of augmentation (2× buffer pressure, paired-sample correlation in batches) outweighs the marginal regularization benefit. If empirical evidence later shows asymmetric teammate play, revisit as on-the-fly batch augmentation (50% prob random swap) — not in the buffer.

**Validation:**
- `config/mlp-512/seed.yaml` parses and `run_train` starts cleanly with `game_variant: 2v2`.
- After ~10K seed steps, training loss decreases (sanity check, not a quality bar).
- A full seed → selfplay → selfplay2 pipeline run produces a 2v2 model that beats Heuristic V17 by a clear margin in 2v2 eval. Quality bar — sets the "super-human" target for 2v2 in line with the 1v1 result.

---

## Order of execution

```
0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → (9)
```

Each task is mergeable on its own (1v1 keeps passing every test at every step). Tasks 3 and 6 are the only ones where a subtle wrong implementation would silently train a bad model — those are the ones to code-review most carefully.

## Risk register

| Risk | Mitigation |
| :---- | :---- |
| Template bloat slows 1v1 | engine_bench gate in Tasks 2 and 7 |
| TD bootstrap miswire trains teammates antagonistically | Mandatory `trajectory_2v2_test.cc` (Task 6) — must assert P0 and P2 share target |
| 1v1 checkpoint loaded into 2v2 process | Variant tag in checkpoint, fail-loud on load (Task 7.3) |
| `compute_duel` shortcut survives the refactor | Task 3 explicitly deletes the `active_crush = max(...)` and the shared `clean_bonus` |
| Tensor permutation bug breaks symmetry | Bit-equal rotation test (Task 5) |
| 2v2 trains slowly due to harder credit assignment | mlp-512 + buffer 2M + longer heuristic decay (Task 9) |
