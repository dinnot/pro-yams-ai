# Pro Yams AI — 2v2 Variant Design

This document is the 2v2-specific companion to `pro_yams_ai_high_level_design.md` (which covers 1v1). It describes the rules, engine choices, tensor layout, RL targets, and configuration that make `Yams2v2` work in the same binary as `Yams1v1`. For the full 2v2 rule sheet, see [`pro_yams_2v2.md`](pro_yams_2v2.md).

> **Status:** fully implemented. Engine, scoring, duel math, heuristic, tensor, TD targets, self-play, training, evaluation, tournament, and UI all support 2v2. A single binary dispatches on `game_variant` at startup. All 11 test suites pass for both variants.

---

## 1. Rules in one paragraph

2v2 is the four-player variant of Pro Yams. Teams: **Team 0 = {P0, P2}** and **Team 1 = {P1, P3}**. Seating is clockwise — `A → B → C → D` with `A=P0, B=P1, C=P2, D=P3` — so every player's two neighbours are always opponents. Each player has their own 13×6 sheet (`Traits::kCellsPerSheet = 78`). The Golden Rule and SS/LS interlock apply across **all four sheets** (including your teammate's). Duels are computed **per pairing** (each player duels both neighbours); the Crush Multiplier and the Clean Column Bonus value (100 with a crush, 200 without) are independent per pairing. Team score = sum of both teammates' duel points; the team with the higher total wins. Full rules in [`pro_yams_2v2.md`](pro_yams_2v2.md).

The key spec-level differences from 1v1 that drive every code change:

1. **4 sheets, not 2.** Player count is a trait constant (`Traits::kNumPlayers`), not a magic number.
2. **Per-pairing duel math.** 1v1 has one pairing per column; 2v2 has four — `{(0,1), (0,3), (2,1), (2,3)}`. The Crush Multiplier and Clean Column Bonus value belong to a pairing, not a column.
3. **Team-based outcomes.** Win is per team, not per player. RL credit assignment must follow teams.
4. **The Golden Rule is global across all sheets.** A teammate's high score in Free/6s raises *your* threshold for Free/6s, exactly like an opponent's would.

---

## 2. Trait system — `Yams1v1` and `Yams2v2`

The codebase is templated on a `Traits` type. The two concrete traits live in `src/engine/game_traits.h`:

```cpp
struct Yams1v1 {
    static constexpr int kNumPlayers      = 2;
    static constexpr int kNumTeams        = 2;
    static constexpr int kPlayersPerTeam  = 1;
    static constexpr int kCellsPerSheet   = 78;
    static constexpr int kTotalCells      = 156;
    static constexpr int kTensorSize      = 986;
    static constexpr int kNumPairings     = 1;
    static constexpr bool are_teammates(int p, int q) { return p == q; }
    static constexpr int kTeam0[1] = {0};
    static constexpr int kTeam1[1] = {1};
    static constexpr int kCanonicalPairingT0[1] = {0};
    static constexpr int kCanonicalPairingT1[1] = {1};
};

struct Yams2v2 {
    static constexpr int kNumPlayers      = 4;
    static constexpr int kNumTeams        = 2;
    static constexpr int kPlayersPerTeam  = 2;
    static constexpr int kCellsPerSheet   = 78;
    static constexpr int kTotalCells      = 312;
    static constexpr int kTensorSize      = 2126;
    static constexpr int kNumPairings     = 4;
    static constexpr bool are_teammates(int p, int q) { return (p & 1) == (q & 1); }
    static constexpr int kTeam0[2] = {0, 2};
    static constexpr int kTeam1[2] = {1, 3};
    // Canonical (relative to active player): {Active, NextOpp, Teammate, PrevOpp}.
    static constexpr int kCanonicalPairingT0[4] = {0, 0, 2, 2};
    static constexpr int kCanonicalPairingT1[4] = {1, 3, 1, 3};
};
```

Every templated function lives behind one of these names — there is no runtime player count. The `are_teammates` contract guarantees `are_teammates(p, p) == true` for both traits; the TD-target code relies on this so 1v1 collapses to the original semantics bit-for-bit.

### Backward-compat aliases

To keep 1v1 callers working without explicit `<Yams1v1>` everywhere, the migration introduced aliases:

```cpp
using BoardState     = BoardStateT<Yams1v1>;
using GameContext    = GameContextT<Yams1v1>;
using GameState      = GameStateT<Yams1v1>;
using GameInstance   = GameInstanceT<Yams1v1>;
using TrainingSample = TrainingSampleT<Yams1v1>;
using ReplayBuffer   = ReplayBufferT<Yams1v1>;
using TrainingLoop   = TrainingLoopT<Yams1v1>;
using GameSession    = GameSessionT<Yams1v1>;
using SessionManager = SessionManagerT<Yams1v1>;
using UIServer       = UIServerT<Yams1v1>;
using TournamentManager = TournamentManagerT<Yams1v1>;
// …and explicit `*2v2` aliases for the Yams2v2 counterparts.
```

---

## 3. Engine and scoring

`src/engine/` is templated end-to-end. `BoardStateT<Traits>` has `cells[Traits::kNumPlayers][kNumColumns][kNumRows]` and the per-player metadata (`upper_sum`, `legal_*`, etc.) widened to `kNumPlayers`. `cells_filled` is `uint16_t` so 312 cells doesn't overflow.

### The Golden Rule

Implemented in `recalc_golden_max` by iterating `for (int p = 0; p < Traits::kNumPlayers; ++p)` over every player's sheet. In 2v2 this naturally includes the teammate's scores — no special-casing.

### SS/LS interlock

`scoring.cc` consults `ctx.golden_max[col][kRowSS]` for the LS-must-be-strictly-greater-than-highest-SS check. Since `golden_max` is built over all players, LS is also gated by the teammate's SS. The `golden_rule_teammate_test.cc` and `ls_interlock_teammate_test.cc` (in `tests/engine/`) regression-test this.

### `compute_duel<Traits>` — the load-bearing rewrite

`src/engine/duel.cc` is **not** a templatised version of the old 1v1 code — the per-pairing rule could not be expressed with the old "one shared crush + one shared bonus_value per column" shortcut. The new shape:

```cpp
template <typename Traits>
int compute_duel(const BoardStateT<Traits>& board,
                 const GameContextT<Traits>& ctx) {
    int total = 0;
    for (int col = 0; col < kNumColumns; ++col) {
        int raw[Traits::kNumPlayers] = {};        // raw = cells + upper_bonus
        // ... compute raw[p] for every p
        for (int pi = 0; pi < Traits::kNumPairings; ++pi) {
            int t0p = Traits::kCanonicalPairingT0[pi];
            int t1p = Traits::kCanonicalPairingT1[pi];
            int crush_t0 = crush_multiplier(raw[t0p], raw[t1p]);
            int crush_t1 = crush_multiplier(raw[t1p], raw[t0p]);
            int active_crush = std::max(crush_t0, crush_t1);
            int bonus_value  = (active_crush > 1) ? 100 : 200;
            int adj0 = raw[t0p] + (is_clean[t0p] ? bonus_value : 0);
            int adj1 = raw[t1p] + (is_clean[t1p] ? bonus_value : 0);
            total += (adj0 - adj1) * active_crush * coeff[col];
        }
    }
    return total;
}
```

Return value: signed Team 0 margin (positive = Team 0 wins). 1v1 instantiations produce identical output to the pre-migration code on every fixture.

Pairing iteration uses the *canonical* index lists in the trait, not raw seat indices — see §5.

---

## 4. Heuristic in 2v2

`heuristic_evaluate_research<Traits>` and its v1–v3 cousins are templated. They build a per-cell expected raw score using the precomputed DP tables (which are variant-agnostic — they take a `(player, col, row, ctx)` and read `ctx.golden_max`).

For move selection, the scalar EV in 2v2 runs all four cross-pairings through the same per-pairing crush + bonus-value pipeline as `compute_duel`, then sums into a Team 0 vs Team 1 margin. This is the same EV-of-margins approximation as the 1v1 heuristic, just over four pairings instead of one. Implementation in `src/heuristic/heuristic_bot.cc`; coverage in `tests/heuristic/heuristic_2v2_test.cc`.

The *tensor* features keep per-player heuristic signals separate — the network learns its own combination. See §5.

---

## 5. Tensor layout (`kTensorSize = 2126`)

The tensor is generated by `generate_tensor<Traits>` and `generate_tensor_batch<Traits>` in `src/engine/tensor.cc`. The defining property in 2v2 is **rotational invariance** — the same board observed from any of the four seats must produce a tensor that is one cyclic permutation of any other. The implementation achieves this by emitting sheets in *canonical relative order*:

```
1v1: [Active, Opp]
2v2: [Active, NextOpp, Teammate, PrevOpp]
```

`Active = active player; NextOpp = (active+1)%4; Teammate = (active+2)%4; PrevOpp = (active+3)%4`. The network never sees raw seat indices.

### Feature groups (2v2 dimensions in parentheses)

| Group | 1v1 size | 2v2 size | Content |
|-------|----------|----------|---------|
| **A** | 312 | **624** | Per-player × column × row × {filled flag, normalised value} (canonical sheet order) |
| **B.1** | 24 | **48** | Per-player × column × {upper_sum/100, E[raw]/500} |
| **B.2** | 84 | **336** | Per-pairing × column × 14 features (rem, margin_now, margin_E, crush probabilities, points-to-crush thresholds) |
| **C** | 156 | **312** | Per-player × column × row single-turn-non-scratch probability |
| **D** | 14 | **14** | Global aggregates: column coefficients (6), filled fraction, total duel margins (now + expected), dominance flags, phase flags |
| **E** | 216 | **432** | Per-player × column × 18 DP upper-section P/EV at three horizons (T_min, T_mid, T_max) |
| **F** | 180 | **360** | Per-player × column × 15 DP mid/low P/EV and P_clean at three horizons |
| Total | 986 | **2126** | |

In 2v2 Groups A/B.1/C/E/F scale by `kNumPlayers / 2 = 2×` because they're per-player; Group B.2 scales by `kNumPairings / 1 = 4×` because it's per-pairing; Group D is a global summary and stays at 14 features.

### Tests pinning the rotational invariance

- `tests/engine/tensor_rotation_test.cc` — cyclically shift the four sheets by one seat, recompute the tensor with `active = 1`. Assert bit-equality vs the original (`active = 0`) tensor.
- `tests/engine/tensor_team_swap_test.cc` — swap P0↔P2 (within-team permutation). The `active = 0` tensor after the swap must equal the `active = 2` tensor before the swap (validates the canonical view sees the teammate correctly).
- 1v1 tensor stays bit-equal to the pre-migration output on every fixture — the canonical view collapses to `[Active, Opp]`.

### Group B.2 — new in 2v2

The per-pairing crush/duel summary expands from one pairing to four. Iteration order matches `Traits::kCanonicalPairingT{0,1}`:

```
1v1: (Active, Opp)
2v2: (Active, NextOpp), (Active, PrevOpp),
     (Teammate, NextOpp), (Teammate, PrevOpp)
```

This means the network sees the pairings in a *seat-relative* order, supporting the rotational-invariance contract.

---

## 6. RL: team-aware TD targets

`extract_training_samples<Traits>` (`src/self_play/training_data.cc`) is templated on the trait so the bootstrap logic consults `Traits::are_teammates(future_player, my_player)`. The semantics in pseudo-code:

```cpp
auto terminal_for = [&](int8_t my_player) {
    bool same_team_as_p0 = Traits::are_teammates(my_player, 0);
    if (use_margin) return same_team_as_p0 ? terminal_p0 : -terminal_p0;
    return same_team_as_p0 ? terminal_p0 : 1.0 - terminal_p0;
};

// TD(0):
target = Traits::are_teammates(next.player, my_player)
       ? next.value        // teammate's value already encodes my-team probability
       : flip(next.value); // opponent's value is the other team's probability

// TD(λ) Watkin's cut: same flip rule applies to the bootstrap term.
```

In 1v1 `are_teammates(p, q) == (p == q)`, so the 2v2 logic collapses to "flip on opponent, no flip on self" — exactly the original 1v1 behaviour. In 2v2 it correctly *does not* flip on the teammate's value.

### Why this is the highest-stakes correctness fix

Without team-aware targets, the bootstrap would treat the teammate's value as if it were an opponent's. P0 and P2 (Team 0) would learn antagonistic targets despite winning or losing together — the network would never converge on coordinated team play. `tests/self_play/trajectory_2v2_test.cc` includes scenarios for:

- MC mode: Team 0 wins ⇒ both P0 and P2 samples target 1.0; P1 and P3 target 0.0.
- TD(0): bootstrap from a teammate's future step uses the value unflipped.
- TD(λ): a Watkin's cut against a teammate's *exploratory* value is also unflipped.

### `GameInstanceT<Traits>::result` and `final_duel_margin`

Both stay as **Team-0-perspective scalars** (no per-player array). `terminal_for` handles each player's perspective via `are_teammates(my_player, 0)`. This matches the 1v1 convention and keeps the trajectory size minimal.

---

## 7. Hyperparameter rationale — `mlp-512`

2v2 differs from 1v1 in three ways that matter for hyperparameters:

1. **Larger state space + harder credit assignment** ⇒ wider model.
2. **2× samples per game** ⇒ a fixed-size buffer holds half as many *independent games*.
3. **Diluted credit per decision** (three other agents act between your turns) ⇒ lower learning rate and longer heuristic decay.

Concrete deltas vs `config/mlp-256/seed.yaml`:

| Knob | mlp-256 (1v1) | mlp-512 (2v2) | Rationale |
| :---- | :---- | :---- | :---- |
| `model.architecture` | mlp | mlp | unchanged |
| `model.hidden_width` | 256 | **512** | ~2× input + harder problem |
| `model.hidden_layers` | 3 | 3 | unchanged |
| `model.learning_rate` | 0.00045 | **0.0003** | noisier per-sample credit |
| `model.input_size` | 986 | **2126** | `Yams2v2::kTensorSize` |
| `game_variant` | "1v1" (default) | **"2v2"** | drives runtime dispatch |
| `td_mode` | tdlambda | tdlambda | unchanged |
| `td_lambda` | 0.95 | 0.95 | per-sample variance coefficient `(1−λ)/(1+λ)` is λ-only, not trajectory-length |
| `train_batch_size` | 8192 | 8192 | unchanged |
| `train_steps_per_collect` | 0.041 | 0.041 | unchanged — already game-heavy |
| `replay_capacity` | 1,000,000 | **2,000,000** | preserves independent-game count |
| `min_buffer_size` | 100,000 | **150,000** | scaled to per-game sample count |
| `heuristic_decay_steps` (seed) | 80,000 | **120,000** | low-variance bootstrap; 2v2 has more to learn |
| `initial_heuristic_weight` (seed) | 1.0 | 1.0 | unchanged |
| `num_steps` (seed) | 100,000 | **150,000** | matches the longer heuristic-decay window |
| temperature schedule | unchanged | unchanged | |
| `past_opponent_probability` (selfplay/2) | 0.2 | 0.2 | unchanged |
| `self_play.num_workers` | 24 | 24 | unchanged |
| `self_play.num_games` | 2048 | 2048 | unchanged |

What is *not* a 2v2 issue:

- **TD(λ) per-sample variance.** Coefficient is `(1−λ)/(1+λ)` — depends only on λ, not trajectory length. Keep `td_lambda = 0.95`.
- **Bias from long on-policy traces.** All four agents share the policy; same situation as 1v1.

### No within-team augmentation

Considered and rejected. The within-team symmetry is a single binary swap (Active sheet ↔ Teammate sheet), not a rich group like Go's dihedral 8. The canonical-view tensor (§5) already bakes in the larger cyclic symmetry across seats. Cost of augmentation (2× buffer pressure, paired-sample correlation in batches) outweighs the marginal regularisation benefit. If empirical evidence later shows asymmetric teammate play, revisit as on-the-fly 50%-probability swap in the training batch.

---

## 8. Self-play, training, evaluation, tournament

Every layer is templated on `Traits`. The cascade:

```
GameInstanceT<T>  → GameQueueT<T>  → BatchManagerT<T>
        ↘                                    ↘
         worker_thread<T>  →  coordinator_thread<T>  →  resolved queue
                                       ↓
                            InferenceEngine (variant-agnostic)
                                       ↓
                        SelfPlayOrchestratorT<T>
                                       ↓
                          TrainingLoopT<T>  ↔  ReplayBufferT<T>
                                       ↓
              run_evaluation<T> + TournamentManagerT<T>
```

`InferenceEngine` is **not** templated — it takes a CPU float tensor of shape `[batch_size, model.input_size]` and reads the model's configured `input_size` at construction. A 1v1 engine with `input_size=986` and a 2v2 engine with `input_size=2126` coexist in the same process (the UI server can spin up either, but a given session is always one variant).

### Checkpoint variant tag

`ModelTrainer::save_checkpoint` writes a `game_variant` field (1 or 2). `load_checkpoint` reads it back and throws `std::runtime_error` if it doesn't match the trainer's `ModelConfig::game_variant`. `mode_eval` reads this tag *before* constructing the trainer and dispatches to `run_eval_variant<Yams1v1>` or `run_eval_variant<Yams2v2>` accordingly — no `--game_variant` flag is needed when evaluating.

### Evaluator

`run_evaluation<Traits>(model, …, num_games, base_seed)` plays `num_games` games of NN-vs-heuristic. In 2v2 the NN drives both seats of one team (anchored at `nn_player ∈ {0, 1}` which means "Team `nn_player`"); the heuristic drives the other team. Games alternate which team the NN anchors. `EvalResult::nn_wins_as_p0` counts games where NN was on Team 0; `nn_wins_as_p1` is Team 1.

### Tournament

`TournamentManagerT<Traits>` runs a round-robin between any number of participants. In 2v2, every game is a 2v2 with one participant driving both seats of one team and the other driving the other team. The grid in `TournamentState` mirrors A-vs-B and B-vs-A so the UI can render team perspectives symmetrically. NN-type participants are rejected at start if their checkpoint's `input_size` doesn't match the active variant.

---

## 9. UI and play binary

`UIServerT<Traits>` / `SessionManagerT<Traits>` / `GameSessionT<Traits>` are header-only templates. The HTTP routes are unchanged; the JSON state now emits two extra top-level fields so the frontend can pick the right layout:

```json
{
  "game_variant": "2v2",
  "num_players": 4,
  "boards": { "player0": …, "player1": …, "player2": …, "player3": … },
  …
}
```

`static/js/game.js` calls `ensureBoardContainers(numPlayers, variant)` which dynamically creates `<div id="board-p2">` and `<div id="board-p3">` if they don't exist yet. CSS:

```css
.boards-area[data-variant="2v2"] {
  display: grid;
  grid-template-columns: repeat(2, minmax(260px, 1fr));
}
```

Team shading: `.board-container.team-0` (P0, P2) gets one accent colour; `.team-1` (P1, P3) gets the other. The frontend computes team membership locally with `(p & 1) === 0`.

The `New Game` HTML form currently exposes selectors for P0 and P1; P2 and P3 default to Heuristic V2 when the server is in 2v2 mode. Full control is available over the JSON API by passing `player0..player3`.

---

## 10. File map

Files that gained 2v2 support (all keep 1v1 working unchanged):

```
src/engine/
  game_traits.h          — Yams1v1 / Yams2v2 traits
  board_state.h          — BoardStateT<Traits>
  game_context.h         — GameContextT<Traits>
  game_state.h           — GameStateT<Traits>
  scoring.{h,cc}         — calculate_score, is_terminal, cells_remaining
  legal_moves.{h,cc}     — rebuild_legal_placements + update
  placement.{h,cc}       — apply_placement
  game_flow.{h,cc}       — init_game, perform_reroll, perform_placement
  duel.{h,cc}            — compute_duel (per-pairing rewrite)
  tensor.{h,cc}          — canonical-view tensor, B.2 expansion

src/solver/
  solver.{h,cc}          — solver_get_requests, solver_resolve

src/heuristic/
  heuristic_bot.{h,cc}   — evaluate_* and play_turn families, 4-pair scalar EV

src/self_play/
  game_instance.h        — GameInstanceT<Traits>, TrajectoryStepT
  game_queues.h          — GameQueueT (header-only)
  batch_manager.h        — BatchManagerT (header-only)
  worker.{h,cc}          — worker_thread<Traits>
  coordinator.{h,cc}     — coordinator_thread<Traits>
  orchestrator.{h,cc}    — SelfPlayOrchestratorT<Traits>
  training_data.{h,cc}   — extract_training_samples<Traits>, TrainingSampleT

src/training/
  replay_buffer.h        — ReplayBufferT (header-only)
  training_loop.{h,cc}   — TrainingLoopT<Traits>
  resume.{h,cc}          — resume_from_checkpoint<Traits>, init_from_checkpoint<Traits>

src/model/
  model_config.h         — kGameVariant1v1 / kGameVariant2v2, ModelConfig::game_variant
  trainer.{h,cc}         — checkpoint variant tag write/read; input_size from cfg
  inference.{h,cc}       — input_size_ from model->config() (no kTensorSize fallback)

src/eval/
  evaluator.{h,cc}       — run_evaluation<Traits>, nn_play_turn<Traits>
  tournament.{h,cc}      — TournamentManagerT<Traits>

src/config/
  config_loader.cc       — parses game_variant, --game_variant flag, auto-promotes input_size
  config_validator.cc    — rejects mismatched (game_variant, input_size)
  config_printer.cc      — emits game_variant

src/ui/
  game_session.h         — GameSessionT<Traits>
  session_manager.{h,cc} — SessionManagerT (header-only)
  json_serialization.{h,cc} — game_state_to_json<Traits>, num_players/game_variant fields
  server.{h,cc}          — UIServerT (header-only)
  ui_main.cpp            — variant dispatch + TournamentManagerT for both
  play_main.cpp          — variant dispatch
  static/js/game.js      — ensureBoardContainers, dynamic 2×2 rendering
  static/css/style.css   — .boards-area[data-variant="2v2"] grid + team shading

src/main.cpp             — run_training_variant<T>, run_eval_variant<T>, mode_eval auto-detect

tests/engine/duel_2v2_test.cc, golden_rule_teammate_test.cc, tensor_rotation_test.cc, tensor_team_swap_test.cc
tests/heuristic/heuristic_2v2_test.cc
tests/self_play/trajectory_2v2_test.cc
tests/training/training_loop_2v2_test.cc
```

---

## 11. What's intentionally still 1v1-only

- **`src/solver/mc_bot.{h,cc}` and `mc_solver`.** MC rollout bot is broken/unused. `SessionManagerT<Yams2v2>::play_mc_turn` falls back to Heuristic V2 via `if constexpr`.
- **UI `New Game` form P2/P3 dropdowns.** The HTML form has P0 and P1 selectors; in 2v2 P2/P3 default to V2. The JSON API accepts all four. Adding the selectors is a small HTML-only follow-up.

Everything else — engine, scoring, duel, heuristic, tensor, RL targets, self-play, training, eval, tournament, web UI, REST API, JSON state, checkpoint format, config loader, validator — works in both variants out of the box.

---

## 12. Pointer summary

- **Rules:** [`pro_yams_2v2.md`](pro_yams_2v2.md)
- **1v1 design doc:** `pro_yams_ai_high_level_design.md`
- **CLI / config / training / eval / UI usage:** [`usage.md`](usage.md)
- **Migration history (now done):** `2v2migration.md`, `2v2migration_remaining.md`
- **Training configs:** `config/mlp-256/` (1v1), `config/mlp-512/` (2v2)
