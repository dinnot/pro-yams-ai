# 2v2 Migration — Remaining Work

This document is the resumption guide for the 2v2 migration. Tasks 1–6 and the structural part of Task 7 are complete; what follows is everything still to do.

> **Out of scope here:** Task 9 (`config/mlp-512/` configs) — the user owns that. Everything else below is fair game.

---

## Current state snapshot (already done)

For reference, **do not re-do these**:

- **Engine** (`src/engine/`): `BoardStateT<Traits>`, `GameContextT<Traits>`, `GameStateT<Traits>` are templated. `BoardState`/`GameContext`/`GameState` are 1v1 aliases. `cells_filled` is `uint16_t`. `apply_placement`, `calculate_score`, `get_legal_placements`, `rebuild_legal_placements`, `update_legal_placements_after_move`, `has_filled_neighbor`, `is_terminal`, `cells_remaining`, `column_cells_remaining`, `rebuild_context_from_board`, `init_board`, `init_context`, all of `game_flow.cc`, and all of `duel.cc` are templated and explicitly instantiated for `Yams1v1` and `Yams2v2`.
- **Game traits** (`src/engine/game_traits.h`): `Yams1v1`, `Yams2v2` with `kNumPlayers`, `kNumTeams`, `kPlayersPerTeam`, `kCellsPerSheet`, `kTotalCells`, `kTensorSize`, `kNumPairings`, `kTeam0[]`, `kTeam1[]`, `kCanonicalPairingT0[]`, `kCanonicalPairingT1[]`, `are_teammates(p1, p2)`. Contract: `are_teammates(p, p) == true`.
- **Tensor helpers** (`src/engine/tensor.cc`, `src/solver/dp_eval.cc`): `count_filled_cells`, `count_empty_cells`, `sum_all_filled`, `compute_column_raw_score`, `compute_column_potential_score`, `compute_total_potential`, `build_Sc`, `get_E_raw`, `get_P_clean`, `get_E_raw_var` all templated.
- **Tensor generation** (`src/engine/tensor.cc`): `generate_tensor<T>`, `generate_tensor_batch<T>` emit in **canonical view** `[Active, NextOpp, Teammate, PrevOpp]` and per-pairing groups. Bit-equal under rotation. 1v1 = 986 features, 2v2 = 2126 features.
- **Heuristic evaluators** (`src/heuristic/heuristic_bot.cc`): `heuristic_evaluate`, `heuristic_evaluate_v2`, `heuristic_evaluate_v3`, `heuristic_evaluate_research` all templated. 2v2 uses proper 4-pair-duel team aggregation. V3 Rule 1 drops the kColumnAim when ALL opps are scratched. `opp_aware_factor` amplifies pairings involving active player when behind worst threat. `dominance_bonus` compares to average opponent.
- **Solver** (`src/solver/solver.cc`): `solver_get_requests<T>`, `solver_resolve<T>`, `solver_resolve_greedy<T>` templated and instantiated for both variants.
- **GameInstance** (`src/self_play/game_instance.h`): `GameInstanceT<Traits>`, `TrajectoryStepT<Traits>` templated. `GameInstance`/`TrajectoryStep` are 1v1 aliases. `kMaxTrajectorySteps = Traits::kTotalCells`.
- **Training data extraction** (`src/self_play/training_data.cc`): `extract_training_samples<Traits = Yams1v1>` templated with team-aware bootstrap (`Traits::are_teammates(future.player, my_player) ? future.value : flip(future.value)`).
- **Model** (`src/model/model_config.h`, `src/model/trainer.cc`): `ModelConfig::game_variant` field exists. Checkpoints serialize the variant tag. `load_checkpoint` throws `std::runtime_error` on mismatch.

**1v1 paths bit-identical.** All 11 test suites pass (155 engine, 33 tensor + 5 rotation, 11 heuristic + 5 2v2, 14 trajectory + 6 2v2, etc.). Engine bench within 2% on hot paths.

**Files where the migration HAS NOT touched:**
- `src/self_play/{worker,batch_manager,coordinator,orchestrator,game_queues}.{h,cc}` — these still hold `GameInstance*` (= 1v1 alias).
- `src/training/training_loop.{h,cc}`
- `src/main.cpp` (no variant dispatch)
- `src/ui/*`, `src/config/*` (no `game_variant` config field)
- `src/eval/*`, `src/solver/mc_bot.*` (1v1-only — see scoping notes below)
- `src/heuristic/heuristic_bot.cc`'s `heuristic_play_turn*` family (explicit `<Yams1v1>` instantiations because the solver wasn't templated at the time — now it is, but the family is only called from 1v1 eval/tournament so it can stay 1v1 if scope-bound).

---

## Architectural patterns / gotchas to remember

These bit me during the migration. Internalise them before touching the cascade.

### Pattern 1: Template + alias for backward compat
The standard pattern across the migration is:

```cpp
template <typename Traits> struct FooT { /* ... */ };
using Foo    = FooT<Yams1v1>;   // backward-compat alias
using Foo2v2 = FooT<Yams2v2>;   // explicit 2v2 alias
```

Existing call sites that use `Foo` keep working unchanged. New 2v2 code uses `FooT<Yams2v2>` or `Foo2v2`. **Do not** try to name the template `Foo` directly with `using Foo = ...` later — it produces a "conflicting declaration" error (the template `Foo` and the alias `Foo` collide).

### Pattern 2: Forward-declaring a templated alias across headers
If a header forward-declares the old non-templated `struct Foo;` (to avoid pulling in heavy headers), the alias breaks it. Fix:

```cpp
#include "engine/game_traits.h"
template <typename Traits> struct FooT;
using Foo = FooT<Yams1v1>;
```

This is what `src/self_play/game_queues.h` does for `GameInstance`.

### Pattern 3: Template-aware function calls inside templated bodies
Inside a templated function, calling another templated function does NOT always deduce — sometimes you need to specify explicitly:

```cpp
template <typename T>
void outer(BoardStateT<T>& b, GameContextT<T>& c) {
    rebuild_legal_placements<T>(p, b, c);  // explicit <T>
}
```

Without the explicit `<T>` the compiler can deduce from arguments, but it can fail in some chains (especially when the function is yet to be defined in this TU). **Always be explicit** when calling templated helpers from inside templated bodies.

### Pattern 4: Explicit instantiation at the bottom of `.cc`
Every templated function or class whose definition lives in a `.cc` (not the header) must add explicit instantiations at the bottom:

```cpp
template void foo<Yams1v1>(/* signature */);
template void foo<Yams2v2>(/* signature */);
```

Otherwise the linker won't find the symbol for callers in other translation units. **Easy to forget; the linker error is unambiguous when you do.**

### Pattern 5: `BoardStateCopy` size gotcha
`BoardStateT<Traits>` size: 1v1 ≈ 166 bytes, 2v2 ≈ 322 bytes. The static asserts at the bottom of `board_state.h` constrain these (≤168 / ≤336). Don't add new per-player fields without re-checking.

### Pattern 6: int8_t overflow in tests
When building synthetic board states by hand in tests, individual cell values must fit in `int8_t` (i.e., ≤ 127). For sums > 100, spread across multiple lower-row cells. See `tests/engine/duel_2v2_test.cc::set_player_col0` for the working pattern.

### Pattern 7: `compute_duel` is not a pure templatisation
The 1v1 → 2v2 transition for duel computation is a **rewrite**, not just a template wrapper. The plan calls this out; the existing `compute_duel<T>` in `src/engine/duel.cc` already handles it. The hot point: **per-pairing crush + per-pairing clean-bonus value**, NOT a column-shared `active_crush`. If you see `active_crush = max(crush0, crush1)` shared across multiple pairings, that's the old bug.

### Pattern 8: TD bootstrap is the highest-stakes correctness fix
`extract_training_samples` in `src/self_play/training_data.cc` MUST flip the bootstrap sign based on `Traits::are_teammates(future.player, my_player)`, **not** `future.player != my_player`. The latter trains teammates antagonistically. The fix is already in place; do not regress.

### Pattern 9: Default template parameter on class templates is annoying
`template <typename T = Yams1v1> class Foo {};` does NOT make `Foo` (bare name, no `<>`) a valid type. Existing call sites with `Foo x;` break. Use the explicit alias pattern (Pattern 1) instead.

### Pattern 10: Watch the per-column recompute in tensor batching
`generate_tensor_batch` post-placement must recompute the placed column's `PCData` for **all players**, not just the active one, because `apply_placement` mutates `ctx.golden_max[col][row]` which affects every player's expected EVs in that column. (Existing 1v1 batch tests catch this regression immediately.)

---

## Task 7 (continued) — the self-play / training cascade

The remaining ~2000 lines of templating. **Do this layered, bottom-up**, building and running tests between each layer. Targets in order of dependency:

```
game_queues  →  batch_manager  →  worker  →  coordinator  →  orchestrator  →  training_loop  →  main.cpp dispatch
```

### Step 7.A — Templatize `game_queues`

**Files:** `src/self_play/game_queues.{h,cc}`

`GameQueue` currently holds `GameInstance*`. For 2v2 it must hold `GameInstanceT<Yams2v2>*`. Simplest: templatize.

```cpp
template <typename Traits>
class GameQueueT {
    // … same body, but with GameInstanceT<Traits>* throughout …
};

using GameQueue    = GameQueueT<Yams1v1>;
using GameQueue2v2 = GameQueueT<Yams2v2>;
```

In `game_queues.h`, drop the forward-decl/alias pair (it was a workaround) and replace with:

```cpp
#include "self_play/game_instance.h"
```

…now that `GameInstance` is a real type (alias), there's no reason to forward-declare.

**Gotchas:**
- `game_queues.cc` defines methods out-of-line. They all need `template <typename Traits>` headers, and either:
  - (a) move the bodies into the header (simplest), or
  - (b) keep them in `.cc` with explicit instantiation at the bottom.
- The class uses `std::deque`, `std::mutex`, `std::condition_variable`. None of these depend on the template parameter — Traits affects only the pointer type held.

**Validation:** existing `tests/self_play/queue_test.cc` should pass unchanged (it uses 1v1 alias).

### Step 7.B — Templatize `BatchManager`

**Files:** `src/self_play/batch_manager.{h,cc}` (265 lines)

`InferenceBatch::Entry` holds `GameInstance* game`. `torch::Tensor storage` is allocated with shape `[max_tensors, kTensorSize]` — must become `[max_tensors, Traits::kTensorSize]`.

```cpp
template <typename Traits>
class InferenceBatchT {
public:
    InferenceBatchT(int max_tensors, bool use_pinned);
    struct Entry { GameInstanceT<Traits>* game; int start_idx; int count; };
    torch::Tensor storage;
    float* data_ptr;
    // … rest unchanged …
};

template <typename Traits>
class BatchManagerT {
    // … API unchanged, types templated …
public:
    float* reserve(int req_count, InferenceBatchT<Traits>*& out_batch, int& out_offset);
    void commit(InferenceBatchT<Traits>* batch, GameInstanceT<Traits>* game,
                int offset, int req_count);
    InferenceBatchT<Traits>* pop_ready_batch();
    void recycle_batch(InferenceBatchT<Traits>* batch);
    // …
};

using InferenceBatch    = InferenceBatchT<Yams1v1>;
using InferenceBatch2v2 = InferenceBatchT<Yams2v2>;
using BatchManager      = BatchManagerT<Yams1v1>;
using BatchManager2v2   = BatchManagerT<Yams2v2>;
```

In the constructor body, change `torch::empty({max_tensors, kTensorSize}, …)` to `torch::empty({max_tensors, Traits::kTensorSize}, …)`.

**Gotchas:**
- `BatchManager`'s `flush_active_batch_locked`, mutex/cv usage — all unchanged.
- The `push_claimed` flag and the active-batch state machine logic is invariant.

**Validation:** `tests/self_play/batch_manager_test.cc` should pass unchanged.

### Step 7.C — Templatize `worker`

**Files:** `src/self_play/worker.{h,cc}` (358 lines, plus the .h declaration)

The function `worker_thread` becomes a function template. Its bodies are the most complex per-file in the migration because they touch many cross-cutting concerns.

```cpp
template <typename Traits>
void worker_thread(GameQueueT<Traits>& available,
                   BatchManagerT<Traits>& batch_manager,
                   BatchManagerT<Traits>* opponent_batch_manager,
                   GameQueueT<Traits>& completed,
                   const PrecomputedTables& tables, const SolverConfig& config,
                   std::atomic<bool>& shutdown);
```

Inside `worker.cc`:
- All `GameInstance*` → `GameInstanceT<Traits>*`.
- All `solver_get_requests(state, ctx, ...)` calls → `solver_get_requests<Traits>(...)` (or rely on deduction since `state` is now `GameStateT<Traits>`).
- All `generate_tensor*` calls → `generate_tensor_batch<Traits>(...)`.
- All `heuristic_evaluate*` calls → `heuristic_evaluate_research<Traits>(...)` etc.
- `apply_placement(…)` → `apply_placement<Traits>(…)`.
- `is_terminal(board)` → `is_terminal<Traits>(board)`.
- `compute_duel(board, ctx)` → `compute_duel<Traits>(board, ctx)`.
- `kTensorSize` references (for memcpy etc.) → `Traits::kTensorSize`.
- `kTotalCells` references → `Traits::kTotalCells`.

**Specific hot spots in `worker.cc` to watch:**
- Around line 290–291: `assert(game->trajectory_length < GameInstance::kMaxTrajectorySteps); TrajectoryStep& step = game->trajectory[game->trajectory_length];` → `GameInstanceT<Traits>::kMaxTrajectorySteps` and `TrajectoryStepT<Traits>& step = …`.
- The heuristic blending block (around line 130–200) calls `heuristic_evaluate*` with the un-templated names — these need `<Traits>` suffixes.
- The `current_player` advancement: now driven by `perform_placement<Traits>` (already done in `game_flow.cc`).

**Explicit instantiation at the bottom:**
```cpp
template void worker_thread<Yams1v1>(GameQueueT<Yams1v1>&, BatchManagerT<Yams1v1>&,
                                     BatchManagerT<Yams1v1>*, GameQueueT<Yams1v1>&,
                                     const PrecomputedTables&, const SolverConfig&,
                                     std::atomic<bool>&);
template void worker_thread<Yams2v2>(/* same with Yams2v2 */);
```

**Validation:** `tests/self_play/worker_test.cc` and `integration_test.cc` should pass unchanged.

### Step 7.D — Templatize `coordinator`

**Files:** `src/self_play/coordinator.{h,cc}` (142 lines)

`coordinator_thread` consumes batches from `BatchManager`, runs inference, and re-pushes games to `available`. Similar shape to `worker_thread`. Same templating pattern.

```cpp
template <typename Traits>
void coordinator_thread(GameQueueT<Traits>& available,
                        BatchManagerT<Traits>& batch_manager,
                        InferenceEngine& inference,
                        std::atomic<bool>& shutdown);
```

The `InferenceEngine` itself isn't templated — it takes raw float tensors of dimensions matching the model's `input_size`. As long as the `BatchManagerT<Traits>::storage` has the right shape, the inference call works.

**Validation:** `tests/self_play/coordinator_test.cc` should pass unchanged.

### Step 7.E — Templatize `SelfPlayOrchestrator`

**Files:** `src/self_play/orchestrator.{h,cc}` (183 lines, plus .h with class definition)

This is the biggest single class. Holds `std::vector<std::unique_ptr<GameInstance>> games_`, owns workers and coordinators, the BatchManager(s), and queues.

```cpp
template <typename Traits>
class SelfPlayOrchestratorT {
public:
    SelfPlayOrchestratorT(const SelfPlayConfig& config,
                          const PrecomputedTables& tables,
                          InferenceEngine& inference,
                          const SolverConfig& solver_config,
                          InferenceEngine* opponent_inference = nullptr);
    ~SelfPlayOrchestratorT();

    void start();
    void stop();
    int  collect_completed(GameInstanceT<Traits>** out, int max_count);
    void recycle_game(GameInstanceT<Traits>* game, uint64_t new_seed,
                      bool use_past_opponent = false,
                      int past_opponent_player = -1);
    // … rest unchanged …

private:
    // … all field types templated:
    std::vector<std::unique_ptr<GameInstanceT<Traits>>> games_;
    GameQueueT<Traits>                                  available_queue_;
    std::unique_ptr<BatchManagerT<Traits>>              batch_manager_;
    std::unique_ptr<BatchManagerT<Traits>>              opponent_batch_manager_;
    GameQueueT<Traits>                                  completed_queue_;
    // … threads etc. unchanged …
};

using SelfPlayOrchestrator    = SelfPlayOrchestratorT<Yams1v1>;
using SelfPlayOrchestrator2v2 = SelfPlayOrchestratorT<Yams2v2>;
```

In the implementation:
- `worker_thread` and `coordinator_thread` calls get `<Traits>` suffix.
- The `recycle_game` body resets the game's trajectory and state — uses templated `init_board`/`init_context`.
- The constructor allocates games: `games_.emplace_back(std::make_unique<GameInstanceT<Traits>>());`.

**Gotchas:**
- The orchestrator passes `BatchManagerT<Traits>*` (raw pointer) to `worker_thread` and `coordinator_thread`. Make sure both accept the templated type.
- `past_opponent_player` is in `{0, 1}` for 1v1 — for 2v2, it could be any of `{0, 1, 2, 3}`. The plan doesn't address past-opponent rotation in 2v2 explicitly; for the smoke run, just leave `past_opponent_probability = 0` in the 2v2 seed config (already implied since `seed.yaml` doesn't enable it; the user's `mlp-256/selfplay.yaml` does set it to 0.2 for 1v1). Document in the orchestrator that `past_opponent_player` in 2v2 means "this single seat plays the old model" — the team semantics need a follow-up if past-opponent rotation is wanted in 2v2.

**Validation:** No standalone orchestrator test exists; integration test in `tests/self_play/integration_test.cc` covers end-to-end.

### Step 7.F — Templatize `TrainingLoop`

**Files:** `src/training/training_loop.{h,cc}` (480 lines)

Holds `SelfPlayOrchestrator`, manages replay buffer, calls `extract_training_samples`, drives the training step loop. Similar templating pattern.

```cpp
template <typename Traits>
class TrainingLoopT {
public:
    TrainingLoopT(const TrainingConfig& config);
    void run(int num_steps);
    // …
private:
    std::unique_ptr<SelfPlayOrchestratorT<Traits>> orchestrator_;
    // … replay buffer, optimiser, etc. unchanged …
};

using TrainingLoop    = TrainingLoopT<Yams1v1>;
using TrainingLoop2v2 = TrainingLoopT<Yams2v2>;
```

**Key call sites inside `training_loop.cc`:**
- `extract_training_samples(…)` → `extract_training_samples<Traits>(…)`.
- `collect_completed(...)` returns `GameInstanceT<Traits>**`.
- The replay buffer holds `TrainingSample` (variant-agnostic — `state[kTensorSize]` is hardcoded to 986). **This is a real problem for 2v2**: TrainingSample needs to be templatized OR the replay buffer needs a variable-size tensor. Easiest fix:

```cpp
// In training_data.h:
template <typename Traits>
struct TrainingSampleT {
    float  state[Traits::kTensorSize];
    double target;
};
using TrainingSample = TrainingSampleT<Yams1v1>;
```

And cascade through `replay_buffer.{h,cc}` and `training_loop.cc`. The `replay_buffer` already stores arrays of `TrainingSample` — make it `TrainingSampleT<Traits>`.

**Replay buffer file:** `src/training/replay_buffer.{h,cc}` (159 lines). Templatize as `ReplayBufferT<Traits>`. The capacity / sampling logic is variant-agnostic; only the storage type changes.

**Trainer call site:** `trainer.train_step(states.data(), targets.data(), batch_size)` — `states` is `float*` of size `batch × kTensorSize`. For 2v2 it's `batch × 2126`. The `ModelTrainer` already takes a raw `float*` and `int batch_size`; it forms a tensor of shape `[batch_size, config_.input_size]` internally (`trainer.cc:37`). As long as `config_.input_size` matches the trait, it works.

**Gotcha:** The smoke run will crash if `config_.input_size` doesn't match `Traits::kTensorSize`. Add an assertion in `TrainingLoopT<Traits>` constructor:
```cpp
assert(config.model.input_size == Traits::kTensorSize &&
       "ModelConfig input_size must match Traits::kTensorSize");
assert(config.model.game_variant ==
       (std::is_same_v<Traits, Yams1v1> ? kGameVariant1v1 : kGameVariant2v2) &&
       "ModelConfig game_variant must match the trait");
```

### Step 7.G — Update tests for templated infrastructure

**Files to scan:** `tests/self_play/*`, `tests/training/*`.

Most tests use `GameInstance` (= 1v1 alias) → unchanged.

Add a 2v2 smoke test: `tests/training/training_loop_2v2_test.cc` (new):
```cpp
// Construct a TrainingLoop<Yams2v2>, run a few self-play games end-to-end,
// extract samples, run a training step. Just verify no crash + loss is finite.
```

This is the only **new** test required for the cascade. Existing tests should keep passing.

### Step 7.5 — `main.cpp` variant dispatch

**File:** `src/main.cpp`

After config is loaded, dispatch:

```cpp
if (cfg.training.model.game_variant == kGameVariant2v2) {
    TrainingLoopT<Yams2v2> loop(cfg.training);
    loop.run(cfg.num_steps);
} else {
    TrainingLoopT<Yams1v1> loop(cfg.training);
    loop.run(cfg.num_steps);
}
```

(Replace `cfg.training` and `cfg.num_steps` with the actual struct paths from `AppConfig`.)

The config loader should accept `game_variant: "1v1"` or `game_variant: "2v2"` strings and set `cfg.training.model.game_variant` to the matching `kGameVariant*` constant. See **Task 8** for the config loader changes.

The same dispatch pattern applies to **other binaries** with their own `main` entry points:
- `src/ui/ui_main.cpp` (the play binary)
- `src/ui/play_main.cpp`

For the eval-only binary (`tournament_main` or similar), the user may choose to keep it 1v1-only since eval/tournament aren't templated — that's a Task 8.4 decision.

### Step 7.6 — Smoke validation

**1v1 smoke (perf-parity gate):**
```bash
# Run a short training job with the existing 1v1 config:
./pro_yams_ai --config config/mlp-256/seed.yaml --num_steps 100
```
Expected: loss curve within numerical noise of the pre-migration baseline. Track total games/sec — should be within 2% of the Task 0.3 baseline.

**2v2 smoke (correctness gate):**
```bash
# Requires Task 9 (user) to produce config/mlp-512/seed.yaml.
./pro_yams_ai --config config/mlp-512/seed.yaml --num_steps 10
```
Expected: program starts, plays a few 2v2 games to completion (312 cells filled per game), trains a few steps, loss is finite and non-degenerate.

**`engine_bench` re-check after Step 7.E (orchestrator templating):**
```bash
/home/sorin/pro_yams_ai/build/tests/benchmarks/engine_bench --benchmark_min_time=1s
```
Compare `CalculateScore`, `ApplyPlacement`, `RebuildLegalPlacements`, `ComputeDuel`, `BoardStateCopy` against the baselines in `2v2migration.md` Task 0.3. Within 2% on each.

---

## Optional pairings with Task 7

These are smaller bits that pair well with the cascade because they touch the same files / call sites.

### Optional 7.X1 — Templatize `heuristic_play_turn*`

**File:** `src/heuristic/heuristic_bot.cc`

When I did Task 4, the play_turn functions (`heuristic_play_turn_research`, `heuristic_play_turn`, `play_heuristic_game`) were left 1v1-only with explicit `<Yams1v1>` instantiations because the solver wasn't templated yet. **The solver is now templated**, so these can be templatized too.

Pattern (apply to all three functions):
```cpp
template <typename Traits>
void heuristic_play_turn_research(GameStateT<Traits>& state,
                                  GameContextT<Traits>& ctx,
                                  const PrecomputedTables& tables,
                                  SolverBuffers& buffers, RNG& rng,
                                  const ResearchConfig& cfg) {
    // remove all explicit <Yams1v1> and replace with <Traits>
}
```

Drop the explicit `<Yams1v1>` calls inside (e.g., `solver_resolve_greedy<Yams1v1>(...)` → `solver_resolve_greedy<Traits>(...)`).

Add explicit instantiations for both traits. This unblocks 2v2 standalone heuristic games (used by future 2v2 eval/tournament).

**Use cases that would benefit:** future 2v2 eval, future 2v2 tournament. For Task 7's scope this is optional, but it's a 30-line change once you're already in heuristic_bot.cc.

### Optional 7.X2 — Templatize evaluator / tournament

**Files:** `src/eval/evaluator.{h,cc}` (~270 lines), `src/eval/tournament.{h,cc}` (~500 lines)

These play full 1v1 games using `GameState`, `compute_duel`, `heuristic_play_turn` etc. For 2v2 eval/tournament, templatize the same way.

**Cost vs benefit:** medium effort, only needed if you want to run **2v2 evaluation tournaments** (heuristic-vs-heuristic, model-vs-heuristic, model-vs-model). For the smoke run in Step 7.6, you don't need this — the 2v2 training loop produces its own loss metrics. Defer unless the user explicitly wants 2v2 eval.

### Optional 7.X3 — Remove PBRS

**Files:** `src/self_play/training_data.cc`, `src/training/training_config.h`

The user said PBRS is unused. Removing it is a clean simplification:
- Drop `use_pbrs`, `pbrs_upper_reward`, `pbrs_clean_reward` from `TrainingConfig`.
- Drop the `use_pbrs` parameter from `extract_training_samples`.
- Drop `pbrs_reward` field from `TrajectoryStep`.
- Remove the `if (use_pbrs) target += step.pbrs_reward;` line.
- Remove the `if (use_pbrs) ...` branches in `worker.cc`.

Roughly 30–50 lines deleted, no behaviour change. Do this before the worker.cc templating in Step 7.C so you don't have to template PBRS-related dead code.

### Optional 7.X4 — Remove backward-compat aliases (`kNumPlayers`, `kTotalCells`, `kTensorSize`)

**File:** `src/engine/constants.h`, `src/engine/tensor.h`

Once Worker/Coordinator/Orchestrator/TrainingLoop are templated, no call site should reference the bare aliases `kNumPlayers`, `kTotalCells`, `kTensorSize`. Remove them from `constants.h` / `tensor.h` and audit:

```bash
grep -rn "kNumPlayers\|kTotalCells\b\|\bkTensorSize\b" src/ tests/
```

Any remaining references must use `Yams1v1::...` or `Traits::...` explicitly. The 1v1 alias `using BoardState = BoardStateT<Yams1v1>;` etc. can stay — they're convenient for tests and 1v1-only code paths.

This is plan item 7.4. Save it for last; the aliases are harmless and removing them is mostly cosmetic. Skip if churn-averse.

---

## Task 8 — CLI, UI, tooling

### Step 8.1 — `AppConfig.game_variant`

**File:** `src/config/app_config.h` (not yet read in this migration — exists, look for `AppConfig` struct), `src/config/config_loader.cc`

Add to `AppConfig` (or wherever `TrainingConfig`'s YAML is parsed):

```cpp
std::string game_variant = "1v1";   // "1v1" or "2v2"
```

In the YAML loader, parse `game_variant` and set `cfg.training.model.game_variant` to `kGameVariant1v1` or `kGameVariant2v2`. Also propagate to `cfg.training.model.input_size` if not explicitly set:
- 1v1 default: `986`
- 2v2 default: `2126`

(The user's Task 9 config will set `input_size` explicitly, so this is just a guard.)

Surface a CLI flag too: `--game_variant 1v1|2v2`. See `src/config/config_loader.cc:117` for the pattern used by `--heuristic_version`.

### Step 8.2 — JSON export for N players

**File:** `src/ui/json_serialization.{h,cc}` (file path may differ; check `src/ui/` and `src/ui/static/`)

The JSON export currently emits 2 sheets. For 2v2 it must emit 4. Pattern:
```cpp
for (int p = 0; p < kNumPlayers; ++p) { /* emit sheet p */ }
```
becomes (after templating the JSON serializer):
```cpp
for (int p = 0; p < Traits::kNumPlayers; ++p) { /* emit sheet p */ }
```

Also emit the variant in the JSON header:
```json
{
  "game_variant": "2v2",
  "sheets": [ … ]
}
```

So the UI can render appropriately.

### Step 8.3 — UI: 2×2 grid rendering

**Files:** `src/ui/static/js/board.js`, `src/ui/static/js/play.js`, possibly templates in `src/ui/static/html/`

When `game_variant == "2v2"` and JSON contains 4 sheets, render them as a 2×2 grid:

```
+----------+----------+
| P0 (T0)  | P1 (T1)  |
+----------+----------+
| P2 (T0)  | P3 (T1)  |
+----------+----------+
```

- Team 0 (P0, P2) on the left, shaded one colour.
- Team 1 (P1, P3) on the right, shaded another colour.
- Existing 1v1 rendering (2 side-by-side sheets) unchanged.

CSS Grid is the natural fit. Add a `body[data-variant="2v2"]` selector to switch grid layout.

### Step 8.4 — Manual play wiring

**File:** `src/ui/play_main.cpp`

For human-vs-bot 2v2 testing:
- Add `--variant 2v2` CLI flag.
- Default: human plays seat 0 (P0); P1, P2, P3 are bots.
- Bot type selection: a flag like `--bot_versions V17,V17,V17` (one per non-human seat). Default to `V2` for all.
- Turn order is clockwise (A→B→C→D) — the engine handles this via `perform_placement<T>`.

The browser UI shows the human's sheet "active" (top-left in 2×2 layout) and the others as "waiting" except when their bot is taking a turn (brief animation if you want).

### Step 8.5 — Validation

Manual Human-vs-3-Bots 2v2 game in the browser:
- Launch: `./pro_yams_play --variant 2v2 --bot_versions V17,V17,V17`
- Open in browser; play 5–10 turns as P0.
- Check that **Golden Rule blocking** works: place a high score (say 24 in 6s), then watch a bot try to score lower — the bot should scratch. (Confirms `golden_max` is global across all 4 sheets.)
- Check that **clean-column highlighting** works on all four sheets.
- Play to terminal (312 cells). Verify the final scoring matches `compute_duel<Yams2v2>` and is shown as a team-vs-team result.

If browser-side dev is out of scope, a minimum acceptance is: the binary runs, accepts dice and placements via the CLI, and produces correct end-of-game team scores.

---

## Suggested execution order

1. **(Optional 7.X3)** Delete PBRS — small, frees `worker.cc` from dead branches before templating it.
2. **7.A** game_queues — smallest, sets template pattern.
3. **7.B** batch_manager — adds tensor-size routing.
4. **7.C** worker — biggest single file, most tricky.
5. **7.D** coordinator.
6. **7.E** orchestrator.
7. **7.F** training_loop + replay_buffer + TrainingSample.
8. **7.G** test updates + new 2v2 smoke test.
9. **(Optional 7.X1)** heuristic_play_turn family — easy now that solver is templated.
10. **8.1** AppConfig.game_variant + config loader.
11. **7.5** main.cpp dispatch.
12. **7.6** smoke validation (1v1 baseline + 2v2 short run).
13. **8.2** JSON export for N players.
14. **8.3** UI 2×2 rendering.
15. **8.4** play_main 2v2 wiring.
16. **8.5** manual browser test.
17. **(Optional 7.X4)** alias cleanup.
18. **(Optional 7.X2)** templatize evaluator / tournament (only if 2v2 eval is wanted).

Each step builds + runs `ctest -j 8` before moving on. The 1v1 path must stay green throughout.

---

## Acceptance criteria for "done"

- All 11 existing test suites pass (currently 100%; preserve this).
- New 2v2 tests pass: trajectory, duel, heuristic, tensor rotation, training loop smoke.
- 1v1 short training run produces a loss curve within numerical noise of the pre-migration baseline.
- 2v2 short training run completes one full game (312 placements), produces finite non-degenerate loss across the first ~10 training steps.
- A 1v1 checkpoint loaded into a 2v2 binary fails fast with the variant-mismatch error message (already implemented; the test `TrainerTest.Checkpoint_VariantMismatch_Throws` covers this).
- `engine_bench` 1v1 throughput within 2% of the Task 0.3 baseline.
- Manual 2v2 game in the browser plays correctly through to terminal.

---

## Pointers for the next session

- This file is the resumption guide. The full plan is `2v2migration.md` (still valid; Tasks 7+ are the parts not yet done).
- The architectural snapshot at the top of this doc lists what's already templated — **don't re-touch those files** unless fixing a bug.
- The patterns/gotchas section is hard-won. Each item bit me at least once during Tasks 1–6 / 7-partial. Re-read before starting a layer.
- When in doubt about test fixtures: existing 1v1 tests are the source of truth for behaviour. They MUST keep passing.
- Bench gate (`engine_bench`) is for the 1v1 path. The 2v2 path doesn't have a perf gate — it just needs to be functionally correct.
