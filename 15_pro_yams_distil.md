# `pro_yams_distil` — Implementation Plan

New training mode that distills a fixed teacher (heuristic bot V1–V17 or a target NN of any architecture) into a student `ProYamsNet`. Used to bootstrap a NN to behave like a heuristic, or to compress / re-shape an existing NN.

## Design decisions (locked in)

- **Driver policy:** teacher plays every seat. Student never picks moves — supervision-only.
- **Teacher types (v1):** both heuristic (V1–V17) and NN checkpoints (any shape, same `game_variant`).
- **Sample granularity:** every visited afterstate becomes a training sample (~30× more than per-placement). For each turn, all `request_count` afterstates that the teacher evaluated are emitted.
- **Sample pipeline:** standalone module — no edits to `TrainingLoopT`, `worker_thread`, `extract_training_samples`, or the existing self-play pipeline. RL paths stay bit-identical.
- **No replay buffer.** Teacher is fixed → no policy drift → no reason to mix samples across time. Each sample trained exactly once. Replaced with a chunked shuffle queue (see below).
- **Convergence-driven stopping.** No pre-specified step count. Train until the student is "distilled" by an eval-win-rate or match-loss criterion (see Convergence section). `max_steps` is a hard cap, not the primary stop condition.
- **Final teacher-vs-student report.** Regardless of how the run stopped, do a high-sample-count final eval of both teacher and student against a reference heuristic and print both rates side-by-side.
- **Variant coverage:** works for both `Yams1v1` and `Yams2v2` via existing `Traits` machinery.

## Architecture: `src/distil/`

```
src/distil/
  CMakeLists.txt
  teacher.h                  Teacher<Traits> interface
  teacher_heuristic.h/.cc    CPU-side heuristic eval wrapper (V1..V17)
  teacher_nn.h/.cc           InferenceEngine wrapper + checkpoint loader
  distil_config.h            DistilConfig struct
  shuffle_queue.h            ShuffleQueueT<Traits> (header-only template)
  distil_worker.h/.cc        Variant-templated worker loop (forked from self_play/worker.cc, slimmed)
  distil_orchestrator.h/.cc  Owns games, workers, optional teacher coordinator(s)
  distil_loop.h/.cc          Top-level loop (drain shuffle queue → train_step → checkpoint)
```

## Key types

### `Teacher<Traits>` (`teacher.h`)

```cpp
template <typename Traits>
class Teacher {
public:
  virtual ~Teacher() = default;
  // Called by the worker after solver_get_requests. Fills evs[] for all n
  // afterstates. May be synchronous (heuristic) or batched through an
  // InferenceEngine (NN). n <= kMaxAfterstateRequests.
  virtual void evaluate(const BoardStateT<Traits>& board,
                        const GameContextT<Traits>& ctx,
                        const AfterstateRequest* requests, int n,
                        const float* tensors,   // [n * kTensorSize], may be null
                        double* evs) = 0;

  // True if evaluate() needs the tensors pre-built (NN teacher).
  virtual bool needs_tensor_input() const = 0;
};
```

- **HeuristicTeacher<Traits>**: dispatches into `heuristic_evaluate{,_v2,_v3,_research}` exactly like `worker.cc` does today, picked by `HeuristicVersion`. CPU-only, synchronous, no batching.
- **NNTeacher<Traits>**: wraps `InferenceEngine::batch_inference(float*, n, double*)`. Validates that the loaded checkpoint's `game_variant` and `input_size` match the student's `Traits`.

### `DistilConfig` (`distil_config.h`)

```cpp
enum class TeacherKind { kHeuristic, kNN };

struct DistilConfig {
    SelfPlayConfig self_play;             // num_workers, num_games, batch params

    TeacherKind  teacher_kind = TeacherKind::kHeuristic;
    int          teacher_heuristic_version = 2;   // V1..V17 if kHeuristic
    std::string  teacher_checkpoint_path;          // if kNN

    ModelConfig  student_model;            // student arch (may differ from teacher)

    // Shuffle queue (replaces replay buffer)
    int   shuffle_chunk_size      = 65536; // Samples per permuted chunk
    int   min_chunk_size_to_start = 16384; // Don't begin training until first chunk reaches this
    int   train_batch_size        = 1024;

    int   checkpoint_interval = 5000;
    int   max_checkpoints     = 5;
    int   eval_interval       = 1000;
    int   eval_games          = 200;

    // --- Convergence / stopping ---
    int    max_steps             = 500'000;  // Hard cap, always honored.
    int    min_steps             = 5'000;    // Don't check convergence before this.
    int    reference_heuristic_version = 2;  // Heuristic both student and teacher are evaluated against.
    double convergence_win_rate_delta  = 0.02;   // Stop when |student_wr - teacher_wr| < this.
    int    convergence_patience        = 3;      // Consecutive eval checks below the delta required.
    double convergence_match_mse       = -1.0;   // Disabled if < 0. Else: stop when probe MSE < this.
    int    final_eval_games            = 1000;   // Bigger sample for the headline end-of-run number.

    std::string  checkpoint_dir = "checkpoints_distil";
    std::string  log_dir        = ".";
    std::string  log_path       = "distil_log.csv";

    bool  debug_mode = false;
};
```

### `ShuffleQueueT<Traits>` (`shuffle_queue.h`)

Replaces the replay buffer. Each sample is consumed exactly once; shuffled within a chunk to decorrelate consecutive worker output.

```cpp
template <typename Traits>
class ShuffleQueueT {
public:
  using Sample = TrainingSampleT<Traits>;

  explicit ShuffleQueueT(int chunk_size, int min_chunk_size_to_start);

  // Worker side: push N samples. Append-only into the accumulating buffer.
  void add_batch(const Sample* s, int n);

  // Trainer side: blocking draw of one mini-batch. Returns 0 only after stop().
  //   - Waits until either (a) the serving buffer has unconsumed samples or
  //     (b) the accumulating buffer has reached chunk_size (or
  //     min_chunk_size_to_start for the very first chunk).
  //   - When the serving buffer empties, atomically rotates: serving <- accumulating,
  //     permutes serving's indices once, resets serve_pos.
  // Writes states (batch_size × kTensorSize) and targets (batch_size) directly.
  int  draw_batch(float* states, double* targets, int batch_size);

  void stop();

  // Diagnostics.
  int  accumulating_size() const;
  int  serving_remaining() const;
  long total_drawn() const;

private:
  int chunk_size_;
  int min_chunk_size_to_start_;
  std::vector<Sample> accumulating_;
  std::vector<Sample> serving_;
  std::vector<int>    perm_;
  int  serve_pos_ = 0;
  bool first_chunk_ = true;
  long total_drawn_ = 0;
  mutable std::mutex      mu_;
  std::condition_variable cv_;
  std::atomic<bool>       stop_{false};
};
```

Properties:
- Each sample appears in exactly one batch.
- Within a chunk, sample order is a uniformly random permutation.
- Bounded memory: accumulating + serving ≤ 2 × `chunk_size`.
- When production outpaces consumption, the accumulating buffer grows (no eviction). When consumption outpaces production, `draw_batch` blocks on the cv.

## Worker loop (`distil_worker.cc`)

Per game pulled from the available queue:

1. `solver_get_requests<Traits>(...)` → fills `solver_buffers.requests`, `request_count`.
2. `request_count == 0` fast path (forced placement / forced reroll, copied from `worker.cc`).
3. `generate_tensor_batch<Traits>(...)` into a per-worker scratch `float[kMaxAfterstateRequests * kTensorSize]`. Done unconditionally — we need the tensors for the shuffle queue regardless of teacher kind.
4. `teacher_->evaluate(board, ctx, requests, request_count, tensors, evs)`:
   - **Heuristic**: synchronous, game stays on the same thread.
   - **NN**: route through a `BatchManagerT<Traits>` → teacher coordinator → back. Same async pattern as today's primary inference. Game cycles `kNeedRequests → kWaitingInference → kNeedResolve`.
5. **Emit samples** (always — both teacher kinds): build a stack `TrainingSampleT<Traits> staging[request_count]`, fill `state` from `tensors[i*kT..]` and `target` from `evs[i]`, then one `shuffle_queue.add_batch(staging, request_count)`.
6. `solver_resolve<Traits>(...)` with greedy `SolverConfig` (no temperature, no exploration, no heuristic blending — teacher's evs are the policy of record).
7. `perform_placement` / `perform_reroll`, advance phase, push back to `available` or `completed`.

No trajectory recording, no MC/TD logic, no `extract_training_samples`. Game results are tracked only for the `games_played` counter and game recycling.

## Orchestrator (`distil_orchestrator.cc`)

Mirrors `SelfPlayOrchestratorT` but simpler:

- Owns: game pool, `GameQueueT<Traits>` available + completed queues, the `Teacher<Traits>`.
- Holds a reference to the externally-owned `ShuffleQueueT<Traits>` (passed in by `DistilLoopT`).
- If `TeacherKind == kNN`: also owns one `BatchManagerT<Traits>` and `num_coordinators` coordinator threads (reusing `coordinator_thread<Traits>` unchanged, pointed at the teacher's `InferenceEngine`).
- Launches `num_workers` `distil_worker_thread<Traits>` instances.

API: `start()`, `stop()`, `collect_completed(...)`, `recycle_game(...)`, `update_solver_config(...)` (debug-only fields).

## Top-level loop (`distil_loop.cc`)

```cpp
template <typename Traits>
class DistilLoopT {
public:
  DistilLoopT(const DistilConfig&, const PrecomputedTables&,
              torch::Device train_device, torch::Device infer_device);

  /// Run until a convergence criterion fires, stop() is called, or
  /// config.max_steps is reached. Always ends with the final teacher/student
  /// eval and a printed report (see "Final report" below).
  void run();
  void stop();
  TrainingMetrics metrics() const;

  // Resume support
  ModelTrainer& trainer() { return *student_trainer_; }

private:
  void do_training_step();
  void maybe_checkpoint();
  void maybe_evaluate();                // student wr + match-mse; updates convergence state
  bool convergence_satisfied() const;   // true if any stop criterion passed
  void final_report();                  // run final eval, print, log
  int  collect_completed_games();       // recycle only, no sample extraction

  DistilConfig                       config_;
  const PrecomputedTables&           tables_;
  std::unique_ptr<ModelTrainer>      student_trainer_;
  std::unique_ptr<Teacher<Traits>>   teacher_;
  std::unique_ptr<ShuffleQueueT<Traits>>      queue_;
  std::unique_ptr<DistilOrchestratorT<Traits>> orchestrator_;

  // Convergence tracking
  double teacher_win_rate_ = 0.0;       // Cached once at startup
  double last_student_win_rate_ = 0.0;
  int    consecutive_passes_ = 0;       // Reset to 0 on any failing check
  double rolling_match_mse_  = 0.0;     // EMA over probe batches if enabled

  // ... metrics / step counters / stop_flag
};
```

Construction:

1. Build the student `ModelTrainer` from `cfg.student_model`. Assert `student_model.input_size == Traits::kTensorSize` and `student_model.game_variant == kGameVariant{1v1,2v2}`.
2. Build the teacher:
   - **Heuristic** → `HeuristicTeacher<Traits>(version)`.
   - **NN** → `ModelTrainer::config_from_checkpoint` to get teacher arch, instantiate teacher `ModelTrainer` on `infer_device`, `clone_for_inference`, wrap in `InferenceEngine`, wrap in `NNTeacher<Traits>`. Reject if teacher's `input_size` / `game_variant` don't match `Traits`.
3. Build the `ShuffleQueueT<Traits>`.
4. Build and `start()` the `DistilOrchestratorT<Traits>`.
5. **Cache the teacher's reference win-rate**: run `run_evaluation<Traits>(teacher, reference_heuristic_version, eval_games)` once. For an NN teacher this is the standard NN-vs-heuristic evaluator. For a heuristic teacher, use a heuristic-vs-heuristic helper (see Evaluation section). Cache `teacher_win_rate_` and log it before training begins.

`run()`:

```
while !stop and step < config.max_steps:
    n = queue_.draw_batch(states, targets, train_batch_size)     // blocks until queue ready
    if n == 0: break                                              // stopped
    last_loss = student_trainer_.train_step(states, targets, n)
    ++step; total_samples_trained += n
    maybe_checkpoint()
    maybe_evaluate()                                              // updates convergence state
    if convergence_satisfied():
        break
    collect_completed_games()                                     // recycle only

final_report()                                                    // always runs
```

No model swap (the student never gets swapped into self-play — there's no student inference at all).

## Evaluation (`maybe_evaluate`)

Runs every `eval_interval` training steps. Two metrics:

- **Student win-rate vs reference heuristic** (load-bearing, drives convergence): `run_evaluation<Traits>(student_model, reference_heuristic_version, eval_games)`. Updates `last_student_win_rate_`.
- **Match MSE** (optional, drives convergence if `convergence_match_mse > 0`): take a probe batch (peek the next training batch before consuming it, or maintain a small held-out probe queue). Compare student outputs to teacher outputs (we already have both — teacher output is the target field). EMA into `rolling_match_mse_`.

CSV columns logged: `step, loss, student_wr_vs_ref, teacher_wr_vs_ref, win_rate_gap, match_mse, samples_drawn, queue_serving_remaining`.

The teacher win-rate is constant for the whole run (teacher is fixed), but it's logged on every row so the CSV is self-contained and downstream plots can draw the convergence target without joining tables.

## Convergence (`convergence_satisfied`)

```cpp
bool DistilLoopT::convergence_satisfied() const {
    if (training_step_ < config_.min_steps) return false;

    bool wr_pass = std::abs(last_student_win_rate_ - teacher_win_rate_)
                   < config_.convergence_win_rate_delta;
    bool mse_pass = (config_.convergence_match_mse > 0.0) &&
                    (rolling_match_mse_ < config_.convergence_match_mse);

    return wr_pass || mse_pass;
}
```

`consecutive_passes_` is incremented inside `maybe_evaluate` on each passing check and reset to 0 on a failure. Convergence fires when `consecutive_passes_ >= convergence_patience` *and* one of the criteria currently holds. Patience exists because a 200-game eval has stdev ~3.5%, so a single passing check can be noise.

To disable convergence and run for exactly `max_steps`: set `convergence_win_rate_delta` to a negative value (sentinel) and leave `convergence_match_mse < 0`. Both criteria then trivially fail forever.

## Final report (`final_report`)

Always runs once, regardless of stop reason (convergence, max-steps, SIGINT). Steps:

1. Re-eval the **student** vs reference heuristic with `final_eval_games` (default 1000) for a tighter CI on the headline number.
2. Re-eval the **teacher** vs reference heuristic with `final_eval_games`. If the teacher is a heuristic *and* its version equals `reference_heuristic_version`, this is V_n-vs-V_n ≈ 50% — run it anyway, the measured number is honest signal and the code path stays uniform.
3. Compute `gap = student_wr - teacher_wr`.
4. Print to stdout and append a single `final` row to the CSV.

Example console output:

```
=== Distillation final report ===
Teacher (heuristic V2) vs V2:  win rate = 0.502  (±0.016 @ 1000 games)
Student (mlp 4x512)    vs V2:  win rate = 0.494  (±0.016 @ 1000 games)
Win-rate gap: -0.008 (target |gap| < 0.020)
Stop reason: convergence (3 consecutive eval checks within delta)
Steps: 42000 / max 500000
```

### Helper needed: `run_evaluation` variants

- Student NN vs heuristic — **already exists** as `run_evaluation<Traits>(model, ..., heuristic_version, ...)`.
- NN teacher vs heuristic — add an overload of `run_evaluation<Traits>` that takes a teacher `InferenceEngine` (or just a `ProYamsNet&` already moved to the inference device — same signature works).
- Heuristic teacher vs heuristic reference — add `run_heuristic_vs_heuristic<Traits>(teacher_version, ref_version, eval_games, seed) → EvalResult`. Small wrapper around `play_heuristic_game<Traits>` plus a per-seat-version override on `heuristic_play_turn<Traits>`.

## Config + CLI wiring

### `config_loader.cc`

- `parse_teacher_kind("heuristic"|"nn") → TeacherKind`.
- `load_distil_config(YAML::Node, DistilConfig&)` reads the `distil:` section. Mirrors `load_training_config` for the shared fields (`self_play`, `student`, training knobs) and adds the teacher fields.
- `AppConfig` gains an `std::optional<DistilConfig> distil;` (or a parallel sibling to `training`).

### CLI flags (new)

- `--teacher_kind {heuristic|nn}`
- `--teacher_heuristic_version <int>`
- `--teacher_checkpoint <path>`
- `--max_steps <int>`
- `--min_steps <int>`
- `--reference_heuristic_version <int>`
- `--convergence_win_rate_delta <float>`
- `--convergence_patience <int>`
- `--final_eval_games <int>`

The existing `--num_workers`, `--num_games`, `--batch_size`, `--learning_rate`, `--hidden_layers`, `--hidden_width`, `--game_variant` flags apply to the distil config when `--mode distil` is active. `--num_steps` is repurposed as `--max_steps`'s synonym for parity with `mode train` (whichever is provided wins; CLI overrides YAML).

### `config_validator.cc`

- `teacher_kind == kNN` ⇒ `teacher_checkpoint_path` non-empty and file exists.
- `teacher_kind == kHeuristic` ⇒ version ∈ [1, 17].
- Student's `game_variant` ↔ `input_size` consistency (same rule as today).
- NN: teacher's `game_variant` + `input_size` must match the student's.

### `main.cpp`

- Add `mode_distil(cfg)`:
  - Builds `PrecomputedTables`.
  - Saves the config alongside checkpoints (matches `mode_train`).
  - Dispatches on `cfg.distil->student_model.game_variant` → `run_distil_variant<Yams1v1>` or `run_distil_variant<Yams2v2>`.
- Add `"distil"` to the mode list and help string.

## Config file examples

### `config/distil/heuristic_v2_1v1.yaml`

```yaml
distil:
  teacher_kind: heuristic
  teacher_heuristic_version: 2
  game_variant: 1v1
  self_play:
    num_workers: 16
    num_games: 512
  student:
    hidden_layers: 4
    hidden_width: 512
    architecture: mlp
    output_activation: tanh
  train_batch_size: 1024
  shuffle_chunk_size: 65536
  min_chunk_size_to_start: 16384
  max_steps: 500000
  min_steps: 5000
  reference_heuristic_version: 2
  convergence_win_rate_delta: 0.02
  convergence_patience: 3
  eval_interval: 1000
  eval_games: 200
  final_eval_games: 1000
  checkpoint_dir: checkpoints/distil_v2
  log_dir: logs/distil_v2
```

### `config/distil/from_nn_2v2.yaml`

```yaml
distil:
  teacher_kind: nn
  teacher_checkpoint: checkpoints/big_teacher/checkpoint_step_500000
  game_variant: 2v2
  self_play:
    num_workers: 16
    num_games: 512
    num_coordinators: 1
  student:
    hidden_layers: 3
    hidden_width: 256
    architecture: mlp
    output_activation: tanh
  train_batch_size: 1024
  shuffle_chunk_size: 65536
  min_chunk_size_to_start: 16384
  max_steps: 1000000
  reference_heuristic_version: 2
  convergence_win_rate_delta: 0.02
  convergence_patience: 3
  final_eval_games: 1000
```

## Tests (`tests/distil/`)

- `shuffle_queue_test.cc` — push 100K samples, draw all of them, verify each appeared exactly once, order is not FIFO, and `draw_batch` blocks correctly when empty.
- `teacher_heuristic_test.cc` — `HeuristicTeacher<Traits>::evaluate` returns the same evs as a direct `heuristic_evaluate_v2` call for V2.
- `teacher_nn_test.cc` — load a known checkpoint, compare `NNTeacher::evaluate` output to direct `InferenceEngine::batch_inference`.
- `distil_worker_test.cc` — drive one turn end-to-end, assert N samples landed in the queue with `target == teacher's EV` and tensors matching `generate_tensor_batch`.
- `distil_loop_test.cc` — `RunsAFewSteps` for both `Yams1v1` and `Yams2v2` with heuristic teacher (finite loss, queue drains, no crash). Same CTest resource lock / MKL env vars as the other libtorch suites (`MKL_THREADING_LAYER=SEQUENTIAL`, `OMP_NUM_THREADS=1`).
- `convergence_test.cc` — unit-test `convergence_satisfied()`: feed synthetic win-rate sequences and verify patience / delta logic; verify negative `convergence_win_rate_delta` disables the criterion; verify `min_steps` gate.
- `final_report_test.cc` — small end-to-end with a tiny `max_steps` (e.g. 10), confirms `final_report()` always runs, the headline rates are populated, and the printed gap matches the computed one.

## Build wiring

- New `src/distil/CMakeLists.txt` static library linked against `engine`, `solver`, `heuristic`, `model`, `self_play` (for `GameInstanceT`, `GameQueueT`, `BatchManagerT`, `coordinator_thread`).
- `src/main.cpp` target adds the new `distil` lib to its link line.
- `tests/distil/CMakeLists.txt` mirrors `tests/training/`.

## Open follow-ups (out of v1)

- Win-rate eval for NN teacher (`run_evaluation` overload accepting a teacher `InferenceEngine`).
- Distill → fine-tune: warm-start `TrainingLoopT` from a distilled checkpoint. Should already work via existing `--checkpoint` flag, but explicitly validate the path.
- Optional `distill_weight` blend with TD targets inside `TrainingLoopT` if mixed objectives turn out useful later.
- Per-worker thread-local sample accumulators flushed in bigger chunks (lower mutex contention) if the single-mutex shuffle queue shows up in profiles.

## Implementation order

1. `ShuffleQueueT<Traits>` + `shuffle_queue_test.cc`.
2. `Teacher<Traits>` interface + `HeuristicTeacher` + `teacher_heuristic_test.cc`.
3. `DistilConfig` + config loader + validator + CLI wiring.
4. `distil_worker.cc` for heuristic teacher only (no BatchManager wiring) + `DistilOrchestratorT` + `DistilLoopT` (stubbed `convergence_satisfied` returning false, runs to `max_steps`).
5. `main.cpp` `mode_distil` + `config/distil/heuristic_v2_1v1.yaml` + end-to-end smoke test on `Yams1v1`.
6. Same end-to-end on `Yams2v2` (should be free given templating).
7. `run_heuristic_vs_heuristic<Traits>` helper + startup teacher win-rate caching + student eval-vs-reference each `eval_interval`.
8. `convergence_satisfied` + patience tracking + `convergence_test.cc`.
9. `final_report()` + CSV `final` row + console output + `final_report_test.cc`.
10. `NNTeacher` + teacher coordinator wiring in `DistilOrchestratorT` + NN-teacher integration test + NN-teacher overload of `run_evaluation`.
11. Optional match-loss-MSE criterion + probe batch plumbing.
12. Real distillation run end-to-end (V2 → mlp 4×512 on Yams1v1) to validate convergence behaviour.
