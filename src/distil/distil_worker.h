#pragma once

#include <atomic>
#include <cstdint>

#include "distil/shuffle_queue.h"
#include "distil/teacher.h"
#include "engine/game_traits.h"
#include "self_play/game_instance.h"
#include "self_play/game_queues.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// Per-worker counters, updated atomically. Aggregated by DistilOrchestratorT
// for run-time metrics.
// ---------------------------------------------------------------------------
struct DistilWorkerStats {
    std::atomic<long> turns_processed{0};
    std::atomic<long> games_completed{0};
    std::atomic<long> samples_emitted{0};
};

// ---------------------------------------------------------------------------
// distil_worker_thread<Traits> — CPU-only distillation worker.
//
// Pops a GameInstanceT<Traits> from `available`, runs one solver iteration
// (one placement or one reroll) driven entirely by the teacher's evaluation,
// emits one (state, teacher_ev) sample per visited afterstate to the shared
// ShuffleQueueT, then either pushes the game back to `available` (if the
// turn continued) or recycles it in-place with a fresh seed and a fresh
// init_game (if the game became terminal).
//
// No BatchManager and no async inference — the heuristic teacher runs
// synchronously inline. An NN-teacher path will land in a later step and
// likely take a different worker variant (with BatchManager wiring).
//
// Exits when a nullptr sentinel is received from `available`.
// ---------------------------------------------------------------------------
template <typename Traits>
void distil_worker_thread(GameQueueT<Traits>& available,
                          Teacher<Traits>& teacher,
                          ShuffleQueueT<Traits>& shuffle_queue,
                          const PrecomputedTables& tables,
                          const SolverConfig& solver_config,
                          uint64_t worker_seed,
                          DistilWorkerStats* stats = nullptr);
