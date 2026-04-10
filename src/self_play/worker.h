#pragma once

#include <atomic>
#include "self_play/game_queues.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// worker_thread — CPU-side game processing loop.
//
// Pulls games from `available`, processes them based on their phase:
//   kNeedRequests → solver_get_requests + generate_tensor_batch → pending
//   kNeedResolve  → solver_resolve → placement/hold → available or completed
//
// Runs until a nullptr sentinel is received from `available`.
// ---------------------------------------------------------------------------
void worker_thread(GameQueue& available, GameQueue& pending, GameQueue& completed,
                   const PrecomputedTables& tables, const SolverConfig& config,
                   std::atomic<bool>& shutdown);
