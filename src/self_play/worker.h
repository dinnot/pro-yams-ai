#pragma once

#include <atomic>
#include "self_play/game_queues.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

#include "self_play/batch_manager.h"

// ---------------------------------------------------------------------------
// worker_thread — CPU-side game processing loop.
//
// Pulls games from `available`, processes them based on their phase:
//   kNeedRequests → solver_get_requests + generate_tensor_batch → batch_manager
//   kNeedResolve  → solver_resolve → placement/hold → available or completed
//
// Runs until a nullptr sentinel is received from `available`.
// ---------------------------------------------------------------------------
void worker_thread(GameQueue& available, BatchManager& batch_manager, GameQueue& completed,
                   const PrecomputedTables& tables, const SolverConfig& config,
                   std::atomic<bool>& shutdown);
