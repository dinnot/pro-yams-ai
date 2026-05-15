#pragma once

#include <atomic>
#include "engine/game_traits.h"
#include "self_play/batch_manager.h"
#include "self_play/game_queues.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// worker_thread<Traits> — CPU-side game processing loop.
//
// Pulls games from `available`, processes them based on their phase:
//   kNeedRequests → solver_get_requests + generate_tensor_batch → batch_manager
//   kNeedResolve  → solver_resolve → placement/hold → available or completed
//
// `opponent_batch_manager` is optional (may be nullptr). When non-null, requests
// from a game where `use_past_opponent && current_player == past_opponent_player`
// are routed to it instead of the primary batch manager — that lets the older
// model evaluate the opponent's afterstates while the current model evaluates
// the rest.
//
// Runs until a nullptr sentinel is received from `available`.
// ---------------------------------------------------------------------------
template <typename Traits>
void worker_thread(GameQueueT<Traits>& available,
                   BatchManagerT<Traits>& batch_manager,
                   BatchManagerT<Traits>* opponent_batch_manager,
                   GameQueueT<Traits>& completed,
                   const PrecomputedTables& tables, const SolverConfig& config,
                   std::atomic<bool>& shutdown);
