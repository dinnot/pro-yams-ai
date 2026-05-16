#pragma once

#include <atomic>
#include "engine/game_traits.h"
#include "self_play/batch_manager.h"
#include "self_play/debug_stats.h"
#include "self_play/game_queues.h"
#include "model/inference.h"

// ---------------------------------------------------------------------------
// SelfPlayConfig — tuning parameters for the async self-play pipeline.
// ---------------------------------------------------------------------------
struct SelfPlayConfig {
    int max_inference_batch = 1024;  // Max tensors per GPU batch
    int min_games_per_batch = 2;      // Min games before sending (or timeout)
    int batch_timeout_ms    = 5;      // Max wait time for batch assembly (ms)
    int num_workers         = 16;     // Number of worker threads
    int num_games           = 512;    // Total concurrent game instances
    int num_coordinators    = 1;      // Coordinator threads (each gets a CUDA stream)
    bool debug_mode         = false;
    std::string debug_log_path = "";
};

// ---------------------------------------------------------------------------
// coordinator_thread<Traits> — manages GPU inference batching.
//
// Collects pending games from `batch_manager`, assembles a contiguous tensor batch,
// runs GPU inference via `inference`, distributes EVs back to each game's
// SolverBuffers, and pushes games to `available` with kNeedResolve phase.
//
// Runs until shutdown.load() returns true.
// ---------------------------------------------------------------------------
template <typename Traits>
void coordinator_thread(BatchManagerT<Traits>& batch_manager,
                        GameQueueT<Traits>& available,
                        InferenceEngine& inference,
                        const SelfPlayConfig& config,
                        std::atomic<bool>& shutdown,
                        int coordinator_id,
                        SelfPlayDebugStats* debug_stats = nullptr);
