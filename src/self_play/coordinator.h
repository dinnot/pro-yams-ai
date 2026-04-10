#pragma once

#include <atomic>
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
};

// ---------------------------------------------------------------------------
// coordinator_thread — manages GPU inference batching.
//
// Collects pending games from `pending`, assembles a contiguous tensor batch,
// runs GPU inference via `inference`, distributes EVs back to each game's
// SolverBuffers, and pushes games to `available` with kNeedResolve phase.
//
// Runs until shutdown.load() returns true.
// ---------------------------------------------------------------------------
void coordinator_thread(GameQueue& pending, GameQueue& available,
                        InferenceEngine& inference,
                        const SelfPlayConfig& config,
                        std::atomic<bool>& shutdown);
