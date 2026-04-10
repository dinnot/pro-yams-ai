#include "self_play/coordinator.h"

#include <cassert>
#include <cstring>
#include <vector>

#include <ATen/Parallel.h>  // at::init_num_threads()

#include "self_play/game_instance.h"
#include "engine/tensor.h"

void coordinator_thread(GameQueue& pending, GameQueue& available,
                        InferenceEngine& inference,
                        const SelfPlayConfig& config,
                        std::atomic<bool>& shutdown) {
    // Warm up the inference engine in THIS thread to amortize any per-thread
    // initialization cost (e.g., TBB task scheduler, oneDNN library init).
    {
        at::init_num_threads();
        float dummy[kTensorSize] = {};
        double dummy_out = 0.0;
        inference.batch_inference(dummy, 1, &dummy_out);
    }

    // Pre-allocate buffers.
    std::vector<float>  batch_tensors(config.max_inference_batch * kTensorSize);
    std::vector<double> batch_results(config.max_inference_batch);
    std::vector<GameInstance*> temp_buffer(config.min_games_per_batch);

    struct BatchEntry {
        GameInstance* game;
        int           start_idx;  // first tensor index in batch_tensors
        int           count;      // number of tensors for this game
    };
    // Upper bound: each game contributes at least 1 tensor.
    std::vector<BatchEntry> batch_entries(config.max_inference_batch);

    while (!shutdown.load(std::memory_order_relaxed)) {
        int num_entries    = 0;
        int total_tensors  = 0;

        // ----------------------------------------------------------------
        // Phase 1: assemble a batch of up to min_games_per_batch games.
        //
        // Use pop_with_timeout so that when only a few (or one) game
        // remains active and the others have completed, we don't wait
        // indefinitely for a full min_games_per_batch.  The rule:
        //   • If pop_with_timeout returns nullptr (timeout) AND the
        //     batch already has at least one game, break immediately
        //     and proceed to inference on the partial batch.
        //   • If the batch is still empty on timeout, loop back to
        //     the outer while so we re-check shutdown.
        // ----------------------------------------------------------------
        for (int slot = 0; slot < config.min_games_per_batch; ++slot) {
            GameInstance* game = pending.pop_with_timeout(config.batch_timeout_ms);
            if (game == nullptr) {
                // Timeout: process partial batch if we have one, else re-check.
                break;
            }
            if (game == nullptr) continue;  // sentinel guard (redundant, safe)
            int req_count = game->solver_buffers.request_count;
            if (req_count <= 0) {
                game->phase = GamePhase::kNeedRequests;
                available.push(game);
                continue;
            }
            // Always admit the very first game to guarantee progress.
            if (num_entries > 0 &&
                total_tensors + req_count > config.max_inference_batch) {
                pending.push_front(game);
                break;  // batch is full enough; don't fetch more
            }
            std::memcpy(batch_tensors.data() + total_tensors * kTensorSize,
                        game->tensor_buffer,
                        static_cast<size_t>(req_count) * kTensorSize * sizeof(float));
            batch_entries[num_entries++] = {game, total_tensors, req_count};
            total_tensors += req_count;
        }

        // ----------------------------------------------------------------
        // Phase 2: greedily fill remaining batch space (non-blocking).
        // ----------------------------------------------------------------
        while (total_tensors < config.max_inference_batch) {
            GameInstance* game = pending.try_pop();
            if (game == nullptr) break;
            int req_count = game->solver_buffers.request_count;
            if (req_count <= 0) {
                game->phase = GamePhase::kNeedRequests;
                available.push(game);
                continue;
            }
            if (total_tensors + req_count > config.max_inference_batch) {
                // Requeue at the front and stop trying — subsequent games won't fit either.
                pending.push_front(game);
                break;
            }
            std::memcpy(batch_tensors.data() + total_tensors * kTensorSize,
                        game->tensor_buffer,
                        static_cast<size_t>(req_count) * kTensorSize * sizeof(float));
            batch_entries[num_entries++] = {game, total_tensors, req_count};
            total_tensors += req_count;
        }

        if (total_tensors == 0) continue;

        // ----------------------------------------------------------------
        // Phase 3: GPU inference.
        // ----------------------------------------------------------------
        inference.batch_inference(batch_tensors.data(), total_tensors,
                                  batch_results.data());

        // ----------------------------------------------------------------
        // Phase 4: distribute EVs back to each game and release to workers.
        // ----------------------------------------------------------------
        std::vector<GameInstance*> resolved_games;
        resolved_games.reserve(num_entries);
        for (int i = 0; i < num_entries; ++i) {
            BatchEntry& e = batch_entries[i];
            std::memcpy(e.game->solver_buffers.evs,
                        batch_results.data() + e.start_idx,
                        static_cast<size_t>(e.count) * sizeof(double));
            e.game->phase = GamePhase::kNeedResolve;
            resolved_games.push_back(e.game);
        }
        if (!resolved_games.empty()) {
            available.push_batch(resolved_games.data(),
                                 static_cast<int>(resolved_games.size()));
        }
    }
}
