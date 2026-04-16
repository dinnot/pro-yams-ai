#include "self_play/coordinator.h"

#include <cassert>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <ATen/Parallel.h>  // at::init_num_threads()
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "self_play/game_instance.h"
#include "engine/tensor.h"

void coordinator_thread(GameQueue& pending, GameQueue& available,
                        InferenceEngine& inference,
                        const SelfPlayConfig& config,
                        std::atomic<bool>& shutdown,
                        int coordinator_id) {
    // Each coordinator gets a dedicated CUDA stream so multiple coordinators
    // can overlap GPU inference with each other's batch assembly.
    std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;
    if (inference.device().is_cuda()) {
        stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
            c10::cuda::getStreamFromPool(false, inference.device().index()));
    }

    // Warm up the inference engine in THIS thread to amortize any per-thread
    // initialization cost (e.g., TBB task scheduler, oneDNN library init).
    {
        at::init_num_threads();
        float dummy[kTensorSize] = {};
        double dummy_out = 0.0;
        inference.batch_inference(dummy, 1, &dummy_out);
    }

    // Pre-allocate batch tensor as pinned memory when using CUDA.
    // Pinned (page-locked) memory enables async DMA transfers to GPU,
    // freeing the CPU thread during the H2D copy.
    auto tensor_opts = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU);
    if (inference.device().is_cuda()) {
        tensor_opts = tensor_opts.pinned_memory(true);
    }
    torch::Tensor batch_tensor_storage = torch::empty(
        {config.max_inference_batch, kTensorSize}, tensor_opts);
    float* batch_tensors_ptr = batch_tensor_storage.data_ptr<float>();

    std::vector<double> batch_results(config.max_inference_batch);
    std::vector<GameInstance*> temp_buffer(config.max_inference_batch);

    struct BatchEntry {
        GameInstance* game;
        int           start_idx;  // first tensor index in batch_tensors
        int           count;      // number of tensors for this game
    };
    // Upper bound: each game contributes at least 1 tensor.
    std::vector<BatchEntry> batch_entries(config.max_inference_batch);

    // --- Timing instrumentation ---
    using Clock = std::chrono::steady_clock;
    int64_t loop_count = 0;
    double accum_assembly_us = 0.0;
    double accum_inference_us = 0.0;
    double accum_distribute_us = 0.0;
    int64_t accum_tensors = 0;
    int64_t accum_games = 0;

    while (!shutdown.load(std::memory_order_relaxed)) {
        int num_entries    = 0;
        int total_tensors  = 0;

        auto t_assembly_start = config.debug_mode ? Clock::now() : Clock::time_point();

        // ----------------------------------------------------------------
        // Phase 1+2: Bulk-collect games with minimal lock acquisitions.
        //
        // First collect() blocks up to batch_timeout_ms for at least one
        // game. Second collect() greedily grabs whatever is instantly
        // available. This replaces per-game pop loops that hammered the
        // queue mutex.
        // ----------------------------------------------------------------
        int collected = 0;
        collected = pending.collect(temp_buffer.data(),
                                    config.max_inference_batch,
                                    config.batch_timeout_ms);
        if (collected == 0) continue;  // timeout with nothing — re-check shutdown

        // Greedily grab any more that arrived while we processed.
        if (collected < config.max_inference_batch) {
            collected += pending.collect(temp_buffer.data() + collected,
                                         config.max_inference_batch - collected,
                                         0);
        }

        for (int i = 0; i < collected; ++i) {
            GameInstance* game = temp_buffer[i];
            int req_count = game->solver_buffers.request_count;
            if (req_count <= 0) {
                game->phase = GamePhase::kNeedRequests;
                available.push(game);
                continue;
            }

            // Always admit the very first game to guarantee progress.
            if (num_entries > 0 &&
                total_tensors + req_count > config.max_inference_batch) {
                // Batch is full. Return this game and all subsequent
                // unprocessed games to the front of the queue.
                pending.push_front_batch(temp_buffer.data() + i, collected - i);
                break;
            }

            std::memcpy(batch_tensors_ptr + total_tensors * kTensorSize,
                        game->tensor_buffer,
                        static_cast<size_t>(req_count) * kTensorSize * sizeof(float));
            batch_entries[num_entries++] = {game, total_tensors, req_count};
            total_tensors += req_count;
        }

        if (total_tensors == 0) continue;

        auto t_inference_start = config.debug_mode ? Clock::now() : Clock::time_point();

        // ----------------------------------------------------------------
        // Phase 3: GPU inference (uses async DMA when buffer is pinned).
        // ----------------------------------------------------------------
        auto input_slice = batch_tensor_storage.narrow(0, 0, total_tensors);
        inference.batch_inference(input_slice, total_tensors,
                                  batch_results.data());

        auto t_distribute_start = config.debug_mode ? Clock::now() : Clock::time_point();

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

        // --- Accumulate timing and log ---
        if (config.debug_mode) {
            auto t_end = Clock::now();
            accum_assembly_us += std::chrono::duration<double, std::micro>(
                t_inference_start - t_assembly_start).count();
            accum_inference_us += std::chrono::duration<double, std::micro>(
                t_distribute_start - t_inference_start).count();
            accum_distribute_us += std::chrono::duration<double, std::micro>(
                t_end - t_distribute_start).count();
            accum_tensors += total_tensors;
            accum_games += num_entries;
            ++loop_count;

            if (loop_count % 1000 == 0) {
                double total_us = accum_assembly_us + accum_inference_us + accum_distribute_us;
                
                std::stringstream ss;
                ss << "[Coordinator " << coordinator_id << " @ iter " << loop_count << "] "
                   << "Assembly: " << static_cast<int>(accum_assembly_us) << " µs ("
                   << static_cast<int>(100.0 * accum_assembly_us / total_us) << "%) | "
                   << "Inference: " << static_cast<int>(accum_inference_us) << " µs ("
                   << static_cast<int>(100.0 * accum_inference_us / total_us) << "%) | "
                   << "Distribute: " << static_cast<int>(accum_distribute_us) << " µs ("
                   << static_cast<int>(100.0 * accum_distribute_us / total_us) << "%) | "
                   << "Tensors: " << accum_tensors << " Games: " << accum_games
                   << " AvgBatch: " << (accum_tensors / 1000) << "\n";

                if (!config.debug_log_path.empty()) {
                    std::ofstream f(config.debug_log_path, std::ios::app);
                    if (f.is_open()) {
                        f << ss.str();
                    }
                } else {
                    std::cerr << ss.str();
                }

                accum_assembly_us = 0.0;
                accum_inference_us = 0.0;
                accum_distribute_us = 0.0;
                accum_tensors = 0;
                accum_games = 0;
            }
        }
    }
}
