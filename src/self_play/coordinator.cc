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

#include <ATen/Parallel.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>

#include "self_play/game_instance.h"
#include "engine/tensor.h"

template <typename Traits>
void coordinator_thread(BatchManagerT<Traits>& batch_manager,
                        GameQueueT<Traits>& available,
                        InferenceEngine& inference,
                        const SelfPlayConfig& config,
                        std::atomic<bool>& shutdown,
                        int coordinator_id) {
    using GI = GameInstanceT<Traits>;
    using IB = InferenceBatchT<Traits>;

    std::unique_ptr<c10::cuda::CUDAStreamGuard> stream_guard;
    if (inference.device().is_cuda()) {
        stream_guard = std::make_unique<c10::cuda::CUDAStreamGuard>(
            c10::cuda::getStreamFromPool(false, inference.device().index()));
    }

    {
        at::init_num_threads();
        float dummy[Traits::kTensorSize] = {};
        double dummy_out = 0.0;
        inference.batch_inference(dummy, 1, &dummy_out);
    }

    std::vector<double> batch_results(config.max_inference_batch);

    using Clock = std::chrono::steady_clock;
    int64_t loop_count = 0;
    double accum_assembly_us = 0.0;
    double accum_inference_us = 0.0;
    double accum_distribute_us = 0.0;
    int64_t accum_tensors = 0;
    int64_t accum_games = 0;

    while (!shutdown.load(std::memory_order_relaxed)) {
        auto t_assembly_start = config.debug_mode ? Clock::now() : Clock::time_point();

        IB* batch = batch_manager.pop_ready_batch();
        if (!batch) break;

        int total_tensors = batch->committed_count.load(std::memory_order_relaxed);
        int num_entries = batch->entries.size();

        if (total_tensors == 0) {
            batch_manager.recycle_batch(batch);
            continue;
        }

        auto t_inference_start = config.debug_mode ? Clock::now() : Clock::time_point();

        auto input_slice = batch->storage.narrow(0, 0, total_tensors);
        inference.batch_inference(input_slice, total_tensors,
                                  batch_results.data());

        auto t_distribute_start = config.debug_mode ? Clock::now() : Clock::time_point();

        std::vector<GI*> resolved_games;
        resolved_games.reserve(num_entries);
        for (int i = 0; i < num_entries; ++i) {
            const typename IB::Entry& e = batch->entries[i];
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

        batch_manager.recycle_batch(batch);

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

// ---------------------------------------------------------------------------
// Explicit instantiations.
// ---------------------------------------------------------------------------
template void coordinator_thread<Yams1v1>(BatchManagerT<Yams1v1>&, GameQueueT<Yams1v1>&,
                                          InferenceEngine&, const SelfPlayConfig&,
                                          std::atomic<bool>&, int);
template void coordinator_thread<Yams2v2>(BatchManagerT<Yams2v2>&, GameQueueT<Yams2v2>&,
                                          InferenceEngine&, const SelfPlayConfig&,
                                          std::atomic<bool>&, int);
