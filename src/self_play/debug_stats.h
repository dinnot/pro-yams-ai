#pragma once

#include <atomic>
#include <cstdint>

struct SelfPlayDebugStatsSnapshot {
    int64_t worker_need_requests = 0;
    int64_t worker_need_resolve = 0;
    int64_t worker_completed_games = 0;
    int64_t worker_rerolls = 0;
    int64_t worker_placements = 0;
    int64_t worker_requests = 0;
    int64_t worker_tensors = 0;

    int64_t solver_get_requests_us = 0;
    int64_t tensor_batch_us = 0;
    int64_t batch_reserve_us = 0;
    int64_t batch_commit_us = 0;
    int64_t pure_resolve_us = 0;
    int64_t heuristic_eval_us = 0;
    int64_t blend_us = 0;
    int64_t solver_resolve_us = 0;
    int64_t chosen_tensor_us = 0;
    int64_t perform_action_us = 0;

    int64_t heuristic_eval_calls = 0;
    int64_t heuristic_requests = 0;
    int64_t heuristic_v_counts[18] = {};

    int64_t coordinator_batches = 0;
    int64_t coordinator_games = 0;
    int64_t coordinator_tensors = 0;
    int64_t coordinator_pop_wait_us = 0;
    int64_t coordinator_inference_us = 0;
    int64_t coordinator_distribute_us = 0;
};

struct SelfPlayDebugStats {
    std::atomic<int64_t> worker_need_requests{0};
    std::atomic<int64_t> worker_need_resolve{0};
    std::atomic<int64_t> worker_completed_games{0};
    std::atomic<int64_t> worker_rerolls{0};
    std::atomic<int64_t> worker_placements{0};
    std::atomic<int64_t> worker_requests{0};
    std::atomic<int64_t> worker_tensors{0};

    std::atomic<int64_t> solver_get_requests_us{0};
    std::atomic<int64_t> tensor_batch_us{0};
    std::atomic<int64_t> batch_reserve_us{0};
    std::atomic<int64_t> batch_commit_us{0};
    std::atomic<int64_t> pure_resolve_us{0};
    std::atomic<int64_t> heuristic_eval_us{0};
    std::atomic<int64_t> blend_us{0};
    std::atomic<int64_t> solver_resolve_us{0};
    std::atomic<int64_t> chosen_tensor_us{0};
    std::atomic<int64_t> perform_action_us{0};

    std::atomic<int64_t> heuristic_eval_calls{0};
    std::atomic<int64_t> heuristic_requests{0};
    std::atomic<int64_t> heuristic_v_counts[18];

    std::atomic<int64_t> coordinator_batches{0};
    std::atomic<int64_t> coordinator_games{0};
    std::atomic<int64_t> coordinator_tensors{0};
    std::atomic<int64_t> coordinator_pop_wait_us{0};
    std::atomic<int64_t> coordinator_inference_us{0};
    std::atomic<int64_t> coordinator_distribute_us{0};

    SelfPlayDebugStats() {
        for (auto& c : heuristic_v_counts) c.store(0, std::memory_order_relaxed);
    }

    SelfPlayDebugStatsSnapshot exchange_snapshot() {
        SelfPlayDebugStatsSnapshot s;
        s.worker_need_requests = worker_need_requests.exchange(0, std::memory_order_relaxed);
        s.worker_need_resolve = worker_need_resolve.exchange(0, std::memory_order_relaxed);
        s.worker_completed_games = worker_completed_games.exchange(0, std::memory_order_relaxed);
        s.worker_rerolls = worker_rerolls.exchange(0, std::memory_order_relaxed);
        s.worker_placements = worker_placements.exchange(0, std::memory_order_relaxed);
        s.worker_requests = worker_requests.exchange(0, std::memory_order_relaxed);
        s.worker_tensors = worker_tensors.exchange(0, std::memory_order_relaxed);

        s.solver_get_requests_us = solver_get_requests_us.exchange(0, std::memory_order_relaxed);
        s.tensor_batch_us = tensor_batch_us.exchange(0, std::memory_order_relaxed);
        s.batch_reserve_us = batch_reserve_us.exchange(0, std::memory_order_relaxed);
        s.batch_commit_us = batch_commit_us.exchange(0, std::memory_order_relaxed);
        s.pure_resolve_us = pure_resolve_us.exchange(0, std::memory_order_relaxed);
        s.heuristic_eval_us = heuristic_eval_us.exchange(0, std::memory_order_relaxed);
        s.blend_us = blend_us.exchange(0, std::memory_order_relaxed);
        s.solver_resolve_us = solver_resolve_us.exchange(0, std::memory_order_relaxed);
        s.chosen_tensor_us = chosen_tensor_us.exchange(0, std::memory_order_relaxed);
        s.perform_action_us = perform_action_us.exchange(0, std::memory_order_relaxed);

        s.heuristic_eval_calls = heuristic_eval_calls.exchange(0, std::memory_order_relaxed);
        s.heuristic_requests = heuristic_requests.exchange(0, std::memory_order_relaxed);
        for (int i = 0; i < 18; ++i) {
            s.heuristic_v_counts[i] = heuristic_v_counts[i].exchange(0, std::memory_order_relaxed);
        }

        s.coordinator_batches = coordinator_batches.exchange(0, std::memory_order_relaxed);
        s.coordinator_games = coordinator_games.exchange(0, std::memory_order_relaxed);
        s.coordinator_tensors = coordinator_tensors.exchange(0, std::memory_order_relaxed);
        s.coordinator_pop_wait_us = coordinator_pop_wait_us.exchange(0, std::memory_order_relaxed);
        s.coordinator_inference_us = coordinator_inference_us.exchange(0, std::memory_order_relaxed);
        s.coordinator_distribute_us = coordinator_distribute_us.exchange(0, std::memory_order_relaxed);
        return s;
    }
};
