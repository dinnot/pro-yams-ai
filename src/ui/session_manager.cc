#include "ui/session_manager.h"

#include <algorithm>
#include <cassert>
#include <cstring>

#include <torch/torch.h>

#include "engine/game_flow.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "heuristic/heuristic_bot.h"
#include "solver/mc_bot.h"
#include "solver/solver.h"

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

SessionManager::SessionManager(const PrecomputedTables& tables,
                               ProYamsNet* nn_model,
                               torch::Device device)
    : tables_(tables), nn_model_(nn_model), device_(device) {}

// ---------------------------------------------------------------------------
// eval_and_store_board_nn
// ---------------------------------------------------------------------------

void SessionManager::eval_and_store_board_nn(GameSession& session,
                                              GameSession::TurnRecord& record) {
    if (!nn_model_) return;
    float buf[kTensorSize];
    generate_tensor(session.state.board, session.ctx, record.player, tables_, buf);
    torch::NoGradGuard no_grad;
    auto input  = torch::from_blob(buf, {1, kTensorSize}, torch::kFloat32).to(device_);
    auto output = nn_model_->forward(input).to(torch::kCPU).contiguous();
    float val = output.data_ptr<float>()[0];
    // Normalize to [0, 1] win-probability space regardless of output activation.
    // tanh models output [-1, 1]: map via (v + 1) / 2.
    // sigmoid models output [0, 1]: already correct.
    if (nn_model_->config().output_activation != "sigmoid") {
        val = (val + 1.0f) / 2.0f;
    }
    record.board_nn_value     = val;
    record.has_board_nn_value = true;
    session.current_board_nn_value = val;
    session.has_current_board_nn   = true;
}

// ---------------------------------------------------------------------------
// get_entry — look up a session entry by ID.  Briefly acquires map_mutex_.
// Returns a shared_ptr so the entry stays alive even if remove_session runs
// concurrently.
// ---------------------------------------------------------------------------

std::shared_ptr<SessionEntry> SessionManager::get_entry(int id) const {
    std::lock_guard<std::mutex> lock(map_mutex_);
    auto it = sessions_.find(id);
    return it != sessions_.end() ? it->second : nullptr;
}

// ---------------------------------------------------------------------------
// create_session
// ---------------------------------------------------------------------------

int SessionManager::create_session(PlayerType p0, PlayerType p1, uint64_t seed, bool debug_mode) {
    std::lock_guard<std::mutex> lock(map_mutex_);
    int id = next_session_id_++;
    auto entry = std::make_shared<SessionEntry>();
    entry->session = std::make_unique<GameSession>();

    GameSession& s = *entry->session;
    s.session_id      = id;
    s.player_types[0] = p0;
    s.player_types[1] = p1;
    s.rng             = RNG(seed);
    s.tensor_buffer.resize(
        static_cast<size_t>(kMaxAfterstateRequests) * kTensorSize, 0.0f);

    init_game(s.state, s.ctx, s.rng);
    s.game_over         = false;
    s.result            = 0.0;
    s.waiting_for_human = false;
    s.debug_mode        = debug_mode;

    // If player 0 is human, start waiting immediately.
    if (p0 == PlayerType::kHuman) {
        s.waiting_for_human = true;
        s.current_turn.player = 0;
        std::memcpy(s.current_turn.initial_dice, s.state.dice,
                    sizeof(s.state.dice));
    }

    sessions_[id] = std::move(entry);
    return id;
}

// ---------------------------------------------------------------------------
// get_session_copy
// ---------------------------------------------------------------------------

bool SessionManager::get_session_copy(int id, GameSession& out) const {
    auto entry = get_entry(id);
    if (!entry) return false;
    std::lock_guard<std::mutex> lock(entry->mutex);
    if (!entry->session) return false;
    out = *entry->session;
    return true;
}

// ---------------------------------------------------------------------------
// Internal helpers: play one complete turn for a bot player.
// Records all holds and the final placement in session.history.
// Caller must hold the session's mutex.
// ---------------------------------------------------------------------------

void SessionManager::play_heuristic_turn(GameSession& session, HeuristicVersion version) {
    GameSession::TurnRecord record;
    record.player = static_cast<int>(session.state.board.current_player);
    std::memcpy(record.initial_dice, session.state.dice, sizeof(session.state.dice));

    // --- CLEAR CACHES AND GET REQUESTS ONCE PER TURN ---
    session.buffers.dp_computed = false;
    session.buffers.evs_blended = false;
    solver_get_requests(session.state, session.ctx, tables_, session.buffers);

    // Handle Turbo-only edge case
    if (session.buffers.request_count == 0) {
        while (can_reroll(session.state, session.ctx)) {
            perform_reroll(session.state, 0, session.rng);
            std::array<int8_t, kNumDice> after;
            std::memcpy(after.data(), session.state.dice, sizeof(session.state.dice));
            record.hold_masks.push_back(0);
            record.dice_after_hold.push_back(after);
            record.is_forced_reroll = true;
        }
        auto& all = session.ctx.legal_all[session.state.board.current_player];
        if (all.count > 0) {
            record.placement = all.placements[0];
            record.score = 0;
            perform_placement(session.state, session.ctx,
                              record.placement.column, record.placement.row, session.rng);
        }
        if (session.debug_mode) eval_and_store_board_nn(session, record);
        session.history.push_back(std::move(record));
        return;
    }

    // Evaluate ONCE
    if (static_cast<int>(version) >= static_cast<int>(HeuristicVersion::V4)) {
        const ResearchConfig& cfg = get_research_config_for(version);
        heuristic_evaluate_research(session.state.board, session.ctx,
                                    session.buffers.requests,
                                    session.buffers.request_count,
                                    session.buffers.evs, tables_, cfg);
    } else if (version == HeuristicVersion::V3) {
        heuristic_evaluate_v3(session.state.board, session.ctx,
                              session.buffers.requests,
                              session.buffers.request_count,
                              session.buffers.evs, tables_);
    } else if (version == HeuristicVersion::V2) {
        heuristic_evaluate_v2(session.state.board, session.ctx,
                              session.buffers.requests,
                              session.buffers.request_count,
                              session.buffers.evs, tables_);
    } else {
        heuristic_evaluate(session.state.board, session.ctx,
                           session.buffers.requests,
                           session.buffers.request_count,
                           session.buffers.evs);
    }

    // Loop only to resolve and reroll dice
    while (true) {
        SolverResult result = solver_resolve_greedy(session.state, session.ctx, tables_, session.buffers);

        if (result.should_place) {
            if (session.debug_mode) {
                // Sort dice once — compute_raw_score requires sorted input.
                int8_t sorted_dice[kNumDice];
                std::memcpy(sorted_dice, session.state.dice, sizeof(sorted_dice));
                std::sort(sorted_dice, sorted_dice + kNumDice);

                std::vector<GameSession::PlacementCandidateEval> pevs;
                for (int i = 0; i < session.buffers.request_count; ++i) {
                    int8_t req_score = session.buffers.requests[i].score;
                    int    row       = session.buffers.requests[i].placement.row;
                    int8_t raw       = static_cast<int8_t>(compute_raw_score(sorted_dice, row));
                    // Only include placements valid for the current dice: the declared
                    // score must match what the dice actually produce, or be a scratch (0).
                    if (req_score == raw || req_score == 0) {
                        pevs.push_back({session.buffers.requests[i].placement,
                                        req_score,
                                        static_cast<float>(session.buffers.evs[i])});
                    }
                }
                std::sort(pevs.begin(), pevs.end(),
                          [](const auto& a, const auto& b) { return a.eval_value > b.eval_value; });
                record.placement_evals = std::move(pevs);
            }
            record.placement = result.placement;
            record.score     = result.score;
            perform_placement(session.state, session.ctx,
                              result.placement.column, result.placement.row, session.rng);
            break;
        }

        if (session.debug_mode) {
            std::vector<GameSession::HoldCandidateEval> hevs;
            hevs.reserve(kNumHoldMasks);
            for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                hevs.push_back({static_cast<uint8_t>(mask),
                                static_cast<float>(session.buffers.mask_evs[mask])});
            }
            std::sort(hevs.begin(), hevs.end(),
                      [](const auto& a, const auto& b) { return a.expected_value > b.expected_value; });
            record.hold_evals.push_back(std::move(hevs));
        }

        assert(can_reroll(session.state, session.ctx));
        record.hold_masks.push_back(result.hold_mask);
        perform_reroll(session.state, result.hold_mask, session.rng);
        std::array<int8_t, kNumDice> after;
        std::memcpy(after.data(), session.state.dice, sizeof(session.state.dice));
        record.dice_after_hold.push_back(after);
    }

    if (session.debug_mode) eval_and_store_board_nn(session, record);
    session.history.push_back(std::move(record));
}

void SessionManager::play_nn_turn(GameSession& session) {
    assert(nn_model_ != nullptr);

    GameSession::TurnRecord record;
    record.player = static_cast<int>(session.state.board.current_player);
    std::memcpy(record.initial_dice, session.state.dice, sizeof(session.state.dice));

    SolverConfig greedy;

    // --- CLEAR CACHES AND GET REQUESTS ONCE PER TURN ---
    session.buffers.dp_computed = false;
    session.buffers.evs_blended = false;
    solver_get_requests(session.state, session.ctx, tables_, session.buffers);

    if (session.buffers.request_count == 0) {
        while (can_reroll(session.state, session.ctx)) {
            perform_reroll(session.state, 0, session.rng);
            std::array<int8_t, kNumDice> after;
            std::memcpy(after.data(), session.state.dice, sizeof(session.state.dice));
            record.hold_masks.push_back(0);
            record.dice_after_hold.push_back(after);
            record.is_forced_reroll = true;
        }
        auto& all = session.ctx.legal_all[session.state.board.current_player];
        if (all.count > 0) {
            record.placement = all.placements[0];
            record.score = 0;
            perform_placement(session.state, session.ctx,
                              record.placement.column, record.placement.row, session.rng);
        }
        if (session.debug_mode) eval_and_store_board_nn(session, record);
        session.history.push_back(std::move(record));
        return;
    }

    // Generate tensors ONCE.
    generate_tensor_batch(session.state.board, session.ctx,
                          static_cast<int>(session.state.board.current_player),
                          session.buffers.requests,
                          session.buffers.request_count,
                          tables_,
                          session.tensor_buffer.data());

    // Synchronous NN inference ONCE
    {
        torch::NoGradGuard no_grad;
        auto input = torch::from_blob(
            session.tensor_buffer.data(),
            {session.buffers.request_count, kTensorSize},
            torch::kFloat32).to(device_);
        auto output = nn_model_->forward(input).to(torch::kCPU).contiguous();
        const float* ptr = output.data_ptr<float>();
        for (int i = 0; i < session.buffers.request_count; ++i)
            session.buffers.evs[i] = static_cast<double>(ptr[i]);
    }

    // Loop only for rerolls
    while (true) {
        SolverResult result = solver_resolve(session.state, session.ctx,
                                              tables_, session.buffers,
                                              greedy, session.rng);

        if (result.should_place) {
            if (session.debug_mode) {
                // Sort dice once — compute_raw_score requires sorted input.
                int8_t sorted_dice[kNumDice];
                std::memcpy(sorted_dice, session.state.dice, sizeof(sorted_dice));
                std::sort(sorted_dice, sorted_dice + kNumDice);

                std::vector<GameSession::PlacementCandidateEval> pevs;
                for (int i = 0; i < session.buffers.request_count; ++i) {
                    int8_t req_score = session.buffers.requests[i].score;
                    int    row       = session.buffers.requests[i].placement.row;
                    int8_t raw       = static_cast<int8_t>(compute_raw_score(sorted_dice, row));
                    // Only include placements valid for the current dice: the declared
                    // score must match what the dice actually produce, or be a scratch (0).
                    if (req_score == raw || req_score == 0) {
                        pevs.push_back({session.buffers.requests[i].placement,
                                        req_score,
                                        static_cast<float>(session.buffers.evs[i])});
                    }
                }
                std::sort(pevs.begin(), pevs.end(),
                          [](const auto& a, const auto& b) { return a.eval_value > b.eval_value; });
                record.placement_evals = std::move(pevs);
            }
            record.placement = result.placement;
            record.score     = result.score;
            perform_placement(session.state, session.ctx,
                              result.placement.column, result.placement.row, session.rng);
            break;
        }

        if (session.debug_mode) {
            std::vector<GameSession::HoldCandidateEval> hevs;
            hevs.reserve(kNumHoldMasks);
            for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                hevs.push_back({static_cast<uint8_t>(mask),
                                static_cast<float>(session.buffers.mask_evs[mask])});
            }
            std::sort(hevs.begin(), hevs.end(),
                      [](const auto& a, const auto& b) { return a.expected_value > b.expected_value; });
            record.hold_evals.push_back(std::move(hevs));
        }

        assert(can_reroll(session.state, session.ctx));
        record.hold_masks.push_back(result.hold_mask);
        perform_reroll(session.state, result.hold_mask, session.rng);
        std::array<int8_t, kNumDice> after;
        std::memcpy(after.data(), session.state.dice, sizeof(session.state.dice));
        record.dice_after_hold.push_back(after);
    }

    if (session.debug_mode) eval_and_store_board_nn(session, record);
    session.history.push_back(std::move(record));
}

void SessionManager::play_mc_turn(GameSession& session) {
    assert(nn_model_ != nullptr);

    GameSession::TurnRecord record;
    int player = static_cast<int>(session.state.board.current_player);
    record.player = player;
    std::memcpy(record.initial_dice, session.state.dice, sizeof(session.state.dice));

    // Snapshot board before the turn to detect what was placed.
    BoardState pre_board = session.state.board;

    MCConfig mc_config;
    mc_play_turn(session.state, session.ctx,
                 *nn_model_, device_, tables_,
                 mc_config, session.rng);

    // Find what cell was placed by diffing the board.
    for (int c = 0; c < kNumColumns; ++c) {
        for (int r = 0; r < kNumRows; ++r) {
            if (pre_board.cells[player][c][r] == kCellEmpty &&
                session.state.board.cells[player][c][r] != kCellEmpty) {
                record.placement = {static_cast<int8_t>(c),
                                    static_cast<int8_t>(r)};
                record.score = session.state.board.cells[player][c][r];
            }
        }
    }

    if (session.debug_mode) eval_and_store_board_nn(session, record);
    session.history.push_back(std::move(record));
}

// ---------------------------------------------------------------------------
// play_one_bot_turn — play one bot turn for the current player.
// Returns true if the game is now terminal.
// Caller must hold the session's mutex.
// ---------------------------------------------------------------------------

bool SessionManager::play_one_bot_turn(GameSession& s) {
    int player = static_cast<int>(s.state.board.current_player);
    PlayerType pt = s.player_types[player];

    if (pt == PlayerType::kHeuristic)
        play_heuristic_turn(s, HeuristicVersion::V1);
    else if (pt == PlayerType::kHeuristicV2)
        play_heuristic_turn(s, HeuristicVersion::V2);
    else if (pt == PlayerType::kHeuristicV3)
        play_heuristic_turn(s, HeuristicVersion::V3);
    else if (pt >= PlayerType::kHeuristicV4 && pt <= PlayerType::kHeuristicV15) {
        const int delta = static_cast<int>(pt) -
                          static_cast<int>(PlayerType::kHeuristicV4);
        play_heuristic_turn(s, static_cast<HeuristicVersion>(
            static_cast<int>(HeuristicVersion::V4) + delta));
    }
    else if (pt == PlayerType::kNNSolver && nn_model_)
        play_nn_turn(s);
    else if (pt == PlayerType::kMCRollout && nn_model_)
        play_mc_turn(s);
    else
        play_heuristic_turn(s, HeuristicVersion::V2);  // fallback when NN unavailable

    if (is_terminal(s.state.board)) {
        s.game_over = true;
        int duel = get_game_result(s.state, s.ctx);
        s.result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// advance_turn — one player's complete turn (bot only).
// ---------------------------------------------------------------------------

GameSession::TurnRecord SessionManager::advance_turn(int session_id) {
    auto entry = get_entry(session_id);
    if (!entry) return {};
    std::lock_guard<std::mutex> lock(entry->mutex);
    GameSession* s = entry->session.get();
    if (!s || s->game_over || s->waiting_for_human)
        return {};

    int player = static_cast<int>(s->state.board.current_player);
    PlayerType pt = s->player_types[player];

    if (pt == PlayerType::kHuman) {
        s->waiting_for_human = true;
        s->current_turn.player = player;
        std::memcpy(s->current_turn.initial_dice, s->state.dice,
                    sizeof(s->state.dice));
        return {};
    }

    bool terminal = play_one_bot_turn(*s);
    if (terminal) return s->history.back();

    // If next player is human, set flag.
    int next_player = static_cast<int>(s->state.board.current_player);
    if (s->player_types[next_player] == PlayerType::kHuman) {
        s->waiting_for_human = true;
        s->current_turn = {};
        s->current_turn.player = next_player;
        std::memcpy(s->current_turn.initial_dice, s->state.dice,
                    sizeof(s->state.dice));
    }

    return s->history.back();
}

// ---------------------------------------------------------------------------
// play_to_completion
// ---------------------------------------------------------------------------

void SessionManager::play_to_completion(int session_id) {
    auto entry = get_entry(session_id);
    if (!entry) return;

    while (true) {
        std::lock_guard<std::mutex> lock(entry->mutex);
        GameSession* s = entry->session.get();
        if (!s || s->game_over || s->waiting_for_human) return;

        int player = static_cast<int>(s->state.board.current_player);
        if (s->player_types[player] == PlayerType::kHuman) {
            s->waiting_for_human = true;
            return;
        }

        if (play_one_bot_turn(*s)) return;

        int next_player = static_cast<int>(s->state.board.current_player);
        if (s->player_types[next_player] == PlayerType::kHuman) {
            s->waiting_for_human = true;
            s->current_turn = {};
            s->current_turn.player = next_player;
            std::memcpy(s->current_turn.initial_dice, s->state.dice,
                        sizeof(s->state.dice));
            return;
        }
    } // Session mutex released between iterations.
}

// ---------------------------------------------------------------------------
// human_hold
// ---------------------------------------------------------------------------

bool SessionManager::human_hold(int session_id, uint8_t hold_mask) {
    auto entry = get_entry(session_id);
    if (!entry) return false;
    std::lock_guard<std::mutex> lock(entry->mutex);
    GameSession* s = entry->session.get();
    if (!s || s->game_over || !s->waiting_for_human) return false;
    if (!can_reroll(s->state, s->ctx)) return false;

    s->current_turn.hold_masks.push_back(hold_mask);
    perform_reroll(s->state, hold_mask, s->rng);
    std::array<int8_t, kNumDice> after;
    std::memcpy(after.data(), s->state.dice, sizeof(s->state.dice));
    s->current_turn.dice_after_hold.push_back(after);
    return true;
}

// ---------------------------------------------------------------------------
// human_place
// ---------------------------------------------------------------------------

bool SessionManager::human_place(int session_id, int column, int row) {
    auto entry = get_entry(session_id);
    if (!entry) return false;

    {
        std::lock_guard<std::mutex> lock(entry->mutex);
        GameSession* s = entry->session.get();
        if (!s || s->game_over || !s->waiting_for_human) return false;

        // Verify legality.
        const LegalPlacementCache& legal = get_legal_placements(s->state, s->ctx);
        if (!legal.is_legal[column][row]) return false;

        int score = calculate_score(row, s->state.dice,
                                    static_cast<int>(s->state.board.current_player),
                                    column, s->state.board, s->ctx);

        s->current_turn.placement = {static_cast<int8_t>(column),
                                      static_cast<int8_t>(row)};
        s->current_turn.score = static_cast<int8_t>(score);

        perform_placement(s->state, s->ctx, column, row, s->rng);
        s->waiting_for_human = false;

        if (s->debug_mode) eval_and_store_board_nn(*s, s->current_turn);

        s->history.push_back(s->current_turn);
        s->current_turn = {};

        if (is_terminal(s->state.board)) {
            s->game_over = true;
            int duel = get_game_result(s->state, s->ctx);
            s->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
            return true;
        }
    } // Session mutex released before bot loop.

    // Advance bot turns until next human turn or game over.
    // Re-acquire the session mutex per iteration so reads (get_session_copy,
    // get_human_options) for this session can interleave between bot turns.
    while (true) {
        std::lock_guard<std::mutex> lock(entry->mutex);
        GameSession* s = entry->session.get();
        if (!s || s->game_over || s->waiting_for_human) break;

        int next = static_cast<int>(s->state.board.current_player);
        if (s->player_types[next] == PlayerType::kHuman) {
            s->waiting_for_human = true;
            s->current_turn.player = next;
            std::memcpy(s->current_turn.initial_dice, s->state.dice,
                        sizeof(s->state.dice));
            break;
        }

        if (play_one_bot_turn(*s)) break;
    }

    return true;
}

// ---------------------------------------------------------------------------
// get_human_options
// ---------------------------------------------------------------------------

std::vector<std::pair<Placement, int>>
SessionManager::get_human_options(int session_id, bool& out_can_reroll) const {
    auto entry = get_entry(session_id);
    if (!entry) {
        out_can_reroll = false;
        return {};
    }
    std::lock_guard<std::mutex> lock(entry->mutex);
    const GameSession* s = entry->session.get();
    if (!s || !s->waiting_for_human) {
        out_can_reroll = false;
        return {};
    }
    out_can_reroll = can_reroll(s->state, s->ctx);

    const LegalPlacementCache& legal = get_legal_placements(s->state, s->ctx);
    int player = static_cast<int>(s->state.board.current_player);

    std::vector<std::pair<Placement, int>> options;
    options.reserve(legal.count);
    for (int i = 0; i < legal.count; ++i) {
        Placement p = legal.placements[i];
        int score = calculate_score(static_cast<int>(p.row), s->state.dice,
                                    player, static_cast<int>(p.column),
                                    s->state.board, s->ctx);
        options.emplace_back(p, score);
    }
    return options;
}

// ---------------------------------------------------------------------------
// compute_current_tensor — generate the 986-feature observation tensor for
// the session's current board state, optionally running NN inference.
// ---------------------------------------------------------------------------

bool SessionManager::compute_current_tensor(int session_id, int player,
                                              std::vector<float>& out_tensor,
                                              float& out_nn_value,
                                              bool& out_has_nn) const {
    auto entry = get_entry(session_id);
    if (!entry) return false;
    std::lock_guard<std::mutex> lock(entry->mutex);
    const GameSession* s = entry->session.get();
    if (!s) return false;

    out_tensor.assign(kTensorSize, 0.0f);
    int p = (player == 0 || player == 1) ? player
                                          : static_cast<int>(s->state.board.current_player);
    generate_tensor(s->state.board, s->ctx, p, tables_, out_tensor.data());

    out_has_nn  = false;
    out_nn_value = 0.0f;
    if (nn_model_) {
        torch::NoGradGuard no_grad;
        auto input  = torch::from_blob(out_tensor.data(), {1, kTensorSize},
                                        torch::kFloat32).to(device_);
        auto output = nn_model_->forward(input).to(torch::kCPU).contiguous();
        float val = output.data_ptr<float>()[0];
        if (nn_model_->config().output_activation != "sigmoid") {
            val = (val + 1.0f) / 2.0f;
        }
        out_nn_value = val;
        out_has_nn   = true;
    }
    return true;
}

// ---------------------------------------------------------------------------
// remove_session
// ---------------------------------------------------------------------------

void SessionManager::remove_session(int session_id) {
    std::shared_ptr<SessionEntry> entry;
    {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = sessions_.find(session_id);
        if (it == sessions_.end()) return;
        entry = std::move(it->second);
        sessions_.erase(it);
    }
    // Wait for any in-progress operation on this session to finish.
    std::lock_guard<std::mutex> session_lock(entry->mutex);
    entry->session.reset();
    // Entry is destroyed when the last shared_ptr (this local) goes out of scope.
}
