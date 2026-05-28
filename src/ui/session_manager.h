#pragma once

#include <algorithm>
#include <cassert>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "engine/context_rebuild.h"
#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "engine/scoring.h"
#include "engine/tensor.h"
#include "heuristic/heuristic_bot.h"
#include "model/model_config.h"
#include "model/pro_yams_net.h"
#include "model/trainer.h"
#include "solver/precomputed_tables.h"
#include "solver/solver.h"
#include "ui/game_session.h"

// MC bot is 1v1-only for now; gated by std::is_same_v<Traits, Yams1v1>.
#include "solver/mc_bot.h"

// ---------------------------------------------------------------------------
// SessionEntryT — wraps a GameSessionT<Traits> with a per-session mutex so
// that operations on one session never block another session.
// ---------------------------------------------------------------------------
template <typename Traits>
struct SessionEntryT {
    using Session = GameSessionT<Traits>;
    std::unique_ptr<Session> session;
    std::mutex mutex;
};

// ---------------------------------------------------------------------------
// SessionManagerT<Traits> — thread-safe manager for active game sessions.
//
// Templated on the game variant so 1v1 and 2v2 sessions are distinct types.
// Header-only so explicit instantiations are not required.
// ---------------------------------------------------------------------------
template <typename Traits>
class SessionManagerT {
public:
    using Session   = GameSessionT<Traits>;
    using TurnRec   = typename Session::TurnRecord;
    using HoldCand  = typename Session::HoldCandidateEval;
    using PlaceCand = typename Session::PlacementCandidateEval;
    using Entry     = SessionEntryT<Traits>;

    SessionManagerT(const PrecomputedTables& tables,
                    ProYamsNet* nn_model,
                    torch::Device device)
        : tables_(tables), nn_model_(nn_model), device_(device) {}

    /// Create a new game session. Returns its ID. `player_type_count` players
    /// are configured; for 1v1 pass {p0_type, p1_type}; for 2v2 pass four.
    int create_session(const PlayerType* player_types, int player_type_count,
                       uint64_t seed, bool debug_mode = false) {
        assert(player_type_count == Traits::kNumPlayers);
        std::lock_guard<std::mutex> lock(map_mutex_);
        int id = next_session_id_++;
        auto entry = std::make_shared<Entry>();
        entry->session = std::make_unique<Session>();

        Session& s = *entry->session;
        s.session_id      = id;
        for (int i = 0; i < Traits::kNumPlayers; ++i)
            s.player_types[i] = player_types[i];
        s.rng             = RNG(seed);
        s.tensor_buffer.resize(
            static_cast<size_t>(kMaxAfterstateRequests) * Traits::kTensorSize, 0.0f);

        init_game<Traits>(s.state, s.ctx, s.rng);
        s.state.board.current_player = 0;
        s.game_over         = false;
        s.result            = 0.0;
        s.waiting_for_human = false;
        s.debug_mode        = debug_mode;

        if (s.player_types[0] == PlayerType::kHuman) {
            s.waiting_for_human = true;
            s.current_turn.player = 0;
            std::memcpy(s.current_turn.initial_dice, s.state.dice,
                        sizeof(s.state.dice));
        }

        sessions_[id] = std::move(entry);
        return id;
    }

    /// Create a session seeded from an explicit board position instead of a
    /// fresh deal. `board` must carry cells, coefficients and current_player;
    /// cells_filled is recomputed and the context is rebuilt by scanning the
    /// cells. Fresh dice are rolled for the player on move (unless the position
    /// is already terminal, in which case the game opens as game_over).
    int create_session_from_board(const PlayerType* player_types,
                                  int player_type_count,
                                  const BoardStateT<Traits>& board,
                                  uint64_t seed, bool debug_mode = false) {
        assert(player_type_count == Traits::kNumPlayers);
        std::lock_guard<std::mutex> lock(map_mutex_);
        int id = next_session_id_++;
        auto entry = std::make_shared<Entry>();
        entry->session = std::make_unique<Session>();

        Session& s = *entry->session;
        s.session_id = id;
        for (int i = 0; i < Traits::kNumPlayers; ++i)
            s.player_types[i] = player_types[i];
        s.rng = RNG(seed);
        s.tensor_buffer.resize(
            static_cast<size_t>(kMaxAfterstateRequests) * Traits::kTensorSize, 0.0f);

        s.state.board = board;
        // Recompute cells_filled authoritatively so terminal detection is sound
        // regardless of what the caller supplied.
        uint16_t filled = 0;
        for (int p = 0; p < Traits::kNumPlayers; ++p)
            for (int c = 0; c < kNumColumns; ++c)
                for (int r = 0; r < kNumRows; ++r)
                    if (s.state.board.cells[p][c][r] != kCellEmpty) ++filled;
        s.state.board.cells_filled = filled;

        rebuild_context_from_board<Traits>(s.state.board, s.ctx);

        s.game_over         = false;
        s.result            = 0.0;
        s.waiting_for_human = false;
        s.debug_mode        = debug_mode;

        if (is_terminal<Traits>(s.state.board)) {
            s.game_over = true;
            int duel = get_game_result<Traits>(s.state, s.ctx);
            s.result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
        } else {
            start_turn<Traits>(s.state, s.rng);
            int cp = static_cast<int>(s.state.board.current_player);
            if (s.player_types[cp] == PlayerType::kHuman) {
                s.waiting_for_human = true;
                s.current_turn.player = cp;
                std::memcpy(s.current_turn.initial_dice, s.state.dice,
                            sizeof(s.state.dice));
            }
        }

        sessions_[id] = std::move(entry);
        return id;
    }

    /// 1v1 convenience overload.
    int create_session(PlayerType p0_type, PlayerType p1_type, uint64_t seed,
                       bool debug_mode = false) {
        static_assert(Traits::kNumPlayers == 2,
                      "Two-arg create_session is for 1v1 only; use the array overload for 2v2");
        PlayerType pts[2] = {p0_type, p1_type};
        return create_session(pts, 2, seed, debug_mode);
    }

    /// Get a copy of the current session state.
    bool get_session_copy(int id, Session& out) const {
        auto entry = get_entry(id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (!entry->session) return false;
        out = *entry->session;
        return true;
    }

    /// Advance one bot turn.
    TurnRec advance_turn(int session_id) {
        auto entry = get_entry(session_id);
        if (!entry) return {};
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s || s->game_over || s->waiting_for_human) return {};

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

    /// Advance one turn, forcing the bot to play even if the current player is human.
    TurnRec advance_turn_bot_override(int session_id) {
        auto entry = get_entry(session_id);
        if (!entry) return {};
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s || s->game_over) return {};

        int player = static_cast<int>(s->state.board.current_player);
        PlayerType old_pt = s->player_types[player];
        if (old_pt == PlayerType::kHuman) {
            s->player_types[player] = PlayerType::kNNSolver;
        }

        s->waiting_for_human = false;
        bool terminal = play_one_bot_turn(*s);

        if (old_pt == PlayerType::kHuman) {
            s->player_types[player] = PlayerType::kHuman;
        }

        if (terminal) return s->history.back();

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

    /// Play all remaining bot turns until the game ends or a human turn.
    void play_to_completion(int session_id) {
        auto entry = get_entry(session_id);
        if (!entry) return;

        while (true) {
            std::lock_guard<std::mutex> lock(entry->mutex);
            Session* s = entry->session.get();
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
        }
    }

    bool human_hold(int session_id, uint8_t hold_mask) {
        auto entry = get_entry(session_id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s || s->game_over || !s->waiting_for_human) return false;
        if (!can_reroll<Traits>(s->state, s->ctx)) return false;

        s->current_turn.hold_masks.push_back(hold_mask);
        perform_reroll<Traits>(s->state, hold_mask, s->rng);
        std::array<int8_t, kNumDice> after;
        std::memcpy(after.data(), s->state.dice, sizeof(s->state.dice));
        s->current_turn.dice_after_hold.push_back(after);
        return true;
    }

    bool human_place(int session_id, int column, int row) {
        auto entry = get_entry(session_id);
        if (!entry) return false;

        {
            std::lock_guard<std::mutex> lock(entry->mutex);
            Session* s = entry->session.get();
            if (!s || s->game_over || !s->waiting_for_human) return false;

            const LegalPlacementCache& legal = get_legal_placements<Traits>(s->state, s->ctx);
            if (!legal.is_legal[column][row]) return false;

            int score = calculate_score<Traits>(row, s->state.dice,
                                                static_cast<int>(s->state.board.current_player),
                                                column, s->state.board, s->ctx);

            s->current_turn.placement = {static_cast<int8_t>(column),
                                          static_cast<int8_t>(row)};
            s->current_turn.score = static_cast<int8_t>(score);

            perform_placement<Traits>(s->state, s->ctx, column, row, s->rng);
            s->waiting_for_human = false;

            if (s->debug_mode) eval_and_store_board_nn(*s, s->current_turn);

            s->history.push_back(s->current_turn);
            s->current_turn = {};

            if (is_terminal<Traits>(s->state.board)) {
                s->game_over = true;
                int duel = get_game_result<Traits>(s->state, s->ctx);
                s->result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
                return true;
            }
        }

        while (true) {
            std::lock_guard<std::mutex> lock(entry->mutex);
            Session* s = entry->session.get();
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

    std::vector<std::pair<Placement, int>>
    get_human_options(int session_id, bool& out_can_reroll) const {
        auto entry = get_entry(session_id);
        if (!entry) {
            out_can_reroll = false;
            return {};
        }
        std::lock_guard<std::mutex> lock(entry->mutex);
        const Session* s = entry->session.get();
        if (!s || !s->waiting_for_human) {
            out_can_reroll = false;
            return {};
        }
        out_can_reroll = can_reroll<Traits>(s->state, s->ctx);

        const LegalPlacementCache& legal = get_legal_placements<Traits>(s->state, s->ctx);
        int player = static_cast<int>(s->state.board.current_player);

        std::vector<std::pair<Placement, int>> options;
        options.reserve(legal.count);
        for (int i = 0; i < legal.count; ++i) {
            Placement p = legal.placements[i];
            int score = calculate_score<Traits>(static_cast<int>(p.row), s->state.dice,
                                                player, static_cast<int>(p.column),
                                                s->state.board, s->ctx);
            options.emplace_back(p, score);
        }
        return options;
    }

    void remove_session(int session_id) {
        std::shared_ptr<Entry> entry;
        {
            std::lock_guard<std::mutex> lock(map_mutex_);
            auto it = sessions_.find(session_id);
            if (it == sessions_.end()) return;
            entry = std::move(it->second);
            sessions_.erase(it);
        }
        std::lock_guard<std::mutex> session_lock(entry->mutex);
        entry->session.reset();
    }

    bool has_nn() const { return nn_model_ != nullptr; }

    // Evaluate an arbitrary board position with a checkpoint loaded on demand,
    // returning the model's win-rate for `player` (same forward + normalization
    // as debug-mode board eval). Used by the recorded-games replay viewer to
    // show, at each step, the win-rate the playing checkpoint assigns to the
    // player who just placed.
    //
    // Models are cached by checkpoint path in eval_models_, so a checkpoint is
    // loaded at most once no matter how many games (or steps) reference it —
    // this both avoids redundant loads and keeps memory bounded by the number
    // of distinct checkpoints, not the number of previews.
    bool evaluate_position(const std::string& checkpoint_path,
                           const BoardStateT<Traits>& board,
                           int player, float& out_value, std::string& err) {
        if (player < 0 || player >= Traits::kNumPlayers) {
            err = "player out of range";
            return false;
        }

        std::shared_ptr<ProYamsNet> model;
        {
            std::lock_guard<std::mutex> lock(eval_models_mutex_);
            auto it = eval_models_.find(checkpoint_path);
            if (it != eval_models_.end()) {
                model = it->second;
            } else {
                try {
                    ModelConfig cfg =
                        ModelTrainer::config_from_checkpoint(checkpoint_path);
                    ModelTrainer trainer(cfg, device_);
                    trainer.load_weights(checkpoint_path);
                    model = trainer.clone_for_inference(device_);
                    model->to(device_);
                    model->eval();
                } catch (const std::exception& e) {
                    err = e.what();
                    return false;
                }
                eval_models_[checkpoint_path] = model;
            }
        }

        // Rebuild context from the supplied cells (cells_filled recomputed so
        // the tensor's stage-dependent features are sound).
        BoardStateT<Traits> b = board;
        uint16_t filled = 0;
        for (int p = 0; p < Traits::kNumPlayers; ++p)
            for (int c = 0; c < kNumColumns; ++c)
                for (int r = 0; r < kNumRows; ++r)
                    if (b.cells[p][c][r] != kCellEmpty) ++filled;
        b.cells_filled = filled;

        GameContextT<Traits> ctx;
        rebuild_context_from_board<Traits>(b, ctx);

        std::vector<float> tensor(Traits::kTensorSize, 0.0f);
        generate_tensor<Traits>(b, ctx, player, tables_, tensor.data());

        torch::NoGradGuard no_grad;
        auto input = torch::from_blob(tensor.data(), {1, Traits::kTensorSize},
                                      torch::kFloat32).to(device_);
        auto output = model->forward(input).to(torch::kCPU).contiguous();
        float val = output.template data_ptr<float>()[0];
        if (model->config().output_activation != "sigmoid")
            val = (val + 1.0f) / 2.0f;
        out_value = val;
        return true;
    }

    bool compute_current_tensor(int session_id, int player,
                                std::vector<float>& out_tensor,
                                float& out_nn_value,
                                bool& out_has_nn) const {
        auto entry = get_entry(session_id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        const Session* s = entry->session.get();
        if (!s) return false;

        out_tensor.assign(Traits::kTensorSize, 0.0f);
        int p = (player >= 0 && player < Traits::kNumPlayers)
                ? player
                : static_cast<int>(s->state.board.current_player);
        generate_tensor<Traits>(s->state.board, s->ctx, p, tables_, out_tensor.data());

        out_has_nn  = false;
        out_nn_value = 0.0f;
        if (nn_model_) {
            torch::NoGradGuard no_grad;
            auto input  = torch::from_blob(out_tensor.data(), {1, Traits::kTensorSize},
                                            torch::kFloat32).to(device_);
            auto output = nn_model_->forward(input).to(torch::kCPU).contiguous();
            float val = output.template data_ptr<float>()[0];
            if (nn_model_->config().output_activation != "sigmoid") {
                val = (val + 1.0f) / 2.0f;
            }
            out_nn_value = val;
            out_has_nn   = true;
        }
        return true;
    }

private:
    std::shared_ptr<Entry> get_entry(int id) const {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = sessions_.find(id);
        return it != sessions_.end() ? it->second : nullptr;
    }

    void eval_and_store_board_nn(Session& session, TurnRec& record) {
        if (!nn_model_) return;
        std::vector<float> buf(Traits::kTensorSize);
        generate_tensor<Traits>(session.state.board, session.ctx,
                                record.player, tables_, buf.data());
        torch::NoGradGuard no_grad;
        auto input  = torch::from_blob(buf.data(), {1, Traits::kTensorSize},
                                        torch::kFloat32).to(device_);
        auto output = nn_model_->forward(input).to(torch::kCPU).contiguous();
        float val = output.template data_ptr<float>()[0];
        if (nn_model_->config().output_activation != "sigmoid") {
            val = (val + 1.0f) / 2.0f;
        }
        record.board_nn_value     = val;
        record.has_board_nn_value = true;
        session.current_board_nn_value = val;
        session.has_current_board_nn   = true;
    }

    void play_heuristic_turn(Session& session, HeuristicVersion version) {
        TurnRec record;
        record.player = static_cast<int>(session.state.board.current_player);
        std::memcpy(record.initial_dice, session.state.dice, sizeof(session.state.dice));

        session.buffers.dp_computed = false;
        session.buffers.evs_blended = false;
        solver_get_requests<Traits>(session.state, session.ctx, tables_, session.buffers);

        if (session.buffers.request_count == 0) {
            while (can_reroll<Traits>(session.state, session.ctx)) {
                perform_reroll<Traits>(session.state, 0, session.rng);
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
                perform_placement<Traits>(session.state, session.ctx,
                                          record.placement.column, record.placement.row, session.rng);
            }
            if (session.debug_mode) eval_and_store_board_nn(session, record);
            session.history.push_back(std::move(record));
            return;
        }

        if (static_cast<int>(version) >= static_cast<int>(HeuristicVersion::V4)) {
            const ResearchConfig& cfg = get_research_config_for(version);
            heuristic_evaluate_research<Traits>(session.state.board, session.ctx,
                                                session.buffers.requests,
                                                session.buffers.request_count,
                                                session.buffers.evs, tables_, cfg);
        } else if (version == HeuristicVersion::V3) {
            heuristic_evaluate_v3<Traits>(session.state.board, session.ctx,
                                          session.buffers.requests,
                                          session.buffers.request_count,
                                          session.buffers.evs, tables_);
        } else if (version == HeuristicVersion::V2) {
            heuristic_evaluate_v2<Traits>(session.state.board, session.ctx,
                                          session.buffers.requests,
                                          session.buffers.request_count,
                                          session.buffers.evs, tables_);
        } else {
            heuristic_evaluate<Traits>(session.state.board, session.ctx,
                                       session.buffers.requests,
                                       session.buffers.request_count,
                                       session.buffers.evs);
        }

        while (true) {
            SolverResult result = solver_resolve_greedy<Traits>(session.state, session.ctx,
                                                                tables_, session.buffers);

            if (result.should_place) {
                if (session.debug_mode) {
                    int8_t sorted_dice[kNumDice];
                    std::memcpy(sorted_dice, session.state.dice, sizeof(sorted_dice));
                    std::sort(sorted_dice, sorted_dice + kNumDice);

                    std::vector<PlaceCand> pevs;
                    for (int i = 0; i < session.buffers.request_count; ++i) {
                        int8_t req_score = session.buffers.requests[i].score;
                        int    row       = session.buffers.requests[i].placement.row;
                        int8_t raw       = static_cast<int8_t>(compute_raw_score(sorted_dice, row));
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
                perform_placement<Traits>(session.state, session.ctx,
                                          result.placement.column, result.placement.row, session.rng);
                break;
            }

            if (session.debug_mode) {
                std::vector<HoldCand> hevs;
                hevs.reserve(kNumHoldMasks);
                for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                    hevs.push_back({static_cast<uint8_t>(mask),
                                    static_cast<float>(session.buffers.mask_evs[mask])});
                }
                std::sort(hevs.begin(), hevs.end(),
                          [](const auto& a, const auto& b) { return a.expected_value > b.expected_value; });
                record.hold_evals.push_back(std::move(hevs));
            }

            assert(can_reroll<Traits>(session.state, session.ctx));
            record.hold_masks.push_back(result.hold_mask);
            perform_reroll<Traits>(session.state, result.hold_mask, session.rng);
            std::array<int8_t, kNumDice> after;
            std::memcpy(after.data(), session.state.dice, sizeof(session.state.dice));
            record.dice_after_hold.push_back(after);
        }

        if (session.debug_mode) eval_and_store_board_nn(session, record);
        session.history.push_back(std::move(record));
    }

    void play_nn_turn(Session& session) {
        assert(nn_model_ != nullptr);

        TurnRec record;
        record.player = static_cast<int>(session.state.board.current_player);
        std::memcpy(record.initial_dice, session.state.dice, sizeof(session.state.dice));

        SolverConfig greedy;

        session.buffers.dp_computed = false;
        session.buffers.evs_blended = false;
        solver_get_requests<Traits>(session.state, session.ctx, tables_, session.buffers);

        if (session.buffers.request_count == 0) {
            while (can_reroll<Traits>(session.state, session.ctx)) {
                perform_reroll<Traits>(session.state, 0, session.rng);
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
                perform_placement<Traits>(session.state, session.ctx,
                                          record.placement.column, record.placement.row, session.rng);
            }
            if (session.debug_mode) eval_and_store_board_nn(session, record);
            session.history.push_back(std::move(record));
            return;
        }

        generate_tensor_batch<Traits>(session.state.board, session.ctx,
                                       static_cast<int>(session.state.board.current_player),
                                       session.buffers.requests,
                                       session.buffers.request_count,
                                       tables_,
                                       session.tensor_buffer.data());

        {
            torch::NoGradGuard no_grad;
            auto input = torch::from_blob(
                session.tensor_buffer.data(),
                {session.buffers.request_count, Traits::kTensorSize},
                torch::kFloat32).to(device_);
            auto output = nn_model_->forward(input).to(torch::kCPU).contiguous();
            const float* ptr = output.template data_ptr<float>();
            for (int i = 0; i < session.buffers.request_count; ++i)
                session.buffers.evs[i] = static_cast<double>(ptr[i]);
        }

        while (true) {
            SolverResult result = solver_resolve<Traits>(session.state, session.ctx,
                                                        tables_, session.buffers,
                                                        greedy, session.rng);

            if (result.should_place) {
                if (session.debug_mode) {
                    int8_t sorted_dice[kNumDice];
                    std::memcpy(sorted_dice, session.state.dice, sizeof(sorted_dice));
                    std::sort(sorted_dice, sorted_dice + kNumDice);

                    std::vector<PlaceCand> pevs;
                    for (int i = 0; i < session.buffers.request_count; ++i) {
                        int8_t req_score = session.buffers.requests[i].score;
                        int    row       = session.buffers.requests[i].placement.row;
                        int8_t raw       = static_cast<int8_t>(compute_raw_score(sorted_dice, row));
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
                perform_placement<Traits>(session.state, session.ctx,
                                          result.placement.column, result.placement.row, session.rng);
                break;
            }

            if (session.debug_mode) {
                std::vector<HoldCand> hevs;
                hevs.reserve(kNumHoldMasks);
                for (int mask = 0; mask < kNumHoldMasks; ++mask) {
                    hevs.push_back({static_cast<uint8_t>(mask),
                                    static_cast<float>(session.buffers.mask_evs[mask])});
                }
                std::sort(hevs.begin(), hevs.end(),
                          [](const auto& a, const auto& b) { return a.expected_value > b.expected_value; });
                record.hold_evals.push_back(std::move(hevs));
            }

            assert(can_reroll<Traits>(session.state, session.ctx));
            record.hold_masks.push_back(result.hold_mask);
            perform_reroll<Traits>(session.state, result.hold_mask, session.rng);
            std::array<int8_t, kNumDice> after;
            std::memcpy(after.data(), session.state.dice, sizeof(session.state.dice));
            record.dice_after_hold.push_back(after);
        }

        if (session.debug_mode) eval_and_store_board_nn(session, record);
        session.history.push_back(std::move(record));
    }

    // MC bot is 1v1-only. In 2v2 we silently fall back to the V2 heuristic.
    void play_mc_turn(Session& session) {
        if constexpr (std::is_same_v<Traits, Yams1v1>) {
            assert(nn_model_ != nullptr);

            TurnRec record;
            int player = static_cast<int>(session.state.board.current_player);
            record.player = player;
            std::memcpy(record.initial_dice, session.state.dice, sizeof(session.state.dice));

            BoardStateT<Traits> pre_board = session.state.board;

            MCConfig mc_config;
            mc_play_turn(session.state, session.ctx,
                         *nn_model_, device_, tables_,
                         mc_config, session.rng);

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
        } else {
            play_heuristic_turn(session, HeuristicVersion::V2);
        }
    }

    bool play_one_bot_turn(Session& s) {
        int player = static_cast<int>(s.state.board.current_player);
        PlayerType pt = s.player_types[player];

        if (pt == PlayerType::kHeuristic)
            play_heuristic_turn(s, HeuristicVersion::V1);
        else if (pt == PlayerType::kHeuristicV2)
            play_heuristic_turn(s, HeuristicVersion::V2);
        else if (pt == PlayerType::kHeuristicV3)
            play_heuristic_turn(s, HeuristicVersion::V3);
        else if (pt >= PlayerType::kHeuristicV4 && pt <= PlayerType::kHeuristicV17) {
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
            play_heuristic_turn(s, HeuristicVersion::V2);

        if (is_terminal<Traits>(s.state.board)) {
            s.game_over = true;
            int duel = get_game_result<Traits>(s.state, s.ctx);
            s.result = (duel > 0) ? 1.0 : (duel < 0) ? 0.0 : 0.5;
            return true;
        }
        return false;
    }

    const PrecomputedTables& tables_;
    ProYamsNet*              nn_model_;
    torch::Device            device_;

    std::map<int, std::shared_ptr<Entry>> sessions_;
    int next_session_id_ = 1;
    mutable std::mutex map_mutex_;

    // Checkpoints loaded on demand for recorded-game replay evaluation, keyed
    // by path so each distinct checkpoint is held exactly once.
    std::map<std::string, std::shared_ptr<ProYamsNet>> eval_models_;
    std::mutex eval_models_mutex_;
};

// Backward-compat aliases.
using SessionManager    = SessionManagerT<Yams1v1>;
using SessionManager2v2 = SessionManagerT<Yams2v2>;
using SessionEntry      = SessionEntryT<Yams1v1>;
