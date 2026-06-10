#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <utility>
#include <vector>

#include <torch/torch.h>

#include "engine/board_init.h"
#include "engine/context_rebuild.h"
#include "engine/game_context.h"
#include "engine/game_flow.h"
#include "engine/game_traits.h"
#include "engine/placement.h"
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
    // Long-poll support: `version` increments on every visible state change and
    // `cv` wakes any client blocked in wait_for_change. Both guarded by `mutex`.
    std::condition_variable cv;
    uint64_t version = 0;
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

    // One enumerated next-move afterstate evaluated with a loaded checkpoint.
    // valid_for_dice mirrors debug-mode's placement filter: the (row, score) is
    // reachable with the dice the player actually rolled that turn AND the cell
    // is a legal final placement (Turbo is excluded — it needs a roll left).
    // requires_roll flags the Turbo column: those afterstates can only be reached
    // with rolls_left > 0, so they are shown for comparison (debugging the
    // roll-vs-place-in-Turbo decision) but are never valid_for_dice.
    struct MoveEval {
        Placement placement;
        int8_t    score;
        float     eval_value;
        bool      valid_for_dice;
        bool      requires_roll;
    };

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

    /// Get a copy plus the current state version (for long-poll clients).
    bool get_session_copy(int id, Session& out, uint64_t& out_version) const {
        auto entry = get_entry(id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        if (!entry->session) return false;
        out = *entry->session;
        out_version = entry->version;
        return true;
    }

    /// Bump a session's state version and wake any long-poll waiters. Called by
    /// the server after a mutating request so blocked clients return at once.
    void notify_changed(int id) {
        auto entry = get_entry(id);
        if (!entry) return;
        std::lock_guard<std::mutex> lock(entry->mutex);
        ++entry->version;
        entry->cv.notify_all();
    }

    /// Block until the session's version differs from `since` (i.e. something
    /// changed) or `timeout_ms` elapses, then return a fresh copy and the
    /// current version. Returns false only if the session is gone. The state is
    /// returned on timeout too (with version == since), so callers can simply
    /// re-issue with the returned version.
    bool wait_for_change(int id, uint64_t since, int64_t timeout_ms,
                         Session& out, uint64_t& out_version) {
        auto entry = get_entry(id);
        if (!entry) return false;
        std::unique_lock<std::mutex> lock(entry->mutex);
        if (!entry->session) return false;
        entry->cv.wait_for(lock, std::chrono::milliseconds(timeout_ms),
            [&] { return !entry->session || entry->version != since; });
        if (!entry->session) return false;
        out = *entry->session;
        out_version = entry->version;
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

        // Capture the pre-roll dice so we can compute which dice carried over.
        int8_t before[kNumDice];
        std::memcpy(before, s->state.dice, sizeof(before));

        perform_reroll<Traits>(s->state, hold_mask, s->rng);

        // Keep the live preview alive on the carried-over dice rather than
        // clearing it. The active player's own client remaps its hold selection
        // to the dice it kept (remapHoldMask); mirroring that here means a
        // spectating teammate sees those dice stay selected through the reroll
        // instead of flickering off (preview_mask=0) and back on when the next
        // hold preview arrives. Greedy value-match, matching build_holds_json's
        // held_flags so the reveal and the post-roll preview agree.
        int8_t held_values[kNumDice];
        int held_count = 0;
        for (int i = 0; i < kNumDice; ++i)
            if ((hold_mask >> i) & 1) held_values[held_count++] = before[i];
        bool mapped[kNumDice] = {false};
        uint8_t carried = 0;
        for (int i = 0; i < kNumDice; ++i) {
            for (int j = 0; j < held_count; ++j) {
                if (!mapped[j] && held_values[j] == s->state.dice[i]) {
                    mapped[j] = true;
                    carried |= static_cast<uint8_t>(1u << i);
                    break;
                }
            }
        }
        s->live_hold_player = static_cast<int>(s->state.board.current_player);
        s->live_hold_mask   = carried;
        s->live_rolling     = false;  // result landed — stop the optimistic spin

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

            int player = static_cast<int>(s->state.board.current_player);
            int score = yams_bonus_active<Traits>(s->state)
                ? calculate_yams_bonus_score<Traits>(row, player, column,
                                                     s->state.board, s->ctx)
                : calculate_score<Traits>(row, s->state.dice, player,
                                          column, s->state.board, s->ctx);

            s->current_turn.placement = {static_cast<int8_t>(column),
                                          static_cast<int8_t>(row)};
            s->current_turn.score = static_cast<int8_t>(score);

            perform_placement<Traits>(s->state, s->ctx, column, row, s->rng);
            s->waiting_for_human = false;
            s->live_hold_player = -1;  // turn over — drop the tentative preview
            s->live_rolling     = false;

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
        bool bonus = yams_bonus_active<Traits>(s->state);

        std::vector<std::pair<Placement, int>> options;
        options.reserve(legal.count);
        for (int i = 0; i < legal.count; ++i) {
            Placement p = legal.placements[i];
            int score = bonus
                ? calculate_yams_bonus_score<Traits>(static_cast<int>(p.row), player,
                                                     static_cast<int>(p.column),
                                                     s->state.board, s->ctx)
                : calculate_score<Traits>(static_cast<int>(p.row), s->state.dice,
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
        ++entry->version;
        entry->cv.notify_all();  // wake any long-poll waiter so it returns 404
    }

    bool has_nn() const { return nn_model_ != nullptr; }

    // ---------------------------------------------------------------------------
    // Shared multiplayer (two humans vs NN) — heartbeat / disconnect takeover.
    // ---------------------------------------------------------------------------

    static int64_t now_ms() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
    }

    // Mark a session as a shared two-human game and stamp the given human seats
    // as just-seen so they don't immediately time out before the first poll.
    void mark_shared(int session_id, const std::vector<int>& human_seats) {
        auto entry = get_entry(session_id);
        if (!entry) return;
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s) return;
        s->shared_multiplayer = true;
        int64_t t = now_ms();
        for (int seat : human_seats)
            if (seat >= 0 && seat < Traits::kNumPlayers)
                s->seat_last_seen_ms[seat] = t;
    }

    // Record the active player's tentative (uncommitted) hold selection so a
    // shared-game teammate can watch the hold pattern form live. Ignored unless
    // the session is waiting on `seat` to move.
    bool set_hold_preview(int session_id, int seat, uint8_t mask) {
        if (seat < 0 || seat >= Traits::kNumPlayers) return false;
        auto entry = get_entry(session_id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s || s->game_over || !s->waiting_for_human) return false;
        if (static_cast<int>(s->state.board.current_player) != seat) return false;
        s->live_hold_player = seat;
        s->live_hold_mask   = mask;
        s->live_rolling     = false;  // a fresh preview means still deliberating
        return true;
    }

    // Mark that `seat` has committed a reroll keeping `mask` — the new dice
    // aren't computed yet (the /hold request is in flight). Lets a spectating
    // teammate start the spin at once. Cleared when the reroll actually lands.
    bool set_rolling(int session_id, int seat, uint8_t mask) {
        if (seat < 0 || seat >= Traits::kNumPlayers) return false;
        auto entry = get_entry(session_id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s || s->game_over || !s->waiting_for_human) return false;
        if (static_cast<int>(s->state.board.current_player) != seat) return false;
        s->live_hold_player = seat;
        s->live_hold_mask   = mask;
        s->live_rolling     = true;
        return true;
    }

    // Record that a client controlling `seat` is still alive.
    void heartbeat(int session_id, int seat) {
        if (seat < 0 || seat >= Traits::kNumPlayers) return;
        auto entry = get_entry(session_id);
        if (!entry) return;
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s) return;
        s->seat_last_seen_ms[seat] = now_ms();
    }

    // In a shared game, mark (but do NOT flip) any human seat whose last
    // heartbeat is older than `timeout_ms` as disconnected — a candidate for NN
    // takeover. The surviving teammate decides whether to actually switch it to
    // the AI (see takeover_seat); a seat that resumes heartbeating clears.
    bool refresh_disconnects(int session_id, int64_t timeout_ms) {
        auto entry = get_entry(session_id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s || !s->shared_multiplayer || s->game_over) return false;

        int64_t cutoff = now_ms() - timeout_ms;
        bool changed = false;
        for (int seat = 0; seat < Traits::kNumPlayers; ++seat) {
            bool want;
            if (s->player_types[seat] != PlayerType::kHuman || s->seat_nn_takeover[seat]) {
                want = false;
            } else {
                // never-seen (0) is treated as not-yet-disconnected.
                want = (s->seat_last_seen_ms[seat] != 0 &&
                        s->seat_last_seen_ms[seat] <= cutoff);
            }
            if (s->seat_disconnected[seat] != want) {
                s->seat_disconnected[seat] = want;
                changed = true;
            }
        }
        // A flag flip (e.g. crossing the grace threshold, or a returning player)
        // is a visible change — wake long-poll waiters.
        if (changed) { ++entry->version; entry->cv.notify_all(); }
        return changed;
    }

    // Flip a disconnected human seat to NN at the surviving teammate's request.
    // Refuses unless the seat really is a silent human past the grace period, so
    // a still-active partner can't be kicked. If the seat is the one on the move,
    // clears waiting_for_human so the next advance plays it as a bot. Returns
    // whether the takeover was applied.
    bool takeover_seat(int session_id, int seat, int64_t timeout_ms) {
        if (seat < 0 || seat >= Traits::kNumPlayers) return false;
        auto entry = get_entry(session_id);
        if (!entry) return false;
        std::lock_guard<std::mutex> lock(entry->mutex);
        Session* s = entry->session.get();
        if (!s || !s->shared_multiplayer || s->game_over) return false;
        if (s->player_types[seat] != PlayerType::kHuman) return false;

        int64_t cutoff = now_ms() - timeout_ms;
        bool stale = (s->seat_last_seen_ms[seat] != 0 && s->seat_last_seen_ms[seat] <= cutoff);
        if (!stale) return false;  // partner is still active — don't kick them

        s->player_types[seat]     = PlayerType::kNNSolver;
        s->seat_nn_takeover[seat]  = true;
        s->seat_disconnected[seat] = false;
        int cur = static_cast<int>(s->state.board.current_player);
        if (seat == cur && s->waiting_for_human) {
            s->waiting_for_human = false;
            s->current_turn = {};
        }
        return true;
    }

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

        std::shared_ptr<ProYamsNet> model = get_eval_model(checkpoint_path, err);
        if (!model) return false;

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
        auto input  = make_nn_input(*model, tensor.data(), 1, device_);
        auto output = model->forward(input).to(torch::kCPU).contiguous();
        float val = output.template data_ptr<float>()[0];
        if (model->config().output_activation != "sigmoid")
            val = (val + 1.0f) / 2.0f;
        out_value = val;
        return true;
    }

    // Enumerate every legal next-move afterstate from `board_in` for `player`
    // and evaluate each with the (on-demand loaded) checkpoint, mirroring the
    // bot's own afterstate evaluation in play_nn_turn. The returned eval_value
    // is the raw model output (matching debug-mode's placement evals). If
    // `dice` is non-null it is the dice the player rolled that turn, used to set
    // each move's valid_for_dice flag (the placement-step subset). Powers the
    // recorded-games replay "all possible moves" table.
    bool evaluate_moves(const std::string& checkpoint_path,
                        const BoardStateT<Traits>& board_in, int player,
                        const int8_t* dice,
                        std::vector<MoveEval>& out, std::string& err) {
        out.clear();
        if (player < 0 || player >= Traits::kNumPlayers) {
            err = "player out of range";
            return false;
        }

        std::shared_ptr<ProYamsNet> model = get_eval_model(checkpoint_path, err);
        if (!model) return false;

        // Build a board + context with `player` on the move.
        BoardStateT<Traits> b = board_in;
        b.current_player = static_cast<int8_t>(player);
        uint16_t filled = 0;
        for (int p = 0; p < Traits::kNumPlayers; ++p)
            for (int c = 0; c < kNumColumns; ++c)
                for (int r = 0; r < kNumRows; ++r)
                    if (b.cells[p][c][r] != kCellEmpty) ++filled;
        b.cells_filled = filled;

        GameContextT<Traits> ctx;
        rebuild_context_from_board<Traits>(b, ctx);

        // solver_get_requests is dice-independent: it enumerates every legal
        // cell × achievable score (+ scratch). A GameState wrapper is needed for
        // the call; dice only matter for the valid_for_dice tagging below.
        GameStateT<Traits> state;
        state.board      = b;
        // rolls_left > 0 makes get_legal_placements return the legal_all cache,
        // which includes the Turbo column. We want Turbo afterstates enumerated
        // here even though they are not valid final placements, so the replay
        // move-eval table can show what placing in Turbo would have been worth
        // (the roll-vs-place-in-Turbo decision). valid_for_dice / requires_roll
        // below keep the placement-step semantics straight.
        state.rolls_left = 2;
        int8_t sorted[kNumDice];
        if (dice) {
            std::memcpy(sorted, dice, sizeof(sorted));
            std::sort(sorted, sorted + kNumDice);
            std::memcpy(state.dice, sorted, sizeof(state.dice));
        } else {
            for (int i = 0; i < kNumDice; ++i) { sorted[i] = 1; state.dice[i] = 1; }
        }

        auto buffers = std::make_unique<SolverBuffers>();
        buffers->dp_computed = false;
        buffers->evs_blended = false;
        solver_get_requests<Traits>(state, ctx, tables_, *buffers);
        if (buffers->request_count == 0) return true;  // nothing to evaluate

        std::vector<float> tbuf(
            static_cast<size_t>(buffers->request_count) * Traits::kTensorSize, 0.0f);
        generate_tensor_batch<Traits>(b, ctx, player, buffers->requests,
                                      buffers->request_count, tables_, tbuf.data());

        std::vector<float> evs(buffers->request_count);
        {
            torch::NoGradGuard no_grad;
            auto input = make_nn_input(*model, tbuf.data(),
                                       buffers->request_count, device_);
            auto output = model->forward(input).to(torch::kCPU).contiguous();
            const float* ptr = output.template data_ptr<float>();
            for (int i = 0; i < buffers->request_count; ++i) evs[i] = ptr[i];
        }

        out.reserve(buffers->request_count);
        for (int i = 0; i < buffers->request_count; ++i) {
            const AfterstateRequest& rq = buffers->requests[i];
            int  row = rq.placement.row;
            bool is_turbo = (rq.placement.column == kColTurbo);
            int8_t raw = static_cast<int8_t>(compute_raw_score(sorted, row));
            // Turbo is never a valid final placement (it needs a roll left), so
            // it stays out of the placement-step "valid after roll" subset.
            bool valid = (dice != nullptr) && !is_turbo &&
                         (rq.score == raw || rq.score == 0);
            out.push_back({rq.placement, rq.score, evs[i], valid, is_turbo});
        }
        return true;
    }

    // Replay one recorded turn's decisions against the checkpoint and measure
    // how much win-probability each human choice gave up versus the solver's
    // best action at that point. Decisions are split into rerolls ("holds") and
    // the final placement.
    //
    //   board_before : board at the start of the turn (player on the move)
    //   initial_dice : the first roll (sorted, as recorded)
    //   reroll_dice[k]/hold_masks[k] : the dice after, and mask used for, the
    //                  k-th reroll the player made (k = 0..rerolls-1)
    //   placement/place_score : the cell the player finally wrote
    //
    // Each returned delta is in win-probability units (0..1), already normalized
    // for the model's output activation, and is >= 0 (best minus chosen). On a
    // forced turn with no legal afterstates nothing is produced (out_has_place
    // false, out_hold_deltas empty) and the call still succeeds.
    bool evaluate_turn_accuracy(const std::string& checkpoint_path,
                                const BoardStateT<Traits>& board_before, int player,
                                const int8_t initial_dice[kNumDice],
                                const std::vector<std::array<int8_t, kNumDice>>& reroll_dice,
                                const std::vector<uint8_t>& hold_masks,
                                Placement placement, int8_t place_score,
                                std::vector<double>& out_hold_deltas,
                                double& out_place_delta, bool& out_has_place,
                                std::vector<double>& out_roll_lucks,
                                std::string& err) {
        out_hold_deltas.clear();
        out_place_delta = 0.0;
        out_has_place   = false;
        out_roll_lucks.clear();
        if (player < 0 || player >= Traits::kNumPlayers) {
            err = "player out of range";
            return false;
        }

        std::shared_ptr<ProYamsNet> model = get_eval_model(checkpoint_path, err);
        if (!model) return false;

        BoardStateT<Traits> b = board_before;
        b.current_player = static_cast<int8_t>(player);
        uint16_t filled = 0;
        for (int p = 0; p < Traits::kNumPlayers; ++p)
            for (int c = 0; c < kNumColumns; ++c)
                for (int r = 0; r < kNumRows; ++r)
                    if (b.cells[p][c][r] != kCellEmpty) ++filled;
        b.cells_filled = filled;

        GameContextT<Traits> ctx;
        rebuild_context_from_board<Traits>(b, ctx);

        GameStateT<Traits> state;
        state.board = b;

        auto buffers = std::make_unique<SolverBuffers>();
        buffers->dp_computed = false;
        buffers->evs_blended = false;
        solver_get_requests<Traits>(state, ctx, tables_, *buffers);
        if (buffers->request_count == 0) return true;  // forced turn, nothing to score

        std::vector<float> tbuf(
            static_cast<size_t>(buffers->request_count) * Traits::kTensorSize, 0.0f);
        generate_tensor_batch<Traits>(b, ctx, player, buffers->requests,
                                      buffers->request_count, tables_, tbuf.data());
        {
            torch::NoGradGuard no_grad;
            auto input = make_nn_input(*model, tbuf.data(),
                                       buffers->request_count, device_);
            auto output = model->forward(input).to(torch::kCPU).contiguous();
            const float* ptr = output.template data_ptr<float>();
            for (int i = 0; i < buffers->request_count; ++i)
                buffers->evs[i] = static_cast<double>(ptr[i]);
        }

        // Win-probability scale: margin-output models are affine-mapped by
        // (v+1)/2, so a raw delta becomes a win-prob delta when halved.
        const double scale =
            (model->config().output_activation == "sigmoid") ? 1.0 : 0.5;

        score_turn_filled(b, ctx, player, initial_dice, reroll_dice, hold_masks,
                          placement, place_score, scale, *buffers,
                          out_hold_deltas, out_place_delta, out_has_place,
                          out_roll_lucks);
        return true;
    }

    // Score one turn's decisions (accuracy) and dice (luck) given a SolverBuffers
    // whose afterstate `requests` are already enumerated (solver_get_requests)
    // and whose `evs` are already filled by the NN, with dp_computed == false on
    // entry. This is pure DP/bookkeeping — no inference — so the whole-game
    // evaluator can run the NN once over every turn's afterstates and then call
    // this per turn. `scale` maps a raw EV delta to win-prob units.
    void score_turn_filled(const BoardStateT<Traits>& board,
                           const GameContextT<Traits>& ctx, int player,
                           const int8_t initial_dice[kNumDice],
                           const std::vector<std::array<int8_t, kNumDice>>& reroll_dice,
                           const std::vector<uint8_t>& hold_masks,
                           Placement placement, int8_t place_score,
                           double scale, SolverBuffers& buffers,
                           std::vector<double>& out_hold_deltas,
                           double& out_place_delta, bool& out_has_place,
                           std::vector<double>& out_roll_lucks) {
        GameStateT<Traits> state;
        state.board = board;

        // Request index for a specific (cell, score), or -1 if not enumerated.
        auto find_req = [&](int col, int row, int score) -> int {
            for (int c = 0; c < buffers.num_legal_cells; ++c) {
                if (buffers.cell_cols[c] != col || buffers.cell_rows[c] != row) continue;
                if (score > 0 && buffers.req_map[c][score] != -1)
                    return buffers.req_map[c][score];
                return buffers.scratch_map[c];
            }
            return -1;
        };

        SolverConfig cfg;   // greedy: fills mask_evs[0..kNumHoldMasks]
        cfg.compute_pre_roll_ev = true;  // also build V2 / pre_roll_ev for luck
        RNG dummy(0);

        int8_t dice[kNumDice];
        std::memcpy(dice, initial_dice, sizeof(dice));
        sort_dice(dice);
        int rolls_left = 2;

        const size_t rerolls = std::min(hold_masks.size(), reroll_dice.size());
        for (size_t k = 0; k < rerolls; ++k) {
            std::memcpy(state.dice, dice, sizeof(state.dice));
            state.rolls_left = static_cast<int8_t>(rolls_left);
            SolverResult r =
                solver_resolve<Traits>(state, ctx, tables_, buffers, cfg, dummy);

            // mask_evs is only valid when a real reroll choice existed (not the
            // forced-place degenerate case); guard so we never read stale EVs.
            bool force_place = (rolls_left == 1 &&
                                ctx.non_turbo_cells_remaining[player] == 0);
            uint8_t m = hold_masks[k];
            if (!force_place && m < kNumHoldMasks) {
                double chosen = buffers.mask_evs[m];
                double best   = r.expected_value;
                out_hold_deltas.push_back(scale * (best - chosen));
            }

            std::memcpy(dice, reroll_dice[k].data(), sizeof(dice));
            sort_dice(dice);
            --rolls_left;
        }

        // Final placement decision.
        std::memcpy(state.dice, dice, sizeof(state.dice));
        state.rolls_left = static_cast<int8_t>(rolls_left);
        SolverResult r =
            solver_resolve<Traits>(state, ctx, tables_, buffers, cfg, dummy);

        int creq = find_req(static_cast<int>(placement.column),
                            static_cast<int>(placement.row),
                            static_cast<int>(place_score));
        if (creq >= 0) {
            double chosen = buffers.evs[creq];
            double best;
            if (rolls_left >= 1) {
                // V_r already accounts for the option to reroll instead.
                best = r.expected_value;
            } else {
                // No rerolls left: best is the strongest placement for the dice.
                best = -std::numeric_limits<double>::infinity();
                for (int c = 0; c < buffers.num_legal_cells; ++c) {
                    int row = buffers.cell_rows[c];
                    int raw = static_cast<int>(compute_raw_score(dice, row));
                    int req = (raw > 0 && buffers.req_map[c][raw] != -1)
                                  ? buffers.req_map[c][raw]
                                  : buffers.scratch_map[c];
                    if (req >= 0 && buffers.evs[req] > best) best = buffers.evs[req];
                }
            }
            out_place_delta = scale * (best - chosen);
            out_has_place   = true;
        }

        // --- Luck (normalized win-chance, -100..+100) ---------------------------
        // For each roll the player actually made, compare the win-chance of the
        // dice they got (R) against the win-chance they *expected* before the roll
        // (B = the probability-weighted mean over outcomes), then divide by the
        // full [worst,best] spread of that roll. This keeps the index mean-zero
        // (E[R-B]=0, denominator constant ⇒ E[luck]=0) and bounded to [-100,100]:
        // 0 = rolled as expected, + = luckier than expected, - = unluckier.
        //
        // NB: an earlier version scaled the +/- sides by (best-B) and (B-worst)
        // separately; that anchors +/-100 to the literal best/worst face but is
        // biased strongly negative because dice-value distributions are right-
        // skewed (rare high outliers inflate best-B). The single-denominator form
        // below is the unbiased fix.
        //
        // The DP layers built above (V2/V1/V0, ev_held_v1/v0, pre_roll_ev) supply
        // R and B; we only need min/max over the same transition lists for the
        // [worst,best] spread. Ratios of win-chance differences are invariant to
        // the model's output activation, so no scale factor is needed here.
        if (buffers.dp_computed) {
            auto minmax = [&](const double* layer, int held_id,
                              double& mn, double& mx) {
                int cnt = tables_.transition_count[held_id];
                const Transition* tr = tables_.all_transitions.data()
                                       + tables_.transition_offset[held_id];
                mn =  std::numeric_limits<double>::infinity();
                mx = -std::numeric_limits<double>::infinity();
                for (int t = 0; t < cnt; ++t) {
                    double v = layer[tr[t].target_state_id];
                    if (v < mn) mn = v;
                    if (v > mx) mx = v;
                }
            };
            auto norm = [](double R, double B, double mn, double mx) -> double {
                double d = mx - mn;
                return d > 1e-9 ? 100.0 * (R - B) / d : 0.0;
            };

            int8_t cur[kNumDice];
            std::memcpy(cur, initial_dice, sizeof(cur));
            sort_dice(cur);

            // Roll 0 — the initial roll. Baseline = pre_roll_ev (E[V2] over the
            // full reroll), range = V2 over the "no dice held" transition list.
            {
                int held_none = tables_.moves[0][0];
                double mn, mx; minmax(buffers.v2, held_none, mn, mx);
                double R = buffers.v2[get_dice_state_id(cur, tables_)];
                out_roll_lucks.push_back(norm(R, buffers.pre_roll_ev, mn, mx));
            }
            // Reroll k lands with (1-k) rerolls left → layer V1 (k=0), V0 (k=1).
            int rl = 2;
            for (size_t k = 0; k < rerolls && k < hold_masks.size(); ++k) {
                uint8_t m = hold_masks[k];
                if (m >= kNumHoldMasks) break;
                int held_id = tables_.moves[get_dice_state_id(cur, tables_)][m];
                --rl;
                const double* layer   = (rl >= 1) ? buffers.v1 : buffers.v0;
                const double* held_ev = (rl >= 1) ? buffers.ev_held_v1 : buffers.ev_held_v0;
                double mn, mx; minmax(layer, held_id, mn, mx);
                int8_t nd[kNumDice];
                std::memcpy(nd, reroll_dice[k].data(), sizeof(nd));
                sort_dice(nd);
                double R = layer[get_dice_state_id(nd, tables_)];
                out_roll_lucks.push_back(norm(R, held_ev[held_id], mn, mx));
                std::memcpy(cur, nd, sizeof(cur));
            }
        }
    }

    // Per-turn input for the batched whole-game evaluator.
    struct TurnEval {
        BoardStateT<Traits> board;                 // board before the turn
        int                 player = 0;
        int8_t              initial_dice[kNumDice] = {};
        std::vector<std::array<int8_t, kNumDice>> reroll_dice;
        std::vector<uint8_t> hold_masks;
        Placement           placement{0, 0};
        int8_t              place_score = 0;
    };
    struct TurnResult {
        std::vector<double> hold_deltas;
        double              place_delta = 0.0;
        bool                has_place   = false;
        std::vector<double> roll_lucks;
        int                 player = 0;
    };

    // Score every turn of a game in a single call. The NN runs ONCE over the
    // afterstates of all turns concatenated, instead of one forward pass per
    // turn; results are returned in input order. Forced turns (no legal
    // afterstates) yield empty deltas/luck and has_place == false.
    bool evaluate_game_accuracy(const std::string& checkpoint_path,
                                std::vector<TurnEval>& turns,
                                std::vector<TurnResult>& out, std::string& err) {
        std::shared_ptr<ProYamsNet> model = get_eval_model(checkpoint_path, err);
        if (!model) return false;
        run_game_eval(*model, turns, out);
        return true;
    }

    // Core batched evaluator shared by the checkpoint-based path (admin) and the
    // live play server (which passes its already-loaded model). Runs the NN once
    // over every turn's afterstates, then per-turn DP/accuracy/luck with no
    // further inference.
    void run_game_eval(ProYamsNet& model, std::vector<TurnEval>& turns,
                       std::vector<TurnResult>& out) {
        out.assign(turns.size(), TurnResult{});
        const double scale =
            (model.config().output_activation == "sigmoid") ? 1.0 : 0.5;

        // Pass 1 — enumerate afterstates and build tensors for every turn.
        auto scratch = std::make_unique<SolverBuffers>();
        std::vector<float> tensors;
        std::vector<int>   offset(turns.size(), 0);  // row offset into tensors
        std::vector<int>   count(turns.size(), 0);   // afterstate count per turn

        for (size_t i = 0; i < turns.size(); ++i) {
            TurnEval& t = turns[i];
            out[i].player = t.player;

            BoardStateT<Traits>& b = t.board;
            b.current_player = static_cast<int8_t>(t.player);
            uint16_t filled = 0;
            for (int p = 0; p < Traits::kNumPlayers; ++p)
                for (int c = 0; c < kNumColumns; ++c)
                    for (int r = 0; r < kNumRows; ++r)
                        if (b.cells[p][c][r] != kCellEmpty) ++filled;
            b.cells_filled = filled;

            GameContextT<Traits> ctx;
            rebuild_context_from_board<Traits>(b, ctx);
            GameStateT<Traits> state;
            state.board = b;

            scratch->dp_computed = false;
            scratch->evs_blended = false;
            solver_get_requests<Traits>(state, ctx, tables_, *scratch);
            int n = scratch->request_count;
            offset[i] = static_cast<int>(tensors.size() / Traits::kTensorSize);
            count[i]  = n;
            if (n == 0) continue;
            size_t base = tensors.size();
            tensors.resize(base + static_cast<size_t>(n) * Traits::kTensorSize, 0.0f);
            generate_tensor_batch<Traits>(b, ctx, t.player, scratch->requests, n,
                                          tables_, tensors.data() + base);
        }

        // One NN forward over every turn's afterstates.
        int total = static_cast<int>(tensors.size() / Traits::kTensorSize);
        std::vector<float> all_evs(static_cast<size_t>(total), 0.0f);
        if (total > 0) {
            torch::NoGradGuard no_grad;
            auto input  = make_nn_input(model, tensors.data(), total, device_);
            auto output = model.forward(input).to(torch::kCPU).contiguous();
            std::memcpy(all_evs.data(), output.template data_ptr<float>(),
                        sizeof(float) * static_cast<size_t>(total));
        }

        // Pass 2 — per-turn DP/accuracy/luck from the batched EVs (no inference).
        auto buffers = std::make_unique<SolverBuffers>();
        for (size_t i = 0; i < turns.size(); ++i) {
            if (count[i] == 0) continue;
            TurnEval& t = turns[i];

            GameContextT<Traits> ctx;
            rebuild_context_from_board<Traits>(t.board, ctx);
            GameStateT<Traits> state;
            state.board = t.board;

            buffers->dp_computed = false;
            buffers->evs_blended = false;
            solver_get_requests<Traits>(state, ctx, tables_, *buffers);
            // Request enumeration is deterministic, so the slice aligns 1:1.
            for (int j = 0; j < count[i]; ++j)
                buffers->evs[j] = static_cast<double>(all_evs[offset[i] + j]);

            score_turn_filled(t.board, ctx, t.player, t.initial_dice, t.reroll_dice,
                              t.hold_masks, t.placement, t.place_score, scale,
                              *buffers, out[i].hold_deltas, out[i].place_delta,
                              out[i].has_place, out[i].roll_lucks);
        }
    }

    // Compute per-seat luck for a finished (or in-progress) session using the
    // server's already-loaded play model. Reconstructs the board before each
    // turn by replaying the recorded move history, evaluates every turn's dice
    // with one batched NN pass, and averages per-roll luck → per-turn → per-seat.
    // out_seat_luck has one entry per player (NaN for seats that took no turns),
    // each a mean-zero index in [-100,100]. Returns false (with err) if there is
    // no model or the session is unknown.
    bool compute_game_luck(int session_id, std::vector<double>& out_seat_luck,
                           std::string& err) {
        out_seat_luck.assign(Traits::kNumPlayers,
                             std::numeric_limits<double>::quiet_NaN());
        if (!nn_model_) { err = "no model loaded"; return false; }

        Session copy;
        if (!get_session_copy(session_id, copy)) { err = "session not found"; return false; }
        if (copy.history.empty()) return true;  // nothing played yet

        // Reconstruct the board before each turn by replaying placements from a
        // fresh board carrying this game's column coefficients.
        BoardStateT<Traits> b;
        for (int p = 0; p < Traits::kNumPlayers; ++p)
            for (int c = 0; c < kNumColumns; ++c)
                for (int r = 0; r < kNumRows; ++r)
                    b.cells[p][c][r] = kCellEmpty;
        std::memcpy(b.coefficients, copy.state.board.coefficients, sizeof(b.coefficients));
        b.current_player = 0;
        b.cells_filled   = 0;

        GameContextT<Traits> ctx;
        init_context<Traits>(ctx, b);

        std::vector<TurnEval> turns;
        turns.reserve(copy.history.size());
        for (const auto& tr : copy.history) {
            TurnEval te;
            te.board  = b;                  // board *before* this turn
            te.player = tr.player;
            std::memcpy(te.initial_dice, tr.initial_dice, sizeof(te.initial_dice));
            te.hold_masks  = tr.hold_masks;
            te.reroll_dice = tr.dice_after_hold;
            te.placement   = tr.placement;
            te.place_score = tr.score;
            turns.push_back(std::move(te));

            apply_placement<Traits>(tr.player, tr.placement.column, tr.placement.row,
                                    tr.score, b, ctx);
        }

        std::vector<TurnResult> results;
        run_game_eval(*nn_model_, turns, results);

        // Aggregate: turn luck = mean of its rolls, seat luck = mean of its turns.
        std::vector<std::vector<double>> seat_turns(Traits::kNumPlayers);
        for (size_t i = 0; i < results.size(); ++i) {
            const auto& rl = results[i].roll_lucks;
            if (rl.empty()) continue;
            double m = 0.0;
            for (double x : rl) m += x;
            seat_turns[turns[i].player].push_back(m / static_cast<double>(rl.size()));
        }
        for (int p = 0; p < Traits::kNumPlayers; ++p) {
            if (seat_turns[p].empty()) continue;
            double s = 0.0;
            for (double x : seat_turns[p]) s += x;
            out_seat_luck[p] = s / static_cast<double>(seat_turns[p].size());
        }
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
            auto input  = make_nn_input(*nn_model_, out_tensor.data(), 1, device_);
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
    // Build a model input tensor from `count` afterstate rows laid out
    // contiguously as count × Traits::kTensorSize floats in `rows`.
    //
    // The tensor layout is append-only (see game_traits.h): an older tensor
    // version is a byte-exact prefix of the newer one. A model trained on an
    // older version therefore consumes only the first config().input_size
    // columns of each row. Slicing to that prefix here is what lets a
    // pre-Group-G checkpoint (input_size == kTensorSizeV1) keep running against
    // the current V2 engine (kTensorSize) instead of throwing a dimension
    // mismatch inside forward(). For a current-version model input_size ==
    // kTensorSize and the slice is a no-op.
    static torch::Tensor make_nn_input(const ProYamsNet& model, float* rows,
                                       int count, torch::Device device) {
        const int in = model.config().input_size;
        auto input = torch::from_blob(rows, {count, Traits::kTensorSize},
                                      torch::kFloat32);
        if (in < Traits::kTensorSize)
            input = input.slice(/*dim=*/1, /*start=*/0, /*end=*/in).contiguous();
        return input.to(device);
    }

    std::shared_ptr<Entry> get_entry(int id) const {
        std::lock_guard<std::mutex> lock(map_mutex_);
        auto it = sessions_.find(id);
        return it != sessions_.end() ? it->second : nullptr;
    }

    // Load (or fetch the cached) inference model for a checkpoint path. Models
    // are cached in eval_models_ so each distinct checkpoint loads at most once.
    // Returns nullptr and sets `err` on failure.
    std::shared_ptr<ProYamsNet> get_eval_model(const std::string& checkpoint_path,
                                               std::string& err) {
        std::lock_guard<std::mutex> lock(eval_models_mutex_);
        auto it = eval_models_.find(checkpoint_path);
        if (it != eval_models_.end()) return it->second;
        std::shared_ptr<ProYamsNet> model;
        try {
            ModelConfig cfg = ModelTrainer::config_from_checkpoint(checkpoint_path);
            ModelTrainer trainer(cfg, device_);
            trainer.load_weights(checkpoint_path);
            model = trainer.clone_for_inference(device_);
            model->to(device_);
            model->eval();
        } catch (const std::exception& e) {
            err = e.what();
            return nullptr;
        }
        eval_models_[checkpoint_path] = model;
        return model;
    }

    void eval_and_store_board_nn(Session& session, TurnRec& record) {
        if (!nn_model_) return;
        std::vector<float> buf(Traits::kTensorSize);
        generate_tensor<Traits>(session.state.board, session.ctx,
                                record.player, tables_, buf.data());
        torch::NoGradGuard no_grad;
        auto input  = make_nn_input(*nn_model_, buf.data(), 1, device_);
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
            auto input = make_nn_input(*nn_model_, session.tensor_buffer.data(),
                                       session.buffers.request_count, device_);
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
