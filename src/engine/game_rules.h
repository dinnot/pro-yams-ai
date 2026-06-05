#pragma once

// ---------------------------------------------------------------------------
// GameRules — process-wide gameplay rule configuration.
//
// These are toggles for optional rules that change game behavior but are fixed
// for the lifetime of a process (a training run or a play server each host a
// single variant + rule set). Set once at startup via set_game_rules(), then
// read-only during play — so reads are thread-safe without synchronization.
// ---------------------------------------------------------------------------
struct GameRules {
    // "Lucky Yams": if a player's very first roll of a turn is a Yams (5 dice of
    // the same face), they may write the MAXIMUM possible score in any one cell
    // they are currently allowed to fill — independent of the dice faces (the
    // row's theoretical max, still subject to the Golden Rule and SS/LS
    // interlock). Re-rolling forfeits the bonus. Default true; can be disabled
    // for training runs that prefer the simpler rule.
    bool yams_first_roll_bonus = true;
};

/// Read the active process-wide game rules.
const GameRules& game_rules();

/// Replace the active process-wide game rules. Call once at startup before any
/// games run; not safe to call concurrently with gameplay.
void set_game_rules(const GameRules& rules);
