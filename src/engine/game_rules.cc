#include "engine/game_rules.h"

namespace {
GameRules g_game_rules{};
}  // namespace

const GameRules& game_rules() { return g_game_rules; }

void set_game_rules(const GameRules& rules) { g_game_rules = rules; }
