# Engine library — zero ML dependencies (no libtorch linkage).
# Any accidental torch inclusion will fail at link time, enforcing the boundary.
add_library(engine STATIC
    solver_tables.cc
    scoring.cc
    legal_moves.cc
    placement.cc
    duel.cc
    board_init.cc
    rng.cc
    game_flow.cc    # Added in Task 03
    tensor.cc       # Added in Task 06
    context_rebuild.cc  # Added in Task 14
)
target_include_directories(engine PUBLIC ${CMAKE_SOURCE_DIR}/src)
