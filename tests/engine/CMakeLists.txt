add_executable(engine_tests
    engine_smoke_test.cc
    scoring_test.cc
    legal_moves_test.cc
    placement_test.cc
    duel_test.cc
    solver_tables_test.cc
    init_test.cc
    # Task 03 tests added below:
    rng_test.cc
    game_flow_test.cc
    turbo_test.cc
    full_game_test.cc
    # 2v2 migration Task 1:
    game_traits_test.cc
    # 2v2 migration Task 2:
    golden_rule_teammate_test.cc
    # 2v2 migration Task 3:
    duel_2v2_test.cc
    # Lucky-Yams first-roll bonus:
    yams_bonus_test.cc
)
target_link_libraries(engine_tests PRIVATE engine gtest_main)
add_test(NAME engine_tests COMMAND engine_tests)
set_tests_properties(engine_tests PROPERTIES
    ENVIRONMENT "OMP_WAIT_POLICY=passive;OMP_NUM_THREADS=1"
)

# Task 06: probability table and tensor tests (require solver for PrecomputedTables)
add_executable(tensor_tests
    probability_table_test.cc
    tensor_test.cc
    tensor_batch_test.cc
    context_rebuild_test.cc
    # 2v2 migration Task 5:
    tensor_rotation_test.cc
)
target_link_libraries(tensor_tests PRIVATE solver heuristic engine gtest_main)
add_test(NAME tensor_tests COMMAND tensor_tests)
set_tests_properties(tensor_tests PROPERTIES
    ENVIRONMENT "OMP_WAIT_POLICY=passive;OMP_NUM_THREADS=1"
)
