add_executable(solver_tests
    dice_state_test.cc
    hold_mask_test.cc
    transition_test.cc
    score_table_test.cc
    solver_request_test.cc
    solver_resolve_test.cc
    softmax_test.cc
    dp_tables_test.cc
    middle_dp_test.cc
)
target_link_libraries(solver_tests PRIVATE solver gtest_main)
add_test(NAME solver_tests COMMAND solver_tests)
set_tests_properties(solver_tests PROPERTIES
    ENVIRONMENT "OMP_WAIT_POLICY=passive;OMP_NUM_THREADS=1"
)

# Heavy DP monotonicity test — separate target since it computes/loads the
# full ~2 GB DP cache. Not registered with CTest by default; run manually.
add_executable(dp_tables_monotonicity_test
    dp_tables_monotonicity_test.cc
)
target_link_libraries(dp_tables_monotonicity_test PRIVATE solver gtest_main)

# MC bot tests (requires libtorch for NN model)
add_executable(mc_bot_tests
    mc_bot_test.cc
)
target_link_libraries(mc_bot_tests PRIVATE
    mc_solver solver heuristic model eval engine
    gtest_main
    ${TORCH_LIBRARIES}
)
add_test(NAME mc_bot_tests COMMAND mc_bot_tests)
set_tests_properties(mc_bot_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
