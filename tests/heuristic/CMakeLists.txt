add_executable(heuristic_tests
    heuristic_bot_test.cc
)
target_link_libraries(heuristic_tests PRIVATE heuristic gtest_main)
add_test(NAME heuristic_tests COMMAND heuristic_tests)
set_tests_properties(heuristic_tests PROPERTIES
    ENVIRONMENT "OMP_WAIT_POLICY=passive;OMP_NUM_THREADS=1"
)
