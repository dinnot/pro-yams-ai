add_executable(eval_tests
    eval_game_test.cc
    evaluator_test.cc
    eval_logging_test.cc
)
target_link_libraries(eval_tests PRIVATE
    eval model solver engine
    gtest_main
    ${TORCH_LIBRARIES}
)
add_test(NAME eval_tests COMMAND eval_tests)
set_tests_properties(eval_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
