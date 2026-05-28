add_executable(training_tests
    replay_buffer_test.cc
    logging_test.cc
    training_loop_test.cc
    training_loop_2v2_test.cc
)
target_link_libraries(training_tests PRIVATE
    training eval model self_play solver engine
    gtest_main
    ${TORCH_LIBRARIES}
)
add_test(NAME training_tests COMMAND training_tests)
set_tests_properties(training_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
