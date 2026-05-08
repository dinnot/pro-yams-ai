add_executable(self_play_tests
    queue_test.cc
    worker_test.cc
    coordinator_test.cc
    integration_test.cc
    trajectory_test.cc
    batch_manager_test.cc
)
target_link_libraries(self_play_tests
    PRIVATE self_play engine solver model gtest_main ${TORCH_LIBRARIES})
add_test(NAME self_play_tests COMMAND self_play_tests)
# Use sequential MKL threading to avoid TBB init hangs when other PyTorch
# processes ran recently (shared TBB state teardown causes 60s+ first-call delay).
# RESOURCE_LOCK prevents concurrent execution with other libtorch tests.
set_tests_properties(self_play_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
