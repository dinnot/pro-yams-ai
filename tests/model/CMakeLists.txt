# Model tests — Task 07: neural network tests
add_executable(model_tests
    model_test.cc
    inference_test.cc
    trainer_test.cc
    init_test.cc
)
target_link_libraries(model_tests PRIVATE model engine gtest_main ${TORCH_LIBRARIES})
add_test(NAME model_tests COMMAND model_tests)
set_tests_properties(model_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
