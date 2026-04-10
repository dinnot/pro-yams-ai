add_executable(config_tests
    config_loader_test.cc
    config_validator_test.cc
    config_printer_test.cc
)
target_link_libraries(config_tests PRIVATE
    config
    gtest_main
    ${TORCH_LIBRARIES}
)
add_test(NAME config_tests COMMAND config_tests)
set_tests_properties(config_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
