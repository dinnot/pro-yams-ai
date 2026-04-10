add_executable(ui_tests
    session_manager_test.cc
    json_test.cc
)
target_link_libraries(ui_tests PRIVATE
    ui_lib config
    gtest_main
    ${TORCH_LIBRARIES}
)
add_test(NAME ui_tests COMMAND ui_tests)
set_tests_properties(ui_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
