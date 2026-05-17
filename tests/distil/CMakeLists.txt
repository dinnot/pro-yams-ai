add_executable(distil_tests
    shuffle_queue_test.cc
    teacher_heuristic_test.cc
    teacher_nn_test.cc
    distil_loop_test.cc
    convergence_test.cc
    final_report_test.cc
    distil_resume_test.cc
)
target_include_directories(distil_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(distil_tests PRIVATE
    distil engine self_play heuristic solver model training
    gtest_main
    ${TORCH_LIBRARIES}
)
add_test(NAME distil_tests COMMAND distil_tests)
# distil_loop_test pulls in libtorch — share the same CTest resource lock
# and MKL env vars as the other libtorch test suites to avoid TBB hangs.
set_tests_properties(distil_tests PROPERTIES
    RESOURCE_LOCK libtorch_cpu
    ENVIRONMENT "MKL_THREADING_LAYER=SEQUENTIAL;OMP_NUM_THREADS=1;OMP_WAIT_POLICY=passive"
)
