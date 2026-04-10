add_executable(engine_bench engine_bench.cc)
target_link_libraries(engine_bench PRIVATE engine benchmark::benchmark_main)

add_executable(solver_bench solver_bench.cc)
target_link_libraries(solver_bench PRIVATE heuristic solver benchmark::benchmark_main)

add_executable(tensor_bench tensor_bench.cc)
target_link_libraries(tensor_bench PRIVATE heuristic solver engine benchmark::benchmark_main)

add_executable(model_bench model_bench.cc)
target_link_libraries(model_bench PRIVATE model engine benchmark::benchmark_main ${TORCH_LIBRARIES})

add_executable(self_play_bench self_play_bench.cc)
target_link_libraries(self_play_bench PRIVATE self_play model solver engine benchmark::benchmark_main ${TORCH_LIBRARIES})

add_executable(training_bench training_bench.cc)
target_link_libraries(training_bench PRIVATE training model self_play solver engine benchmark::benchmark_main ${TORCH_LIBRARIES})

add_executable(eval_bench eval_bench.cc)
target_link_libraries(eval_bench PRIVATE eval model solver engine benchmark::benchmark_main ${TORCH_LIBRARIES})
