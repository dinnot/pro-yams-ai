# Eval library — depends on engine, solver, model, and heuristic.
add_library(eval STATIC
    evaluator.cc
    eval_logging.cc
)
target_include_directories(eval PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(eval PUBLIC engine solver model heuristic ${TORCH_LIBRARIES})
