# Self-play library — depends on engine, solver, and model.
add_library(self_play STATIC
    game_queues.cc
    worker.cc
    coordinator.cc
    orchestrator.cc
    training_data.cc
)
target_include_directories(self_play PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(self_play PUBLIC engine solver model heuristic)
