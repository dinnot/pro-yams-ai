# Distil library — distillation pipeline for the pro_yams_distil training mode.
add_library(distil STATIC
    replay_buffer.cc
    teacher_heuristic.cc
    teacher_nn.cc
    distil_worker.cc
    distil_orchestrator.cc
    distil_loop.cc
    distil_resume.cc
)
target_include_directories(distil PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(distil PUBLIC engine self_play heuristic solver model training eval ${TORCH_LIBRARIES})
