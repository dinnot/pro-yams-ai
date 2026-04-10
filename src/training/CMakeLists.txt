# Training library — depends on model and self_play.
add_library(training STATIC
    replay_buffer.cc
    logging.cc
    training_loop.cc
    resume.cc
)
target_include_directories(training PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(training PUBLIC model self_play ${TORCH_LIBRARIES})
