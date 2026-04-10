# Model library — depends on libtorch and engine (for kTensorSize).
add_library(model STATIC
    pro_yams_net.cc
    inference.cc
    trainer.cc
)
target_include_directories(model PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(model PUBLIC ${TORCH_LIBRARIES} engine)
