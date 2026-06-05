cmake_minimum_required(VERSION 3.24)
project(pro_yams_ai LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# Build type configuration
# ---------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -fsanitize=address,undefined")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -march=native -DNDEBUG")

# ---------------------------------------------------------------------------
# libtorch integration
# ---------------------------------------------------------------------------
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests and benchmarks" ON)

# ---------------------------------------------------------------------------
# Subdirectories (libraries)
# ---------------------------------------------------------------------------
add_subdirectory(src/engine)
add_subdirectory(src/solver)
add_subdirectory(src/model)
add_subdirectory(src/heuristic)
add_subdirectory(src/self_play)
add_subdirectory(src/training)
add_subdirectory(src/distil)
add_subdirectory(src/eval)
add_subdirectory(src/ui)
add_subdirectory(src/config)
add_subdirectory(src/tools)

# ---------------------------------------------------------------------------
# Main executable
# ---------------------------------------------------------------------------
add_executable(pro_yams_ai src/main.cpp)
target_link_libraries(pro_yams_ai
    PRIVATE engine solver model heuristic self_play training eval config
    PRIVATE ${TORCH_LIBRARIES}
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
