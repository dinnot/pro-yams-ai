include(FetchContent)

# ---------------------------------------------------------------------------
# Google Test
# ---------------------------------------------------------------------------
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

# ---------------------------------------------------------------------------
# Google Benchmark
# ---------------------------------------------------------------------------
FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG v1.8.3
)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_subdirectory(engine)
add_subdirectory(solver)
add_subdirectory(heuristic)
add_subdirectory(model)
add_subdirectory(self_play)
add_subdirectory(training)
add_subdirectory(eval)
add_subdirectory(config)
add_subdirectory(ui)
add_subdirectory(benchmarks)
