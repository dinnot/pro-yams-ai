# Heuristic library — depends on engine and solver (no ML).
add_library(heuristic STATIC
    heuristic_bot.cc
    placeholder.cc
)
target_include_directories(heuristic PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(heuristic PUBLIC engine solver)
