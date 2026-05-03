# Solver library — depends on engine only (no ML).
add_library(solver STATIC
    precomputed_tables.cc
    solver.cc
    dp_tables.cc
    dp_tables_compute.cc
)
target_include_directories(solver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(solver PUBLIC engine)

# OpenMP for the DP table compute kernels.
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(solver PUBLIC OpenMP::OpenMP_CXX)
endif()

# MC rollout bot — depends on solver + model + eval (has ML dependency).
add_library(mc_solver STATIC
    mc_bot.cc
)
target_include_directories(mc_solver PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(mc_solver PUBLIC solver model eval heuristic ${TORCH_LIBRARIES})
