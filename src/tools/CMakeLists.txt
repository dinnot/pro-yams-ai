# Golden test corpus generator.
#
# Deliberately depends on ONLY the engine + heuristic (+ its solver) libraries,
# never libtorch. This keeps the generator a small, fast, standalone binary that
# can regenerate the entire corpus with a single command. The heuristic bot
# supplies the "strong/realistic" action policy without dragging in the NN stack.
#
# nlohmann_json::nlohmann_json is made available by src/ui (added before this
# subdirectory in the top-level CMakeLists.txt).
add_executable(golden_gen golden_gen.cc)
target_link_libraries(golden_gen PRIVATE engine heuristic solver nlohmann_json::nlohmann_json)
target_include_directories(golden_gen PRIVATE ${CMAKE_SOURCE_DIR}/src)
