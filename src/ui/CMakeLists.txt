include(FetchContent)

# ---------------------------------------------------------------------------
# cpp-httplib (header-only HTTP server)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG v0.15.3
)
FetchContent_MakeAvailable(httplib)

# ---------------------------------------------------------------------------
# nlohmann/json (header-only JSON library)
# ---------------------------------------------------------------------------
FetchContent_Declare(
    nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# ui_lib — session management, HTTP server, JSON serialization
# ---------------------------------------------------------------------------
add_library(ui_lib STATIC
    session_manager.cc
    server.cc
    json_serialization.cc
)
target_include_directories(ui_lib PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(ui_lib PUBLIC
    engine solver mc_solver heuristic model eval
    httplib::httplib
    nlohmann_json::nlohmann_json
    ${TORCH_LIBRARIES}
)

# ---------------------------------------------------------------------------
# pro_yams_ui — standalone web UI binary
# ---------------------------------------------------------------------------
add_executable(pro_yams_ui ui_main.cpp)
target_link_libraries(pro_yams_ui PRIVATE
    ui_lib config
    ${TORCH_LIBRARIES}
)

# ---------------------------------------------------------------------------
# pro_yams_play — public play-against-the-NN web app (separate port)
# ---------------------------------------------------------------------------
add_executable(pro_yams_play play_main.cpp)
target_link_libraries(pro_yams_play PRIVATE
    ui_lib config
    ${TORCH_LIBRARIES}
)
