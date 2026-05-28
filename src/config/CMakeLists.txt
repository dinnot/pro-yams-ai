# Config library — parses YAML using yaml-cpp.
include(FetchContent)

FetchContent_Declare(
    yaml-cpp
    GIT_REPOSITORY https://github.com/jbeder/yaml-cpp.git
    GIT_TAG 0.8.0
)
FetchContent_MakeAvailable(yaml-cpp)

add_library(config STATIC
    config_loader.cc
    config_validator.cc
    config_printer.cc
)
target_include_directories(config PUBLIC ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(config PUBLIC training distil yaml-cpp::yaml-cpp)
