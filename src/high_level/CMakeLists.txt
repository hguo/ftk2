# FTK2 High-Level API

# Find yaml-cpp
find_package(yaml-cpp QUIET)

if(NOT yaml-cpp_FOUND)
    message(WARNING "yaml-cpp not found. High-level API will not be built.")
    message(WARNING "To install yaml-cpp:")
    message(WARNING "  Ubuntu/Debian: sudo apt-get install libyaml-cpp-dev")
    message(WARNING "  macOS: brew install yaml-cpp")
    message(WARNING "  Or build from source: https://github.com/jbeder/yaml-cpp")
    return()
endif()

# High-level API library
add_library(ftk2_high_level
    tracking_config.cpp
    feature_tracker.cpp
    data_source.cpp
)

target_include_directories(ftk2_high_level
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(ftk2_high_level
    PUBLIC
        yaml-cpp
        ndarray::ndarray
)

# Set C++17 standard
target_compile_features(ftk2_high_level PUBLIC cxx_std_17)

# Installation
install(TARGETS ftk2_high_level
    EXPORT ftk2Targets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ftk2/high_level
    DESTINATION include/ftk2
    FILES_MATCHING PATTERN "*.hpp"
)
