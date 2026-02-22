cmake_minimum_required(VERSION 3.15)
project(FTK2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find ndarray (use -Dndarray_DIR=... or -DCMAKE_PREFIX_PATH=... to specify the path)
find_package(ndarray REQUIRED)

# FTK2 library
add_library(ftk2 INTERFACE)
target_include_directories(ftk2 INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ftk2 INTERFACE ndarray::ndarray)

# Tests
enable_testing()
add_subdirectory(tests)

# Examples
add_subdirectory(examples)
