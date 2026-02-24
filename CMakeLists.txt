cmake_minimum_required(VERSION 3.15)
project(FTK2 VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package (CUDAToolkit)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    add_definitions(-DFTK_HAVE_CUDA=1)
endif()

# Find ndarray (use -Dndarray_DIR=... or -DCMAKE_PREFIX_PATH=... to specify the path)
find_package(ndarray REQUIRED)

find_package(OpenMP REQUIRED)

# FTK2 library
add_library(ftk2 STATIC 
    src/core/mesh.cpp
)
target_include_directories(ftk2 PUBLIC 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(ftk2 PUBLIC ndarray::ndarray OpenMP::OpenMP_CXX)

# Tests
enable_testing()
add_subdirectory(tests)

# Examples
add_subdirectory(examples)
