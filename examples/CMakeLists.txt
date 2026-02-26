add_executable(ftk2_example main.cpp)
target_link_libraries(ftk2_example PRIVATE ftk2)

add_executable(ftk2_critical_point_2d critical_point_2d.cpp)
target_link_libraries(ftk2_critical_point_2d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_levelset_2d levelset_2d.cpp)
target_link_libraries(ftk2_levelset_2d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_levelset_3d levelset_3d.cpp)
target_link_libraries(ftk2_levelset_3d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_fiber_3d fiber_3d.cpp)
target_link_libraries(ftk2_fiber_3d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_tdgl_vortex_2d tdgl_vortex_2d.cpp)
target_link_libraries(ftk2_tdgl_vortex_2d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_parallel_vector_approximate parallel_vector_approximate.cpp)
target_link_libraries(ftk2_parallel_vector_approximate PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_exact_pv_simple exact_pv_simple.cpp)
target_link_libraries(ftk2_exact_pv_simple PRIVATE ftk2)

add_executable(ftk2_critical_point_unstructured_2d critical_point_unstructured_2d.cpp)
target_link_libraries(ftk2_critical_point_unstructured_2d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d unstructured_3d.cu)
target_link_libraries(ftk2_unstructured_3d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_synthetic unstructured_3d_synthetic.cu)
target_link_libraries(ftk2_unstructured_3d_synthetic PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_moving_maximum unstructured_3d_moving_maximum.cu)
target_link_libraries(ftk2_unstructured_3d_moving_maximum PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_single_trajectory unstructured_3d_single_trajectory.cu)
target_link_libraries(ftk2_unstructured_3d_single_trajectory PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_clean_test unstructured_3d_clean_test.cu)
target_link_libraries(ftk2_unstructured_3d_clean_test PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_from_vtu unstructured_3d_from_vtu.cu)
target_link_libraries(ftk2_unstructured_3d_from_vtu PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_stream_test unstructured_3d_stream_test.cu)
target_link_libraries(ftk2_unstructured_3d_stream_test PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_extruded_test unstructured_3d_extruded_test.cu)
target_link_libraries(ftk2_unstructured_3d_extruded_test PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_slow_trajectory unstructured_3d_slow_trajectory.cu)
target_link_libraries(ftk2_unstructured_3d_slow_trajectory PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_pentatope_debug unstructured_3d_pentatope_debug.cu)
target_link_libraries(ftk2_unstructured_3d_pentatope_debug PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

add_executable(ftk2_unstructured_3d_manifold_debug unstructured_3d_manifold_debug.cu)
target_link_libraries(ftk2_unstructured_3d_manifold_debug PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

if (CMAKE_CUDA_COMPILER)
    add_executable(ftk2_levelset_3d_cuda levelset_3d_cuda.cu)
    target_link_libraries(ftk2_levelset_3d_cuda PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

    add_executable(ftk2_levelset_2d_cuda levelset_2d_cuda.cu)
    target_link_libraries(ftk2_levelset_2d_cuda PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

    add_executable(ftk2_fiber_3d_cuda fiber_3d_cuda.cu)
    target_link_libraries(ftk2_fiber_3d_cuda PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

    add_executable(ftk2_critical_point_3d critical_point_3d.cpp)
    target_link_libraries(ftk2_critical_point_3d PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

    add_executable(ftk2_critical_point_3d_cuda critical_point_3d_cuda.cu)
    target_link_libraries(ftk2_critical_point_3d_cuda PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

    add_executable(ftk2_critical_point_2d_cuda critical_point_2d_cuda.cu)
    target_link_libraries(ftk2_critical_point_2d_cuda PRIVATE ftk2 VTK::IOXML VTK::CommonCore VTK::CommonDataModel)

    # Streaming CUDA example - demonstrates memory-efficient 2-timestep GPU execution
    if(yaml-cpp_FOUND)
        add_executable(ftk2_critical_point_2d_streaming_cuda critical_point_2d_streaming_cuda.cu)
        target_link_libraries(ftk2_critical_point_2d_streaming_cuda PRIVATE ftk2 yaml-cpp::yaml-cpp)
        message(STATUS "Building streaming CUDA example (requires yaml-cpp)")
    endif()
endif()

# High-level API demo (Layer 2)
find_package(yaml-cpp QUIET)
if(yaml-cpp_FOUND)
    add_executable(ftk2_high_level_api_demo high_level_api_demo.cpp)
    target_link_libraries(ftk2_high_level_api_demo PRIVATE ftk2_high_level ftk2)

    add_executable(ftk2_high_level_tracking_test high_level_tracking_test.cpp)
    target_link_libraries(ftk2_high_level_tracking_test PRIVATE ftk2_high_level ftk2)

    add_executable(ftk2_approx_pv_highlevel_example approx_pv_highlevel_example.cpp)
    target_link_libraries(ftk2_approx_pv_highlevel_example PRIVATE ftk2_high_level ftk2)

    message(STATUS "Building high-level API demo, tracking test, and ApproxPV example")
else()
    message(STATUS "Skipping high-level API demo (yaml-cpp not found)")
endif()

