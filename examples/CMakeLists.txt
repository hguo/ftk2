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
