add_executable(ftk2_tests 
    test_main.cpp
    test_zero_crossing.cpp
    test_mesh.cpp
    test_engine.cpp
)
target_link_libraries(ftk2_tests PRIVATE ftk2)
add_test(NAME ftk2_tests COMMAND ftk2_tests)
