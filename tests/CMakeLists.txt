# Individual test executables for faster compilation
add_executable(ftk2_test_zero_crossing test_zero_crossing.cpp)
target_link_libraries(ftk2_test_zero_crossing PRIVATE ftk2)
add_test(NAME test_zero_crossing COMMAND ftk2_test_zero_crossing)

add_executable(ftk2_test_mesh test_mesh.cpp)
target_link_libraries(ftk2_test_mesh PRIVATE ftk2)
add_test(NAME test_mesh COMMAND ftk2_test_mesh)

add_executable(ftk2_test_engine test_engine.cpp)
target_link_libraries(ftk2_test_engine PRIVATE ftk2)
add_test(NAME test_engine COMMAND ftk2_test_engine)

add_executable(ftk2_test_sos test_sos.cpp)
target_link_libraries(ftk2_test_sos PRIVATE ftk2)
add_test(NAME test_sos COMMAND ftk2_test_sos)

add_executable(ftk2_test_parity test_parity.cu)
target_link_libraries(ftk2_test_parity PRIVATE ftk2)
add_test(NAME test_parity COMMAND ftk2_test_parity)

add_executable(ftk2_test_examples test_examples.cpp)
target_link_libraries(ftk2_test_examples PRIVATE ftk2)
add_test(NAME test_examples COMMAND ftk2_test_examples)

add_executable(ftk2_test_unstructured test_unstructured.cpp)
target_link_libraries(ftk2_test_unstructured PRIVATE ftk2)
add_test(NAME test_unstructured COMMAND ftk2_test_unstructured)

add_executable(ftk2_test_3d_cp test_3d_cp.cpp)
target_link_libraries(ftk2_test_3d_cp PRIVATE ftk2)
add_test(NAME test_3d_cp COMMAND ftk2_test_3d_cp)

add_executable(ftk2_test_exactpv test_exactpv.cpp)
target_link_libraries(ftk2_test_exactpv PRIVATE ftk2)
add_test(NAME test_exactpv COMMAND ftk2_test_exactpv)
