cmake_minimum_required(VERSION 3.0.0)
project(flint-pack VERSION 0.1.0)

add_executable(flint-test test.cpp)
target_include_directories(flint-test PUBLIC /home/mahrud/.linuxbrew/include)
target_link_directories(flint-test PUBLIC /home/mahrud/.linuxbrew/lib /usr/lib64)
target_link_libraries(flint-test flint gmp openblas)

include(CTest)
enable_testing()

add_test(NAME mat11 COMMAND flint-test 32003 ../mat11.sms)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
