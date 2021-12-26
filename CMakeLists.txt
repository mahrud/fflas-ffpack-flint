cmake_minimum_required(VERSION 3.0.0)
project(flint-pack VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(test-flint test.cpp modular-flint.h) # modular-flint.inc
target_include_directories(test-flint PUBLIC /home/mahrud/.linuxbrew/include)
target_link_directories(test-flint PUBLIC /home/mahrud/.linuxbrew/lib /usr/lib64)
target_link_libraries(test-flint flint gmp openblas)
add_test(NAME mat11 COMMAND test-flint 32003 ../mat11.sms)

add_executable(bench-flint bench/rank.cpp modular-flint.h) # modular-flint.inc
target_compile_definitions(bench-flint PUBLIC USE_FLINT_RANK)
target_include_directories(bench-flint PUBLIC /home/mahrud/.linuxbrew/include)
target_link_directories(bench-flint PUBLIC /home/mahrud/.linuxbrew/lib /usr/lib64)
target_link_libraries(bench-flint flint givaro gmp openblas)
add_test(NAME flint       COMMAND bench-flint -f Y)
add_test(NAME flint-fflas COMMAND bench-flint)

add_executable(bench-givaro bench/rank.cpp)
target_include_directories(bench-givaro PUBLIC /home/mahrud/.linuxbrew/include)
target_link_directories(bench-givaro PUBLIC /home/mahrud/.linuxbrew/lib /usr/lib64)
target_link_libraries(bench-givaro flint givaro gmp openblas)
add_test(NAME givaro      COMMAND bench-givaro)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
