project(bench_proj)



set(APPLICATION_NAME "benchBasic")

add_executable(${APPLICATION_NAME} src/benchBasic.cpp)

target_link_libraries(${APPLICATION_NAME} proj_lib armadillo chronoP)

target_compile_options(${APPLICATION_NAME} PRIVATE ${CXX_FLAGS_REQUIRED})
