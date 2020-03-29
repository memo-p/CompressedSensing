project(bench_proj)


set(APP_HEADERS
    src/analyze.hpp
    src/benchResults.hpp
    src/benchs.hpp
    src/candes.hpp
    src/lasso.hpp
    src/lq.hpp
)

set(APP_SOURCES
    src/main.cpp
)

set(APPLICATION_NAME "benchSolver")

add_executable(${APPLICATION_NAME} ${APP_SOURCES} ${APP_HEADERS})

target_link_libraries(${APPLICATION_NAME} solver_lib proj_lib armadillo chronoP)

target_compile_options(${APPLICATION_NAME} PRIVATE ${CXX_FLAGS_REQUIRED})

