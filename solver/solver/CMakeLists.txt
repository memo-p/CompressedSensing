cmake_minimum_required(VERSION 3.1)

# The project name
project (solver_lib)

# Header only library, therefore INTERFACE
add_library(solver_lib INTERFACE)

# INTERFACE targets only have INTERFACE properties
target_include_directories(solver_lib INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include)


