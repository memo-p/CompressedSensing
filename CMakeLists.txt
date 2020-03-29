cmake_minimum_required(VERSION 3.1)

# The project name
project(global)

#### COMPILER OPTIONS 
set(CMAKE_CXX_STANDARD 11)

set(CXX_FLAGS_REQUIRED -Wall -Wextra)

if(DCMAKE_BUILD_TYPE STREQUAL "Debug")
  set(CXX_FLAGS_REQUIRED ${CXX_FLAGS_REQUIRED} -g2)
endif()

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)


add_subdirectory(3rdparty/weighted_projection)
add_subdirectory(solver)

