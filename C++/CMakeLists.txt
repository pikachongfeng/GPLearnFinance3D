cmake_minimum_required(VERSION 3.27.0)
project(cpp_fitness)
set(CMAKE_CXX_STANDARD 11)

if(NOT CMAKE_BUILD_TYPE)
set(CMAKE_BUILD_TYPE Debug)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -msse2 -msse3 -msse4 -fopenmp")

add_definition(-DEXPORT)

include_directories(include)
include_directories(eigen-3.4.0)

add_subdirectory(pybind11)

pybind11_add_module(pybind11_eigen ./src/fitness.cpp)

find_package(OpenMP REQUIRED)
target_link_libraries(pybind11_eigen PRIVATE OpenMP::OpenMP_CXX)
