cmake_minimum_required(VERSION 3.2)
project(BP)

set(CMAKE_CXX_STANDARD 11)

set(SOURCE_FILES main.cpp BPnet.cpp BPnet.h)
add_executable(BP ${SOURCE_FILES})
target_link_libraries(BP armadillo)
