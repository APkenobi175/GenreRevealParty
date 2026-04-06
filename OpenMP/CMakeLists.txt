cmake_minimum_required(VERSION 3.13)
project(GenreRevealPartyOpenMP)

set(SOURCE_FILES mainOpenMP.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
add_executable(GenreRevealPartyOpenMP ${SOURCE_FILES})
target_include_directories(GenreRevealPartyOpenMP PUBLIC "../include")
