cmake_minimum_required(VERSION 3.18)
project(GenreRevealPartyCUDA CUDA CXX)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 52)

set(SOURCE_FILES main.cu)
add_executable(GenreRevealPartyCUDA ${SOURCE_FILES})
target_include_directories(GenreRevealPartyCUDA PUBLIC "../include")